import torch
import numpy as np
import sounddevice as sd
import logging
import threading
from typing import Optional, Dict, List

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel
from PySide6.QtCore import Qt, QTimer, QSignalBlocker, Signal

# ANode Imports
from base import Node, IClockProvider, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# Global lock for PortAudio (sounddevice) stream operations to prevent Segfaults
_sounddevice_lock = threading.Lock()

# ==============================================================================
# High-Performance Ring Buffer
# ==============================================================================


class AudioRingBuffer:
    """
    Lock-free single-producer / single-consumer ring buffer.

    Each counter is written by exactly one thread:
      - write_count: written only by the producer (engine thread for output,
        hardware callback for input).
      - read_count:  written only by the consumer (hardware callback for output,
        engine thread for input).

    Under CPython's GIL, integer reads/writes are atomic at the Python level, so
    no mutex is needed for the normal read/write paths.

    clear() is the one exception: it resets both counters and zeroes the storage.
    It is only called during stream teardown (via _stop_stream_sync, which runs on
    an NRT pool thread after the stream has been stopped), so both the producer and
    consumer have ceased activity by the time it executes. The one-call-site
    comment in _stop_stream_sync documents this assumption.
    """

    def __init__(self, capacity_blocks=32, block_size=BLOCK_SIZE, channels=CHANNELS):
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.channels = channels
        # Flat storage: rows are frames, columns are channels.
        # Sliced by block boundaries: block i lives at rows [i*block_size, (i+1)*block_size).
        self.storage = np.zeros(
            (capacity_blocks * block_size, channels), dtype=np.float32
        )
        # Monotonic counters. Each is written by its owning thread only.
        self.write_count = 0  # producer thread
        self.read_count = 0   # consumer thread

    def write(self, data: np.ndarray) -> bool:
        """Called by the producer thread only."""
        if self.write_count - self.read_count >= self.capacity_blocks:
            return False  # overrun — caller receives silence or drops block
        start = (self.write_count % self.capacity_blocks) * self.block_size
        frames = min(self.block_size, data.shape[0])
        self.storage[start:start + frames, :] = data[:frames]
        self.write_count += 1
        return True

    def read(self, outdata: np.ndarray) -> bool:
        """Called by the consumer thread only."""
        if self.write_count - self.read_count < 1:
            return False  # underrun
        start = (self.read_count % self.capacity_blocks) * self.block_size
        outdata[:] = self.storage[start:start + self.block_size]
        self.read_count += 1
        return True

    def clear(self):
        """
        Reset the buffer. Only safe to call after the stream has stopped and
        neither the producer nor the consumer is active. Resetting both counters
        to 0 is correct here because the buffer is about to be reused for a
        new stream; any in-flight producer/consumer activity would already have
        been terminated by _stop_stream_sync before this is reached.
        """
        self.write_count = 0
        self.read_count = 0
        self.storage.fill(0)


# ==============================================================================
# Audio Device Management
# ==============================================================================


class AudioDeviceManager:
    @staticmethod
    def get_compatible_devices(is_input: bool, target_rate: int = SAMPLE_RATE) -> List[Dict]:
        with _sounddevice_lock:
            devices = []
            try:
                host_apis = sd.query_hostapis()
                all_devices = sd.query_devices()

                for idx, dev in enumerate(all_devices):
                    max_ch = dev.get("max_input_channels" if is_input else "max_output_channels", 0)
                    if max_ch <= 0:
                        continue

                    # Strict Sample Rate Check
                    try:
                        if is_input:
                            sd.check_input_settings(
                                device=idx, channels=min(2, max_ch), samplerate=target_rate, dtype="float32"
                            )
                        else:
                            sd.check_output_settings(
                                device=idx, channels=min(2, max_ch), samplerate=target_rate, dtype="float32"
                            )
                    except Exception:
                        continue

                    api_index = dev["hostapi"]
                    api_name = host_apis[api_index]["name"] if api_index < len(host_apis) else "Unknown"

                    dev_info = dict(dev)
                    dev_info["id"] = idx
                    dev_info["display_name"] = f"{dev['name']} [{api_name}]"
                    devices.append(dev_info)

            except Exception as e:
                logger.error(f"Device Query Error: {e}")

            return devices

    @staticmethod
    def get_default_id(is_input: bool) -> int:
        with _sounddevice_lock:
            try:
                return sd.default.device[0] if is_input else sd.default.device[1]
            except:
                return -1


# ==============================================================================
# Unified Audio Node Logic
# ==============================================================================


class BaseAudioDeviceNode(Node):
    category, label = "I/O", "Base Audio Device"
    description = (
        "Base class for hardware audio device nodes. Manages device selection, "
        "a lock-free SPSC ring buffer between the engine and the PortAudio "
        "callback thread, and NRT-based stream open/teardown with request-ID "
        "epoch checks. Not instantiated directly; use AudioDeviceInput or "
        "AudioDeviceOutput."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_int_param("device_index", -1, min_v=-1, max_v=999,
                           help="Hardware device index (-1 = system default). Changes restart the stream off the audio path.")
        self.ring_buffer = AudioRingBuffer(capacity_blocks=32)
        self.stream: Optional[sd.Stream] = None
        self._device_state = {"active": False, "status": "Inactive", "latency": 0.0, "idx": -1}
        # Request identity for stream lifecycle operations. Each start/stop
        # request bumps the counter; an NRT task carrying an older id is a
        # superseded request and skips its stream work. Combined with the
        # global _sounddevice_lock this guarantees only one owner manipulates
        # the PortAudio stream at a time, in request order.
        self._stream_request_id = 0

    def _next_stream_request_id(self):
        # Control thread only (on_ui_param_change / start / stop callbacks).
        self._stream_request_id += 1
        return self._stream_request_id

    def _start_stream(self, StreamClass, callback, channels=None):
        self.submit_nrt(self._start_stream_sync, StreamClass, callback, channels,
                        self._next_stream_request_id())

    def _start_stream_sync(self, StreamClass, callback, channels=None, req_id=None):
        with _sounddevice_lock:
            if req_id is not None and req_id != self._stream_request_id:
                return  # superseded by a newer start/stop request
            self._stop_stream_sync_internal()

            # KEY CHANGE: Ensure we are reading the synced value
            requested_idx = self.params["device_index"].value
            target_idx = requested_idx

            # 1. Resolve Default
            if target_idx == -1:
                try:
                    target_idx = sd.default.device[0 if StreamClass == sd.InputStream else 1]
                except Exception:
                    self._device_state = {"active": False, "status": "No Default Device", "latency": 0.0, "idx": -2}
                    return

            # 2. Query Capabilities
            try:
                info = sd.query_devices(target_idx)
            except Exception as e:
                self._device_state = {"active": False, "status": "Device Not Found", "latency": 0.0, "idx": -2}
                return

            # 3. Channel Logic (Clamp to Hardware Max)
            desired_channels = channels or CHANNELS
            hw_max = info.get("max_input_channels" if StreamClass == sd.InputStream else "max_output_channels", 0)
            actual_channels = min(desired_channels, hw_max)

            if actual_channels < 1:
                self._device_state = {"active": False, "status": "Device has 0 channels", "latency": 0.0, "idx": -2}
                return

            # 4. Attempt Stream Open
            try:
                self.stream = StreamClass(
                    device=target_idx,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    channels=actual_channels,
                    dtype="float32",
                    callback=callback,
                )
                try:
                    self.stream.start()
                except Exception as e:
                    if self.stream:
                        self.stream.close()
                        self.stream = None
                    raise e

                ch_str = "Mono" if actual_channels == 1 else f"{actual_channels}ch"
                self._device_state = {
                    "active": True,
                    "status": f"{info['name']} ({ch_str})",
                    "latency": self.stream.latency * 1000.0,
                    "idx": target_idx,
                }

            except Exception as e:
                logger.error(f"Stream Open Failed: {e}")
                self._device_state = {"active": False, "status": f"Error: {str(e)[:20]}...", "latency": 0.0, "idx": -2}

    def _stop_stream(self):
        self.submit_nrt(self._stop_stream_sync, self._next_stream_request_id())

    def _stop_stream_sync(self, req_id=None):
        with _sounddevice_lock:
            if req_id is not None and req_id != self._stream_request_id:
                return  # superseded by a newer start/stop request
            self._stop_stream_sync_internal()

    def _stop_stream_sync_internal(self):
        self._device_state = {"active": False, "status": "Inactive", "latency": 0.0, "idx": -1}
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.ring_buffer.clear()

    def stop(self):
        self._stop_stream()

    def remove(self):
        self._stop_stream()

    def get_telemetry(self) -> dict:
        state = self._device_state
        msg = state["status"]
        if state["active"]:
            msg += f" [{state['latency']:.0f}ms]"

        return {"status": msg, "actual_device_idx": state["idx"]}

    def on_ui_param_change(self, param_name):
        if param_name == "device_index":
            # The engine committed the staged value (set()+sync()) before
            # invoking this callback (AGENTS.md §5), so
            # params["device_index"].value is already the new selection.

            # Device change is an explicit control operation: request a stream
            # restart on the NRT path (stop old stream -> open new stream ->
            # publish state). It never blocks and never touches the stream
            # directly from this callback; the request-id check discards any
            # superseded restart still queued on the pool.
            self.start()


class AudioDeviceInput(BaseAudioDeviceNode):
    category, label = "I/O", "Audio Device Input"
    description = (
        "Captures live audio from a hardware input device via PortAudio. The "
        "hardware callback thread writes into a lock-free ring buffer that the "
        "engine consumes per block; underruns produce silence. Mono hardware "
        "inputs are duplicated to stereo."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.out = self.add_output("out", help="Live hardware input as a stereo signal (silence on underrun).")
        # Pre-allocated numpy scratch buffer for zero-allocation process()
        self._numpy_scratch = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

    def start(self):
        self._start_stream(sd.InputStream, self._callback)

    def _callback(self, indata, frames, time, status):
        # Handle Mono -> Stereo upmix if necessary
        if indata.shape[1] == self.ring_buffer.channels:
            self.ring_buffer.write(indata)
        elif indata.shape[1] == 1 and self.ring_buffer.channels == 2:
            expanded = np.hstack([indata, indata])
            self.ring_buffer.write(expanded)
        else:
            min_ch = min(indata.shape[1], self.ring_buffer.channels)
            temp = np.zeros((frames, self.ring_buffer.channels), dtype=np.float32)
            temp[:, :min_ch] = indata[:, :min_ch]
            self.ring_buffer.write(temp)

    def process(self):
        if self.ring_buffer.read(self._numpy_scratch):
            self.out.buffer.copy_(torch.from_numpy(self._numpy_scratch.T))
        else:
            self.out.buffer.zero_()


class AudioDeviceOutput(BaseAudioDeviceNode, IClockProvider):
    category, label = "I/O", "Audio Device Output"
    description = (
        "Plays the graph's audio through a hardware output device and acts as "
        "the engine's clock provider when set as master: the hardware callback "
        "drives block processing. Uses a lock-free ring buffer; underruns "
        "output silence. Device changes restart the stream off the audio path."
    )

    def __init__(self, name=""):
        BaseAudioDeviceNode.__init__(self, name)
        IClockProvider.__init__(self)
        self.inp = self.add_input("audio_in", help="Signal to play through the selected output device.")
        self._tick_callback = None

        # PRE-ALLOCATION: Create a Numpy array in Interleaved format [Block, Channels]
        # We will copy into this, avoiding new object creation every frame.
        self._scratch_buffer = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

    def start_clock(self, tick_callback):
        self._tick_callback = tick_callback
        self.start()

    def stop_clock(self):
        self._tick_callback = None
        self.stop()

    def start(self):
        self._start_stream(sd.OutputStream, self._callback)

    def _callback(self, outdata, frames, time, status):
        if self._tick_callback:
            self._tick_callback()

        success = self.ring_buffer.read(self._scratch_buffer)

        if not success:
            outdata.fill(0)
            return

        hw_channels = outdata.shape[1]

        if hw_channels == CHANNELS:
            outdata[:] = self._scratch_buffer
        elif hw_channels == 1:
            outdata[:, 0] = (self._scratch_buffer[:, 0] + self._scratch_buffer[:, 1]) * 0.5
        else:
            k = min(hw_channels, CHANNELS)
            outdata[:, :k] = self._scratch_buffer[:, :k]
            # Anti-ghosting: channels beyond the engine format must still be
            # written every callback; PortAudio buffers are not zeroed.
            if hw_channels > k:
                outdata[:, k:] = 0.0

    def process(self):
        # 1. Get Tensor (on CPU)
        tensor_data = self.inp.get_tensor()
        if tensor_data.device.type != "cpu":
            tensor_data = tensor_data.cpu()

        # 2. Copy to Numpy Scratch Buffer (Handling Layout Conversion)
        # PyTorch [2, 512] -> Numpy [512, 2]
        # We use 'out=' to force writing into existing memory

        # Option A: If tensor is strictly [2, 512]
        # torch.t() creates a transposed view, .numpy() creates a view of that.
        # copyto is the actual data movement (interleaving).
        np.copyto(self._scratch_buffer, tensor_data.numpy().T)

        # 3. Write to Ring Buffer
        # Now we are passing a persistent pointer, not a new object
        self.ring_buffer.write(self._scratch_buffer)


# ==============================================================================
# UI Widgets
# ==============================================================================


class AudioDeviceWidget(QWidget):
    devicesQueried = Signal(list)

    def __init__(self, proxy, is_input):
        super().__init__()
        self.proxy, self.is_input = proxy, is_input

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.activated.connect(self._on_combo_user_action)

        btn = QPushButton("⟳")
        btn.setFixedWidth(25)
        btn.setToolTip("Refresh List")
        btn.clicked.connect(self._refresh)

        row.addWidget(self.combo)
        row.addWidget(btn)
        layout.addLayout(row)

        self.lbl_status = QLabel("Inactive")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.lbl_status)

        self.setMinimumWidth(250)
        self.devicesQueried.connect(self._on_devices_queried)
        QTimer.singleShot(100, self._refresh)

    def _refresh(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("Scanning...", -1)
        self.combo.setEnabled(False)
        self.combo.blockSignals(False)

        def worker():
            devices = AudioDeviceManager.get_compatible_devices(self.is_input)
            self.devicesQueried.emit(devices)

        threading.Thread(target=worker, daemon=True).start()

    def _on_devices_queried(self, devices):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("System Default", -1)

        for d in devices:
            name = d["display_name"]
            self.combo.addItem(name, d["id"])

        current_val = self.proxy.node_item.params["device_index"]["value"]
        idx = self.combo.findData(current_val)
        if idx != -1:
            self.combo.setCurrentIndex(idx)
        else:
            self.combo.setCurrentIndex(0)

        self.combo.setEnabled(True)
        self.combo.blockSignals(False)

    def _on_combo_user_action(self, index):
        val = self.combo.currentData()
        self.proxy.set_parameter("device_index", val)

    def on_telemetry(self, data):
        if "status" in data:
            self.lbl_status.setText(data["status"])
            if "On" in data["status"]:
                self.lbl_status.setStyleSheet("color: #55ff55; font-size: 10px; font-weight: bold;")
            elif "Error" in data["status"]:
                self.lbl_status.setStyleSheet("color: #ff5555; font-size: 10px;")
            else:
                self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")


class AudioInputWidget(AudioDeviceWidget):
    IS_NODE_UI, NODE_CLASS_NAME = True, "AudioDeviceInput"

    def __init__(self, proxy):
        super().__init__(proxy, True)


class AudioOutputWidget(AudioDeviceWidget):
    IS_NODE_UI, NODE_CLASS_NAME = True, "AudioDeviceOutput"

    def __init__(self, proxy):
        super().__init__(proxy, False)