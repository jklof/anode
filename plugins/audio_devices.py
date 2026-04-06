import torch
import numpy as np
import sounddevice as sd
import logging
import threading
import queue
from typing import Optional, Dict, List

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel
from PySide6.QtCore import Qt, QTimer, QSignalBlocker

# ANode Imports
from base import Node, IClockProvider, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# ==============================================================================
# High-Performance Ring Buffer
# ==============================================================================


class AudioRingBuffer:
    def __init__(self, capacity_blocks=32, block_size=BLOCK_SIZE, channels=CHANNELS):
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        self.channels = channels
        self.total_frames = capacity_blocks * block_size
        self.storage = np.zeros((self.total_frames, channels), dtype=np.float32)
        self.write_count = 0
        self.read_count = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> bool:
        with self.lock:
            available_space = self.capacity_blocks - (self.write_count - self.read_count)
            if available_space < 1:
                return False  # Overrun

            start_idx = (self.write_count % self.capacity_blocks) * self.block_size
            frames_to_write = min(self.block_size, data.shape[0])
            self.storage[start_idx : start_idx + frames_to_write, :] = data[:frames_to_write]
            self.write_count += 1
            return True

    def read(self, outdata: np.ndarray) -> bool:
        with self.lock:
            available_data = self.write_count - self.read_count
            if available_data < 1:
                return False  # Underrun

            start_idx = (self.read_count % self.capacity_blocks) * self.block_size
            outdata[:] = self.storage[start_idx : start_idx + self.block_size, :]
            self.read_count += 1
            return True

    def clear(self):
        with self.lock:
            self.write_count = 0
            self.read_count = 0
            self.storage.fill(0)


# ==============================================================================
# Audio Device Management
# ==============================================================================


class AudioDeviceManager:
    @staticmethod
    def get_compatible_devices(is_input: bool, target_rate: int = SAMPLE_RATE) -> List[Dict]:
        devices = []
        try:
            # NOTE: Removed sd._terminate() / sd._initialize() here.
            # Doing a full reset while other streams might be running (e.g. Input running, Output refreshing)
            # causes instability and crashes. standard query_devices is usually sufficient.

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
        try:
            return sd.default.device[0] if is_input else sd.default.device[1]
        except:
            return -1


# ==============================================================================
# Unified Audio Node Logic
# ==============================================================================


class BaseAudioDeviceNode(Node):
    category, label = "I/O", "Base Audio Device"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_int_param("device_index", -1, min_v=-1, max_v=999)
        self.ring_buffer = AudioRingBuffer(capacity_blocks=32)
        self.stream: Optional[sd.Stream] = None
        self._status_msg = "Inactive"
        self._active = False
        self._latency_ms = 0.0
        self._actual_device_idx = -1
        
        self._action_queue = queue.Queue()
        self._action_thread = threading.Thread(target=self._action_worker, daemon=True)
        self._action_thread.start()

    def _action_worker(self):
        while True:
            action = self._action_queue.get()
            if action is None:
                break
            func, args = action
            try:
                func(*args)
            except Exception as e:
                logger.error(f"Device Node Task Error: {e}")

    def _start_stream(self, StreamClass, callback, channels=None):
        self._action_queue.put((self._start_stream_sync, (StreamClass, callback, channels)))

    def _start_stream_sync(self, StreamClass, callback, channels=None):
        self._stop_stream_sync()

        # KEY CHANGE: Ensure we are reading the synced value
        requested_idx = self.params["device_index"].value
        target_idx = requested_idx

        # 1. Resolve Default
        if target_idx == -1:
            try:
                target_idx = sd.default.device[0 if StreamClass == sd.InputStream else 1]
            except Exception:
                self._status_msg = "No Default Device"
                self._actual_device_idx = -2
                return

        # 2. Query Capabilities
        try:
            info = sd.query_devices(target_idx)
        except Exception as e:
            self._status_msg = "Device Not Found"
            self._actual_device_idx = -2
            return

        # 3. Channel Logic (Clamp to Hardware Max)
        desired_channels = channels or CHANNELS
        hw_max = info.get("max_input_channels" if StreamClass == sd.InputStream else "max_output_channels", 0)
        actual_channels = min(desired_channels, hw_max)

        if actual_channels < 1:
            self._status_msg = "Device has 0 channels"
            self._actual_device_idx = -2
            return

        # 4. Attempt Stream Open
        try:
            # print(f"DEBUG: Opening Device ID {target_idx} ({info['name']}) Ch: {actual_channels}")
            self.stream = StreamClass(
                device=target_idx,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=actual_channels,
                dtype="float32",
                callback=callback,
            )
            self.stream.start()
            self._active = True
            self._actual_device_idx = target_idx
            self._latency_ms = self.stream.latency * 1000.0

            ch_str = "Mono" if actual_channels == 1 else f"{actual_channels}ch"
            self._status_msg = f"{info['name']} ({ch_str})"

        except Exception as e:
            logger.error(f"Stream Open Failed: {e}")
            self._status_msg = f"Error: {str(e)[:20]}..."
            self._active = False
            self._actual_device_idx = -2

    def _stop_stream(self):
        self._action_queue.put((self._stop_stream_sync, ()))

    def _stop_stream_sync(self):
        self._active = False
        self._actual_device_idx = -1
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.ring_buffer.clear()
        self._status_msg = "Inactive"

    def stop(self):
        self._stop_stream()

    def remove(self):
        self._stop_stream()
        self._action_queue.put(None)  # Signal worker thread to quit

    def get_telemetry(self) -> dict:
        msg = self._status_msg
        if self._active:
            msg += f" [{self._latency_ms:.0f}ms]"

        return {"status": msg, "actual_device_idx": self._actual_device_idx}

    def on_ui_param_change(self, param_name):
        if param_name == "device_index":
            # CRITICAL FIX: The Engine 'param' command sets _staging,
            # but hasn't called sync() yet. We MUST sync manually
            # so that _start_stream() sees the NEW selection, not the old one.
            self.params["device_index"].sync()

            # Restart immediately (safe even if engine is running)
            self.start()


class AudioDeviceInput(BaseAudioDeviceNode):
    category, label = "I/O", "Audio Device Input"

    def __init__(self, name=""):
        super().__init__(name)
        self.out = self.add_output("out")

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
        temp = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)
        if self.ring_buffer.read(temp):
            self.out.buffer.copy_(torch.from_numpy(temp.T))
        else:
            self.out.buffer.zero_()


class AudioDeviceOutput(BaseAudioDeviceNode, IClockProvider):
    category, label = "I/O", "Audio Device Output"

    def __init__(self, name=""):
        BaseAudioDeviceNode.__init__(self, name)
        IClockProvider.__init__(self)
        self.inp = self.add_input("audio_in")
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
        np.copyto(self._scratch_buffer, tensor_data.t().numpy())

        # 3. Write to Ring Buffer
        # Now we are passing a persistent pointer, not a new object
        self.ring_buffer.write(self._scratch_buffer)


# ==============================================================================
# UI Widgets
# ==============================================================================


class AudioDeviceWidget(QWidget):
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
        QTimer.singleShot(100, self._refresh)

    def _refresh(self):
        self.combo.blockSignals(True)
        self.combo.clear()

        # Don't destroy active streams during refresh
        devices = AudioDeviceManager.get_compatible_devices(self.is_input)

        self.combo.addItem(f"System Default", -1)

        for d in devices:
            name = d["display_name"]
            self.combo.addItem(name, d["id"])

        current_val = self.proxy.node_item.params["device_index"]["value"]
        idx = self.combo.findData(current_val)
        if idx != -1:
            self.combo.setCurrentIndex(idx)
        else:
            self.combo.setCurrentIndex(0)

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
