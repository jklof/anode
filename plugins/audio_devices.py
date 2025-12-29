import torch
import numpy as np
import sounddevice as sd
import logging
import threading
from typing import Optional, Dict, List

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QTimer

# ANode Imports
from base import Node, IClockProvider, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# ==============================================================================
# High-Performance Ring Buffer (Merged from audio_output.py)
# ==============================================================================


class AudioRingBuffer:
    def __init__(self, capacity_blocks=16, block_size=BLOCK_SIZE, channels=CHANNELS):
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
                return False
            start_idx = (self.write_count % self.capacity_blocks) * self.block_size
            self.storage[start_idx : start_idx + self.block_size, :] = data
            self.write_count += 1
            return True

    def read(self, outdata: np.ndarray) -> bool:
        with self.lock:
            available_data = self.write_count - self.read_count
            if available_data < 1:
                return False
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
# Audio Device Management (From audio_devices.py)
# ==============================================================================


class AudioDeviceManager:
    @staticmethod
    def get_compatible_devices(is_input: bool) -> List[Dict]:
        devices = []
        try:
            sd._terminate()
            sd._initialize()  # Force rescan
            for idx, dev in enumerate(sd.query_devices()):
                max_ch = dev.get("max_input_channels" if is_input else "max_output_channels", 0)
                if max_ch > 0:
                    dev_info = dict(dev)
                    dev_info["id"] = idx
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
    def __init__(self, name=""):
        super().__init__(name)
        self.add_int_param("device_index", -1, min_v=-1, max_v=999)
        self.ring_buffer = AudioRingBuffer(capacity_blocks=16)
        self.stream: Optional[sd.Stream] = None
        self._status_msg = "Inactive"
        self._active = False

    def _start_stream(self, StreamClass, callback, channels=None, samplerate=None):
        self._stop_stream()
        dev_id = self.params["device_index"].value
        if dev_id == -1:
            dev_id = None  # sounddevice default

        channels = channels or CHANNELS
        samplerate = samplerate or SAMPLE_RATE

        try:
            self.stream = StreamClass(
                device=dev_id,
                samplerate=samplerate,
                blocksize=BLOCK_SIZE,
                channels=channels,
                dtype="float32",
                callback=callback,
            )
            self.stream.start()
            self._active = True
            info = sd.query_devices(dev_id) if dev_id is not None else sd.query_devices(sd.default.device[1])
            self._status_msg = f"Active: {info['name']}"
        except sd.PortAudioError as e:
            logger.error(f"PortAudioError opening stream: {e}")
            if len(e.args) > 1:
                logger.error(f"Host error details: {e.args[1]}")
            self._status_msg = f"Error: {str(e)[:50]}"
            self._active = False
        except Exception as e:
            logger.error(f"Unexpected error opening stream: {e}")
            self._status_msg = f"Error: {str(e)[:50]}"
            self._active = False

    def _stop_stream(self):
        self._active = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.ring_buffer.clear()
        self._status_msg = "Inactive"

    def stop(self):
        self._stop_stream()

    def remove(self):
        self._stop_stream()

    def get_telemetry(self) -> dict:
        return {"status": self._status_msg}

    def on_ui_param_change(self, param_name):
        if param_name == "device_index" and self._active:
            self.start()  # Restart stream with new device


class AudioDeviceInput(BaseAudioDeviceNode):
    category, label = "I/O", "Audio Device Input"

    def __init__(self, name=""):
        super().__init__(name)
        self.out = self.add_output("out")
        self._actual_channels = CHANNELS  # Track actual channels used

    def start(self):
        dev_id = self.params["device_index"].value
        if dev_id == -1:
            dev_id = None

        # Smart settings fallback
        channels = CHANNELS
        samplerate = SAMPLE_RATE

        try:
            if dev_id is not None:
                info = sd.query_devices(dev_id)
            else:
                info = sd.query_devices(sd.default.device[0])

            # Check input channels
            max_input_ch = info.get("max_input_channels", CHANNELS)
            if max_input_ch < channels:
                logger.warning(f"Device supports only {max_input_ch} input channels, falling back from {channels}")
                channels = max_input_ch

            # Check sample rate support
            supported_rates = info.get("supported_samplerates", [])
            if supported_rates and SAMPLE_RATE not in supported_rates:
                logger.warning(
                    f"Sample rate {SAMPLE_RATE} not supported, using default {info.get('default_samplerate', SAMPLE_RATE)}"
                )
                samplerate = info.get("default_samplerate", SAMPLE_RATE)

        except Exception as e:
            logger.warning(f"Could not query device capabilities: {e}, using defaults")

        self._actual_channels = channels
        self._start_stream(sd.InputStream, self._callback, channels=channels, samplerate=samplerate)

    def _callback(self, indata, frames, time, status):
        self.ring_buffer.write(indata)

    def process(self):
        # Read from ring buffer into Torch output
        temp = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)
        if self.ring_buffer.read(temp):
            # Handle mono to stereo duplication if device is mono
            if self._actual_channels == 1:
                temp[:, 1] = temp[:, 0]  # Duplicate mono channel to right channel
            # Transpose [Frames, Ch] to [Ch, Frames] for ANode
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
        if not self.ring_buffer.read(outdata):
            outdata.fill(0)

    def process(self):
        # ANode uses [Ch, Frames]. sounddevice uses [Frames, Ch].
        tensor_data = self.inp.get_tensor()
        np_data = tensor_data.detach().cpu().numpy()

        # Handle mono-to-stereo or multi-channel alignment
        if np_data.shape[0] == 1:
            # Mono to Stereo duplication
            np_data = np.tile(np_data.T, (1, CHANNELS))
        else:
            # Transpose to [Frames, Ch]
            np_data = np_data[:CHANNELS, :].T

        # Try to write to the ring buffer once
        self.ring_buffer.write(np_data)


# ==============================================================================
# UI Widgets (Identical to audio_devices.py but updated class names)
# ==============================================================================


class AudioDeviceWidget(QWidget):
    def __init__(self, proxy, is_input):
        super().__init__()
        self.proxy, self.is_input = proxy, is_input
        self.device_map = {}
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.activated.connect(self._on_combo)
        btn = QPushButton("⟳")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._refresh)
        row.addWidget(self.combo)
        row.addWidget(btn)
        layout.addLayout(row)

        self.lbl_status = QLabel("Inactive")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)
        self.setMinimumWidth(300)
        QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        self.combo.clear()
        devices = AudioDeviceManager.get_compatible_devices(self.is_input)
        default_id = AudioDeviceManager.get_default_id(self.is_input)

        self.combo.addItem(f"Default ({default_id})", -1)
        for d in devices:
            self.combo.addItem(f"{d['name']} [{d['id']}]", d["id"])

        # Sync combo to current param
        current = self.proxy.node_item.params["device_index"]["value"]
        idx = self.combo.findData(current)
        if idx != -1:
            self.combo.setCurrentIndex(idx)

    def _on_combo(self, index):
        self.proxy.set_parameter("device_index", self.combo.currentData())

    def on_telemetry(self, data):
        if "status" in data:
            self.lbl_status.setText(data["status"])
            color = "#55ff55" if "Active" in data["status"] else "#ff5555" if "Error" in data["status"] else "#888"
            self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px;")


class AudioInputWidget(AudioDeviceWidget):
    IS_NODE_UI, NODE_CLASS_NAME = True, "AudioDeviceInput"

    def __init__(self, proxy):
        super().__init__(proxy, True)


class AudioOutputWidget(AudioDeviceWidget):
    IS_NODE_UI, NODE_CLASS_NAME = True, "AudioDeviceOutput"

    def __init__(self, proxy):
        super().__init__(proxy, False)
