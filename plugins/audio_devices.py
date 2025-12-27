import torch
import numpy as np
import sounddevice as sd
import logging
import threading
import time
from collections import deque
from typing import Optional, Dict, List, Any, Callable

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer

# ANode Imports
from base import Node, IClockProvider, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BUFFER_SIZE_BLOCKS = 16

# ==============================================================================
# Audio Device Manager
# ==============================================================================


class AudioDeviceManager:
    """Manages audio device discovery and compatibility checks."""

    @staticmethod
    def rescan_devices():
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            logger.error(f"AudioDeviceManager: Error during device re-scan: {e}")

    @staticmethod
    def is_device_compatible(index: int, is_input: bool) -> bool:
        try:
            info = sd.query_devices(index)
            if not info:
                return False
            max_ch = info.get("max_input_channels" if is_input else "max_output_channels", 0)
            return max_ch > 0
        except Exception:
            return False

    @staticmethod
    def get_compatible_devices(is_input: bool) -> List[Dict]:
        compatible_devices = []
        try:
            AudioDeviceManager.rescan_devices()
            devices = sd.query_devices()
            for idx, dev_info in enumerate(devices):
                if AudioDeviceManager.is_device_compatible(idx, is_input):
                    info = dict(dev_info)
                    info["id"] = idx
                    compatible_devices.append(info)
        except Exception as e:
            logger.error(f"AudioDeviceManager: Error querying devices: {e}")
        return compatible_devices

    @staticmethod
    def get_default_device_index(is_input: bool) -> Optional[int]:
        try:
            default_dev = sd.default.device
            idx = default_dev[0] if is_input else default_dev[1]
            return idx if idx is not None and idx >= 0 else None
        except Exception:
            return None

    @staticmethod
    def get_device_info(index: int) -> Optional[Dict]:
        try:
            return dict(sd.query_devices(index))
        except Exception:
            return None


# ==============================================================================
# UI Widget
# ==============================================================================


class AudioDeviceWidget(QWidget):
    """
    Shared Base Widget for Audio Devices.
    """

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.device_map = {}

        self.setMinimumWidth(250)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Device Selection Row
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo.activated.connect(self._on_combo_changed)

        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(25)
        self.btn_refresh.setToolTip("Refresh Devices")
        self.btn_refresh.clicked.connect(self._refresh_device_list)

        row.addWidget(self.combo)
        row.addWidget(self.btn_refresh)
        layout.addLayout(row)

        # Status Label
        self.lbl_status = QLabel("Status: Initializing...")
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # Populate immediately
        QTimer.singleShot(0, self._refresh_device_list)

    def is_input_node(self):
        # Determine type based on class name string provided by proxy/system
        # Updated to match "AudioDeviceInput"
        return "Input" in self.proxy.node_item.node_type

    def _refresh_device_list(self):
        """Scans devices directly in UI thread to ensure availability even when engine stopped."""
        self.combo.blockSignals(True)
        self.combo.clear()
        self.device_map = {}

        is_input = self.is_input_node()
        devices = AudioDeviceManager.get_compatible_devices(is_input)
        default_id = AudioDeviceManager.get_default_device_index(is_input)

        # Get current selection from node param
        current_val = -1
        if "device_index" in self.proxy.node_item.params:
            current_val = self.proxy.node_item.params["device_index"]["value"]

        # Add "Default" option explicitly
        self.combo.addItem(f"Default (System ID: {default_id})")
        self.device_map[0] = -1  # Map index 0 to ID -1

        combo_idx = 1
        for dev in devices:
            dev_id = dev["id"]
            name = f"{dev['name']} [{dev_id}]"

            # Visual marker for system default
            if dev_id == default_id:
                name = f"* {name}"

            self.combo.addItem(name)
            self.device_map[combo_idx] = dev_id

            # If this matches our saved param, select it
            if dev_id == current_val:
                self.combo.setCurrentIndex(combo_idx)
            combo_idx += 1

        # If current val was -1, select 0 (Default)
        if current_val == -1:
            self.combo.setCurrentIndex(0)

        self.combo.blockSignals(False)

    def _on_combo_changed(self, index):
        if index in self.device_map:
            dev_id = self.device_map[index]
            self.proxy.set_parameter("device_index", int(dev_id))

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
            if "Error" in data["status"]:
                self.lbl_status.setStyleSheet("color: #ff5555; font-size: 10px;")
            elif "Active" in data["status"]:
                self.lbl_status.setStyleSheet("color: #55ff55; font-size: 10px;")
            else:
                self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")


class AudioInputWidget(AudioDeviceWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "AudioDeviceInput"  # Matches class name below


class AudioOutputWidget(AudioDeviceWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "AudioDeviceOutput"  # Matches class name below


# ==============================================================================
# Base Audio Node Logic
# ==============================================================================


class BaseAudioNode(Node):
    def __init__(self, name=""):
        super().__init__(name)

        self.add_int_param("device_index", -1, min_v=-1, max_v=999)

        self._lock = threading.RLock()
        self._buffer = deque(maxlen=DEFAULT_BUFFER_SIZE_BLOCKS)
        self._stream: Optional[sd.Stream] = None
        self._stream_active_flag = False
        self._status_msg = "Inactive"
        self._stream_error_count = 0

    def _get_is_input_node(self) -> bool:
        raise NotImplementedError

    def _get_sounddevice_stream_class(self) -> type:
        raise NotImplementedError

    def _get_audio_callback(self) -> Callable:
        raise NotImplementedError

    def start(self):
        self._restart_stream()

    def stop(self):
        self._stop_stream()

    def remove(self):
        self._stop_stream()

    def _stop_stream(self):
        with self._lock:
            self._stream_active_flag = False
            if self._stream:
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self._buffer.clear()
            self._status_msg = "Inactive"

    def _restart_stream(self):
        self._stop_stream()
        is_input = self._get_is_input_node()

        with self._lock:
            req_id = self.params["device_index"].value

            # Resolve Default if -1 or not found
            dev_id = req_id
            if dev_id == -1:
                dev_id = AudioDeviceManager.get_default_device_index(is_input)

            if dev_id is None:
                self._status_msg = "Error: No Device Available"
                return

            if not AudioDeviceManager.is_device_compatible(dev_id, is_input):
                # Fallback to default if selected device went missing
                logger.warning(f"Device {dev_id} incompatible or missing. Trying default.")
                dev_id = AudioDeviceManager.get_default_device_index(is_input)
                if not AudioDeviceManager.is_device_compatible(dev_id, is_input):
                    self._status_msg = f"Error: Device {req_id} Failed"
                    return

            try:
                StreamClass = self._get_sounddevice_stream_class()
                self._stream = StreamClass(
                    device=dev_id,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._get_audio_callback(),
                )
                self._stream.start()
                self._stream_active_flag = True

                info = AudioDeviceManager.get_device_info(dev_id)
                name = info["name"] if info else f"ID {dev_id}"
                self._status_msg = f"Active: {name}"

            except Exception as e:
                self._status_msg = f"Start Failed: {str(e)[:20]}"
                logger.error(f"[{self.name}] Stream Start Error: {e}")
                self._stream = None
                self._stream_active_flag = False

    def on_ui_param_change(self, param_name):
        if param_name == "device_index":
            self.params[param_name].sync()
            if self._stream_active_flag:
                self._restart_stream()
            else:
                # Just update status text
                dev_id = self.params["device_index"].value
                if dev_id == -1:
                    self._status_msg = "Ready (Default)"
                else:
                    self._status_msg = f"Ready: ID {dev_id}"

    def get_telemetry(self) -> dict:
        return {
            "status": self._status_msg,
            "current_id": self.params["device_index"].value,
        }


# ==============================================================================
# Audio Input Implementation
# ==============================================================================


class AudioDeviceInput(BaseAudioNode):
    category = "I/O"
    label = "Audio Device Input"

    def __init__(self, name=""):
        super().__init__(name)
        self.out = self.add_output("out")

    def _get_is_input_node(self) -> bool:
        return True

    def _get_sounddevice_stream_class(self):
        return sd.InputStream

    def _get_audio_callback(self):
        return self._audio_callback_impl

    def _audio_callback_impl(self, indata, frames, time_info, status):
        # Run in audio thread
        try:
            data_tensor = torch.from_numpy(indata.copy().T)
            with self._lock:
                self._buffer.append(data_tensor)
        except Exception:
            pass

    def process(self):
        target = self.out.buffer
        with self._lock:
            if self._buffer:
                block = self._buffer.popleft()
                if block.shape == target.shape:
                    target.copy_(block)
                else:
                    target.zero_()
            else:
                target.zero_()


# ==============================================================================
# Audio Output Implementation
# ==============================================================================


class AudioDeviceOutput(BaseAudioNode, IClockProvider):
    category = "I/O"
    label = "Audio Device Output"

    def __init__(self, name=""):
        BaseAudioNode.__init__(self, name)
        IClockProvider.__init__(self)
        self.inp = self.add_input("audio_in")

    def _get_is_input_node(self) -> bool:
        return False

    def _get_sounddevice_stream_class(self):
        return sd.OutputStream

    def _get_audio_callback(self):
        return self._audio_callback_impl

    def _audio_callback_impl(self, outdata, frames, time_info, status):
        # Run in audio thread
        with self._lock:
            if self._buffer:
                data_tensor = self._buffer.popleft()
                np_data = data_tensor.numpy().T
                if np_data.shape == outdata.shape:
                    outdata[:] = np_data
                else:
                    outdata.fill(0)
            else:
                outdata.fill(0)

    # --- Clock Provider Implementation ---

    def start_clock(self):
        pass

    def stop_clock(self):
        pass

    def wait_for_sync(self):
        """
        Blocks the Engine thread to throttle it to audio speed.
        Guards against crashes by checking state safely under lock.
        """
        if not self.is_master:
            return

        while True:
            should_break = False

            with self._lock:
                if not self._stream or not self._stream_active_flag:
                    should_break = True
                else:
                    try:
                        if len(self._buffer) < DEFAULT_BUFFER_SIZE_BLOCKS:
                            should_break = True
                    except Exception:
                        should_break = True

            if should_break:
                break

            if self.abort_flag:
                break

            time.sleep(0.001)

    def process(self):
        in_tensor = self.inp.get_tensor()
        data_clone = in_tensor.clone().detach().cpu()

        with self._lock:
            self._buffer.append(data_clone)
