import ctypes
import logging
import os
import queue
import torch
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
)
from PySide6.QtCore import Qt, QTimer, QSignalBlocker

from ffi_base import FFINode
from base import SAMPLE_RATE, BLOCK_SIZE

logger = logging.getLogger(__name__)


class NamWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "NamNode"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- Section 1: File Info ---
        self.lbl_file = QLabel("No Model Loaded")
        self.lbl_file.setStyleSheet("color: #aaa; font-size: 10px; margin-bottom: 2px;")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

        # --- Section 2: Load Button ---
        btn_load = QPushButton("Load NAM Model")
        btn_load.clicked.connect(self.browse)
        layout.addWidget(btn_load)

        # --- Section 3: Status ---
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #666; margin-bottom: 5px;")
        layout.addWidget(self.lbl_status)

        # --- Section 4: Gain Controls ---
        # NAM models often need +/- gain adjustment
        self.drive_widget = self.proxy.create_param_widget("drive")
        self.level_widget = self.proxy.create_param_widget("level")

        layout.addWidget(self.drive_widget)
        layout.addWidget(self.level_widget)

    def browse(self):
        # Use None parent to prevent embedding crashes
        f, _ = QFileDialog.getOpenFileName(None, "Open NAM Model", "", "NAM Models (*.nam);;All Files (*.*)")
        if f:
            self.lbl_status.setText("Loading...")
            self.lbl_status.setStyleSheet("color: #FFaa00")
            self.proxy.set_parameter("model_path", f)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
            style = "color: #00FF00" if data["status"] == "Active" else "color: #FFaa00"
            self.lbl_status.setStyleSheet(style)
        if "filename" in data:
            self.lbl_file.setText(data["filename"])

    def update_from_params(self, params):
        # Update smart widgets
        if "drive" in params:
            self.drive_widget.update_from_backend(params["drive"])
        if "level" in params:
            self.level_widget.update_from_backend(params["level"])

        # Update Filename Label if path is present
        if "model_path" in params and params["model_path"]:
            self.lbl_file.setText(os.path.basename(params["model_path"]))


class NamNode(FFINode):
    LIB_NAME = "neural_amp"
    category = "Effects"
    label = "Neural Amp Modeler"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")

        # Internal params
        self.add_string_param("model_path", "")
        self.add_float_param("drive", 1.0, 0.0, 4.0)
        self.add_float_param("level", 1.0, 0.0, 4.0)

        # Status tracking for telemetry
        self._status = "Idle"
        self._current_filename = "No Model"

        # Bind Custom Function
        if self.lib:
            try:
                self.lib.load_nam_model.restype = None
                self.lib.load_nam_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_int]

                if hasattr(self.lib, "reset"):
                    self.lib.reset.restype = None
                    self.lib.reset.argtypes = [ctypes.c_void_p]
            except AttributeError as e:
                print(f"Error: 'load_nam_model' not found in DLL: {e}")

    def on_ui_param_change(self, param_name: str):
        super().on_ui_param_change(param_name)

        if param_name == "model_path":
            self.params[param_name].sync()
            path = self.params["model_path"].value
            if self.lib and self.dsp_handle and path:
                # Update status
                self._status = "Active"
                self._current_filename = os.path.basename(path)

                # Trigger C++ Load
                b_path = path.encode("utf-8")
                self.lib.load_nam_model(self.dsp_handle, b_path, float(SAMPLE_RATE), int(BLOCK_SIZE))

    def get_telemetry(self) -> dict:
        return {"status": self._status, "filename": self._current_filename}

    def _preprocess_input(self, in_tensor: torch.Tensor, scratch_buffer: torch.Tensor) -> torch.Tensor:
        gain = self.params["drive"].value
        if gain == 1.0:
            return in_tensor  # Zero-copy path
        else:
            scratch_buffer.copy_(in_tensor)
            scratch_buffer.mul_(gain)
            return scratch_buffer

    def process(self):
        # Run C++ Processing (which includes _preprocess_input for input gain)
        super().process()

        # Apply Output Gain (Post-NAM)
        out_gain = self.params["level"].value
        if out_gain != 1.0:
            # We modify the output buffer directly
            self.outputs["out"].buffer.mul_(out_gain)

    def load_state(self, data: dict):
        super().load_state(data)
        # Trigger reload of model if path exists
        if "model_path" in self.params and self.params["model_path"].value:
            self.on_ui_param_change("model_path")

    def start(self):
        if self.lib and self.dsp_handle and hasattr(self.lib, "reset"):
            try:
                self.lib.reset(self.dsp_handle)
            except Exception as e:
                logging.error(f"NAM Reset failed: {e}")
