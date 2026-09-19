import ctypes
import logging
import os
import torch
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt

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

        # --- Section 1: File Parameter ---
        # Unified File Parameter Widget
        self.file_widget = self.proxy.create_param_widget("model_path")
        layout.addWidget(self.file_widget)

        # --- Section 2: Gain Controls ---
        # NAM models often need +/- gain adjustment
        self.drive_widget = self.proxy.create_param_widget("drive")
        self.level_widget = self.proxy.create_param_widget("level")

        layout.addWidget(self.drive_widget)
        layout.addWidget(self.level_widget)

        # --- Section 3: Status ---
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #666; margin-top: 5px;")
        layout.addWidget(self.lbl_status)

        self.lbl_file = QLabel("No Model Loaded")
        self.lbl_file.setStyleSheet("color: #aaa; font-size: 10px;")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

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
        if "model_path" in params:
            self.file_widget.update_from_backend(params["model_path"])


class NamNode(FFINode):
    LIB_NAME = "neural_amp"
    category = "Distortion & Amp"
    label = "Neural Amp Modeler"
    description = (
        "Neural network amplifier/emulator capture player backed by native C++ "
        "inference (libneural_amp). .nam model files are loaded on a background "
        "NRT worker and installed at a safe boundary; replaced models are "
        "deallocated off the audio thread. Drive and level provide input/output "
        "gain trimming."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in", help="Instrument-level signal to run through the neural model.")
        self.add_output("out", help="Modeled amplifier output with level gain applied.")

        # Internal params
        # CHANGED: Use add_file_param to enable generic UI widget
        self.add_file_param("model_path", "", filter="NAM Models (*.nam);;All Files (*.*)",
                            help="Neural Amp Modeler .nam capture file; loaded on a background worker.")
        self.add_float_param("drive", 1.0, 0.0, 4.0, unit="x",
                             help="Input gain applied before the model.")
        self.add_float_param("level", 1.0, 0.0, 4.0, unit="x",
                             help="Output gain applied after the model.")

        # Status tracking for telemetry
        self._status = "Idle"
        self._current_filename = "No Model"

        # Epoch for NRT model loads: stale (superseded) results are rejected.
        self._load_epoch = 0

        # Bind Custom Function
        if self.lib:
            try:
                self.lib.load_model_sync.restype = ctypes.c_int
                self.lib.load_model_sync.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_int]

                if hasattr(self.lib, "reset"):
                    self.lib.reset.restype = None
                    self.lib.reset.argtypes = [ctypes.c_void_p]
            except AttributeError as e:
                logger.error(f"'load_model_sync' not found in DLL: {e}")
                self._status = "Init failed"

    def on_ui_param_change(self, param_name: str):
        super().on_ui_param_change(param_name)

        if param_name == "model_path":
            # The engine committed the staged value (set()+sync()) before
            # invoking this callback (AGENTS.md §5); load_state() also commits
            # parameters before re-triggering this path. No re-sync here.
            path = self.params["model_path"].value
            if self.lib and path:
                self._status = "Loading..."
                self._load_epoch += 1
                self.submit_nrt(self._load_blocking, path, self._load_epoch, tag="load_model")

    def _destroy_handle_blocking(self, handle):
        """NRT thread: deallocate a native DSP handle. Destroying large
        neural-network weight structures can stall, so it must never run on
        the engine/audio thread."""
        try:
            self.lib.destroy(handle)
        except Exception as e:
            logger.error(f"NAM handle destroy failed: {e}")

    def _destroy_handle_background(self, handle):
        """Retire a native handle off the engine/audio thread, without
        bumping the shared NRT epoch: teardown traffic must not invalidate
        in-flight loads (the load epoch lives in _load_epoch)."""
        if not handle or not self.lib:
            return
        engine = getattr(getattr(self, "graph", None), "engine", None)
        nrt = getattr(engine, "nrt", None) if engine is not None else None
        if nrt is not None:
            nrt.submit_detached(self._destroy_handle_blocking, handle)
        else:
            # Headless / no engine: synchronous fallback on the control thread.
            self._destroy_handle_blocking(handle)

    def _load_blocking(self, path, epoch):
        """NRT thread: builds a NEW, independently owned native DSP state and
        loads the model into it. The live handle being processed by the audio
        path is never touched here."""
        new_handle = self.lib.create()
        if not new_handle:
            raise RuntimeError("Failed to create NAM DSP instance for model load")
        try:
            if hasattr(self.lib, "set_samplerate"):
                self.lib.set_samplerate(new_handle, float(SAMPLE_RATE))
            res = self.lib.load_model_sync(new_handle, path.encode("utf-8"),
                                           float(SAMPLE_RATE), int(BLOCK_SIZE))
        except Exception:
            self.lib.destroy(new_handle)
            raise
        if not res:
            self.lib.destroy(new_handle)
            raise RuntimeError(f"Failed to load NAM model from {path}")
        return (new_handle, os.path.basename(path), epoch)

    def on_nrt_complete(self, tag, ok, result):
        if tag != "load_model":
            return
        if ok:
            new_handle, filename, epoch = result
            if new_handle is None:
                self._status = "Error"
                return
            if epoch != self._load_epoch:
                # Superseded by a newer load request: retire the prepared
                # state in the background (engine/control thread must never
                # synchronously destroy large native models — AGENTS.md §6).
                self._destroy_handle_background(new_handle)
                self._status = "Idle"
                return
            # Install replacement state and retire the old one outside of
            # audio processing (this runs from sync(), between blocks on the
            # engine thread — never concurrently with process()).
            old_handle = self.dsp_handle
            self.dsp_handle = new_handle
            self._native_params_dirty = True
            if old_handle:
                # Deallocate large native weight structures on the NRT pool,
                # not on the engine/audio thread (destroy() of big models can
                # stall long enough to cause dropouts). Detached submit: no
                # shared-epoch bump, so in-flight loads stay valid.
                self._destroy_handle_background(old_handle)
            self._status, self._current_filename = "Active", filename
        else:
            self._status = "Error"
            self._current_filename = "Load Failed"

    def on_nrt_discarded(self, tag, ok, result):
        if tag == "load_model" and ok and result:
            # Superseded load: the fully prepared native handle is never
            # installed, so destroy it in the background (never synchronously
            # on the engine thread) to avoid leaking the model weights.
            new_handle = result[0] if isinstance(result, tuple) else None
            self._destroy_handle_background(new_handle)

    def remove(self):
        # Detach the live handle so the audio path can never touch it again,
        # then destroy the large native weight structures on the NRT pool —
        # never synchronously on the engine thread (AGENTS.md §6).
        handle = self.dsp_handle
        self.dsp_handle = None
        self._destroy_handle_background(handle)

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
        # FFINode.start() performs the standard reset contract (_call_reset())
        # and marks native params dirty so they are re-pushed on the next block.
        super().start()
