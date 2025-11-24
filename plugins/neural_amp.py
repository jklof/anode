import ctypes
import logging
from ffi_base import FFINode
from base import SAMPLE_RATE, BLOCK_SIZE


class NamNode(FFINode):
    LIB_NAME = "neural_amp"

    def __init__(self, name="Neural Amp"):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")
        self.add_file_param("model_path", "", filter="NAM Models (*.nam);;All Files (*.*)")

        # Bind Custom Function
        if self.lib:
            try:
                # Bind load function
                self.lib.load_nam_model.restype = None
                # Args: handle, path, sample_rate, block_size
                self.lib.load_nam_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_int]

                # Bind Reset function (if available in DLL)
                if hasattr(self.lib, "reset"):
                    self.lib.reset.restype = None
                    self.lib.reset.argtypes = [ctypes.c_void_p]

            except AttributeError as e:
                print(f"Error: 'load_nam_model' or 'reset' not found in DLL: {e}")

    def on_ui_param_change(self, param_name: str):
        super().on_ui_param_change(param_name)

        if param_name == "model_path":
            self.params[param_name].sync()
            path = self.params["model_path"].value
            if self.lib and self.dsp_handle and path:
                b_path = path.encode("utf-8")
                # Pass Global Config to C++
                # The C++ side now handles this asynchronously/safely
                self.lib.load_nam_model(self.dsp_handle, b_path, float(SAMPLE_RATE), int(BLOCK_SIZE))

    def start(self):
        """Called when audio engine starts. Used to reset DSP history."""
        if self.lib and self.dsp_handle and hasattr(self.lib, "reset"):
            try:
                self.lib.reset(self.dsp_handle)
            except Exception as e:
                logging.error(f"NAM Reset failed: {e}")
