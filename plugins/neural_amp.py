import ctypes
from ffi_base import FFINode
from base import SAMPLE_RATE, BLOCK_SIZE


class NamNode(FFINode):
    LIB_NAME = "neural_amp"

    def __init__(self, name="NAM Amp"):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")
        self.add_file_param("model_path", "", filter="NAM Models (*.nam);;All Files (*.*)")

        # Bind Custom Function
        if self.lib:
            try:
                self.lib.load_nam_model.restype = None
                # Args: handle, path, sample_rate, block_size
                self.lib.load_nam_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_int]
            except AttributeError:
                print("Error: 'load_nam_model' not found in DLL")

    def on_ui_param_change(self, param_name: str):
        super().on_ui_param_change(param_name)

        if param_name == "model_path":
            self.params[param_name].sync()
            path = self.params["model_path"].value
            if self.lib and self.dsp_handle and path:
                b_path = path.encode("utf-8")
                # Pass Global Config to C++
                self.lib.load_nam_model(self.dsp_handle, b_path, float(SAMPLE_RATE), int(BLOCK_SIZE))
