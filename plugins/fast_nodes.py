from ffi_base import FFINode


class CppGain(FFINode):
    # This must match the compiled filename (gain.dll / gain.so)
    LIB_NAME = "gain"

    # Map 'vol' param to ID 0 in C++
    PARAM_MAP = {"vol": 0}

    category = "Utilities"
    label = "Fast Gain (C++)"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in")
        self.add_output("out")
        # Param name matches PARAM_MAP key
        self.add_float_param("vol", 1.0, 0.0, 2.0)
