from ffi_base import FFINode


class SimpleDelay(FFINode):
    # Matches compiled library name (delay.dll / libdelay.so)
    LIB_NAME = "delay"

    category = "Effects"
    label = "Digital Delay"

    # Matches C++ set_param switch-case
    PARAM_MAP = {"time": 0, "feedback": 1, "mix": 2}

    def __init__(self, name=""):
        super().__init__(name)

        self.add_input("in")
        self.add_output("out")

        # Parameters
        self.add_float_param("time", 250.0, 1.0, 2000.0)  # ms
        self.add_float_param("feedback", 0.5, 0.0, 1.1)  # >1.0 allows for dub-style self-oscillation
        self.add_float_param("mix", 0.5, 0.0, 1.0)  # Dry/Wet

    # No need to override process() or __init__ further;
    # FFINode handles the flat buffer pointers and param syncing automatically.
