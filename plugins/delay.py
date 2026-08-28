from ffi_base import FFINode


class SimpleDelay(FFINode):
    # Matches compiled library name (delay.dll / libdelay.so)
    LIB_NAME = "delay"

    category = "Effects"
    label = "Digital Delay"
    description = (
        "Native C++ feedback delay line with dry/wet mix. Feedback above 1.0 "
        "allows dub-style self-oscillation. Delay time range is 1-2000 ms; "
        "all per-sample processing runs natively on the CPU."
    )

    # Matches C++ set_param switch-case
    PARAM_MAP = {"time": 0, "feedback": 1, "mix": 2}

    def __init__(self, name=""):
        super().__init__(name)

        self.add_input("in", help="Signal to delay; mono inputs are duplicated to stereo.")
        self.add_output("out", help="Wet/dry mixed stereo output.")

        # Parameters
        self.add_float_param("time", 250.0, 1.0, 2000.0, unit="ms",
                             help="Delay time before the first repeat.")  # ms
        self.add_float_param("feedback", 0.5, 0.0, 1.1,
                             help="Amount of output fed back into the delay line; >1.0 self-oscillates.")  # >1.0 allows for dub-style self-oscillation
        self.add_float_param("mix", 0.5, 0.0, 1.0,
                             help="Dry/wet balance: 0 = dry only, 1 = wet only.")  # Dry/Wet

    # No need to override process() or __init__ further;
    # FFINode handles the flat buffer pointers and param syncing automatically.
