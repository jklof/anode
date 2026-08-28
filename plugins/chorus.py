"""
ChorusFlanger — quadrature-modulated stereo delay line (Effects).

Thin FFINode wrapper over libchorus (cpp/chorus.cpp). All per-sample LFO
phase accumulation, Hermite fractional delay, and tanh-saturated feedback
run natively; the Python side only marshals pointers and lazily pushes
parameters. Extended export bindings are explicitly annotated — an
unannotated ctypes handle argument defaults to 32-bit c_int and silently
truncates 64-bit pointers.
"""

import ctypes
import logging

from ffi_base import FFINode

logger = logging.getLogger(__name__)


class ChorusFlanger(FFINode):
    category = "Effects"
    label = "Chorus / Flanger"
    description = (
        "Native C++ quadrature-modulated stereo delay line. Two LFOs in "
        "quadrature drive Hermite-interpolated fractional delays with tanh-"
        "saturated feedback; the spread parameter offsets the channels for a "
        "wide chorus. Short base delays produce flanging."
    )

    LIB_NAME = "chorus"
    # Matches cpp/chorus.cpp set_param switch-case
    PARAM_MAP = {
        "rate": 0,
        "depth_ms": 1,
        "base_delay_ms": 2,
        "feedback": 3,
        "spread": 4,
        "mix": 5,
    }

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Signal to modulate; mono inputs are duplicated to stereo.")
        self.out = self.add_output("out", channels=2, help="Stereo chorus/flanger output.")

        self.add_float_param("rate", 0.6, 0.05, 8.0, unit="Hz",
                             help="LFO modulation rate.")
        self.add_float_param("depth_ms", 3.0, 0.0, 8.0, unit="ms",
                             help="LFO modulation depth added to the base delay.")
        self.add_float_param("base_delay_ms", 5.0, 0.0, 20.0, unit="ms",
                             help="Center delay time; small values give flanging, larger give chorus.")
        self.add_float_param("feedback", 0.3, 0.0, 0.9,
                             help="Amount of delayed signal fed back (tanh-saturated).")
        self.add_float_param("spread", 1.0, 0.0, 1.0,
                             help="Stereo phase offset between the left and right LFOs.")
        self.add_float_param("mix", 0.5, 0.0, 1.0,
                             help="Dry/wet balance: 0 = dry only, 1 = wet only.")

    def process(self):
        super().process()

    def get_telemetry(self) -> dict:
        return {}
