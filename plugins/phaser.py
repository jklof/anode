import logging
from ffi_base import FFINode
from base import CHANNELS

logger = logging.getLogger(__name__)


class Phaser(FFINode):
    category = "Effects"
    label = "Phaser (6-Stage)"
    description = (
        "Native C++ 6-stage stereo allpass phaser. Features exponential sweep "
        "modulation, quadrature stereo spread, and tanh-saturated feedback."
    )

    LIB_NAME = "phaser"
    PARAM_MAP = {
        "rate": 0,
        "depth": 1,
        "base_freq": 2,
        "feedback": 3,
        "spread": 4,
        "mix": 5,
    }

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Signal to phase; mono inputs are duplicated to stereo.")
        self.out = self.add_output("out", channels=CHANNELS, help="Stereo phaser output.")

        self.add_float_param("rate", 0.5, 0.05, 8.0, unit="Hz", help="LFO modulation rate.")
        self.add_float_param("depth", 0.7, 0.0, 1.0, help="Frequency sweep depth.")
        self.add_float_param("base_freq", 400.0, 50.0, 3000.0, unit="Hz", help="Center/base notch frequency.")
        self.add_float_param("feedback", 0.5, 0.0, 0.95, help="Resonance feedback amount (tanh saturated).")
        self.add_float_param("spread", 0.5, 0.0, 1.0, help="Stereo LFO phase offset between channels.")
        self.add_float_param("mix", 0.5, 0.0, 1.0, help="Dry/wet crossfade: 0.5 gives maximum notch cancellation.")