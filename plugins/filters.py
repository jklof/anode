"""
Filter / EQ nodes backed by native C++ DSP (libbiquad / libfir_eq).

Both nodes are thin FFINode wrappers: parameter values are pushed to the C++
side via PARAM_MAP, where coefficient design and per-sample processing run.
Class names, params and ranges match the original Python implementations so
saved patches stay loadable.

Real-time notes:
- All per-sample math runs in C++ (~tens of microseconds per stereo block).
- Coefficients update at block rate (93.75 Hz); fast parameter moves may
  zipper. Per-sample coefficient smoothing is deliberately out of scope.
"""

import ctypes
import logging

from ffi_base import FFINode
from base import BLOCK_SIZE, SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)

CUTOFF_MIN = 20.0
CUTOFF_MAX = 20000.0


class BiquadFilter(FFINode):
    category = "Effects"
    label = "Biquad Filter (IIR)"

    LIB_NAME = "biquad"
    # Matches cpp/biquad.cpp set_param switch-case
    PARAM_MAP = {"type": 0, "cutoff": 1, "q": 2, "gain_db": 3}

    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Notch",
                    "Peaking", "Low Shelf", "High Shelf"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("type", self.FILTER_TYPES, 0)
        self.add_float_param("cutoff", 1000.0, CUTOFF_MIN, CUTOFF_MAX)
        self.add_float_param("q", 0.707, 0.1, 10.0)
        self.add_float_param("gain_db", 0.0, -24.0, 24.0)

        self.inp = self.add_input("in")
        # Modulation socket: connected signal is used as cutoff (Hz).
        self.in_mod = self.add_input("mod_cutoff")
        self.out = self.add_output("out", channels=CHANNELS)

    def process(self):
        # Base class _sync_params_to_cpp() called first in FFINode.process()
        # Audio-rate modulation: if mod_cutoff connected, push directly after sync
        if self.in_mod.connected_outputs and self.lib and self.dsp_handle:
            # Block mean of the modulation signal; the C++ side clamps to its
            # stable range before designing coefficients.
            sig = self.in_mod.get_tensor()
            eff = float(sig[0].mean().item())
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["cutoff"], eff)

        super().process()


class LinearPhaseEQ(FFINode):
    category = "Effects"
    label = "Linear Phase EQ (FIR)"

    LIB_NAME = "fir_eq"
    # Matches cpp/fir_eq.cpp set_param switch-case
    PARAM_MAP = {"type": 0, "cutoff": 1, "q": 2}

    NUM_TAPS = 255  # odd -> Type I symmetric FIR, integer delay (N-1)/2
    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Band Stop (Notch)"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("type", self.FILTER_TYPES, 0)
        self.add_float_param("cutoff", 1000.0, CUTOFF_MIN, CUTOFF_MAX)
        self.add_float_param("q", 1.0, 0.1, 10.0)

        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=CHANNELS)

    def process(self):
        super().process()

    def get_telemetry(self) -> dict:
        samples = (self.NUM_TAPS - 1) // 2
        return {
            "latency_samples": samples,
            "latency_ms": round(samples / SAMPLE_RATE * 1000.0, 2),
        }
