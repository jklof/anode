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


class _CppParamMixin:
    """Pushes parameter changes into the native processor.

    Plain mixin (does NOT subclass Node, so plugin_system never registers it).
    Params are synced lazily in process() by comparing a value tuple against
    the last-pushed state. This covers every path uniformly: UI edits
    (on_ui_param_change), save/load restore (load_state), and direct
    Parameter.set()+sync() from scripts/tests."""

    PARAM_MAP = {}
    _cpp_param_state = None

    def _bind_reset(self):
        if self.lib and hasattr(self.lib, "reset"):
            self.lib.reset.restype = None
            self.lib.reset.argtypes = [ctypes.c_void_p]

    def _call_reset(self):
        if self.lib and self.dsp_handle and hasattr(self.lib, "reset"):
            try:
                self.lib.reset(self.dsp_handle)
            except Exception as e:
                logger.error(f"[{self.name}] reset failed: {e}")

    def _sync_params_to_cpp(self):
        if not (self.lib and self.dsp_handle):
            return
        state = tuple(float(self.params[name].value) for name in self.PARAM_MAP)
        if state != self._cpp_param_state:
            for (name, pid), val in zip(self.PARAM_MAP.items(), state):
                self.lib.set_param(self.dsp_handle, pid, val)
            self._cpp_param_state = state

    def start(self):
        self._call_reset()
        self._cpp_param_state = None  # force re-push on next block

    def load_state(self, data: dict):
        super().load_state(data)
        self._sync_params_to_cpp()


class BiquadFilter(_CppParamMixin, FFINode):
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

        self._bind_reset()

    def process(self):
        self._sync_params_to_cpp()

        if self.in_mod.connected_outputs and self.lib and self.dsp_handle:
            # Block mean of the modulation signal; the C++ side clamps to its
            # stable range before designing coefficients.
            sig = self.in_mod.get_tensor()
            eff = float(sig[0].mean().item())
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["cutoff"], eff)

        super().process()


class LinearPhaseEQ(_CppParamMixin, FFINode):
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

        self._bind_reset()

    def process(self):
        self._sync_params_to_cpp()
        super().process()

    def get_telemetry(self) -> dict:
        samples = (self.NUM_TAPS - 1) // 2
        return {
            "latency_samples": samples,
            "latency_ms": round(samples / SAMPLE_RATE * 1000.0, 2),
        }
