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


class _CppParamMixin:
    """Lazily pushes parameter changes into the native processor by
    comparing a value tuple against the last-pushed state. Covers every
    path uniformly: UI edits, save/load restore, and direct
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


class ChorusFlanger(_CppParamMixin, FFINode):
    category = "Effects"
    label = "Chorus / Flanger"

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
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=2)

        self.add_float_param("rate", 0.6, 0.05, 8.0)
        self.add_float_param("depth_ms", 3.0, 0.0, 8.0)
        self.add_float_param("base_delay_ms", 5.0, 0.0, 20.0)
        self.add_float_param("feedback", 0.3, 0.0, 0.9)
        self.add_float_param("spread", 1.0, 0.0, 1.0)
        self.add_float_param("mix", 0.5, 0.0, 1.0)

        self._bind_reset()
        self._sync_params_to_cpp()

    def process(self):
        if not self.lib or not self.dsp_handle:
            return
        super().process()   # standard ABI (in -> out); params pushed below
        self._sync_params_to_cpp()

    def start(self):
        self._call_reset()
        self._cpp_param_state = None   # force re-push on next block

    def load_state(self, data: dict):
        super().load_state(data)
        self._sync_params_to_cpp()
