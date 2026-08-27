"""
EnvelopeFollower — peak/RMS envelope + hysteresis gate (Utilities).

Thin FFINode wrapper over libenvelope (cpp/envelope.cpp). All per-sample
ballistics and gate hysteresis run natively; the Python side only marshals
pointers. Extended export bindings are explicitly annotated — an
unannotated ctypes handle argument defaults to 32-bit c_int and silently
truncates 64-bit pointers.

CV output range is [0, gain] (unclamped); use MathOp "Clamp" downstream for
a strict [0, 1] CV.
"""

import ctypes
import logging

from ffi_base import FFINode
from base import BLOCK_SIZE

logger = logging.getLogger(__name__)


class EnvelopeFollower(FFINode):
    category = "Utilities"
    label = "Envelope Follower"

    LIB_NAME = "envelope"
    # Matches cpp/envelope.cpp set_param switch-case
    PARAM_MAP = {"mode": 0, "attack_ms": 1, "release_ms": 2, "gain": 3, "gate_thresh": 4}

    MODES = ["Peak", "RMS"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in")
        self.add_output("cv_out", channels=1)
        self.add_output("gate_out", channels=1)

        self.add_menu_param("mode", self.MODES, 0)
        self.add_float_param("attack_ms", 10.0, 0.1, 500.0)
        self.add_float_param("release_ms", 100.0, 1.0, 2000.0)
        self.add_float_param("gain", 1.0, 0.1, 10.0)
        self.add_float_param("gate_thresh", 0.1, 0.0, 1.0)

    def _bind_functions(self):
        super()._bind_functions()
        # MANDATORY annotations: extended dual-output process signature
        self.lib.process.restype = None
        self.lib.process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),   # in (planar)
            ctypes.POINTER(ctypes.c_float),   # cv out (mono planar)
            ctypes.POINTER(ctypes.c_float),   # gate out (mono planar)
            ctypes.c_int,
            ctypes.c_int,
        ]

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # MANDATORY: Sync parameters before native processing
        self._sync_params_to_cpp()

        sig = self.inputs["in"].get_tensor()
        if sig.device.type != "cpu":
            sig = sig.cpu()
        if not sig.is_contiguous():
            self._ffi_in_buffer.copy_(sig)
            sig = self._ffi_in_buffer

        cv = self.outputs["cv_out"].buffer[0]
        gate = self.outputs["gate_out"].buffer[0]

        self.lib.process(
            self.dsp_handle,
            ctypes.cast(sig.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(cv.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(gate.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            int(sig.shape[0]),
            BLOCK_SIZE,
        )
