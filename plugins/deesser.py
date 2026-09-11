"""
DeEsser — split-band dynamic sibilance attenuator (Effects).

Thin FFINode wrapper over libdeesser (cpp/deesser.cpp). A Linkwitz-Riley 4
crossover splits the signal at `frequency`; a linked peak detector on an
independent bandpass (Q ~= 1) drives HF-only gain reduction (fixed 3:1
ratio, clamped to `depth`), so vowel formants and low harmonics pass
untouched while ess energy is ducked. All per-sample ballistics run
natively; the Python side only marshals pointers and pushes staged
parameters. `get_gr_db` is an extended export with explicit ctypes
annotations (cf. plugins/envelope.py).
"""

import ctypes
import logging

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


class DeEsser(FFINode):
    category = "Effects"
    label = "De-Esser"
    description = (
        "Split-band de-esser: attenuates only the high-frequency band above "
        "'frequency' when sibilant energy exceeds 'threshold' (up to 'depth' "
        "dB), leaving vowel body untouched. Detection is linked across "
        "channels; 'listen' solos the HF crossover band (the ducked signal) for tuning. Zero added "
        "latency; mix blends dry/wet without comb filtering."
    )

    LIB_NAME = "deesser"
    # Matches cpp/deesser.cpp ParamId
    PARAM_MAP = {
        "frequency": 0,
        "threshold": 1,
        "depth": 2,
        "attack_ms": 3,
        "release_ms": 4,
        "listen": 5,
        "mix": 6,
    }

    def __init__(self, name=""):
        super().__init__(name)

        self.inp = self.add_input(
            "in", help="Signal to de-ess; mono inputs are duplicated to stereo.")
        self.out = self.add_output(
            "out", channels=CHANNELS, help="De-essed stereo output.")

        self.add_float_param("frequency", 6500.0, 3000.0, 9000.0, unit="Hz",
                             help="Crossover/detector center frequency for the sibilant band.")
        self.add_float_param("threshold", -18.0, -40.0, 0.0, unit="dB",
                             help="Sibilant-band level above which reduction engages.")
        self.add_float_param("depth", 6.0, 0.0, 12.0, unit="dB",
                             help="Maximum HF attenuation applied to hot sibilants.")
        self.add_float_param("attack_ms", 1.0, 0.1, 10.0, unit="ms",
                             help="Detector attack time (catches ess onsets).")
        self.add_float_param("release_ms", 60.0, 10.0, 300.0, unit="ms",
                             help="Detector release time (smooth recovery).")
        self.add_bool_param("listen", False,
                            help="Solo the HF crossover band (the ducked signal) to tune frequency/threshold.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet balance (0.0 = bit-exact bypass). Intermediate "
                                  "values blend inside the crossover's phase-consistent "
                                  "domain, so no comb filtering occurs. Note: moving off "
                                  "0.0 engages the crossover phase and fresh detector "
                                  "state (clean engagement, no ghost of bypassed audio).")

    def _bind_functions(self):
        super()._bind_functions()
        if hasattr(self.lib, "get_gr_db"):
            self.lib.get_gr_db.restype = ctypes.c_float
            self.lib.get_gr_db.argtypes = [ctypes.c_void_p]

    def process(self):
        # Standard FFINode dispatch with mono->stereo duplication and
        # anti-ghosting when the backend is unavailable.
        if not self.lib or not self.dsp_handle:
            out_slot = self.outputs.get("out")
            if out_slot:
                out_slot.buffer.zero_()
            return

        self._sync_params_to_cpp()

        raw_tensor = self.inp.get_tensor()
        processed_tensor = self._preprocess_input(raw_tensor, self._ffi_in_buffer)
        in_channels = processed_tensor.shape[0]

        out_slot = self.outputs.get("out")
        if not out_slot:
            return
        out_tensor = out_slot.buffer
        out_channels = out_tensor.shape[0]

        if not out_tensor.is_contiguous():
            raise RuntimeError(f"Output tensor is not contiguous. Node: {self.name}")

        if processed_tensor.device.type != "cpu":
            processed_tensor = processed_tensor.cpu()

        if processed_tensor.is_contiguous():
            processing_tensor = processed_tensor
        else:
            self._ffi_in_buffer.copy_(processed_tensor)
            processing_tensor = self._ffi_in_buffer

        if in_channels == 1 and out_channels == 2:
            self._ffi_in_buffer[0].copy_(processed_tensor[0])
            self._ffi_in_buffer[1].copy_(processed_tensor[0])
            processing_tensor = self._ffi_in_buffer
            process_channels = 2
        else:
            process_channels = min(in_channels, out_channels)
            if process_channels < out_channels:
                out_tensor[process_channels:].zero_()

        in_ptr = ctypes.cast(processing_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        self.lib.process(self.dsp_handle, in_ptr, out_ptr, process_channels, BLOCK_SIZE)

    def get_telemetry(self) -> dict:
        gr = 0.0
        if self.lib and self.dsp_handle and hasattr(self.lib, "get_gr_db"):
            try:
                gr = float(self.lib.get_gr_db(self.dsp_handle))
            except Exception:
                pass
        return {
            "gr_db": round(gr, 2),
            "latency_samples": 0,
            "latency_ms": 0.0,
        }
