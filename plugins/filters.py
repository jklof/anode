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
    description = (
        "Native C++ RBJ biquad IIR filter (Low Pass, High Pass, Band Pass, Notch, "
        "Peaking, Low Shelf, High Shelf). Coefficients are recomputed at block rate "
        "from the cutoff/Q/gain parameters, so fast parameter moves may zipper. "
        "Mono inputs are duplicated to both output channels."
    )

    LIB_NAME = "biquad"
    # Matches cpp/biquad.cpp set_param switch-case
    PARAM_MAP = {"type": 0, "cutoff": 1, "q": 2, "gain_db": 3}

    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Notch",
                    "Peaking", "Low Shelf", "High Shelf"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("type", self.FILTER_TYPES, 0,
                            help="Filter response type.")
        self.add_float_param("cutoff", 1000.0, CUTOFF_MIN, CUTOFF_MAX, unit="Hz",
                             help="Cutoff/corner frequency of the filter.")
        self.add_float_param("q", 0.707, 0.1, 10.0, unit="Q",
                             help="Resonance (quality factor); higher values ring more.")
        self.add_float_param("gain_db", 0.0, -24.0, 24.0, unit="dB",
                             help="Boost/cut used by Peaking and Shelf filter types.")

        self.inp = self.add_input("in", help="Signal to filter; mono inputs are duplicated to stereo.")
        # Modulation socket: connected signal is used as cutoff (Hz).
        self.in_mod = self.add_input("mod_cutoff",
                                     help="Audio-rate cutoff modulation (Hz, uses the mean of the block). Unconnected: uses 'cutoff' parameter.")
        self.out = self.add_output("out", channels=CHANNELS, help="Filtered stereo output.")
        self._was_cutoff_mod_connected = False

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # 1. Sync staged parameters first
        self._sync_params_to_cpp()

        # 2. Audio-rate modulation: if mod_cutoff connected, push directly after staged sync
        if self.in_mod.connected_outputs:
            sig = self.in_mod.get_tensor()
            eff = float(sig[0].mean().item())
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["cutoff"], eff)
            self._was_cutoff_mod_connected = True
        elif self._was_cutoff_mod_connected:
            # Disconnect: _sync_params_to_cpp() sees nothing dirty, so the
            # native filter would otherwise stay stuck on the last modulated
            # cutoff. Re-push the staged parameter once on the disconnect.
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["cutoff"],
                               float(self.params["cutoff"].value))
            self._was_cutoff_mod_connected = False

        # 3. Preprocess and call native process
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

        process_channels = min(in_channels, out_channels)
        # Channel adaptation: duplicate a mono input to both output channels
        # (must mirror ffi_base.FFINode.process(); see the policy comment there).
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


class LinearPhaseEQ(FFINode):
    category = "Effects"
    label = "Linear Phase EQ (FIR)"
    description = (
        "Native C++ 255-tap linear-phase FIR equalizer (Low Pass, High Pass, Band "
        "Pass, Band Stop). The odd-length Type-I symmetric FIR introduces a constant "
        "integer group delay of (N-1)/2 = 127 samples (~2.65 ms at 48 kHz), reported "
        "in the node telemetry. Zero phase distortion across the passband."
    )

    LIB_NAME = "fir_eq"
    # Matches cpp/fir_eq.cpp set_param switch-case
    PARAM_MAP = {"type": 0, "cutoff": 1, "q": 2}

    NUM_TAPS = 255  # odd -> Type I symmetric FIR, integer delay (N-1)/2
    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Band Stop (Notch)"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("type", self.FILTER_TYPES, 0,
                            help="Filter response type.")
        self.add_float_param("cutoff", 1000.0, CUTOFF_MIN, CUTOFF_MAX, unit="Hz",
                             help="Primary cutoff/corner frequency.")
        self.add_float_param("q", 1.0, 0.1, 10.0, unit="Q",
                             help="Bandwidth/resonance control for band responses.")

        self.inp = self.add_input("in", help="Signal to filter; mono inputs are duplicated to stereo.")
        self.out = self.add_output("out", channels=CHANNELS,
                                   help="Filtered stereo output, delayed by 127 samples (linear phase).")

    def process(self):
        super().process()

    def get_telemetry(self) -> dict:
        samples = (self.NUM_TAPS - 1) // 2
        return {
            "latency_samples": samples,
            "latency_ms": round(samples / SAMPLE_RATE * 1000.0, 2),
        }
