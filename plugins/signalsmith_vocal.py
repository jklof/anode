"""
SignalsmithVocal — sub-band pitch/formant vocal transformer (Effects).

Thin FFINode wrapper over libsignalsmith_vocal
(cpp/signalsmith_vocal.cpp). The native side runs Signalsmith Stretch v1.3.2
(MIT, header-only) as a pitch-shifter at 1x rate (equal input/output block
sizes, no time-stretch):

- Sub-band transient-aware phase locking: no glottal-closure marking, no
  phase-vocoder smearing on pitch shifts of multiple octaves.
- Pitch transpose (semitones, optional tonality limit) and formant shift
  (semitones, pitch-compensated so formants stay absolute) plus a gender
  morph that adds 3.5 semitones of formant shift per unit.
- Fixed algorithmic latency of inputLatency()+outputLatency() samples
  (5760 samples / 120 ms at 48 kHz in Studio mode, 1920 / 40 ms in Live
  mode); `mix` crossfades with the latency-aligned dry path, and `mix = 0`
  is a bit-exact zero-latency bypass.

The Python side only marshals pointers, pushes staged parameters once per
change, and forwards block-rate CV from the modulation sockets.
"""

import ctypes

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


class SignalsmithVocal(FFINode):
    category = "Voice & Pitch"
    label = "Signalsmith Vocal Transformer"
    description = (
        "Sub-band pitch and formant transformer powered by Signalsmith "
        "Stretch (transient-aware phase locking, no glottal marking, no "
        "phase-vocoder smearing). Independent pitch transposition with "
        "optional tonality limit, pitch-compensated formant shifting, and "
        "acoustic gender morphing. Algorithmic latency of 5760 samples "
        "(120 ms at 48 kHz) in Studio mode, 1920 (40 ms) in Live mode; mix "
        "crossfades cleanly with the "
        "latency-aligned dry path."
    )

    LIB_NAME = "signalsmith_vocal"
    # Matches cpp/signalsmith_vocal.cpp ParamId
    PARAM_MAP = {
        "pitch_shift": 0,
        "formant_shift": 1,
        "gender_morph": 2,
        "tonality_limit": 3,
        "mix": 4,
        "latency_mode": 5,
    }

    LATENCY_MODES = [
        "Studio (HQ / 120ms)",
        "Live (40ms)",
    ]

    # Presets mirror the VocalTransformer schema for the ui_system NodeItem menu.
    PRESETS = {
        "Male -> Female Pop": {
            "pitch_shift": 10.0,
            "formant_shift": 1.0,
            "gender_morph": 0.85,
            "tonality_limit": 0.0,
            "mix": 1.0,
        },
        "Female -> Male Deep": {
            "pitch_shift": -10.0,
            "formant_shift": -1.0,
            "gender_morph": -0.80,
            "tonality_limit": 0.0,
            "mix": 1.0,
        },
        "Roland VT-4 Lead Style": {
            "pitch_shift": 0.0,
            "formant_shift": 2.5,
            "gender_morph": 0.5,
            "tonality_limit": 0.0,
            "mix": 1.0,
        },
        "Neutral (Bypass Tuning)": {
            "pitch_shift": 0.0,
            "formant_shift": 0.0,
            "gender_morph": 0.0,
            "tonality_limit": 0.0,
            "mix": 1.0,
        },
    }

    def __init__(self, name=""):
        super().__init__(name)

        # Modulation disconnect detection (karplus_strong pattern). Removing a
        # wire changes no parameter value, so without these flags the native
        # DSP would stay stuck at the last CV value (_sync_params_to_cpp sees
        # nothing dirty after the disconnect).
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False

        self.inp = self.add_input(
            "in", help="Vocal signal to transform; mono inputs are duplicated to stereo.")
        self.pitch_mod = self.add_input(
            "pitch_mod", "pitch_shift",
            help="Block-rate pitch CV in semitones (bound to 'pitch_shift'; "
                 "first sample of each block). Unconnected: uses the parameter value.")
        self.formant_mod = self.add_input(
            "formant_mod", "formant_shift",
            help="Block-rate formant CV in semitones (bound to 'formant_shift'; "
                 "first sample of each block). Unconnected: uses the parameter value.")
        self.out = self.add_output(
            "out", channels=CHANNELS, help="Transformed vocal stereo output.")

        self.add_float_param("pitch_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Fundamental pitch shift in semitones.")
        self.add_float_param("formant_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Vocal tract resonance / formant shift in semitones "
                                  "(pitch-compensated, stays absolute).")
        self.add_float_param("gender_morph", 0.0, -1.0, 1.0, unit="",
                             help="Gender coloration (+1.0 = Feminine, -1.0 = Masculine). "
                                  "Adds 3.5 semitones of formant shift per unit.")
        self.add_float_param("tonality_limit", 0.0, 0.0, 8000.0, unit="Hz",
                             help="Tonality limit in Hz: non-linear frequency map that "
                                  "preserves timbre above this frequency. 0 = off.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet crossfade (0.0 = dry bypass, 1.0 = transformed "
                                  "vocal). The dry path is latency-aligned inside the DSP, "
                                  "so intermediate values crossfade without comb filtering.")
        self.add_menu_param(
            "latency_mode", self.LATENCY_MODES, initial_idx=0,
            help="Algorithmic latency tradeoff: Studio (default preset geometry, "
                 "120 ms) or Live (40 ms block / 10 ms interval, 40 ms). "
                 "Switching clears the transient pipeline (one latency cycle "
                 "of silence).")

    def _bind_functions(self):
        super()._bind_functions()
        if hasattr(self.lib, "get_latency_samples"):
            self.lib.get_latency_samples.restype = ctypes.c_int
            self.lib.get_latency_samples.argtypes = [ctypes.c_void_p]

    def start(self):
        super().start()
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False

    def load_state(self, data: dict):
        super().load_state(data)
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False

    def process(self):
        # Mirrors plugins/vocal_transformer.py: replicate the FFINode
        # dispatch so block-rate CV can be pushed BETWEEN staged parameter
        # sync and the native process() call.
        if not self.lib or not self.dsp_handle:
            # Anti-ghosting: never leave stale audio in the output buffer
            # (AGENTS.md §2), even when the native backend is unavailable.
            out_slot = self.outputs.get("out")
            if out_slot:
                out_slot.buffer.zero_()
            return

        # 1. Sync staged parameters (canonical path)
        self._sync_params_to_cpp()

        # 2. Block-rate modulation with disconnect detection: re-push the
        #    staged parameter ONCE when a wire is removed.
        if self.pitch_mod.connected_outputs:
            eff = float(self.pitch_mod.get_tensor()[0, 0].item())
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["pitch_shift"], eff)
            self._was_pitch_mod_connected = True
        elif self._was_pitch_mod_connected:
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["pitch_shift"],
                               float(self.params["pitch_shift"].value))
            self._was_pitch_mod_connected = False

        if self.formant_mod.connected_outputs:
            eff = float(self.formant_mod.get_tensor()[0, 0].item())
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["formant_shift"], eff)
            self._was_formant_mod_connected = True
        elif self._was_formant_mod_connected:
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["formant_shift"],
                               float(self.params["formant_shift"].value))
            self._was_formant_mod_connected = False

        # 3. Native dispatch with FFINode's channel-adaptation policy
        #    (mono -> stereo duplication; never shrink outputs via out=).
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
        # Fixed emission latency (inputLatency + outputLatency inside the
        # DSP); the dry path is delayed by the same amount.
        latency_samples = 5760
        if self.lib and self.dsp_handle and hasattr(self.lib, "get_latency_samples"):
            try:
                latency_samples = int(self.lib.get_latency_samples(self.dsp_handle))
            except Exception:
                pass
        return {
            "latency_samples": latency_samples,
            "latency_ms": round(latency_samples / float(SAMPLE_RATE) * 1000.0, 2),
        }
