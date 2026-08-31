"""
VocalTransformer — studio-grade human voice pitch, formant, and gender
transformation (Effects).

Thin FFINode wrapper over libvocal_transformer (cpp/vocal_transformer.cpp).
All spectral processing (true-envelope cepstral fitting, destination-grid
bin mapping, peak-locked phase vocoding, VTLN warping) runs natively; the
Python side only marshals pointers, pushes staged parameters once per
change, and forwards block-rate CV from the modulation sockets.

Latency: 1024 samples (one FFT window) — see get_telemetry(). The 'mix'
parameter is NOT latency-compensated: below 1.0 it is a comb-filtering
special effect, matching the RubberbandPitchShifter disclosure.
"""

import ctypes

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


class VocalTransformer(FFINode):
    category = "Effects"
    label = "Vocal Transformer"
    description = (
        "Studio-grade real-time vocal pitch, formant, and gender transformer. "
        "Employs True-Envelope peak cepstral estimation, peak-locked phase "
        "vocoding, and vocal tract length normalization (VTLN) to eliminate "
        "metallic phasiness, prevent pitch-harmonic comb ripples, and preserve "
        "crisp unvoiced sibilants. Algorithmic latency 1024 samples at neutral "
        "pitch (up to ~85 ms at extreme shifts); mix below 1.0 is a "
        "comb-filtering effect, not a compensated crossfade."
    )

    LIB_NAME = "vocal_transformer"
    # Matches cpp/vocal_transformer.cpp set_param switch-case
    PARAM_MAP = {
        "pitch_shift": 0,
        "formant_shift": 1,
        "gender_morph": 2,
        "breathiness": 3,
        "sibilant_bypass": 4,
        "mix": 5,
    }

    def __init__(self, name=""):
        super().__init__(name)

        # Modulation disconnect detection (karplus_strong pattern). Removing a
        # wire changes no parameter value, so without these flags the native
        # DSP would stay stuck at the last CV value (_sync_params_to_cpp sees
        # nothing dirty after the disconnect).
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False

        # Audio sockets
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

        # Parameters (defaults/ranges match the native constructor defaults)
        self.add_float_param("pitch_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Fundamental pitch shift in semitones.")
        self.add_float_param("formant_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Vocal tract resonance / formant shift in semitones.")
        self.add_float_param("gender_morph", 0.0, -1.0, 1.0, unit="",
                             help="Vocal tract length morphing (-1.0 = feminine/child, "
                                  "0.0 = neutral, +1.0 = masculine/deep).")
        self.add_float_param("breathiness", 0.0, 0.0, 1.0, unit="",
                             help="Vocal aspiration noise level.")
        self.add_float_param("sibilant_bypass", 0.8, 0.0, 1.0, unit="",
                             help="Preserves natural unvoiced consonants (/s/, /t/, /k/) "
                                  "without pitch artifacts.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet crossfade (0.0 = dry only, 1.0 = transformed). "
                                  "NOT latency-compensated: below 1.0 this combs the "
                                  "spectrum because the wet path lags the dry path by "
                                  "up to 4096 samples.")

    def process(self):
        # Mirrors plugins/filters.py BiquadFilter.process(): replicate the
        # FFINode dispatch so block-rate CV can be pushed BETWEEN staged
        # parameter sync and the native process() call.
        if not self.lib or not self.dsp_handle:
            return

        # 1. Sync staged parameters (canonical path)
        self._sync_params_to_cpp()

        # 2. Block-rate modulation: push directly after staged sync.
        #    First sample of the block, matching RubberbandPitchShifter.
        #    Disconnect contract: when a mod input is disconnected, re-push the
        #    staged parameter ONCE — _sync_params_to_cpp() would otherwise skip
        #    the push (nothing is dirty) and the native DSP would stay stuck at
        #    the last CV value. Mirrors plugins/karplus_strong.py.
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
            # Mono -> stereo duplication (ffi_base policy; see ffi_base.process)
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
        # Fixed emission latency (see cpp kLatency): frames span 1024*ratio
        # input samples (ratio <= 4 at +-24 st) and the read pointer trails the
        # input stream by L = 512 + 1024*4 = 4608 samples so every emitted
        # sample is fully accumulated.
        latency_samples = 4608
        return {
            "latency_samples": latency_samples,
            "latency_ms": round(latency_samples / float(SAMPLE_RATE) * 1000.0, 2),
        }
