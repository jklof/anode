"""
WorldVoiceTransformer — real-time voice pitch/formant transformer (Effects).

Thin FFINode wrapper over libworld_transformer
(cpp/world_transformer.cpp). The native side combines ReIm streaming
analysis (5 ms framing, silence gating, DIO + instantaneous-frequency +
SRH F0 tracking) with WORLD v1.0.1 acoustic analysis (CheapTrick spectral
envelope, D4C band-aperiodicity) and WORLD sequential real-time synthesis
(WorldSynthesizer / Synthesis2):

- F0 analysis runs once per hop on the mono downmix; both stereo channels
  are synthesized from the shared parameters (independent synthesis noise
  gives a natural stereo image; bit-identical mono inputs still produce
  bit-identical outputs).
- D4C runs at half rate (every 2nd hop); aperiodicity evolves slowly and
  the held frame stays valid across one hop.
- Fixed algorithmic latency of 1024 samples (21.3 ms at 48 kHz); `mix`
  crossfades with the latency-aligned dry path, and `mix = 0` is a
  bit-exact zero-latency bypass.

The Python side only marshals pointers, pushes staged parameters once per
change, and forwards block-rate CV from the modulation sockets.
"""

import ctypes

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


class WorldVoiceTransformer(FFINode):
    category = "Effects"
    label = "WORLD Voice Transformer"
    description = (
        "Real-time voice pitch and formant transformer combining ReIm "
        "streaming F0 tracking with WORLD vocoding (CheapTrick envelope, "
        "D4C aperiodicity, real-time synthesis). Independent pitch "
        "transposition, formant warping, acoustic gender morphing, and "
        "native aperiodicity breathiness. Fixed algorithmic latency of "
        "1024 samples (21.3 ms at 48 kHz); mix crossfades cleanly with the "
        "latency-aligned dry path."
    )

    LIB_NAME = "world_transformer"
    # Matches cpp/world_transformer.cpp ParamId
    PARAM_MAP = {
        "pitch_shift": 0,
        "formant_shift": 1,
        "mix": 2,
        "output_gain": 3,
        "gender_morph": 4,
        "breathiness": 5,
    }

    # Presets mirror the VocalTransformer schema for the ui_system NodeItem menu.
    PRESETS = {
        "Male -> Female": {
            "pitch_shift": 9.5,
            "formant_shift": 1.0,
            "gender_morph": 0.85,
            "breathiness": 0.20,
            "output_gain": 0.0,
            "mix": 1.0,
        },
        "Female -> Male": {
            "pitch_shift": -9.0,
            "formant_shift": -0.8,
            "gender_morph": -0.80,
            "breathiness": 0.05,
            "output_gain": 0.0,
            "mix": 1.0,
        },
        "Neutral (Reset)": {
            "pitch_shift": 0.0,
            "formant_shift": 0.0,
            "gender_morph": 0.0,
            "breathiness": 0.0,
            "output_gain": 0.0,
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

        self.add_float_param("pitch_shift", 0.0, -12.0, 12.0, unit="st",
                             help="Fundamental pitch shift in semitones.")
        self.add_float_param("formant_shift", 0.0, -12.0, 12.0, unit="st",
                             help="Vocal tract resonance / formant shift in semitones.")
        self.add_float_param("gender_morph", 0.0, -1.0, 1.0, unit="",
                             help="Vocal tract length morphing & glottal character (+1.0 = Feminine, "
                                  "-1.0 = Masculine). Shifts formants ±3 st, applies glottal spectral "
                                  "tilt, and shapes H1/H2 fundamental balance.")
        self.add_float_param("breathiness", 0.0, 0.0, 1.0, unit="",
                             help="Vocal cord aspiration / breathiness level. Injects native aperiodicity "
                                  "into the 1.5-7 kHz band on voiced speech.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet crossfade (0.0 = dry bypass, 1.0 = transformed "
                                  "vocal). The dry path is latency-aligned inside the DSP, "
                                  "so intermediate values crossfade without comb filtering.")
        self.add_float_param("output_gain", 0.0, -12.0, 12.0, unit="dB",
                             help="Post-synthesis output trim applied to the wet path.")

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
        # Fixed emission latency (see cpp kLatency); the dry path is delayed
        # by the same amount inside the DSP.
        latency_samples = 1024
        if self.lib and self.dsp_handle and hasattr(self.lib, "get_latency_samples"):
            try:
                latency_samples = int(self.lib.get_latency_samples(self.dsp_handle))
            except Exception:
                pass
        return {
            "latency_samples": latency_samples,
            "latency_ms": round(latency_samples / float(SAMPLE_RATE) * 1000.0, 2),
        }
