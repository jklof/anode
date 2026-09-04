"""
StudioVocalTransformer — studio-grade vocal pitch correction, formant sculpting,
and acoustic gender transformation (Effects).

Thin FFINode wrapper over libstudio_vocal_transformer
(cpp/studio_vocal_transformer.cpp). The native side unites a real-time retune
front end with the full VocalTransformer spectral core:

Retune front end:
  - NSDF (Normalized Square Difference Function) real-time pitch tracker with
    parabolic peak interpolation on a 12 kHz analysis buffer (sub-cent F0).
  - Dual-mode retune: scale snapping (12-bit pitch-class bitmask rotated to a
    root) or live MIDI note targeting via the `midi_in` port. An exponential
    target-approach glide governs retune speed (0 ms hard T-Pain snap .. 100 ms
    transparent studio correction).
  - Synthesized vibrato (depth / rate) summed on top of the retune shift.

Spectral core (identical to VocalTransformer):
  - PCHIP monotonic true-envelope, peak-locked phase vocoder, F1-decoupled VTLN,
    H1/H2 glottal shaping, pitch-synchronous aspiration, unvoiced sibilant
    bypass, transient-gated phase reset, and a fixed 9216-sample (192 ms @ 48 kHz)
    latency-aligned OLA ring buffer; `mix` crossfades cleanly with the dry path.

The Python side only marshals pointers, pushes staged parameters once per change,
forwards block-rate CV from the modulation sockets, and resolves MIDI note
targets. Block-rate pitch/formant CV and the MIDI path push native parameters
directly once per block, exactly like the other FFI voice/vocoder nodes.
"""

import ctypes

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

# Musical scale definitions (bitmasks over 12 semitones: C=0, C#=1, ... B=11).
SCALES = {
    "Chromatic": 0b111111111111,
    "Major":     0b101011010101,  # C, D, E, F, G, A, B
    "Minor":     0b101101011010,  # natural minor
    "Harmonic Minor": 0b101101011001,
    "Pentatonic": 0b101001010010,
    "Bypass":    0b000000000000,
}

ROOT_NOTES = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}


class StudioVocalTransformer(FFINode):
    category = "Effects"
    label = "Studio Vocal Transformer"
    description = (
        "Studio-grade vocal processing suite. Real-time pitch correction "
        "(hard-tune to natural glide), scale snapping, MIDI note targeting, "
        "advanced acoustic gender transformation (F1-decoupled VTLN, H1/H2 "
        "glottal reshaping, pitch-synchronous aspiration), and "
        "transient-preserved phase vocoding."
    )

    LIB_NAME = "studio_vocal_transformer"
    # Matches cpp/studio_vocal_transformer.cpp set_param switch-case. Only the
    # 1:1 float parameters are listed here; scale_root/scale_mask/midi_mode/
    # target_midi_note are derived (menu / MIDI packet) and pushed manually so
    # _sync_params_to_cpp() never dereferences a non-existent parameter.
    PARAM_MAP = {
        "correction_enable": 0,
        "retune_speed": 3,
        "pitch_shift": 4,
        "formant_shift": 5,
        "gender_morph": 6,
        "vibrato_depth": 7,
        "vibrato_rate": 8,
        "breathiness": 9,
        "sibilant_bypass": 10,
        "mix": 11,
    }

    # Derived native IDs (not part of PARAM_MAP; see _sync_scale_parameters /
    # the MIDI block in process()).
    PARAM_SCALE_ROOT_ID = 1
    PARAM_SCALE_MASK_ID = 2
    PARAM_MIDI_MODE_ID = 12
    PARAM_TARGET_MIDI_NOTE_ID = 13

    # Gender-transformation / FX macro presets. Pitch relocates F0, formant_shift
    # trims vocal-tract length on top of the gender_morph VTLN, breathiness adds
    # glottal aspiration, sibilant_bypass preserves consonants.
    PRESETS = {
        "Male -> Female Pop Lead": {
            "correction_enable": 1.0,
            "retune_speed": 15.0,
            "pitch_shift": 10.0,
            "formant_shift": 1.5,
            "gender_morph": 0.85,
            "breathiness": 0.20,
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
        "Female -> Male Deep Chest": {
            "correction_enable": 1.0,
            "retune_speed": 20.0,
            "pitch_shift": -10.0,
            "formant_shift": -1.2,
            "gender_morph": -0.80,
            "breathiness": 0.05,
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
        "Hard-Tune (T-Pain FX)": {
            "correction_enable": 1.0,
            "retune_speed": 0.0,     # instantaneous snap
            "pitch_shift": 0.0,
            "formant_shift": 0.0,
            "gender_morph": 0.0,
            "breathiness": 0.0,
            "sibilant_bypass": 0.9,
            "mix": 1.0,
        },
        "Transparent Vocal Polisher": {
            "correction_enable": 1.0,
            "retune_speed": 45.0,    # transparent natural glide
            "pitch_shift": 0.0,
            "formant_shift": 0.2,
            "gender_morph": 0.1,
            "breathiness": 0.08,
            "sibilant_bypass": 0.95,
            "mix": 1.0,
        },
    }

    def __init__(self, name=""):
        super().__init__(name)

        # Modulation / MIDI disconnect detection (karplus_strong pattern).
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False
        self._active_midi_note = -1.0
        self._last_scale_root = None
        self._last_scale_mask = None

        # ---- Ports ----
        self.inp = self.add_input(
            "in", help="Vocal audio input; mono inputs are duplicated to stereo.")
        self.midi_in = self.add_midi_input(
            "midi_in", help="Optional MIDI input for live note targeting.")
        self.pitch_mod = self.add_input(
            "pitch_mod", "pitch_shift",
            help="Block-rate pitch CV in semitones (bound to 'pitch_shift'; "
                 "first sample of each block). Unconnected: uses the parameter.")
        self.formant_mod = self.add_input(
            "formant_mod", "formant_shift",
            help="Block-rate formant CV in semitones (bound to 'formant_shift'; "
                 "first sample of each block). Unconnected: uses the parameter.")
        self.out = self.add_output(
            "out", channels=CHANNELS, help="Transformed stereo vocal output.")

        # ---- Parameters ----
        self.add_float_param("correction_enable", 1.0, 0.0, 1.0, unit="",
                             help="Enable real-time pitch correction (1.0 = On, 0.0 = Off).")
        self.add_menu_param("scale_root", list(ROOT_NOTES.keys()), initial_idx=0,
                            help="Root note of the musical scale.")
        self.add_menu_param("scale_type", list(SCALES.keys()), initial_idx=1,
                            help="Musical scale type used for snapping.")
        self.add_float_param("retune_speed", 20.0, 0.0, 100.0, unit="ms",
                             help="Pitch snapping transition time (0 ms = hard snap, 100 ms = natural).")
        self.add_float_param("pitch_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Manual pitch relocation in semitones.")
        self.add_float_param("formant_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Formant / resonance shift in semitones.")
        self.add_float_param("gender_morph", 0.0, -1.0, 1.0, unit="",
                             help="Vocal tract length & glottal balance (+1 = Feminine, -1 = Masculine).")
        self.add_float_param("vibrato_depth", 0.0, 0.0, 2.0, unit="st",
                             help="Synthesized vibrato depth in semitones.")
        self.add_float_param("vibrato_rate", 5.5, 2.0, 9.0, unit="Hz",
                             help="Synthesized vibrato modulation rate in Hz.")
        self.add_float_param("breathiness", 0.0, 0.0, 1.0, unit="",
                             help="Glottally modulated aspiration noise level.")
        self.add_float_param("sibilant_bypass", 0.85, 0.0, 1.0, unit="",
                             help="High-frequency unvoiced consonant preservation.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Wet/dry mix (latency-aligned dry path).")

    def _sync_scale_parameters(self):
        """Map the menu params to the native scale_root / scale_mask and push
        them (only on change, so per-block automation never spams set_param).
        menu .value is the item index, not the display string."""
        if not self.lib or not self.dsp_handle:
            return
        root_names = list(ROOT_NOTES.keys())
        scale_names = list(SCALES.keys())
        root_val = float(ROOT_NOTES[root_names[int(self.params["scale_root"].value)]])
        mask_val = float(SCALES[scale_names[int(self.params["scale_type"].value)]])
        if root_val != self._last_scale_root or mask_val != self._last_scale_mask:
            self.lib.set_param(self.dsp_handle, self.PARAM_SCALE_ROOT_ID, root_val)
            self.lib.set_param(self.dsp_handle, self.PARAM_SCALE_MASK_ID, mask_val)
            self._last_scale_root = root_val
            self._last_scale_mask = mask_val

    def process(self):
        out_slot = self.outputs.get("out")
        if not self.lib or not self.dsp_handle:
            # Anti-ghosting: never leave stale audio in the output buffer.
            if out_slot:
                out_slot.buffer.zero_()
            return

        # 1. Sync staged UI parameters (canonical path) + derived scale params.
        self._sync_params_to_cpp()
        self._sync_scale_parameters()

        # 2. MIDI note target: track the active note_on; hold until its
        #    note_off. Push the target note and midi_mode every block.
        packet = self.midi_in.get_packet()
        midi_active = 0.0
        if packet.messages:
            for _, msg in packet.messages:
                mtype = getattr(msg, "type", "")
                if mtype == "note_on" and getattr(msg, "velocity", 0) > 0:
                    self._active_midi_note = float(msg.note)
                elif mtype == "note_off" or (mtype == "note_on" and getattr(msg, "velocity", 0) == 0):
                    if float(msg.note) == self._active_midi_note:
                        self._active_midi_note = -1.0

        if self._active_midi_note >= 0.0:
            midi_active = 1.0
            self.lib.set_param(self.dsp_handle, self.PARAM_TARGET_MIDI_NOTE_ID,
                               self._active_midi_note)
        self.lib.set_param(self.dsp_handle, self.PARAM_MIDI_MODE_ID, midi_active)

        # 3. Block-rate CV with disconnect detection (first sample of block).
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

        # 4. Channel adaptation + native dispatch (FFINode policy).
        raw_tensor = self.inp.get_tensor()
        processed_tensor = self._preprocess_input(raw_tensor, self._ffi_in_buffer)
        in_channels = processed_tensor.shape[0]

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
            # Mono -> stereo duplication (ffi_base policy).
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
        # Fixed emission latency (see cpp kLatency = 9216).
        latency_samples = 9216
        return {
            "latency_samples": latency_samples,
            "latency_ms": round(latency_samples / float(SAMPLE_RATE) * 1000.0, 2),
        }