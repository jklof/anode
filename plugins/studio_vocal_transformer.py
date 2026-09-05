"""
StudioVocalTransformer — studio-grade vocal pitch correction, formant sculpting,
and acoustic gender transformation (Effects).

Thin FFINode wrapper over libstudio_vocal_transformer
(cpp/studio_vocal_transformer.cpp). The native side unites a real-time retune
front end with a spectral vocal pipeline (true-envelope phase vocoder with
VTLN formant warp):

Retune front end:
  - Anti-aliased (2-pole Butterworth lowpass, fc = 1.2 kHz) NSDF real-time pitch
    tracker with continuity scoring, fundamental-preference harmonic
    unwinding, octave-jump guard, and parabolic peak refinement on a 12 kHz
    analysis buffer.
  - Dual-mode retune: scale snapping (12-bit pitch-class bitmask rotated to a
    root) or live MIDI note targeting via the `midi_in` port. An exponential
    target-approach glide governs retune speed (0 ms hard T-Pain snap .. 100 ms
    transparent studio correction).
  - Synthesized vibrato (depth / rate) summed on top of the retune shift.

Spectral core:
  - Resampled-frame pitch shifting (the shift happens in the frame read),
    so pitch and formants stay decoupled (no chipmunk effect).
  - 2048-pt True-Envelope estimation (PCHIP peak hull + symmetric cepstral
    liftering) with same-grid peak-locked phase vocoding.
  - Piecewise-linear knee VTLN warp with F1 decoupling, log-octave spectral
    tilt, and dynamic H1 harmonic shaping for the gender morph.
  - Voiced/unvoiced-gated sibilant bypass (dry consonants pass through
    untouched) and transient-gated phase reset (plosives never smear).
  - Tract-shaped 1.5-7 kHz aspiration noise with pitch-synchronous
    glottal modulation.
  - Algorithmic latency of 9216 samples (192.0 ms @ 48 kHz); `mix`
    crossfades cleanly with the latency-aligned dry path.

Threading / RT notes (AGENTS.md):
  - Node construction (including native library load) happens off the audio
    thread. process() performs no I/O, no allocations in steady state, and
    only pushes staged parameters / block-rate CV via set_param().
  - Scale and MIDI derived parameters are change-detected in Python so the
    native side sees no redundant set_param() traffic.
"""

import ctypes

from ffi_base import FFINode
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

SCALES = {
    "Chromatic": 0b111111111111,
    "Major":     0b101011010101,
    "Minor":     0b101101011010,
    "Harmonic Minor": 0b101101011001,
    "Pentatonic": 0b101001010010,
    "Bypass":    0b000000000000,
}

ROOT_NOTES = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}

_ROOT_NAMES = tuple(ROOT_NOTES.keys())
_SCALE_NAMES = tuple(SCALES.keys())
_ROOT_VALUES = tuple(float(ROOT_NOTES[k]) for k in _ROOT_NAMES)
_SCALE_VALUES = tuple(float(SCALES[k]) for k in _SCALE_NAMES)


class StudioVocalTransformer(FFINode):
    category = "Effects"
    label = "Studio Vocal Transformer"
    description = (
        "Studio-grade vocal processing suite. Real-time pitch correction "
        "(hard-tune to natural glide), scale snapping, MIDI note targeting, "
        "advanced vocal timbre and gender coloration, "
        "gentle aspiration, and "
        "spectral pitch shifting with independent formant control."
    )

    LIB_NAME = "studio_vocal_transformer"
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

    PARAM_SCALE_ROOT_ID = 1
    PARAM_SCALE_MASK_ID = 2
    PARAM_MIDI_MODE_ID = 12
    PARAM_TARGET_MIDI_NOTE_ID = 13

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
            "retune_speed": 0.0,
            "pitch_shift": 0.0,
            "formant_shift": 0.0,
            "gender_morph": 0.0,
            "breathiness": 0.0,
            "sibilant_bypass": 0.9,
            "mix": 1.0,
        },
        "Transparent Vocal Polisher": {
            "correction_enable": 1.0,
            "retune_speed": 45.0,
            "pitch_shift": 0.0,
            "formant_shift": 0.2,
            "gender_morph": 0.1,
            "breathiness": 0.08,
            "sibilant_bypass": 0.95,
            "mix": 1.0,
        },
        # Speech conversion: pitch correction stays OFF (scale snapping
        # warbles prosody). Pitch + formants do the gender work; the VTLN
        # warp and H1 shaping inside gender_morph carry the timbre.
        "Male -> Female Speech": {
            "correction_enable": 0.0,
            "retune_speed": 20.0,
            "pitch_shift": 9.0,
            "formant_shift": 1.0,
            "gender_morph": 0.85,
            "breathiness": 0.20,
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
        "Female -> Male Speech": {
            "correction_enable": 0.0,
            "retune_speed": 20.0,
            "pitch_shift": -9.0,
            "formant_shift": -1.0,
            "gender_morph": -0.80,
            "breathiness": 0.05,
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
    }

    def __init__(self, name=""):
        super().__init__(name)

        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False
        self._active_midi_note = -1.0
        self._last_scale_root = None
        self._last_scale_mask = None
        # Change-detected MIDI derived params: avoids two redundant native
        # set_param() calls on every block when nothing changed.
        self._last_midi_mode = None
        self._last_midi_target = None

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
                             help="Very subtle band-limited vocal air level.")
        self.add_float_param("sibilant_bypass", 0.85, 0.0, 1.0, unit="",
                             help="High-frequency unvoiced consonant preservation.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Wet/dry mix (latency-aligned dry path).")

    def start(self):
        super().start()
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False
        self._active_midi_note = -1.0
        self._last_scale_root = None
        self._last_scale_mask = None
        self._last_midi_mode = None
        self._last_midi_target = None

    def load_state(self, data: dict):
        super().load_state(data)
        # Force derived params to re-push on the next block; the active MIDI
        # note itself is performance state (not saved), so clear it.
        self._last_scale_root = None
        self._last_scale_mask = None
        self._last_midi_mode = None
        self._last_midi_target = None
        self._active_midi_note = -1.0

    def _sync_scale_parameters(self):
        if not self.lib or not self.dsp_handle:
            return

        root_p = self.params["scale_root"].value
        if isinstance(root_p, str):
            root_val = float(ROOT_NOTES.get(root_p, 0))
        else:
            idx = int(root_p)
            root_val = _ROOT_VALUES[idx] if 0 <= idx < len(_ROOT_VALUES) else 0.0

        scale_p = self.params["scale_type"].value
        if isinstance(scale_p, str):
            mask_val = float(SCALES.get(scale_p, SCALES["Major"]))
        else:
            idx = int(scale_p)
            mask_val = _SCALE_VALUES[idx] if 0 <= idx < len(_SCALE_VALUES) else float(SCALES["Major"])

        if root_val != self._last_scale_root or mask_val != self._last_scale_mask:
            self.lib.set_param(self.dsp_handle, self.PARAM_SCALE_ROOT_ID, root_val)
            self.lib.set_param(self.dsp_handle, self.PARAM_SCALE_MASK_ID, mask_val)
            self._last_scale_root = root_val
            self._last_scale_mask = mask_val

    def _sync_midi_parameters(self):
        # Fold the block's MIDI packet into the latched target note, then
        # push the derived (mode, target) pair only when it changed. Steady
        # state with no MIDI traffic performs zero native calls here.
        packet = self.midi_in.get_packet()
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
            midi_target = self._active_midi_note
        else:
            midi_active = 0.0
            # Keep the last target value stable when idle; only the mode
            # matters to the native side in that state.
            midi_target = self._last_midi_target if self._last_midi_target is not None else -1.0

        if midi_active != self._last_midi_mode:
            self.lib.set_param(self.dsp_handle, self.PARAM_MIDI_MODE_ID, midi_active)
            self._last_midi_mode = midi_active
        if self._active_midi_note >= 0.0 and midi_target != self._last_midi_target:
            self.lib.set_param(self.dsp_handle, self.PARAM_TARGET_MIDI_NOTE_ID, midi_target)
            self._last_midi_target = midi_target
        elif self._last_midi_target is None:
            # First block with no MIDI: publish the idle target once so the
            # native side starts from a defined state.
            self.lib.set_param(self.dsp_handle, self.PARAM_TARGET_MIDI_NOTE_ID, -1.0)
            self._last_midi_target = -1.0

    def process(self):
        out_slot = self.outputs.get("out")
        if not self.lib or not self.dsp_handle:
            if out_slot:
                out_slot.buffer.zero_()
            return

        # 1. Sync staged parameters & derived scale parameters
        self._sync_params_to_cpp()
        self._sync_scale_parameters()

        # 2. MIDI note targeting (change-detected; silent when idle)
        self._sync_midi_parameters()

        # 3. Block-rate CV with disconnect detection
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

        # 4. Channel adaptation & FFI dispatch (mirrors FFINode policy:
        #    mono -> stereo duplication; never shrink outputs via out=).
        raw_tensor = self.inp.get_tensor()
        processed_tensor = self._preprocess_input(raw_tensor, self._ffi_in_buffer)
        in_channels = processed_tensor.shape[0]

        if out_slot is None:
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
        latency_samples = 9216
        return {
            "latency_samples": latency_samples,
            "latency_ms": round(latency_samples / float(SAMPLE_RATE) * 1000.0, 2),
        }
