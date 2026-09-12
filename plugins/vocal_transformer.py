"""
VocalTransformer — studio-grade human voice pitch, formant, and gender
transformation with optional real-time pitch correction (Effects).

Thin FFINode wrapper over libvocal_transformer (cpp/vocal_transformer.cpp).
All spectral processing (true-envelope cepstral fitting, resampled-frame
pitch shifting, peak-locked phase vocoding, asymmetric multi-band VTLN
warping with precomputed spectral-tilt / H1 harmonic shaping, and
tract-shaped 1.5-7 kHz aspiration noise) runs natively; the Python side only
marshals pointers, pushes staged parameters once per change, and forwards
block-rate CV from the modulation sockets.

Gender morphing uses an asymmetric vocal-tract-length warp: the F1 region
(< 1 kHz) is decoupled from F2/F3 (1-4 kHz) because the adult male pharynx
is disproportionately long relative to the oral cavity. Feminine morphs
(+1) additionally apply a precomputed spectral tilt (HF attenuation,
clamped to +-8 dB) plus H1 harmonic emphasis (0-350 Hz, gated off DC) to
avoid a buzzy/pinched sound when shifting up, and broaden formant bandwidths
via an adaptive cepstral lifter cutoff.

Optional retune front end (off by default): a 12 kHz NSDF pitch tracker
with continuity scoring and octave-jump guard drives scale snapping (12-bit
pitch-class mask rotated to a root), live MIDI note targeting via the
`midi_in` port, an exponential target-approach glide (retune speed), and
synthesized vibrato. With correction and MIDI both off, tracking is skipped
entirely and the node is a pure manual pitch/formant/gender shifter.

Latency: 9216 samples (192 ms @ 48 kHz) in Studio mode across the whole
pitch range — see get_telemetry(). Live Tracking (2560 spls / 53 ms) and
Ultra-Low (2048 spls / 42 ms) modes trade FFT resolution and pitch range
for monitor-friendly latency. The 'mix' parameter is latency-compensated:
intermediate values crossfade cleanly with the dry path without comb filtering.
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

LATENCY_MODES = [
    "Studio (HQ / 192ms)",
    "Live Tracking (53ms)",
    "Ultra-Low (42ms)",
]


class VocalTransformer(FFINode):
    category = "Effects"
    label = "Vocal Transformer"
    description = (
        "Studio-grade real-time vocal pitch, formant, and gender transformer. "
        "Employs True-Envelope peak cepstral estimation, peak-locked phase "
        "vocoding, and asymmetric multi-band vocal tract length normalization "
        "(VTLN) — decoupling the F1 region from F2/F3, with precomputed "
        "spectral-tilt and H1 harmonic shaping to eliminate the buzzy/pinched "
        "quality on upward shifts and dullness on downward shifts, plus "
        "tract-filtered 1.5-7 kHz aspiration noise. Optional auto-tune style "
        "pitch correction (hard-tune to natural glide), scale snapping, MIDI "
        "note targeting, and synthesized vibrato. Algorithmic latency of "
        "9216 samples (192 ms at 48 kHz) in Studio mode, 2560 (53 ms) Live, "
        "2048 (42 ms) Ultra-Low; mix "
        "crossfades cleanly with the latency-aligned dry path."
    )

    LIB_NAME = "vocal_transformer"
    # Matches cpp/vocal_transformer.cpp set_param switch-case
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
        "latency_mode": 14,
    }

    PARAM_SCALE_ROOT_ID = 1
    PARAM_SCALE_MASK_ID = 2
    PARAM_MIDI_MODE_ID = 12
    PARAM_TARGET_MIDI_NOTE_ID = 13

    # Gender-transformation macro presets (surfaced in the node's
    # right-click "Presets" context menu; see ui_system.NodeItem).
    #
    # The acoustics they encode:
    #   - pitch: F0 relocation (male 85-155 Hz <-> female 165-255 Hz)
    #   - formant_shift: fine vocal-tract length trim on top of gender_morph VTLN
    #   - gender_morph: vocal-tract length warp + glottal-source shaping —
    #     feminine needs a steeper HF tilt + dominant fundamental (H1 >> H2);
    #     masculine needs a flatter tilt for chest resonance
    #   - breathiness: incomplete posterior glottal closure (female) vs
    #     tight closure (male)
    #   - sibilant_bypass: unvoiced-consonant dry blend (V/UV-gated in the
    #     native DSP, so vowels never comb-filter against it)
    PRESETS = {
        "Male -> Female": {
            "pitch_shift": 9.5,      # e.g. 110 Hz -> ~190 Hz
            "formant_shift": 1.0,    # fine trim on top of gender_morph VTLN (+3.5 st net)
            "gender_morph": 0.85,    # asymmetric VTLN, H1>>H2, glottal tilt, bandwidth broadening
            "breathiness": 0.20,     # natural vocal-cord leakage
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
        "Female -> Male": {
            "pitch_shift": -9.0,     # e.g. 220 Hz -> ~130 Hz
            "formant_shift": -0.8,   # fine trim on top of gender_morph VTLN (-3.2 st net)
            "gender_morph": -0.80,   # flatter tilt, richer chest resonance
            "breathiness": 0.05,     # tighter glottal closure
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
        "Neutral (Reset)": {
            "pitch_shift": 0.0,
            "formant_shift": 0.0,
            "gender_morph": 0.0,
            "breathiness": 0.0,
            "sibilant_bypass": 0.85,
            "mix": 1.0,
        },
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

        # Modulation disconnect detection (karplus_strong pattern). Removing a
        # wire changes no parameter value, so without these flags the native
        # DSP would stay stuck at the last CV value (_sync_params_to_cpp sees
        # nothing dirty after the disconnect).
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False
        self._active_midi_note = -1.0
        self._last_scale_root = None
        self._last_scale_mask = None
        # Change-detected MIDI derived params: avoids two redundant native
        # set_param() calls on every block when nothing changed.
        self._last_midi_mode = None
        self._last_midi_target = None

        # Audio sockets
        self.inp = self.add_input(
            "in", help="Vocal signal to transform; mono inputs are duplicated to stereo.")
        self.midi_in = self.add_midi_input(
            "midi_in", help="Optional MIDI input for live note targeting.")
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
        self.add_float_param("correction_enable", 0.0, 0.0, 1.0, unit="",
                             help="Enable real-time pitch correction (1.0 = On, 0.0 = Off).")
        self.add_menu_param("scale_root", list(ROOT_NOTES.keys()), initial_idx=0,
                            help="Root note of the musical scale.")
        self.add_menu_param("scale_type", list(SCALES.keys()), initial_idx=1,
                            help="Musical scale type used for snapping.")
        self.add_float_param("retune_speed", 20.0, 0.0, 100.0, unit="ms",
                             help="Pitch snapping transition time (0 ms = hard snap, 100 ms = natural).")
        self.add_float_param("pitch_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Fundamental pitch shift in semitones.")
        self.add_float_param("formant_shift", 0.0, -24.0, 24.0, unit="st",
                             help="Vocal tract resonance / formant shift in semitones.")
        self.add_float_param("gender_morph", 0.0, -1.0, 1.0, unit="",
                             help="Vocal tract length morphing (+1.0 = feminine/child "
                                  "short tract, 0.0 = neutral, -1.0 = masculine/deep). "
                                  "F1 is warped at reduced intensity; positive values "
                                  "add spectral-tilt + H1 harmonic emphasis and broaden "
                                  "formant bandwidths.")
        self.add_float_param("vibrato_depth", 0.0, 0.0, 2.0, unit="st",
                             help="Synthesized vibrato depth in semitones.")
        self.add_float_param("vibrato_rate", 5.5, 2.0, 9.0, unit="Hz",
                             help="Synthesized vibrato modulation rate in Hz.")
        self.add_float_param("breathiness", 0.0, 0.0, 1.0, unit="",
                             help="Vocal aspiration noise level. Deterministic, "
                                  "tract-shaped noise injected into the 1.5-7 kHz "
                                  "band only (avoids low-frequency rumble).")
        self.add_float_param("sibilant_bypass", 0.85, 0.0, 1.0, unit="",
                             help="Preserves natural unvoiced consonants (/s/, /t/, /k/) "
                                  "without pitch artifacts.")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet crossfade (0.0 = dry bypass, 1.0 = transformed "
                                  "vocal). The dry path is latency-aligned inside the DSP, "
                                  "so intermediate values crossfade without comb filtering.")
        self.add_menu_param(
            "latency_mode", LATENCY_MODES, initial_idx=0,
            help="Algorithmic latency tradeoff: Studio (2048-pt FFT, highest "
                 "frequency resolution down to 80 Hz, +-24 st, 192 ms), Live "
                 "Tracking (1024-pt FFT, 53 ms, +-12 st, >= 140 Hz material), "
                 "or Ultra-Low (1024-pt FFT, 42 ms, +-7 st, in-ear monitoring). "
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
        self._was_pitch_mod_connected = False
        self._was_formant_mod_connected = False

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
        # Mirrors plugins/filters.py BiquadFilter.process(): replicate the
        # FFINode dispatch so block-rate CV can be pushed BETWEEN staged
        # parameter sync and the native process() call.
        if not self.lib or not self.dsp_handle:
            # Anti-ghosting: never leave stale audio in the output buffer
            # (AGENTS.md §2 — every audio output must be fully written every
            # processed block), even when the native backend is unavailable.
            out_slot = self.outputs.get("out")
            if out_slot:
                out_slot.buffer.zero_()
            return

        # 1. Sync staged parameters (canonical path) & derived scale parameters
        self._sync_params_to_cpp()
        self._sync_scale_parameters()

        # 2. MIDI note targeting (change-detected; silent when idle)
        self._sync_midi_parameters()

        # 3. Block-rate modulation: push directly after staged sync.
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

        # 4. Native dispatch with FFINode's channel-adaptation policy
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
        # Emission latency depends on the latency mode (see cpp
        # configure_mode); query the native side so the UI always reports
        # the active value.
        latency_samples = 9216
        if self.lib and self.dsp_handle and hasattr(self.lib, "get_latency_samples"):
            try:
                latency_samples = int(self.lib.get_latency_samples(self.dsp_handle))
            except Exception:
                pass
        return {
            "latency_samples": latency_samples,
            "latency_ms": round(latency_samples / float(SAMPLE_RATE) * 1000.0, 2),
        }
