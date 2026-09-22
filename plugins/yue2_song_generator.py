"""
YuE2SongGenerator — offline lyrics-to-song generator backed by audio.cpp (Sources).

Generates complete songs (vocals + accompaniment) from a style prompt and
lyrics using the YuE2-3B GGUF model through the in-tree audio.cpp sidecar
(tools/audiocpp/). The minutes-long generation runs as a blocking call on
an NRT worker and the finished WAV is installed as RAM-cached playback —
the real-time audio thread only ever does cached-tensor playback, exactly
like SamplePlayer.

Requirements (fetched once, never auto-downloaded by the node):
  python tools/audiocpp/fetch_audiocpp.py    # native runtime (v0.8.1)
  python tools/audiocpp/fetch_yue2_gguf.py   # Q4_K_M weights + VAE + sidecars

Measured on an RTX 5070 Laptop (8 GB): Q4_K_M peaks at ~4.3 GB VRAM for a
77 s song at 2.1x realtime. Model weights are CC BY-NC 4.0
(non-commercial).

With cot=melody/full and no input score, the v0.8.1 runtime exports the
model-generated ABC plan as `score.abc` (via `--out-dir`); the node keeps
it as `plan.abc`, publishes it on the plan URI output, and shows it in the
widget score viewer, closing the plan -> edit -> re-render loop (wire plan
to `abc_uri`, edit a copy, re-render with cot=melody/full).
"""

import logging
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from PySide6.QtWidgets import QPushButton
from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    AUDIOCPP_PIN_VERSION,
    YUE2_ATTENTION_MODES,
    YUE2_FPS,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    abc_section_names,
    default_backend,
    lyric_section_names,
    run_yue2_gen,
    score_lyrics_fit_warning,
    yue2_fetch_specs,
)
from base import SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)

COT_MODES = ("off", "melody", "full")
FORMATS = ("mp3", "wav")
LORA_FILTER = "SafeTensors (*.safetensors);;All Files (*.*)"
# Fixed CLI WAV sample format: float32 is lossless (best kept WAV and best
# source for the MP3 re-encode); it costs ~2x the bytes of pcm16 with no
# VRAM or speed effect, so no user-facing choice is needed.
CLI_OUT_FORMAT = "float32"
# CLI defaults, kept in one place: the spec omits guidance/steps when they
# equal these so the CLI's own (cot-dependent) defaults apply.
DEFAULT_GUIDANCE = 1.0
DEFAULT_STEPS = 8
# Song length cap in seconds (user-facing unit). Converted to
# semantic_max_tokens = round(seconds * YUE2_FPS) for the CLI; always sent
# explicitly (240 s -> 6000 is a deliberate behavior change from the old
# uncapped CLI default of ~9000 tokens).
DEFAULT_MAX_DURATION = 240.0
MIN_MAX_DURATION = 30.0
MAX_MAX_DURATION = 900.0
PROFILES = (
    ("Q4_K_M (8 GB VRAM)", "yue2-3b-q4_k_m.gguf"),
    ("Q8_0 (16 GB VRAM)", "yue2-3b-q8_0.gguf"),
    ("BF16 (24 GB VRAM)", "yue2-3b-bf16.gguf"),
)
VAE_GGUF = "yue2-vae-f16.gguf"
# Weight compute format (yue2.model_weight_type session option). Measured on
# an 8 GB laptop GPU at equal work (1500 forced tokens): native peaks at
# ~4.0 GB / ~29 s, q4_0 at ~3.4 GB / ~43 s. q4_0 trades speed (and a small
# requantization quality cost) for headroom — the pick when native pressure
# causes paging slowdowns. Deliberately only these two: q4_k aborts the CLI
# on CUDA (unsupported getrows src0 type), and q8_0/f16/bf16 only add VRAM.
WTYPES = (
    ("native (balanced)", "native"),
    ("q4_0 (low VRAM)", "q4_0"),
)


def load_song_file(path):
    """Decode a kept song (WAV or MP3) into a (2, N) contiguous float32 CPU
    tensor at the engine rate.

    Thin wrapper over :func:`audio_io.to_engine_audio` (shared decode +
    mono-dup + NRT resample): the NRT worker calls this, tests call it
    directly. Raises RuntimeError with a human-readable message on failure.
    """
    from audio_io import to_engine_audio
    audio = to_engine_audio(path, target_sr=SAMPLE_RATE,
                            target_channels=CHANNELS, label="song")
    tensor = torch.from_numpy(audio)
    if tensor.shape[0] != CHANNELS or tensor.ndim != 2 or tensor.shape[1] < 1:
        raise RuntimeError(f"decoded song has unexpected shape {tuple(tensor.shape)}")
    return tensor


class YuE2Widget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "YuE2SongGenerator"

    PARAM_KEYS = ("style", "lyrics_file", "abc_file", "profile", "weight_type",
                  "format", "cot", "max_duration", "ar_lora", "ar_lora_scale",
                  "nar_lora", "nar_lora_scale", "guidance_scale",
                  "num_inference_steps", "attention",
                  "seed", "auto_seed")
    ACTION_LABEL = "Generate song"
    ACTION_PARAM = "generate"
    DOWNLOAD_LABEL = "Download runtime + model (~3.8 GB)"
    SHOW_SCORE = True

    def _build_after_param(self, layout, key):
        if key == "seed":
            self.btn_seed = QPushButton("Randomize seed")
            self.btn_seed.clicked.connect(self._on_seed_pressed)
            layout.addWidget(self.btn_seed)

    def _on_seed_pressed(self):
        self.proxy.set_parameter("seed", random.randrange(0, 2 ** 31))

    def _on_generate_pressed(self):
        self._on_action_pressed()

    def _on_action_pressed(self):
        # Commit the style editor text first: set_parameter only debounces,
        # and the flush preserves insertion order, so style is guaranteed
        # to be applied before generate (no stale-prompt race).
        style_widget = self.param_widgets.get("style")
        get_text = getattr(style_widget, "text", None)
        if callable(get_text):
            self.proxy.set_parameter("style", get_text())
        else:
            # Fallback for stub widgets exposing the raw line edit.
            line_edit = getattr(style_widget, "line_edit", None)
            if line_edit is not None:
                self.proxy.set_parameter("style", line_edit.text())
        self.proxy.set_parameter("generate", True)


class YuE2SongGenerator(AudioCppJob):
    category = "Offline"
    label = "YuE2 Song Generator"
    description = (
        "Offline lyrics-to-song generator (YuE2-3B GGUF via the in-tree "
        "audio.cpp sidecar). Generation runs on a background NRT worker "
        "(~2x realtime, ~4.3 GB peak VRAM with Q4_K_M on an 8 GB GPU; the "
        "weight_type menu trades speed for headroom when that pressure "
        "causes slowdowns) and "
        "publishes the kept song on the song URI output with a one-block "
        "pulse on ready for auto-chaining (e.g. into SamplePlayer). "
        "With cot=melody/full and no input score the model-generated plan "
        "is kept as plan.abc on the plan URI output for review and "
        "cover re-rendering. "
        "Generation starts from the Generate button or a rising edge on "
        "trigger_in (e.g. wired from a done pulse); a new edge restarts "
        "generation, cancelling the previous run. "
        "Playback lives in SamplePlayer; this node holds no audio. Missing "
        "runtime/models can be fetched from the node itself (Download "
        "button: resumable background download with progress) or via "
        "tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_yue2_gguf.py. "
        "Max duration caps the song length (default 240 s; the song may end "
        "earlier and is cut with a warning if longer) and bounds peak VRAM. "
        "Weights are CC BY-NC 4.0 (non-commercial)."
    )

    PULSE_OUTPUT_NAME = "ready"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background generation "
                            "(same as the Generate button), e.g. wired from a done pulse.")
        self.add_uri_input("abc_uri",
                           help="Wired score path (e.g. from SheetSage2Transcriber). While connected "
                                "and non-empty it overrides abc_file; snapshots at Generate time.")
        self.add_uri_input("lyrics_uri",
                           help="Wired lyrics path (e.g. from LyricFitter). While connected "
                                "and non-empty it overrides lyrics_file; snapshots at Generate time.")
        self.add_uri_output("song",
                            help="Kept song path (MP3 or WAV per format); published on completion and on patch-load relink.")
        self.add_uri_output("plan",
                            help="Kept model-generated ABC plan (plan.abc) for cot=melody/full without an input score; "
                                 "empty for cot=off or external-score runs. Wire to abc_uri (after editing a copy) to re-render.")
        self.ready = self.add_output("ready", channels=1,
                                     help="One-block 1.0 pulse when a new song is ready (wire to a trigger input).")

        self.add_string_param("style",
                              "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal",
                              multiline=True,
                              help="Genre, instruments, vocal character, language, tempo.")
        self.add_file_param("lyrics_file", "", filter="Text Files (*.txt *.md);;All Files (*.*)",
                            help="Lyrics file with section tags like [Verse]/[Chorus]; read by the background worker. "
                                 "A connected lyrics_uri input overrides this.")
        self.add_file_param("abc_file", "", filter="ABC Files (*.abc);;All Files (*.*)",
                            help="Optional melody/chord score for covers; requires cot=melody or full. "
                                 "A connected abc_uri input overrides this.")
        self.add_menu_param("profile", [label for label, _ in PROFILES], initial_idx=0,
                            help="GGUF precision profile; larger profiles need more VRAM.")
        self.add_menu_param("weight_type", [label for label, _ in WTYPES], initial_idx=0,
                            help="Weight compute format: q4_0 cuts peak VRAM ~15% for ~1.5x slower "
                                 "generation (and slight requantization quality cost) — use it when "
                                 "native memory pressure causes slowdowns.")
        self.add_menu_param("format", ["MP3 (320 kbps)", "WAV (lossless)"], initial_idx=0,
                            help="Kept song format — exactly one file is kept. MP3 shares cheaply; "
                                 "WAV keeps full quality for further processing.")
        self.add_menu_param("cot", list(COT_MODES), initial_idx=2,
                            help="Symbolic planning: off = direct, melody = melody plan, full = melody+chords.")
        self.add_file_param("ar_lora", "", filter=LORA_FILTER,
                            help="Optional unfused AR (planning) LoRA safetensors; empty = none. "
                                 "Changing adapters starts a fresh session per generation.")
        self.add_float_param("ar_lora_scale", 1.0, 0.0, 10.0,
                             help="AR LoRA delta scale; 0 disables the adapter.")
        self.add_file_param("nar_lora", "", filter=LORA_FILTER,
                            help="Optional unfused NAR (acoustic) LoRA safetensors; empty = none. "
                                 "Can be used alone or with an AR adapter.")
        self.add_float_param("nar_lora_scale", 1.0, 0.0, 10.0,
                             help="NAR LoRA delta scale (projection replacements stay full strength); "
                                  "0 disables the entire adapter.")
        self.add_float_param("guidance_scale", DEFAULT_GUIDANCE, 0.0, 20.0,
                             help="Semantic classifier-free guidance scale; at the 1.0 default the "
                                  "CLI's own cot-dependent default applies.")
        self.add_int_param("num_inference_steps", DEFAULT_STEPS, 1, 64,
                           help="NAR midpoint ODE steps; at the default 8 the CLI default applies.")
        self.add_float_param("max_duration", DEFAULT_MAX_DURATION,
                             MIN_MAX_DURATION, MAX_MAX_DURATION, unit="s",
                             help="Max song length in seconds (upper bound; the song may end "
                                  "earlier and is cut with a warning if it needs longer). "
                                  "Shorter caps use less VRAM.")
        self.add_menu_param("attention", list(YUE2_ATTENTION_MODES), initial_idx=0,
                            help="NAR acoustic-flow attention kernel; auto picks flash except where "
                                 "eager is faster (Turing CUDA, Intel Vulkan).")
        # Capped at 2**31 - 1: IntParamWidget is a 32-bit QSpinBox
        # (ui_system.py); the backend accepts the full [0, 2**63) range.
        self.add_int_param("seed", 831001, 0, 2 ** 31 - 1,
                           help="Generation seed; same inputs + seed reproduce the song.")
        self.add_bool_param("auto_seed", False,
                            help="Draw a fresh random seed on every Generate (written back to seed).")
        self.add_string_param("model_dir", "models/YuE2-3B-GGUF",
                              help="Model directory holding the GGUF files and sidecars/ (repo-relative).")
        self.add_bool_param("generate", False,
                            help="Transient trigger: stages a background generation, then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads missing runtime/models in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight generation.")
        # Last kept song, for save/load re-linking. Written on completion.
        # The key stays "last_wav" for saved-patch compatibility; the value
        # is whatever format was kept (song.mp3 or song.wav).
        self.add_string_param("last_wav", "",
                              help="Path of the last generated song (MP3 or WAV); re-linked on patch load.")
        self.add_string_param("last_plan", "",
                              help="Path of the last generated ABC plan (plan.abc); re-linked on patch load.")

        self._pulse = False  # emitted as a one-block pulse on ready
        self._last_trig = 0.0
        self._gen_t0 = None  # monotonic start of the in-flight job (elapsed display)
        self._plan_text = None  # generated plan text for the score viewer
        self._plan_path = ""
        self._truncated = False  # last song hit the max-duration cap
        self._cap_seconds = None  # cap in seconds for the last song
        self._status = "Idle"
        self._status_detail = "No song generated"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def _resolve_score(self):
        """Wired abc_uri wins over the abc_file param (both snapshot at
        Generate time). Raises ValueError with a clear message."""
        abc_in = self.inputs.get("abc_uri")
        if abc_in is not None and abc_in.connected_outputs:
            wired = abc_in.get_uri()
            if not wired:
                raise ValueError(
                    "Wired score is empty — transcribe first, then generate.")
            if not Path(wired).exists():
                raise ValueError(f"Wired score file not found: {wired}")
            return wired
        abc_file = self.params["abc_file"].value or None
        if abc_file and not Path(abc_file).exists():
            raise ValueError(f"ABC score file not found: {abc_file}")
        return abc_file

    def _resolve_lyrics(self):
        """Wired lyrics_uri wins over the lyrics_file param (both snapshot
        at Generate time). Raises ValueError with a clear message."""
        lyrics_in = self.inputs.get("lyrics_uri")
        if lyrics_in is not None and lyrics_in.connected_outputs:
            wired = lyrics_in.get_uri()
            if not wired:
                raise ValueError(
                    "Wired lyrics are empty — fit first, then generate.")
            if not Path(wired).exists():
                raise ValueError(f"Wired lyrics file not found: {wired}")
            return wired
        lyrics_file = self.params["lyrics_file"].value
        if not lyrics_file or not Path(lyrics_file).exists():
            raise ValueError("Pick a lyrics file first (lyrics_file is empty or missing).")
        return lyrics_file

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        runtime = AudioCppRuntime()
        ok, hint = runtime.check_ready()
        if not ok:
            raise ValueError(hint)
        model_dir = self._resolve_model_dir()
        profile_idx = int(self.params["profile"].value)
        main_gguf = PROFILES[profile_idx][1]
        if not (model_dir / main_gguf).exists():
            raise ValueError(
                f"GGUF profile {main_gguf} not found in {model_dir}. "
                "Run: python tools/audiocpp/fetch_yue2_gguf.py"
            )
        lyrics_file = self._resolve_lyrics()
        abc_file = self._resolve_score()
        style = self.params["style"].value
        if not style or not style.strip():
            raise ValueError("Style prompt is empty.")
        ar_lora = self.params["ar_lora"].value or None
        if ar_lora and not Path(ar_lora).exists():
            raise ValueError(f"AR LoRA file not found: {ar_lora}")
        nar_lora = self.params["nar_lora"].value or None
        if nar_lora and not Path(nar_lora).exists():
            raise ValueError(f"NAR LoRA file not found: {nar_lora}")
        ar_scale = float(self.params["ar_lora_scale"].value)
        nar_scale = float(self.params["nar_lora_scale"].value)
        if ar_lora and not math.isfinite(ar_scale):
            raise ValueError(f"AR LoRA scale must be finite, got {ar_scale!r}.")
        if nar_lora and not math.isfinite(nar_scale):
            raise ValueError(f"NAR LoRA scale must be finite, got {nar_scale!r}.")
        guidance = float(self.params["guidance_scale"].value)
        if not math.isfinite(guidance) or not 0 <= guidance <= 20:
            raise ValueError(f"Guidance scale must be in [0, 20], got {guidance!r}.")
        steps = int(self.params["num_inference_steps"].value)
        if steps < 1:
            raise ValueError(f"Inference steps must be >= 1, got {steps!r}.")
        seconds = float(self.params["max_duration"].value)
        if (not math.isfinite(seconds)
                or not MIN_MAX_DURATION <= seconds <= MAX_MAX_DURATION):
            raise ValueError(
                f"max_duration must be in [{MIN_MAX_DURATION:.0f}, "
                f"{MAX_MAX_DURATION:.0f}] s, got {seconds!r}.")
        tokens = max(1, round(seconds * YUE2_FPS))
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "lyrics_file": lyrics_file,
            "abc_file": abc_file,
            "style": style,
            "cot": COT_MODES[int(self.params["cot"].value)],
            "seed": self._pick_seed(),
            "main_gguf": main_gguf,
            "weight_type": WTYPES[int(self.params["weight_type"].value)][1],
            "format": FORMATS[int(self.params["format"].value)],
            "backend": default_backend(),
            "ar_lora": ar_lora,
            "ar_lora_scale": ar_scale,
            "nar_lora": nar_lora,
            "nar_lora_scale": nar_scale,
            # None keeps the CLI's own (cot-dependent) default.
            "guidance_scale": None if guidance == DEFAULT_GUIDANCE else guidance,
            "num_inference_steps": None if steps == DEFAULT_STEPS else steps,
            "attention": YUE2_ATTENTION_MODES[int(self.params["attention"].value)],
            "out_format": CLI_OUT_FORMAT,
            "max_duration": seconds,
            "semantic_max_tokens": tokens,
        }

    def _pick_seed(self):
        """Committed seed, or a fresh random one with auto_seed on.

        The drawn value is written back to the seed param so the UI, undo
        history and kept request.json all record what actually ran.
        """
        if self.params["auto_seed"].value:
            seed = random.randrange(0, 2 ** 31)
            self.params["seed"].set(seed)
            self.params["seed"].sync()
            return seed
        return int(self.params["seed"].value)

    def _model_fetch_specs(self, model_dir):
        return yue2_fetch_specs(model_dir)

    def _running_status(self, spec):
        lora = (" +LoRA" if spec.get("ar_lora") or spec.get("nar_lora")
                else "")
        cap = spec.get("max_duration")
        cap_text = f", cap {cap:.0f}s" if isinstance(cap, (int, float)) else ""
        return ("Generating",
                f"seed {spec['seed']}, {spec['cot']}, "
                f"{spec['main_gguf']}, {spec.get('weight_type', 'native')}"
                f"{lora}{cap_text}…")

    def _spec_warnings(self, spec):
        """Warn when the cover score and lyrics are shaped too differently
        (stretch/cram garbles vocals). Reads two small text files on the
        engine/control thread — same I/O class as the existence checks in
        _build_spec. Warn-only; a missing file here is the worker's error
        to raise, so read failures simply skip the warning."""
        if not spec.get("abc_file"):
            return []
        try:
            lyrics_text = Path(spec["lyrics_file"]).read_text(encoding="utf-8")
            abc_text = Path(spec["abc_file"]).read_text(encoding="utf-8")
        except OSError:
            return []
        warning = score_lyrics_fit_warning(
            abc_sections=abc_section_names(abc_text),
            lyric_sections=lyric_section_names(lyrics_text))
        return [warning] if warning else []

    def _on_job_submitted(self, spec):
        self._gen_t0 = time.monotonic()

    def _on_job_cancelled(self):
        self._gen_t0 = None

    def _fail(self, message):
        self._gen_t0 = None  # terminal state; stops the elapsed clock
        super()._fail(message)

    # ------------------------------------------------------------------
    # NRT worker
    # ------------------------------------------------------------------
    def _run_nrt(self, cancel_event, spec):
        """Full blocking pipeline: read inputs, run CLI, decode, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "yue2")
        try:
            lyrics = Path(spec["lyrics_file"]).read_text(encoding="utf-8")
            if not lyrics.strip():
                raise ValueError(f"Lyrics file is empty: {spec['lyrics_file']}")
            abc_text = None
            if spec["abc_file"]:
                abc_text = Path(spec["abc_file"]).read_text(encoding="utf-8")
            out_wav = run_dir / "song.wav"
            gen = run_yue2_gen(
                spec["cli"], spec["model_dir"], lyrics=lyrics, style=spec["style"],
                cot=spec["cot"], seed=spec["seed"], main_gguf=spec["main_gguf"],
                vae_gguf=VAE_GGUF, abc_file=spec["abc_file"], out_wav=out_wav,
                cancel_event=cancel_event, backend=spec.get("backend"),
                weight_type=spec.get("weight_type", "native"),
                out_dir=run_dir,
                ar_lora=spec.get("ar_lora"),
                ar_lora_scale=spec.get("ar_lora_scale", 1.0),
                nar_lora=spec.get("nar_lora"),
                nar_lora_scale=spec.get("nar_lora_scale", 1.0),
                guidance_scale=spec.get("guidance_scale"),
                num_inference_steps=spec.get("num_inference_steps"),
                attention=spec.get("attention", "auto"),
                out_format=spec.get("out_format", "pcm16"),
                semantic_max_tokens=spec.get("semantic_max_tokens"),
            )
            audio = load_song_file(gen["wav"])
            plan_text = None
            if gen.get("score"):
                try:
                    plan_text = Path(gen["score"]).read_text(encoding="utf-8")
                    if not plan_text.strip():
                        plan_text = None
                except OSError:
                    plan_text = None
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_seed{spec['seed']}_{spec['cot']}"
            semantic = gen.get("semantic") or {"frames": None, "truncated": None}
            kept_song = self._keep_song(run_dir, keep, audio, spec, gen["metrics"],
                                        lyrics_text=lyrics, abc_text=abc_text,
                                        plan_text=plan_text, semantic=semantic)
            kept_plan = str(keep / "plan.abc") if plan_text is not None else None
            return {"audio": audio, "wav": str(kept_song),
                    "metrics": gen["metrics"], "seed": spec["seed"], "cot": spec["cot"],
                    "plan": kept_plan, "plan_text": plan_text,
                    "semantic": semantic,
                    "max_duration": spec.get("max_duration"),
                    "semantic_max_tokens": spec.get("semantic_max_tokens")}
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    def _keep_song(self, run_dir, keep, audio, spec, metrics,
                   lyrics_text=None, abc_text=None, plan_text=None,
                   semantic=None):
        """Keep exactly one song file plus provenance records. Returns kept path.

        request.json carries everything needed to recreate the song: the
        full lyrics/score texts alongside their source paths (paths rot;
        text doesn't), the resolved seed/profile/VAE, and the sidecar pin.
        The model-generated plan (v0.8.1+ score.abc) is kept as plan.abc
        when present; input-score runs and cot=off keep no plan.
        """
        if spec["format"] == "mp3":
            # Encode MP3 from the validated tensor straight into the keep
            # dir (libmp3lame 320k); the CLI WAV stays in run_dir and is
            # cleaned up with it.
            from audio_io import encode_mp3_file
            keep.mkdir(parents=True, exist_ok=True)
            encode_mp3_file(keep / "song.mp3", audio.numpy(), SAMPLE_RATE)
            kept_song = keep / "song.mp3"
            self.keep_artifacts(run_dir, keep, [])
        else:
            kept_song = keep / "song.wav"
            self.keep_artifacts(run_dir, keep, ["song.wav"])
        if plan_text is not None:
            (keep / "plan.abc").write_text(plan_text, encoding="utf-8")
        self.write_json(keep / "request.json", {
            "style": spec["style"],
            "lyrics_file": spec["lyrics_file"],
            "lyrics_text": lyrics_text,
            "abc_file": spec["abc_file"],
            "abc_text": abc_text,
            "cot": spec["cot"],
            "seed": spec["seed"], "profile": spec["main_gguf"],
            "vae_gguf": VAE_GGUF,
            "weight_type": spec.get("weight_type", "native"),
            "format": spec["format"],
            "backend": spec.get("backend", "cuda"),
            "runtime": {"audio_cpp": AUDIOCPP_PIN_VERSION},
            "generated_plan": plan_text is not None,
            "plan_text": plan_text,
            "ar_lora": spec.get("ar_lora"),
            "ar_lora_scale": spec.get("ar_lora_scale", 1.0),
            "nar_lora": spec.get("nar_lora"),
            "nar_lora_scale": spec.get("nar_lora_scale", 1.0),
            "guidance_scale": spec.get("guidance_scale"),
            "num_inference_steps": spec.get("num_inference_steps"),
            "attention": spec.get("attention", "auto"),
            "out_format": spec.get("out_format", "pcm16"),
            "max_duration": spec.get("max_duration"),
            "semantic_max_tokens": spec.get("semantic_max_tokens"),
            "semantic": semantic,
        })
        self.write_json(keep / "metrics.json", metrics)
        return kept_song

    def _relink_nrt(self, cancel_event, wav_path, plan_path=None):
        plan_text = None
        if plan_path:
            try:
                plan_text = Path(plan_path).read_text(encoding="utf-8")
                if not plan_text.strip():
                    plan_text = None
            except OSError:
                plan_text = None
        return {"audio": load_song_file(wav_path), "wav": wav_path,
                "metrics": {}, "seed": None, "cot": None,
                "plan": plan_path, "plan_text": plan_text}

    def on_nrt_complete(self, tag, ok, result):
        if tag == "gen":
            self._gen_t0 = None  # terminal state either way; stops the elapsed clock
            if not ok:
                if isinstance(result, GenerationCancelled):
                    self._status = "Cancelled"
                    self._status_detail = "Cancelled by user"
                    self.error_msg = None
                else:
                    self._fail(f"Generation failed: {result}")
                return
            audio = result.get("audio")
            if audio is None or audio.ndim != 2 or audio.shape[0] != CHANNELS:
                self._fail(f"Worker returned an invalid audio tensor: {type(audio)}")
                return
            # Generate-only node: the decode above only validates the kept
            # WAV and measures its duration. Playback lives in SamplePlayer;
            # downstream discovers the song through the song URI output and
            # the one-block pulse on ready.
            self.outputs["song"].uri = result["wav"]
            plan = result.get("plan")
            plan_text = result.get("plan_text")
            if plan and plan_text:
                self.outputs["plan"].uri = plan
                self._plan_text = plan_text
                self._plan_path = plan
            else:
                # Anti-ghost: a plan-less run must not leave a stale plan URI.
                self.outputs["plan"].uri = ""
                self._plan_text = None
                self._plan_path = ""
                plan = None
            self._pulse = True
            self.error_msg = None
            secs = audio.shape[1] / SAMPLE_RATE
            rtf = result["metrics"].get("rtf")
            sem = result.get("semantic") or {}
            truncated = sem.get("truncated") is True
            cap = result.get("max_duration")
            cut = ""
            if truncated:
                if isinstance(cap, (int, float)):
                    cut = f", cut at {cap:.0f}s cap"
                else:
                    cut = ", cut at cap"
                logger.warning("YuE2 song hit the max-duration cap%s; kept.",
                               cut.replace(",", ""))
            self._truncated = truncated
            self._cap_seconds = cap if isinstance(cap, (int, float)) else None
            self._status = "Ready"
            self._status_detail = (
                f"{secs:.1f} s song (seed {result['seed']}, {result['cot']}"
                + (f", RTF {rtf:.2f}" if rtf else "")
                + (" + plan" if plan else "") + cut + ")"
            )
            self.params["last_wav"].set(result["wav"])
            self.params["last_wav"].sync()
            self.params["last_plan"].set(plan or "")
            self.params["last_plan"].sync()
        elif tag == "relink":
            self._truncated = False
            self._cap_seconds = None
            if ok and isinstance(result.get("audio"), torch.Tensor):
                self.outputs["song"].uri = result["wav"]
                plan = result.get("plan")
                plan_text = result.get("plan_text")
                if plan and plan_text and Path(plan).exists():
                    self.outputs["plan"].uri = plan
                    self._plan_text = plan_text
                    self._plan_path = plan
                else:
                    self.outputs["plan"].uri = ""
                    self._plan_text = None
                    self._plan_path = ""
                # No pulse: a re-linked song is not newly ready, so a wired
                # SamplePlayer must not autoplay on patch load.
                self.error_msg = None
                secs = result["audio"].shape[1] / SAMPLE_RATE
                self._status = "Ready"
                self._status_detail = f"Re-linked {secs:.1f} s song"
            else:
                self.outputs["song"].uri = ""
                self.outputs["plan"].uri = ""
                self._plan_text = None
                self._plan_path = ""
                self._status = "Idle"
                self._status_detail = "Previous song file is missing; generate again"
                self.error_msg = None
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        # Old patches predate max_duration: the param keeps its default, but
        # a stored out-of-range value (e.g. 0) is reset to the default.
        try:
            v = float(self.params["max_duration"].value)
        except (KeyError, TypeError, ValueError):
            v = DEFAULT_MAX_DURATION
        if not (MIN_MAX_DURATION <= v <= MAX_MAX_DURATION):
            self.params["max_duration"].set(DEFAULT_MAX_DURATION)
            self.params["max_duration"].sync()
        # Re-link the kept WAV (+ plan when still present) without
        # regenerating. Decoding runs on NRT. Old patches have no last_plan
        # key: the param keeps its "" default and only the song re-links.
        wav = self.params["last_wav"].value if "last_wav" in self.params else ""
        plan = self.params["last_plan"].value if "last_plan" in self.params else ""
        if plan and not Path(plan).exists():
            plan = ""
        if wav and Path(wav).exists():
            self._status = "Loading"
            self._status_detail = "Re-linking previous song…"
            if getattr(self, "graph", None) is not None and self.graph.engine is not None:
                self.submit_job("relink", self._relink_nrt, wav, plan or None)
            # Headless (tests): caller drives _relink_nrt directly.
        elif plan and Path(plan).exists():
            # Song gone but plan survives: surface the plan without a pulse.
            try:
                self._plan_text = Path(plan).read_text(encoding="utf-8")
                self._plan_path = plan
                self.outputs["plan"].uri = plan
            except OSError:
                self._plan_text = None
                self._plan_path = ""

    def get_telemetry(self) -> dict:
        detail = self._status_detail
        if self._status == "Generating" and self._gen_t0 is not None:
            # Slow-vs-stuck discriminator: the CLI emits no progress, so a
            # silent tail (e.g. VAE decode of a long song under memory
            # pressure) is otherwise indistinguishable from a hung job.
            elapsed = time.monotonic() - self._gen_t0
            detail = f"{detail} ({elapsed:.0f} s elapsed)"
        elif (self._status == "Ready" and getattr(self, "_truncated", False)
                and "cut at" not in detail):
            cap = getattr(self, "_cap_seconds", None)
            cut = (f", cut at {cap:.0f}s cap"
                   if isinstance(cap, (int, float)) else ", cut at cap")
            if detail.endswith(")"):
                detail = detail[:-1] + cut + ")"
            else:
                detail = detail + cut
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag(),
                "score_text": self._plan_text or ""}

    def _request_job(self):
        """Ask the engine thread to stage a generation (one-shot per edge).

        The audio thread must never build the job spec (file existence
        checks) or touch params directly (AGENTS.md section 5); instead it
        queues a ("param", ...) command — the unbounded command queue never
        blocks — and the engine thread runs the normal Generate path,
        including validation, cancel-previous, and instant telemetry.
        """
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "generate", True))
