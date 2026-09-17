"""
YuE2SongGenerator — offline lyrics-to-song generator backed by audio.cpp (Sources).

Generates complete songs (vocals + accompaniment) from a style prompt and
lyrics using the YuE2-3B GGUF model through the in-tree audio.cpp sidecar
(tools/audiocpp/). The minutes-long generation runs as a blocking call on
an NRT worker and the finished WAV is installed as RAM-cached playback —
the real-time audio thread only ever does cached-tensor playback, exactly
like SamplePlayer.

Requirements (fetched once, never auto-downloaded by the node):
  python tools/audiocpp/fetch_audiocpp.py    # native runtime (v0.8.0)
  python tools/audiocpp/fetch_yue2_gguf.py   # Q4_K_M weights + VAE + sidecars

Measured on an RTX 5070 Laptop (8 GB): Q4_K_M peaks at ~4.3 GB VRAM for a
77 s song at 2.1x realtime. Model weights are CC BY-NC 4.0
(non-commercial).

Known audio.cpp v0.8.0 gap: model-generated ABC plans cannot be exported
yet, so the plan -> edit -> re-render loop is limited to user-supplied
scores via `abc_file` until a newer runtime is pinned. (Fixed upstream on
main after v0.8.0 — `score.abc` via `--out-dir` — but unreleased, and main
also renames `yue2.lora` to `yue2.ar_lora`, so wait for the next release
rather than tracking main.)
"""

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

COT_MODES = ("off", "melody", "full")
FORMATS = ("mp3", "wav")
PROFILES = (
    ("Q4_K_M (8 GB VRAM)", "yue2-3b-q4_k_m.gguf"),
    ("Q8_0 (16 GB VRAM)", "yue2-3b-q8_0.gguf"),
    ("BF16 (24 GB VRAM)", "yue2-3b-bf16.gguf"),
)
VAE_GGUF = "yue2-vae-f16.gguf"


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

    PARAM_KEYS = ("style", "lyrics_file", "abc_file", "profile", "format",
                  "cot", "seed", "auto_seed")
    ACTION_LABEL = "Generate song"
    ACTION_PARAM = "generate"
    DOWNLOAD_LABEL = "Download runtime + model (~3.8 GB)"

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
        line_edit = getattr(
            self.param_widgets.get("style"), "line_edit", None)
        if line_edit is not None:
            self.proxy.set_parameter("style", line_edit.text())
        self.proxy.set_parameter("generate", True)


class YuE2SongGenerator(AudioCppJob):
    category = "Sources"
    label = "YuE2 Song Generator"
    description = (
        "Offline lyrics-to-song generator (YuE2-3B GGUF via the in-tree "
        "audio.cpp sidecar). Generation runs on a background NRT worker "
        "(~2x realtime, ~4.3 GB peak VRAM with Q4_K_M on an 8 GB GPU) and "
        "publishes the kept song on the song URI output with a one-block "
        "pulse on ready for auto-chaining (e.g. into SamplePlayer). "
        "Generation starts from the Generate button or a rising edge on "
        "trigger_in (e.g. wired from a done pulse); a new edge restarts "
        "generation, cancelling the previous run. "
        "Playback lives in SamplePlayer; this node holds no audio. Missing "
        "runtime/models can be fetched from the node itself (Download "
        "button: resumable background download with progress) or via "
        "tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_yue2_gguf.py. "
        "Weights are CC BY-NC 4.0 (non-commercial)."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background generation "
                            "(same as the Generate button), e.g. wired from a done pulse.")
        self.add_uri_input("abc_uri",
                           help="Wired score path (e.g. from SheetSage2Transcriber). While connected "
                                "and non-empty it overrides abc_file; snapshots at Generate time.")
        self.add_uri_output("song",
                            help="Kept song path (MP3 or WAV per format); published on completion and on patch-load relink.")
        self.ready = self.add_output("ready", channels=1,
                                     help="One-block 1.0 pulse when a new song is ready (wire to a trigger input).")

        self.add_string_param("style",
                              "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal",
                              help="Genre, instruments, vocal character, language, tempo.")
        self.add_file_param("lyrics_file", "", filter="Text Files (*.txt *.md);;All Files (*.*)",
                            help="Lyrics file with section tags like [Verse]/[Chorus]; read by the background worker.")
        self.add_file_param("abc_file", "", filter="ABC Files (*.abc);;All Files (*.*)",
                            help="Optional melody/chord score for covers; requires cot=melody or full. "
                                 "A connected abc_uri input overrides this.")
        self.add_menu_param("profile", [label for label, _ in PROFILES], initial_idx=0,
                            help="GGUF precision profile; larger profiles need more VRAM.")
        self.add_menu_param("format", ["MP3 (320 kbps)", "WAV (lossless)"], initial_idx=0,
                            help="Kept song format — exactly one file is kept. MP3 shares cheaply; "
                                 "WAV keeps full quality for further processing.")
        self.add_menu_param("cot", list(COT_MODES), initial_idx=2,
                            help="Symbolic planning: off = direct, melody = melody plan, full = melody+chords.")
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

        self._ready_pulse = False  # emitted as a one-block pulse on ready
        self._last_trig = 0.0
        self._gen_t0 = None  # monotonic start of the in-flight job (elapsed display)
        self._status = "Idle"
        self._status_detail = "No song generated"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def start(self):
        self._ready_pulse = False
        self._last_trig = 0.0

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
        lyrics_file = self.params["lyrics_file"].value
        if not lyrics_file or not Path(lyrics_file).exists():
            raise ValueError("Pick a lyrics file first (lyrics_file is empty or missing).")
        abc_file = self._resolve_score()
        style = self.params["style"].value
        if not style or not style.strip():
            raise ValueError("Style prompt is empty.")
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "lyrics_file": lyrics_file,
            "abc_file": abc_file,
            "style": style,
            "cot": COT_MODES[int(self.params["cot"].value)],
            "seed": self._pick_seed(),
            "main_gguf": main_gguf,
            "format": FORMATS[int(self.params["format"].value)],
            "backend": default_backend(),
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
        return ("Generating",
                f"seed {spec['seed']}, {spec['cot']}, {spec['main_gguf']}…")

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
            )
            audio = load_song_file(gen["wav"])
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_seed{spec['seed']}_{spec['cot']}"
            kept_song = self._keep_song(run_dir, keep, audio, spec, gen["metrics"],
                                        lyrics_text=lyrics, abc_text=abc_text)
            return {"audio": audio, "wav": str(kept_song),
                    "metrics": gen["metrics"], "seed": spec["seed"], "cot": spec["cot"]}
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    def _keep_song(self, run_dir, keep, audio, spec, metrics,
                   lyrics_text=None, abc_text=None):
        """Keep exactly one song file plus provenance records. Returns kept path.

        request.json carries everything needed to recreate the song: the
        full lyrics/score texts alongside their source paths (paths rot;
        text doesn't), the resolved seed/profile/VAE, and the sidecar pin.
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
        self.write_json(keep / "request.json", {
            "style": spec["style"],
            "lyrics_file": spec["lyrics_file"],
            "lyrics_text": lyrics_text,
            "abc_file": spec["abc_file"],
            "abc_text": abc_text,
            "cot": spec["cot"],
            "seed": spec["seed"], "profile": spec["main_gguf"],
            "vae_gguf": VAE_GGUF,
            "format": spec["format"],
            "backend": spec.get("backend", "cuda"),
            "runtime": {"audio_cpp": AUDIOCPP_PIN_VERSION},
        })
        self.write_json(keep / "metrics.json", metrics)
        return kept_song

    def _relink_nrt(self, cancel_event, wav_path):
        return {"audio": load_song_file(wav_path), "wav": wav_path,
                "metrics": {}, "seed": None, "cot": None}

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
            self._ready_pulse = True
            self.error_msg = None
            secs = audio.shape[1] / SAMPLE_RATE
            rtf = result["metrics"].get("rtf")
            self._status = "Ready"
            self._status_detail = (
                f"{secs:.1f} s song (seed {result['seed']}, {result['cot']}"
                + (f", RTF {rtf:.2f}" if rtf else "") + ")"
            )
            self.params["last_wav"].set(result["wav"])
            self.params["last_wav"].sync()
        elif tag == "relink":
            if ok and isinstance(result.get("audio"), torch.Tensor):
                self.outputs["song"].uri = result["wav"]
                # No pulse: a re-linked song is not newly ready, so a wired
                # SamplePlayer must not autoplay on patch load.
                self.error_msg = None
                secs = result["audio"].shape[1] / SAMPLE_RATE
                self._status = "Ready"
                self._status_detail = f"Re-linked {secs:.1f} s song"
            else:
                self.outputs["song"].uri = ""
                self._status = "Idle"
                self._status_detail = "Previous song file is missing; generate again"
                self.error_msg = None
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        # Re-link the kept WAV without regenerating. Decoding runs on NRT.
        wav = self.params["last_wav"].value if "last_wav" in self.params else ""
        if wav and Path(wav).exists():
            self._status = "Loading"
            self._status_detail = "Re-linking previous song…"
            if getattr(self, "graph", None) is not None and self.graph.engine is not None:
                self.submit_job("relink", self._relink_nrt, wav)
            # Headless (tests): caller drives _relink_nrt directly.

    def get_telemetry(self) -> dict:
        detail = self._status_detail
        if self._status == "Generating" and self._gen_t0 is not None:
            # Slow-vs-stuck discriminator: the CLI emits no progress, so a
            # silent tail (e.g. VAE decode of a long song under memory
            # pressure) is otherwise indistinguishable from a hung job.
            elapsed = time.monotonic() - self._gen_t0
            detail = f"{detail} ({elapsed:.0f} s elapsed)"
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag()}

    # ------------------------------------------------------------------
    # audio thread: one-block ready pulse + trigger edge detect (both
    # allocation-free; no file I/O, no param writes here)
    # ------------------------------------------------------------------
    def process(self):
        buf = self.ready.buffer
        buf.zero_()  # anti-ghost: a stale pulse must never retrigger downstream
        if self._ready_pulse:
            self._ready_pulse = False
            buf.fill_(1.0)
        trig = self.inputs["trigger_in"].get_tensor()[0]
        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            self._request_generate()
        self._last_trig = float(trig[-1].item())

    def _request_generate(self):
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
