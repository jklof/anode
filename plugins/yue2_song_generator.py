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
scores via `abc_file` until a newer runtime is pinned.
"""

import logging
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtCore import Qt

from audiocpp_backend import (
    REPO_ROOT,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    install_runtime_archives,
    missing_specs,
    run_yue2_gen,
    runtime_fetch_specs,
    yue2_fetch_specs,
)
from download_util import (
    DownloadCancelled,
    DownloadSpec,
    fetch_all,
    format_bytes,
)
from base import SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    sf = None
    _SF_AVAILABLE = False

try:
    import resampy
    _RESAMPY_AVAILABLE = True
except ImportError:
    resampy = None
    _RESAMPY_AVAILABLE = False

COT_MODES = ("off", "melody", "full")
PROFILES = (
    ("Q4_K_M (8 GB VRAM)", "yue2-3b-q4_k_m.gguf"),
    ("Q8_0 (16 GB VRAM)", "yue2-3b-q8_0.gguf"),
    ("BF16 (24 GB VRAM)", "yue2-3b-bf16.gguf"),
)
VAE_GGUF = "yue2-vae-f16.gguf"


def load_song_wav(path):
    """Decode a song WAV into a (2, N) contiguous float32 CPU tensor.

    Pure helper (no node state): the NRT worker calls this, tests call it
    directly. Raises RuntimeError with a human-readable message on failure.
    """
    if sf is None:
        raise RuntimeError(
            "YuE2SongGenerator: 'soundfile' is required but is not installed"
        )
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise RuntimeError(f"could not decode {path}: {e}")
    import numpy as np
    audio = data.T
    if audio.shape[0] == 1:
        audio = np.vstack([audio[0], audio[0]])  # mono -> stereo dup
    else:
        audio = audio[:CHANNELS]
    if sr != SAMPLE_RATE:
        if resampy is None:
            raise RuntimeError(
                f"song is {sr} Hz but the engine runs at {SAMPLE_RATE} Hz and "
                "'resampy' (needed to convert) is not installed"
            )
        audio = resampy.resample(audio, sr, SAMPLE_RATE, axis=-1)
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if tensor.shape[0] != CHANNELS or tensor.ndim != 2 or tensor.shape[1] < 1:
        raise RuntimeError(f"decoded song has unexpected shape {tuple(tensor.shape)}")
    return tensor


class YuE2Widget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "YuE2SongGenerator"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.style_widget = self.proxy.create_param_widget("style")
        layout.addWidget(self.style_widget)
        self.lyrics_widget = self.proxy.create_param_widget("lyrics_file")
        layout.addWidget(self.lyrics_widget)
        self.abc_widget = self.proxy.create_param_widget("abc_file")
        layout.addWidget(self.abc_widget)
        self.profile_widget = self.proxy.create_param_widget("profile")
        layout.addWidget(self.profile_widget)
        self.cot_widget = self.proxy.create_param_widget("cot")
        layout.addWidget(self.cot_widget)
        self.seed_widget = self.proxy.create_param_widget("seed")
        layout.addWidget(self.seed_widget)

        self.btn_seed = QPushButton("Randomize seed")
        self.btn_seed.clicked.connect(self._on_seed_pressed)
        layout.addWidget(self.btn_seed)

        self.btn_download = QPushButton("Download runtime + model (~3.8 GB)")
        self.btn_download.clicked.connect(
            lambda: self.proxy.set_parameter("download", True))
        layout.addWidget(self.btn_download)

        self.btn_generate = QPushButton("Generate song")
        self.btn_generate.clicked.connect(self._on_generate_pressed)
        layout.addWidget(self.btn_generate)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(
            lambda: self.proxy.set_parameter("cancel", True))
        layout.addWidget(self.btn_cancel)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.lbl_audio = QLabel("No song generated")
        self.lbl_audio.setStyleSheet("color: #aaa; font-size: 10px;")
        self.lbl_audio.setWordWrap(True)
        layout.addWidget(self.lbl_audio)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
        if "audio" in data:
            self.lbl_audio.setText(data["audio"])

    def _on_seed_pressed(self):
        self.proxy.set_parameter("seed", random.randrange(0, 2 ** 31))

    def _on_generate_pressed(self):
        # Commit the style editor text first: set_parameter only debounces,
        # and the flush preserves insertion order, so style is guaranteed
        # to be applied before generate (no stale-prompt race).
        line_edit = getattr(self.style_widget, "line_edit", None)
        if line_edit is not None:
            self.proxy.set_parameter("style", line_edit.text())
        self.proxy.set_parameter("generate", True)

    def update_from_params(self, params):
        for key, widget in (
            ("style", self.style_widget),
            ("lyrics_file", self.lyrics_widget),
            ("abc_file", self.abc_widget),
            ("profile", self.profile_widget),
            ("cot", self.cot_widget),
            ("seed", self.seed_widget),
        ):
            if key in params:
                widget.update_from_backend(params[key])


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
                            help="Kept song WAV path; published on completion and on patch-load relink.")
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
        self.add_string_param("last_wav", "",
                              help="Path of the last generated song; re-linked on patch load.")

        self._ready_pulse = False  # emitted as a one-block pulse on ready
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No song generated"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def start(self):
        self._ready_pulse = False
        self._last_trig = 0.0

    def _restage(self, name, value):
        self.params[name].set(value)
        self.params[name].sync()

    def _resolve_model_dir(self):
        raw = self.params["model_dir"].value
        d = Path(raw)
        if not d.is_absolute():
            d = REPO_ROOT / d
        return d

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

    def _spec_to_dict(self, spec):
        return {"url": spec.url, "dest": str(spec.dest), "size": spec.size,
                "sha256": spec.sha256, "label": spec.label}

    def _build_fetch_payload(self):
        """Plain-data fetch plan for the NRT worker. Raises ValueError."""
        runtime = AudioCppRuntime()
        model_dir = self._resolve_model_dir()
        try:
            runtime_specs, staging = runtime_fetch_specs(runtime.root)
        except RuntimeError as e:
            raise ValueError(str(e))
        payload = {
            "runtime": [self._spec_to_dict(s)
                        for s in missing_specs(runtime_specs)
                        if not runtime.cli.exists()],
            "staging": str(staging),
            "bin_dir": str(runtime.bin_dir),
            "models": [self._spec_to_dict(s)
                       for s in missing_specs(yue2_fetch_specs(model_dir))],
        }
        return payload

    def on_ui_param_change(self, param_name: str):
        if param_name == "generate":
            if not self.params["generate"].value:
                return
            # Deliberate re-stage of the transient trigger (AGENTS.md section 5).
            self._restage("generate", False)
            if getattr(self, "graph", None) is None or self.graph.engine is None:
                self._fail("Node is not attached to an engine.")
                return
            try:
                spec = self._build_spec()
            except ValueError as e:
                self._fail(str(e))
                self._push_telemetry()
                return
            self.error_msg = None
            self._status = "Generating"
            self._status_detail = f"seed {spec['seed']}, {spec['cot']}, {spec['main_gguf']}…"
            self.submit_job("gen", self._generate_nrt, spec)
            self._push_telemetry()
        elif param_name == "cancel":
            if self.params["cancel"].value:
                self._restage("cancel", False)
                self._cancel_job()
                if self._status in ("Generating", "Downloading"):
                    self._status = "Cancelled"
                    self._status_detail = "Cancelled by user"
                self._push_telemetry()
        elif param_name == "download":
            if not self.params["download"].value:
                return
            self._restage("download", False)
            if getattr(self, "graph", None) is None or self.graph.engine is None:
                self._fail("Node is not attached to an engine.")
                self._push_telemetry()
                return
            try:
                payload = self._build_fetch_payload()
            except ValueError as e:
                self._fail(str(e))
                self._push_telemetry()
                return
            if not payload["runtime"] and not payload["models"]:
                self.error_msg = None
                self._status = "Idle"
                self._status_detail = "Everything already downloaded"
                self._push_telemetry()
                return
            self.error_msg = None
            self._status = "Downloading"
            self._status_detail = "Starting download…"
            self.submit_job("fetch", self._fetch_nrt, payload)
            self._push_telemetry()

    def _fail(self, message):
        self._status = "Error"
        self._status_detail = message
        self.error_msg = message

    # ------------------------------------------------------------------
    # NRT worker
    # ------------------------------------------------------------------
    def _generate_nrt(self, cancel_event, spec):
        """Full blocking pipeline: read inputs, run CLI, decode, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "yue2")
        try:
            lyrics = Path(spec["lyrics_file"]).read_text(encoding="utf-8")
            if not lyrics.strip():
                raise ValueError(f"Lyrics file is empty: {spec['lyrics_file']}")
            out_wav = run_dir / "song.wav"
            gen = run_yue2_gen(
                spec["cli"], spec["model_dir"], lyrics=lyrics, style=spec["style"],
                cot=spec["cot"], seed=spec["seed"], main_gguf=spec["main_gguf"],
                vae_gguf=VAE_GGUF, abc_file=spec["abc_file"], out_wav=out_wav,
                cancel_event=cancel_event,
            )
            audio = load_song_wav(gen["wav"])
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_seed{spec['seed']}_{spec['cot']}"
            self.keep_artifacts(run_dir, keep, ["song.wav"])
            self.write_json(keep / "request.json", {
                "style": spec["style"], "lyrics_file": spec["lyrics_file"],
                "abc_file": spec["abc_file"], "cot": spec["cot"],
                "seed": spec["seed"], "profile": spec["main_gguf"],
            })
            self.write_json(keep / "metrics.json", gen["metrics"])
            return {"audio": audio, "wav": str(keep / "song.wav"),
                    "metrics": gen["metrics"], "seed": spec["seed"], "cot": spec["cot"]}
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    def _fetch_nrt(self, cancel_event, payload):
        """Download missing runtime/models with inbox progress. NRT only."""
        epoch = self._nrt_epoch
        inbox = self._nrt_inbox

        def _progress(label, done, total, state):
            if inbox is None:
                return
            try:
                inbox.put((epoch, "fetch_progress", True,
                           {"label": label, "done": done, "total": total,
                            "state": state}))
            except Exception:
                pass

        fetched = []
        model_specs = [DownloadSpec(s["url"], Path(s["dest"]), s["size"],
                                    s["sha256"], s["label"])
                       for s in payload["models"]]
        if model_specs:
            fetch_all(model_specs, progress_cb=_progress, cancel_event=cancel_event)
            fetched += [s.label for s in model_specs]
        runtime_specs = [DownloadSpec(s["url"], Path(s["dest"]), s["size"],
                                      s["sha256"], s["label"])
                         for s in payload["runtime"]]
        if runtime_specs:
            fetch_all(runtime_specs, progress_cb=_progress, cancel_event=cancel_event)
            install_runtime_archives(Path(payload["staging"]), Path(payload["bin_dir"]))
            fetched += [s.label for s in runtime_specs]
        return {"fetched": fetched}

    def _relink_nrt(self, cancel_event, wav_path):
        return {"audio": load_song_wav(wav_path), "wav": wav_path,
                "metrics": {}, "seed": None, "cot": None}

    def on_nrt_complete(self, tag, ok, result):
        if tag == "gen":
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
            if ok:
                self._status = "Downloading"
                self._status_detail = self._format_fetch_progress(result)
        elif tag == "fetch":
            if not ok:
                if isinstance(result, (GenerationCancelled, DownloadCancelled)):
                    self._status = "Cancelled"
                    self._status_detail = "Download cancelled — re-run to resume"
                    self.error_msg = None
                else:
                    self._fail(f"Download failed: {result}")
                return
            self.error_msg = None
            self._status = "Idle"
            self._status_detail = "Download complete — press Generate"

    @staticmethod
    def _format_fetch_progress(progress):
        label = progress.get("label", "download")
        done, total, state = (progress.get("done", 0), progress.get("total"),
                              progress.get("state", ""))
        if state == "present":
            return f"{label}: already present"
        if state == "verifying":
            return f"{label}: verifying…"
        if total:
            pct = 100.0 * done / total
            return (f"{label}: {pct:.1f}% "
                    f"({format_bytes(done)} / {format_bytes(total)})")
        return f"{label}: {format_bytes(done)}"

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
        return {"status": self._status, "audio": self._status_detail}

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
