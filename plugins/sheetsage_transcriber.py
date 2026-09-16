"""
SheetSage2Transcriber — audio-to-ABC score transcription via audio.cpp (Utilities).

Transcribes a recording into a symbolic ABC score (melody + chords) using
the SheetSage2 GGUF model through the in-tree audio.cpp sidecar
(tools/audiocpp/). The minutes-long transcription runs as a blocking call
on an NRT worker; the resulting score.abc is kept on disk for review and
for cover rendering in YuE2SongGenerator (abc_file + cot=melody).

What this node does NOT do: transcribe lyrics. Only the melody/chords
come out; words stay a manual lyrics file on the YuE2 side (a future ASR
family can fill that slot without touching this node).

Requirements (fetched once, never auto-downloaded by the node):
  python tools/audiocpp/fetch_audiocpp.py       # native runtime (v0.8.0)
  python tools/audiocpp/fetch_sheetsage2_gguf.py # weights (orig dtype)

Measured on an RTX 5070 Laptop (8 GB): a 240 s song transcribes in
~312 s at ~7.6 GB peak VRAM — close other GPU apps first. Model weights
are CC BY-NC 4.0 (non-commercial).
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtCore import Qt

from audiocpp_backend import (
    REPO_ROOT,
    SHEETSAGE_DEFAULT_MAX_TOKENS,
    SHEETSAGE_GGUF,
    SHEETSAGE_WEIGHT_TYPES,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    install_runtime_archives,
    missing_specs,
    run_sheetsage_transcribe,
    runtime_fetch_specs,
    sheetsage_fetch_specs,
    strip_abc_chords,
)
from download_util import (
    DownloadCancelled,
    DownloadSpec,
    fetch_all,
    format_bytes,
)

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

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    av = None
    _AV_AVAILABLE = False

INPUT_RATE = 48000
INPUT_CHANNELS = 2
AUDIO_FILTER = "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*.*)"


def _orient_channels(data):
    """Orient a decoded array to (channels, samples). Decoders disagree on
    axis order, so assume the small dim (<= 8) is channels."""
    import numpy as np
    data = np.asarray(data)
    if data.ndim == 1:
        return data[None, :]
    if data.ndim != 2:
        raise ValueError(f"unsupported decoded shape {data.shape}")
    if data.shape[0] <= 8:
        return data
    return data.T


def load_input_wav(src_path, out_wav):
    """Decode an audio file to a 48 kHz stereo WAV for the transcriber.

    Pure helper (no node state): the NRT worker calls this, tests call it
    directly. WAV/FLAC/OGG go through soundfile; MP3 (and anything else
    soundfile rejects) falls back to PyAV. Raises RuntimeError on failure.
    """
    import numpy as np
    src_path, out_wav = Path(src_path), Path(out_wav)
    if not src_path.exists():
        raise RuntimeError(f"input audio not found: {src_path}")
    audio, sr = None, None
    if src_path.suffix.lower() != ".mp3" and sf is not None:
        try:
            data, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
            audio = _orient_channels(data.T).astype(np.float32)
        except Exception:
            audio, sr = None, None
    if audio is None:
        if not _AV_AVAILABLE:
            raise RuntimeError(
                f"could not decode {src_path.name} with soundfile and PyAV "
                "is not installed (MP3 needs PyAV)"
            )
        try:
            container = av.open(str(src_path))
            stream = container.streams.audio[0]
            sr = stream.sample_rate
            # Orient frame-by-frame: PyAV frame layouts are not guaranteed
            # uniform (mixed planar/packed or trailing mono flush frames
            # break a single bulk concatenate).
            parts = []
            for frame in container.decode(audio=0):
                parts.append(_orient_channels(frame.to_ndarray()).astype(np.float32))
            if not parts:
                raise RuntimeError(f"no audio frames decoded from {src_path.name}")
            channels = max(p.shape[0] for p in parts)
            aligned = []
            for p in parts:
                if p.shape[0] == 1 and channels > 1:
                    p = np.repeat(p, channels, axis=0)  # mono flush frame
                if p.shape[0] != channels:
                    raise RuntimeError(
                        f"inconsistent channel counts while decoding {src_path.name}")
                aligned.append(p)
            audio = np.concatenate(aligned, axis=-1)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"could not decode {src_path.name}: {e}")
    if audio.shape[0] == 1:
        audio = np.vstack([audio[0], audio[0]])  # mono -> stereo dup
    else:
        audio = audio[:INPUT_CHANNELS]
    if sr != INPUT_RATE:
        if resampy is None:
            raise RuntimeError(
                f"input is {sr} Hz but the transcriber needs {INPUT_RATE} Hz "
                "and 'resampy' is not installed"
            )
        audio = resampy.resample(audio, sr, INPUT_RATE, axis=-1)
    if sf is None:  # pragma: no cover - soundfile is a hard app dependency
        raise RuntimeError("'soundfile' is required to write the input WAV")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), np.ascontiguousarray(audio.T), INPUT_RATE)
    return out_wav


def section_names(abc_text):
    """Section labels from `% name` comment lines, in order."""
    sections = []
    for line in abc_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") and len(stripped) > 1:
            sections.append(stripped[1:].strip())
    return sections


class SheetSageWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "SheetSage2Transcriber"

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.audio_widget = self.proxy.create_param_widget("audio_file")
        layout.addWidget(self.audio_widget)
        self.weight_widget = self.proxy.create_param_widget("weight_type")
        layout.addWidget(self.weight_widget)
        self.tokens_widget = self.proxy.create_param_widget("max_tokens")
        layout.addWidget(self.tokens_widget)
        self.strip_widget = self.proxy.create_param_widget("strip_chords")
        layout.addWidget(self.strip_widget)

        self.btn_download = QPushButton("Download runtime + model (~3.5 GB)")
        self.btn_download.clicked.connect(
            lambda: self.proxy.set_parameter("download", True))
        layout.addWidget(self.btn_download)

        self.btn_transcribe = QPushButton("Transcribe")
        self.btn_transcribe.clicked.connect(
            lambda: self.proxy.set_parameter("transcribe", True))
        layout.addWidget(self.btn_transcribe)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(
            lambda: self.proxy.set_parameter("cancel", True))
        layout.addWidget(self.btn_cancel)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.lbl_score = QLabel("No transcription yet")
        self.lbl_score.setStyleSheet("color: #aaa; font-size: 10px;")
        self.lbl_score.setWordWrap(True)
        layout.addWidget(self.lbl_score)

    def on_telemetry(self, data: dict):
        if "status" in data:
            self.lbl_status.setText(data["status"])
        if "audio" in data:
            self.lbl_score.setText(data["audio"])

    def update_from_params(self, params):
        for key, widget in (
            ("audio_file", self.audio_widget),
            ("weight_type", self.weight_widget),
            ("max_tokens", self.tokens_widget),
            ("strip_chords", self.strip_widget),
        ):
            if key in params:
                widget.update_from_backend(params[key])


class SheetSage2Transcriber(AudioCppJob):
    category = "Utilities"
    label = "SheetSage2 Transcriber"
    description = (
        "Offline audio-to-ABC transcription (SheetSage2 GGUF via the in-tree "
        "audio.cpp sidecar). A recording becomes a melody+chord score for "
        "review and for cover rendering in YuE2SongGenerator (abc_file + "
        "cot=melody). Runs on a background NRT worker (~0.8x realtime, "
        "~7.6 GB peak VRAM for a 4 min song — close other GPU apps). "
        "Lyrics are NOT transcribed; words stay manual. Needs "
        "tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_sheetsage2_gguf.py run once (or the Download "
        "button). Weights are CC BY-NC 4.0 (non-commercial)."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_file_param("audio_file", "", filter=AUDIO_FILTER,
                            help="Recording to transcribe (WAV/FLAC/OGG/MP3); converted on the background worker.")
        self.add_menu_param("weight_type", list(SHEETSAGE_WEIGHT_TYPES), initial_idx=0,
                            help="Decoder weight dtype; native = best quality.")
        self.add_int_param("max_tokens", SHEETSAGE_DEFAULT_MAX_TOKENS, 1, 32768,
                           help="Decoder token cap; long songs may truncate at the cap.")
        self.add_bool_param("strip_chords", True,
                            help="Also write a melody-only score variant (chord symbols removed) for cot=melody covers.")
        self.add_string_param("model_dir", "models/SheetSage2-GGUF",
                              help="Model directory holding the GGUF file (repo-relative).")
        self.add_bool_param("transcribe", False,
                            help="Transient trigger: stages a background transcription, then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads missing runtime/model in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight job.")
        # Last kept score, for save/load re-linking. Written on completion.
        self.add_string_param("last_abc", "",
                              help="Path of the last transcribed score; re-linked on patch load.")

        self._abc_text = None
        self._abc_path = ""
        self._status = "Idle"
        self._status_detail = "No transcription yet"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def process(self):
        """Analysis node: no audio ports, nothing to do per block."""

    def _restage(self, name, value):
        self.params[name].set(value)
        self.params[name].sync()

    def _resolve_model_dir(self):
        raw = self.params["model_dir"].value
        d = Path(raw)
        if not d.is_absolute():
            d = REPO_ROOT / d
        return d

    def _fail(self, message):
        self._status = "Error"
        self._status_detail = message
        self.error_msg = message

    def _need_engine(self):
        if getattr(self, "graph", None) is None or self.graph.engine is None:
            self._fail("Node is not attached to an engine.")
            self._push_telemetry()
            return False
        return True

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        runtime = AudioCppRuntime()
        ok, hint = runtime.check_ready()
        if not ok:
            raise ValueError(hint)
        model_dir = self._resolve_model_dir()
        if not (model_dir / SHEETSAGE_GGUF).exists():
            raise ValueError(
                f"SheetSage2 weights not found in {model_dir}. "
                "Run: python tools/audiocpp/fetch_sheetsage2_gguf.py "
                "(or press Download)"
            )
        audio_file = self.params["audio_file"].value
        if not audio_file or not Path(audio_file).exists():
            raise ValueError("Pick a recording first (audio_file is empty or missing).")
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "audio_file": audio_file,
            "weight_type": SHEETSAGE_WEIGHT_TYPES[int(self.params["weight_type"].value)],
            "max_tokens": int(self.params["max_tokens"].value),
            "strip_chords": bool(self.params["strip_chords"].value),
        }

    def _spec_to_dict(self, spec):
        return {"url": spec.url, "dest": str(spec.dest), "size": spec.size,
                "sha256": spec.sha256, "label": spec.label}

    def _build_fetch_payload(self):
        """Plain-data fetch plan for the NRT worker. Raises ValueError."""
        from audiocpp_backend import sheetsage_fetch_specs
        runtime = AudioCppRuntime()
        model_dir = self._resolve_model_dir()
        try:
            runtime_specs, staging = runtime_fetch_specs(runtime.root)
        except RuntimeError as e:
            raise ValueError(str(e))
        return {
            "runtime": [self._spec_to_dict(s)
                        for s in missing_specs(runtime_specs)
                        if not runtime.cli.exists()],
            "staging": str(staging),
            "bin_dir": str(runtime.bin_dir),
            "models": [self._spec_to_dict(s)
                       for s in missing_specs(sheetsage_fetch_specs(model_dir))],
        }

    def on_ui_param_change(self, param_name: str):
        if param_name == "transcribe":
            if not self.params["transcribe"].value:
                return
            # Deliberate re-stage of the transient trigger (AGENTS.md section 5).
            self._restage("transcribe", False)
            if not self._need_engine():
                return
            try:
                spec = self._build_spec()
            except ValueError as e:
                self._fail(str(e))
                self._push_telemetry()
                return
            self.error_msg = None
            self._status = "Transcribing"
            self._status_detail = f"{Path(spec['audio_file']).name} ({spec['weight_type']})…"
            self.submit_job("job", self._transcribe_nrt, spec)
            self._push_telemetry()
        elif param_name == "cancel":
            if self.params["cancel"].value:
                self._restage("cancel", False)
                self._cancel_job()
                if self._status in ("Transcribing", "Downloading"):
                    self._status = "Cancelled"
                    self._status_detail = "Cancelled by user"
                self._push_telemetry()
        elif param_name == "download":
            if not self.params["download"].value:
                return
            self._restage("download", False)
            if not self._need_engine():
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

    # ------------------------------------------------------------------
    # NRT worker
    # ------------------------------------------------------------------
    def _transcribe_nrt(self, cancel_event, spec):
        """Full blocking pipeline: convert input, run CLI, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "sheetsage")
        try:
            input_wav = load_input_wav(spec["audio_file"], run_dir / "input.wav")
            out_abc = run_dir / "score.abc"
            result = run_sheetsage_transcribe(
                spec["cli"], spec["model_dir"], audio_wav=input_wav,
                weight_type=spec["weight_type"], max_tokens=spec["max_tokens"],
                out_abc=out_abc, cancel_event=cancel_event,
            )
            abc_text = Path(result["abc"]).read_text(encoding="utf-8")
            if not abc_text.strip():
                raise RuntimeError("transcriber returned an empty score")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = Path(spec["audio_file"]).stem
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_{stem}"
            kept = ["score.abc"]
            melody_path = None
            if spec["strip_chords"]:
                (run_dir / "score-melody.abc").write_text(
                    strip_abc_chords(abc_text), encoding="utf-8")
                kept.append("score-melody.abc")
            self.keep_artifacts(run_dir, keep, kept)
            if spec["strip_chords"]:
                melody_path = str(keep / "score-melody.abc")
            self.write_json(keep / "request.json", {
                "audio_file": spec["audio_file"],
                "weight_type": spec["weight_type"],
                "max_tokens": spec["max_tokens"],
                "strip_chords": spec["strip_chords"],
            })
            self.write_json(keep / "metrics.json", result["metrics"])
            return {"abc": abc_text, "abc_path": str(keep / "score.abc"),
                    "melody_path": melody_path,
                    "metrics": result["metrics"],
                    "sections": section_names(abc_text)}
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    def _fetch_nrt(self, cancel_event, payload):
        """Download missing runtime/model with inbox progress. NRT only."""
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

    def on_nrt_complete(self, tag, ok, result):
        if tag == "job":
            if not ok:
                if isinstance(result, GenerationCancelled):
                    self._status = "Cancelled"
                    self._status_detail = "Cancelled by user"
                    self.error_msg = None
                else:
                    self._fail(f"Transcription failed: {result}")
                return
            self._abc_text = result.get("abc", "")
            self._abc_path = result.get("abc_path", "")
            if not self._abc_text.strip():
                self._fail("Worker returned an empty score.")
                return
            self.error_msg = None
            sections = result.get("sections", [])
            self._status = "Ready"
            self._status_detail = (
                f"{Path(self._abc_path).name}: {len(sections)} sections"
                + (f" ({', '.join(sections[:4])}…)" if len(sections) > 4
                   else (f" ({', '.join(sections)})" if sections else ""))
            )
            self.params["last_abc"].set(self._abc_path)
            self.params["last_abc"].sync()
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
            self._status_detail = "Download complete — press Transcribe"

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
        # Scores are small text: re-link synchronously on the control thread.
        abc = self.params["last_abc"].value if "last_abc" in self.params else ""
        if abc and Path(abc).exists():
            try:
                self._abc_text = Path(abc).read_text(encoding="utf-8")
                self._abc_path = abc
                self._status = "Ready"
                self._status_detail = f"Re-linked {Path(abc).name}"
                self.error_msg = None
            except OSError:
                self._status = "Idle"
                self._status_detail = "Previous score is missing; transcribe again"
        elif abc:
            self._status = "Idle"
            self._status_detail = "Previous score is missing; transcribe again"

    def get_telemetry(self) -> dict:
        preview = ""
        if self._abc_text:
            lines = [ln for ln in self._abc_text.splitlines() if ln.strip()][:3]
            preview = " / ".join(lines)[:160]
        detail = self._status_detail
        if self._abc_path:
            detail = f"{detail}\n{self._abc_path}"
            if preview:
                detail = f"{detail}\n{preview}"
        return {"status": self._status, "audio": detail}
