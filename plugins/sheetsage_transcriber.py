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
  python tools/audiocpp/fetch_audiocpp.py       # native runtime (v0.8.1)
  python tools/audiocpp/fetch_sheetsage2_gguf.py # weights (orig dtype)

Measured on an RTX 5070 Laptop (8 GB): a 240 s song transcribes in
~312 s at ~7.6 GB peak VRAM — close other GPU apps first. Model weights
are CC BY-NC 4.0 (non-commercial).
"""

import shutil
from datetime import datetime
from pathlib import Path

from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    SHEETSAGE_DEFAULT_MAX_TOKENS,
    SHEETSAGE_GGUF,
    SHEETSAGE_WEIGHT_TYPES,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    abc_section_names,
    default_backend,
    run_sheetsage_transcribe,
    sheetsage_fetch_specs,
    strip_abc_chords,
)

INPUT_RATE = 48000
INPUT_CHANNELS = 2
AUDIO_FILTER = "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*.*)"


def load_input_wav(src_path, out_wav):
    """Decode an audio file to a 48 kHz stereo WAV for the transcriber.

    Thin wrapper over :func:`audio_io.to_engine_audio` (shared decode +
    mono-dup + NRT resample): the NRT worker calls this, tests call it
    directly. Decoding is shared with SamplePlayer (audio_io): WAV/FLAC/OGG
    via soundfile, MP3 via PyAV. Raises RuntimeError on failure.
    """
    from audio_io import to_engine_audio, write_wav_file
    audio = to_engine_audio(src_path, target_sr=INPUT_RATE,
                            target_channels=INPUT_CHANNELS,
                            label="transcriber input")
    return write_wav_file(out_wav, audio, INPUT_RATE)


def section_names(abc_text):
    """Section labels from `% name` comment lines, in order.

    Alias of :func:`audiocpp_backend.abc_section_names` (kept here so the
    transcriber module stays the obvious home for score helpers).
    """
    return abc_section_names(abc_text)


class SheetSageWidget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "SheetSage2Transcriber"

    PARAM_KEYS = ("audio_file", "weight_type", "max_tokens", "strip_chords")
    ACTION_LABEL = "Transcribe"
    ACTION_PARAM = "transcribe"
    DOWNLOAD_LABEL = "Download runtime + model (~3.5 GB)"
    SHOW_SCORE = True


class SheetSage2Transcriber(AudioCppJob):
    category = "Offline"
    label = "SheetSage2 Transcriber"
    description = (
        "Offline audio-to-ABC transcription (SheetSage2 GGUF via the in-tree "
        "audio.cpp sidecar). A recording becomes a melody+chord score for "
        "review and for cover rendering: wire the melody URI output to "
        "YuE2SongGenerator's abc_uri input (or pick a kept score file) with "
        "cot=melody. Publishes score/melody URI outputs with a one-block "
        "pulse on done. Transcription starts from the Transcribe button or "
        "a rising edge on trigger_in (e.g. wired from a done pulse); a new "
        "edge restarts transcription, cancelling the previous run. "
        "Runs on a background NRT worker (~0.8x realtime, "
        "~7.6 GB peak VRAM for a 4 min song — close other GPU apps). "
        "Lyrics are NOT transcribed; words stay manual. Needs "
        "tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_sheetsage2_gguf.py run once (or the Download "
        "button). Weights are CC BY-NC 4.0 (non-commercial)."
    )

    RUN_PARAM = "transcribe"
    RUN_TAG = "job"
    RUNNING_STATES = ("Transcribing", "Downloading")
    ACTION_VERB = "Transcribe"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background transcription "
                            "(same as the Transcribe button), e.g. wired from a done pulse.")
        self.add_uri_input("audio_uri",
                           help="Wired recording path (e.g. from a File node). While connected "
                                "and non-empty it overrides audio_file; snapshots at Transcribe time.")
        self.add_uri_output("score",
                            help="Kept full score path (melody + chords); published on completion and on patch-load relink.")
        self.add_uri_output("melody",
                            help="Kept melody-only score path for cot=melody covers (empty when strip_chords is off).")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when a new score is ready (wire to a trigger input).")
        self.add_file_param("audio_file", "", filter=AUDIO_FILTER,
                            help="Recording to transcribe (WAV/FLAC/OGG/MP3); converted on the background worker. "
                                 "A connected audio_uri input overrides this.")
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
        self.add_string_param("last_melody", "",
                              help="Path of the last melody-only score variant; re-linked on patch load.")

        self._abc_text = None
        self._abc_path = ""
        self._pulse = False  # emitted as a one-block pulse on done
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No transcription yet"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def _request_job(self):
        """Ask the engine thread to stage a transcription (one-shot per edge).

        The audio thread must never build the job spec (file existence
        checks) or touch params directly (AGENTS.md section 5); instead it
        queues a ("param", ...) command — the unbounded command queue never
        blocks — and the engine thread runs the normal Transcribe path,
        including validation, cancel-previous, and instant telemetry.
        """
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "transcribe", True))

    def _resolve_audio(self):
        """Wired audio_uri wins over the audio_file param (both snapshot at
        Transcribe time). Raises ValueError with a clear message."""
        audio_in = self.inputs.get("audio_uri")
        if audio_in is not None and audio_in.connected_outputs:
            wired = audio_in.get_uri()
            if not wired:
                raise ValueError(
                    "Wired recording is empty — publish a file first, then transcribe.")
            if not Path(wired).exists():
                raise ValueError(f"Wired recording file not found: {wired}")
            return wired
        audio_file = self.params["audio_file"].value
        if not audio_file or not Path(audio_file).exists():
            raise ValueError("Pick a recording first (audio_file is empty or missing).")
        return audio_file

    def _model_fetch_specs(self, model_dir):
        return sheetsage_fetch_specs(model_dir)

    def _running_status(self, spec):
        return ("Transcribing",
                f"{Path(spec['audio_file']).name} ({spec['weight_type']})…")

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
        audio_file = self._resolve_audio()
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "audio_file": audio_file,
            "weight_type": SHEETSAGE_WEIGHT_TYPES[int(self.params["weight_type"].value)],
            "max_tokens": int(self.params["max_tokens"].value),
            "strip_chords": bool(self.params["strip_chords"].value),
            "backend": default_backend(),
        }

    def _run_nrt(self, cancel_event, spec):
        """Full blocking pipeline: convert input, run CLI, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "sheetsage")
        try:
            input_wav = load_input_wav(spec["audio_file"], run_dir / "input.wav")
            out_abc = run_dir / "score.abc"
            result = run_sheetsage_transcribe(
                spec["cli"], spec["model_dir"], audio_wav=input_wav,
                weight_type=spec["weight_type"], max_tokens=spec["max_tokens"],
                out_abc=out_abc, cancel_event=cancel_event,
                backend=spec.get("backend"),
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
                "backend": spec.get("backend", "cuda"),
            })
            self.write_json(keep / "metrics.json", result["metrics"])
            return {"abc": abc_text, "abc_path": str(keep / "score.abc"),
                    "melody_path": melody_path,
                    "metrics": result["metrics"],
                    "sections": section_names(abc_text)}
        except BaseException:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

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
            self.outputs["score"].uri = self._abc_path
            melody_path = result.get("melody_path") or ""
            self.outputs["melody"].uri = melody_path
            self._pulse = True
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
            self.params["last_melody"].set(melody_path)
            self.params["last_melody"].sync()
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        # Scores are small text: re-link synchronously on the control thread.
        # No done pulse: a re-linked score is not newly ready.
        abc = self.params["last_abc"].value if "last_abc" in self.params else ""
        melody = self.params["last_melody"].value if "last_melody" in self.params else ""
        if abc and Path(abc).exists():
            try:
                self._abc_text = Path(abc).read_text(encoding="utf-8")
                self._abc_path = abc
                self.outputs["score"].uri = abc
                self.outputs["melody"].uri = melody if melody and Path(melody).exists() else ""
                self._status = "Ready"
                self._status_detail = f"Re-linked {Path(abc).name}"
                self.error_msg = None
            except OSError:
                self._status = "Idle"
                self._status_detail = "Previous score is missing; transcribe again"
        elif abc:
            self.outputs["score"].uri = ""
            self.outputs["melody"].uri = ""
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
        # Full text for the read-only score viewer (OfflineJobWidget
        # re-renders only on change; a few KB per 100 ms tick is cheap).
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag(),
                "score_text": self._abc_text or ""}
