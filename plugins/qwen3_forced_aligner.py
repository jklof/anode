"""
Qwen3ForcedAligner — word-timestamp alignment via audio.cpp (Utilities).

Aligns an exact transcript onto a speech/vocal recording using the
Qwen3-ForcedAligner GGUF model through the in-tree audio.cpp sidecar
(tools/audiocpp/). The alignment runs as a blocking call on an NRT
worker; the resulting words.json is kept on disk for review and wires
straight into downstream timing consumers as a URI.

What this node does NOT do: speech recognition. It is not an ASR route —
the transcript is required input (wire Qwen3ASRTranscriber's lyrics
output into lyrics_uri, or pick a transcript file). For long-audio
timestamping without an exact transcript, use Qwen3 ASR with --words-out
instead (the sidecar chunks audio first, then aligns per chunk).

Requirements (fetched once, never auto-downloaded by the node):
  python tools/audiocpp/fetch_audiocpp.py       # native runtime (v0.8.1)
  python tools/audiocpp/fetch_qwen3_align_gguf.py # weights (0.6B Q8, ~1.1 GB)

Model weights are Apache 2.0.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    QWEN3_ALIGN_DEFAULT_GGUF,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    default_backend,
    run_qwen3_align,
    qwen3_align_fetch_specs,
)

INPUT_RATE = 16000
INPUT_CHANNELS = 1
AUDIO_FILTER = "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*.*)"
TEXT_FILTER = "Text Files (*.txt *.md);;All Files (*.*)"
# Single-shot alignment runs the whole clip through the Qwen audio-tower
# encoder at once (no chunking — transcript/chunk pairing cannot be inferred
# safely), so clips longer than one encoder window overflow
# max_source_positions. 30 s matches the ASR chunk default; longer audio
# belongs on the Qwen3ASRTranscriber word_timestamps path, which chunks
# first and aligns per chunk.
ALIGN_MAX_SECONDS = 30.0


def audio_seconds(path):
    """Clip duration in seconds from the file header, or None when the
    header is unreadable. Pure metadata read (never decodes audio).

    WAV/FLAC/OGG go through soundfile; MP3 (and anything soundfile
    rejects) falls back to PyAV container metadata so long MP3s still hit
    the ALIGN_MAX_SECONDS pre-flight guard instead of overflowing the
    encoder deep inside the sidecar.
    """
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        import av
        container = av.open(str(path))
        try:
            if container.duration is not None and container.duration > 0:
                # PyAV duration is in microseconds (AV_TIME_BASE).
                return float(container.duration) / 1_000_000.0
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
        finally:
            close = getattr(container, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    except Exception:
        pass
    return None


def long_audio_message(secs):
    return (
        f"Recording is {secs:.0f} s — single-shot alignment covers clips up "
        f"to ~{ALIGN_MAX_SECONDS:.0f} s (the encoder overflows beyond that). "
        "For songs, enable word_timestamps on Qwen3ASRTranscriber instead: "
        "it chunks the audio first and aligns per chunk."
    )


def load_input_wav(src_path, out_wav):
    """Decode an audio file to a 16 kHz mono WAV for the aligner.

    Thin wrapper over :func:`audio_io.to_engine_audio` (shared decode +
    NRT resample): the NRT worker calls this, tests call it directly.
    Decoding is shared with SamplePlayer (audio_io): WAV/FLAC/OGG via
    soundfile, MP3 via PyAV. Raises RuntimeError on failure.
    """
    from audio_io import to_engine_audio, write_wav_file
    audio = to_engine_audio(src_path, target_sr=INPUT_RATE,
                            target_channels=INPUT_CHANNELS,
                            label="aligner input")
    return write_wav_file(out_wav, audio, INPUT_RATE)


def count_words(words_text):
    """Count word entries in a words.json document. Pure function.

    Accepts a top-level list (one entry per word) or a dict carrying the
    list under ``words``/``tokens``/``segments``; anything else counts 0.
    Malformed JSON counts 0 (the worker rejects empty text separately).
    """
    try:
        data = json.loads(words_text)
    except ValueError:
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("words", "tokens", "segments"):
            entries = data.get(key)
            if isinstance(entries, list):
                return len(entries)
    return 0


class Qwen3ForcedAlignerWidget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "Qwen3ForcedAligner"

    PARAM_KEYS = ("audio_file", "transcript_file", "language")
    ACTION_LABEL = "Align"
    ACTION_PARAM = "align"
    DOWNLOAD_LABEL = "Download runtime + model (~1.1 GB)"


class Qwen3ForcedAligner(AudioCppJob):
    category = "Offline"
    label = "Qwen3 Forced Aligner"
    description = (
        "Offline word-timestamp alignment (Qwen3-ForcedAligner GGUF via the "
        "in-tree audio.cpp sidecar). An exact transcript becomes word "
        "timestamps for a vocal recording: wire Qwen3ASRTranscriber's "
        "lyrics output into lyrics_uri (or pick a transcript file). "
        "Single-shot alignment covers clips up to ~30 s; longer songs "
        "belong on Qwen3ASRTranscriber's word_timestamps path. "
        "Publishes the words URI output with a one-block pulse on done. "
        "Alignment starts from the Align button or a rising edge on "
        "trigger_in (e.g. wired from a done pulse); a new edge restarts "
        "alignment, cancelling the previous run. Runs on a background NRT "
        "worker. This node does NOT transcribe (transcript input is "
        "required). Needs tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_qwen3_align_gguf.py run once (or the "
        "Download button). Weights are Apache 2.0."
    )

    RUN_PARAM = "align"
    RUN_TAG = "job"
    RUNNING_STATES = ("Aligning", "Downloading")
    ACTION_VERB = "Align"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background alignment "
                            "(same as the Align button), e.g. wired from a done pulse.")
        self.add_uri_input("audio_uri",
                           help="Wired recording path (e.g. from a File node). While connected "
                                "and non-empty it overrides audio_file; snapshots at Align time.")
        self.add_uri_input("lyrics_uri",
                           help="Wired transcript path (e.g. from Qwen3ASRTranscriber). While connected "
                                "and non-empty it overrides transcript_file; snapshots at Align time.")
        self.add_uri_output("words",
                            help="Kept word-timestamp JSON path; published on completion and on patch-load relink.")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when new words are ready (wire to a trigger input).")
        self.add_file_param("audio_file", "", filter=AUDIO_FILTER,
                            help="Speech recording to align (WAV/FLAC/OGG/MP3); converted on the background worker. "
                                 "A connected audio_uri input overrides this.")
        self.add_file_param("transcript_file", "", filter=TEXT_FILTER,
                            help="Exact transcript to align; read by the background worker. "
                                 "A connected lyrics_uri input overrides this.")
        self.add_string_param("language", "en",
                              help="Transcript language (e.g. en); required by the aligner, "
                                   "no auto-detect unlike ASR.")
        self.add_string_param("model_dir", "models/Qwen3-ForcedAligner-GGUF",
                              help="Model directory holding the GGUF file (repo-relative).")
        self.add_bool_param("align", False,
                            help="Transient trigger: stages a background alignment, then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads missing runtime/model in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight job.")
        # Last kept words, for save/load re-linking. Written on completion.
        self.add_string_param("last_words", "",
                              help="Path of the last aligned words; re-linked on patch load.")

        self._words_text = None
        self._words_path = ""
        self._word_count = 0
        self._pulse = False  # emitted as a one-block pulse on done
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No alignment yet"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def _request_job(self):
        """Ask the engine thread to stage an alignment (one-shot per edge).

        The audio thread must never build the job spec (file existence
        checks) or touch params directly (AGENTS.md section 5); instead it
        queues a ("param", ...) command — the unbounded command queue never
        blocks — and the engine thread runs the normal Align path,
        including validation, cancel-previous, and instant telemetry.
        """
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "align", True))

    def _model_fetch_specs(self, model_dir):
        return qwen3_align_fetch_specs(model_dir)

    def _running_status(self, spec):
        return ("Aligning",
                f"{Path(spec['audio_file']).name} ({spec['gguf']})…")

    def _resolve_audio(self):
        """Wired audio_uri wins over the audio_file param (both snapshot at
        Align time). Raises ValueError with a clear message."""
        audio_in = self.inputs.get("audio_uri")
        if audio_in is not None and audio_in.connected_outputs:
            wired = audio_in.get_uri()
            if not wired:
                raise ValueError(
                    "Wired recording is empty — publish a file first, then align.")
            if not Path(wired).exists():
                raise ValueError(f"Wired recording file not found: {wired}")
            return wired
        audio_file = self.params["audio_file"].value
        if not audio_file or not Path(audio_file).exists():
            raise ValueError("Pick a recording first (audio_file is empty or missing).")
        return audio_file

    def _resolve_transcript(self):
        """Wired lyrics_uri wins over the transcript_file param (both snapshot
        at Align time). Raises ValueError with a clear message."""
        lyrics_in = self.inputs.get("lyrics_uri")
        if lyrics_in is not None and lyrics_in.connected_outputs:
            wired = lyrics_in.get_uri()
            if not wired:
                raise ValueError(
                    "Wired transcript is empty — transcribe first, then align.")
            if not Path(wired).exists():
                raise ValueError(f"Wired transcript file not found: {wired}")
            return wired
        transcript_file = self.params["transcript_file"].value
        if not transcript_file or not Path(transcript_file).exists():
            raise ValueError("Pick a transcript file first (transcript_file is empty or missing).")
        return transcript_file

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        runtime = AudioCppRuntime()
        ok, hint = runtime.check_ready()
        if not ok:
            raise ValueError(hint)
        model_dir = self._resolve_model_dir()
        gguf = QWEN3_ALIGN_DEFAULT_GGUF
        if not (model_dir / gguf).exists():
            raise ValueError(
                f"Qwen3-ForcedAligner weights {gguf} not found in {model_dir}. "
                "Run: python tools/audiocpp/fetch_qwen3_align_gguf.py "
                "(or press Download)"
            )
        audio_file = self._resolve_audio()
        secs = audio_seconds(audio_file)
        if secs is not None and secs > ALIGN_MAX_SECONDS:
            raise ValueError(long_audio_message(secs))
        transcript_file = self._resolve_transcript()
        language = (self.params["language"].value or "").strip()
        if not language:
            raise ValueError("Pick a language first (language is empty).")
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "gguf": gguf,
            "audio_file": audio_file,
            "transcript_file": transcript_file,
            "language": language,
            "backend": default_backend(),
        }

    def _run_nrt(self, cancel_event, spec):
        """Full blocking pipeline: convert input, run CLI, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "qwen3align")
        try:
            input_wav = load_input_wav(spec["audio_file"], run_dir / "input.wav")
            transcript = Path(spec["transcript_file"]).read_text(encoding="utf-8")
            if not transcript.strip():
                raise ValueError(f"Transcript file is empty: {spec['transcript_file']}")
            out_words = run_dir / "words.json"
            try:
                result = run_qwen3_align(
                    spec["cli"], Path(spec["model_dir"]) / spec["gguf"],
                    audio_wav=input_wav, text=transcript,
                    language=spec["language"], out_words=out_words,
                    cancel_event=cancel_event, backend=spec.get("backend"),
                )
            except RuntimeError as e:
                # Header-unreadable inputs (MP3) skip the _build_spec guard;
                # translate the CLI's encoder overflow into the same guidance.
                if "max_source_positions" in str(e):
                    raise RuntimeError(long_audio_message(
                        audio_seconds(spec["audio_file"])
                        or ALIGN_MAX_SECONDS + 1)) from e
                raise
            words_text = Path(result["words"]).read_text(encoding="utf-8")
            if not words_text.strip():
                raise RuntimeError("aligner returned empty words")
            word_count = count_words(words_text)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = Path(spec["audio_file"]).stem
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_{stem}"
            keep.mkdir(parents=True, exist_ok=True)
            self.keep_artifacts(run_dir, keep, ["words.json"])
            self.write_json(keep / "request.json", {
                "audio_file": spec["audio_file"],
                "transcript_file": spec["transcript_file"],
                "gguf": spec["gguf"],
                "language": spec["language"],
                "backend": spec.get("backend", "cuda"),
            })
            self.write_json(keep / "metrics.json", result["metrics"])
            return {"words_text": words_text,
                    "words_path": str(keep / "words.json"),
                    "word_count": word_count,
                    "metrics": result["metrics"]}
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
                    self._fail(f"Alignment failed: {result}")
                return
            self._words_text = result.get("words_text", "")
            self._words_path = result.get("words_path", "")
            self._word_count = int(result.get("word_count", 0) or 0)
            if not self._words_text.strip():
                self._fail("Worker returned empty words.")
                return
            self.outputs["words"].uri = self._words_path
            self._pulse = True
            self.error_msg = None
            self._status = "Ready"
            self._status_detail = (
                f"{Path(self._words_path).name}: {self._word_count} words"
            )
            self.params["last_words"].set(self._words_path)
            self.params["last_words"].sync()
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        # Words are small JSON text: re-link synchronously on the control thread.
        # No done pulse: re-linked words are not newly ready.
        words = self.params["last_words"].value if "last_words" in self.params else ""
        if words and Path(words).exists():
            try:
                self._words_text = Path(words).read_text(encoding="utf-8")
                self._words_path = words
                self._word_count = count_words(self._words_text)
                self.outputs["words"].uri = words
                self._status = "Ready"
                self._status_detail = f"Re-linked {Path(words).name}"
                self.error_msg = None
            except OSError:
                self._status = "Idle"
                self._status_detail = "Previous words are missing; align again"
        elif words:
            self.outputs["words"].uri = ""
            self._status = "Idle"
            self._status_detail = "Previous words are missing; align again"

    def get_telemetry(self) -> dict:
        preview = ""
        if self._words_text:
            preview = " ".join(self._words_text.split())[:160]
        detail = self._status_detail
        if self._words_path:
            detail = f"{detail}\n{self._words_path}"
            if preview:
                detail = f"{detail}\n{preview}"
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag()}
