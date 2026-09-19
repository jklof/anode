"""
Qwen3ASRTranscriber — speech-to-lyrics transcription via audio.cpp (Utilities).

Transcribes speech/vocal recordings into plain lyrics text using the
Qwen3-ASR GGUF model through the in-tree audio.cpp sidecar
(tools/audiocpp/). The transcription runs as a blocking call on an NRT
worker; the resulting lyrics.txt is kept on disk for review and wires
straight into YuE2SongGenerator's lyrics_uri input (or LyricFitter for
structuring with [Verse]/[Chorus] tags — raw ASR output is a single
flowing paragraph without section tags).

What this node does NOT do: melody/chord transcription (SheetSage2Transcriber
covers scores). Word timestamps are an opt-in: with word_timestamps on, the
sidecar also runs the Qwen3-ForcedAligner model (chunking long audio first,
then aligning per chunk) and keeps words.json alongside lyrics.txt.

Requirements (fetched once, never auto-downloaded by the node):
  python tools/audiocpp/fetch_audiocpp.py       # native runtime (v0.8.1)
  python tools/audiocpp/fetch_qwen3_asr_gguf.py  # weights (0.6B Q8, ~1.2 GB)
  python tools/audiocpp/fetch_qwen3_align_gguf.py # only for word_timestamps
  python tools/audiocpp/fetch_silero_vad.py      # only for word_timestamps

Model weights are Apache 2.0.
"""

import shutil
from datetime import datetime
from pathlib import Path

from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    QWEN3_ALIGN_DEFAULT_GGUF,
    QWEN3_ASR_DEFAULT_MAX_TOKENS,
    REPO_ROOT,
    SILERO_VAD_FILENAME,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    default_backend,
    run_qwen3_asr,
    qwen3_align_fetch_specs,
    qwen3_asr_fetch_specs,
    silero_vad_fetch_specs,
)

INPUT_RATE = 16000
INPUT_CHANNELS = 1
AUDIO_FILTER = "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*.*)"
PROFILES = (
    ("0.6B Q8 (~1.2 GB)", "qwen3-asr-0.6b-q8_0.gguf"),
    ("1.7B Q8 (~2.5 GB, manual download)", "qwen3-asr-1.7b-q8_0.gguf"),
)
# Forced-aligner model for the word_timestamps path (fixed repo-relative
# location, shared with Qwen3ForcedAligner via fetch_qwen3_align_gguf.py).
ALIGNER_DIR = REPO_ROOT / "models" / "Qwen3-ForcedAligner-GGUF"
# Silero VAD asset for timestamp chunking (fixed repo-relative location).
# The CLI default (assets/framework/models/silero_vad, relative to the
# audio.cpp checkout) is not shipped with the runtime archives, so the node
# always overrides it via qwen3_asr.vad_model_path when words are requested.
VAD_DIR = REPO_ROOT / "models" / "Silero-VAD"


def load_input_wav(src_path, out_wav):
    """Decode an audio file to a 16 kHz mono WAV for the transcriber.

    Thin wrapper over :func:`audio_io.to_engine_audio` (shared decode +
    NRT resample): the NRT worker calls this, tests call it directly.
    Decoding is shared with SamplePlayer (audio_io): WAV/FLAC/OGG via
    soundfile, MP3 via PyAV. Raises RuntimeError on failure.
    """
    from audio_io import to_engine_audio, write_wav_file
    audio = to_engine_audio(src_path, target_sr=INPUT_RATE,
                            target_channels=INPUT_CHANNELS,
                            label="transcriber input")
    return write_wav_file(out_wav, audio, INPUT_RATE)


class Qwen3ASRWidget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "Qwen3ASRTranscriber"

    PARAM_KEYS = ("audio_file", "profile", "language", "max_tokens",
                  "word_timestamps")
    ACTION_LABEL = "Transcribe"
    ACTION_PARAM = "transcribe"
    DOWNLOAD_LABEL = "Download runtime + models (~2.0 GB, ~3.1 GB with timestamps)"


class Qwen3ASRTranscriber(AudioCppJob):
    category = "Offline"
    label = "Qwen3 ASR Transcriber"
    description = (
        "Offline speech-to-lyrics transcription (Qwen3-ASR GGUF via the in-tree "
        "audio.cpp sidecar). A vocal recording becomes plain lyrics text for "
        "review and for song generation: wire the lyrics URI output to "
        "YuE2SongGenerator's lyrics_uri input (via LyricFitter for section "
        "tags — raw output has none). Publishes the lyrics URI output with "
        "a one-block pulse on done. With word_timestamps on, the sidecar "
        "additionally keeps words.json (chunked long-audio alignment via "
        "the Qwen3-ForcedAligner model) on the words URI output. "
        "Transcription starts from the Transcribe "
        "button or a rising edge on trigger_in (e.g. wired from a done pulse); "
        "a new edge restarts transcription, cancelling the previous run. "
        "Runs on a background NRT worker. "
        "Melody/chords are NOT transcribed (see SheetSage2Transcriber). Needs "
        "tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_qwen3_asr_gguf.py run once (or the Download "
        "button; timestamps additionally need fetch_qwen3_align_gguf.py). "
        "Weights are Apache 2.0."
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
        self.add_uri_output("lyrics",
                            help="Kept lyrics text path; published on completion and on patch-load relink.")
        self.add_uri_output("words",
                            help="Kept word-timestamp JSON path (only with word_timestamps on); "
                                 "published on completion and on patch-load relink.")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when new lyrics are ready (wire to a trigger input).")
        self.add_file_param("audio_file", "", filter=AUDIO_FILTER,
                            help="Vocal recording to transcribe (WAV/FLAC/OGG/MP3); converted on the background worker. "
                                 "A connected audio_uri input overrides this.")
        self.add_menu_param("profile", [label for label, _ in PROFILES], initial_idx=0,
                            help="GGUF precision profile; the 1.7B file is not auto-fetched, place it manually.")
        self.add_string_param("language", "",
                              help="Recognition language hint as a full name (e.g. English); empty lets "
                                   "the model detect it. For known-language audio a hint is more "
                                   "reliable than auto-detect.")
        self.add_int_param("max_tokens", QWEN3_ASR_DEFAULT_MAX_TOKENS, 1, 32768,
                           help="Decoder token cap; long songs may truncate at the cap.")
        self.add_bool_param("word_timestamps", False,
                            help="Also keep word timestamps (words.json) via the Qwen3-ForcedAligner "
                                 "model — the long-audio path that chunks first and aligns per chunk. "
                                 "Needs the aligner weights plus the Silero VAD asset "
                                 "(Download button or fetch_qwen3_align_gguf.py + fetch_silero_vad.py).")
        self.add_string_param("model_dir", "models/Qwen3-ASR-GGUF",
                              help="Model directory holding the GGUF file (repo-relative).")
        self.add_bool_param("transcribe", False,
                            help="Transient trigger: stages a background transcription, then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads missing runtime/model in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight job.")
        # Last kept lyrics, for save/load re-linking. Written on completion.
        self.add_string_param("last_txt", "",
                              help="Path of the last transcribed lyrics; re-linked on patch load.")
        self.add_string_param("last_words", "",
                              help="Path of the last word timestamps; re-linked on patch load.")

        self._lyrics_text = None
        self._lyrics_path = ""
        self._words_path = ""
        self._done_pulse = False  # emitted as a one-block pulse on done
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No transcription yet"

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def start(self):
        self._done_pulse = False
        self._last_trig = 0.0

    def process(self):
        """Emit the one-block done pulse plus trigger edge detect (both
        allocation-free; no file I/O, no param writes here)."""
        buf = self.done.buffer
        buf.zero_()  # anti-ghost: a stale pulse must never retrigger downstream
        if self._done_pulse:
            self._done_pulse = False
            buf.fill_(1.0)
        trig = self.inputs["trigger_in"].get_tensor()[0]
        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            self._request_transcribe()
        self._last_trig = float(trig[-1].item())

    def _request_transcribe(self):
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
        specs = qwen3_asr_fetch_specs(model_dir)
        if self.params["word_timestamps"].value:
            specs = (specs + qwen3_align_fetch_specs(ALIGNER_DIR)
                     + silero_vad_fetch_specs(VAD_DIR))
        return specs

    def _running_status(self, spec):
        words = " +words" if spec.get("word_timestamps") else ""
        return ("Transcribing",
                f"{Path(spec['audio_file']).name} ({spec['gguf']}{words})…")

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        runtime = AudioCppRuntime()
        ok, hint = runtime.check_ready()
        if not ok:
            raise ValueError(hint)
        model_dir = self._resolve_model_dir()
        profile_idx = int(self.params["profile"].value)
        gguf = PROFILES[profile_idx][1]
        if not (model_dir / gguf).exists():
            raise ValueError(
                f"Qwen3-ASR weights {gguf} not found in {model_dir}. "
                "Run: python tools/audiocpp/fetch_qwen3_asr_gguf.py "
                "(or press Download)"
            )
        audio_file = self._resolve_audio()
        language = self.params["language"].value.strip()
        word_timestamps = bool(self.params["word_timestamps"].value)
        aligner_gguf = None
        vad_model = None
        if word_timestamps:
            aligner_gguf = ALIGNER_DIR / QWEN3_ALIGN_DEFAULT_GGUF
            if not aligner_gguf.exists():
                raise ValueError(
                    f"Qwen3-ForcedAligner weights not found: {aligner_gguf}. "
                    "Run: python tools/audiocpp/fetch_qwen3_align_gguf.py "
                    "(or press Download)"
                )
            vad_model = VAD_DIR / SILERO_VAD_FILENAME
            if not vad_model.exists():
                raise ValueError(
                    f"Silero VAD model not found: {vad_model}. "
                    "Run: python tools/audiocpp/fetch_silero_vad.py "
                    "(or press Download)"
                )
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "gguf": gguf,
            "audio_file": audio_file,
            "language": language or None,
            "max_tokens": int(self.params["max_tokens"].value),
            "backend": default_backend(),
            "word_timestamps": word_timestamps,
            "aligner_gguf": str(aligner_gguf) if aligner_gguf else None,
            "vad_model": str(vad_model) if vad_model else None,
        }

    def _run_nrt(self, cancel_event, spec):
        """Full blocking pipeline: convert input, run CLI, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "qwen3asr")
        try:
            input_wav = load_input_wav(spec["audio_file"], run_dir / "input.wav")
            out_txt = run_dir / "transcript.txt"
            want_words = bool(spec.get("word_timestamps"))
            out_words = run_dir / "words.json" if want_words else None
            result = run_qwen3_asr(
                spec["cli"], Path(spec["model_dir"]) / spec["gguf"],
                audio_wav=input_wav, language=spec.get("language"),
                max_tokens=spec["max_tokens"], out_txt=out_txt,
                cancel_event=cancel_event, backend=spec.get("backend"),
                forced_aligner=spec.get("aligner_gguf"),
                words_out=out_words,
                vad_model=spec.get("vad_model"),
            )
            lyrics_text = Path(result["txt"]).read_text(encoding="utf-8").strip()
            if not lyrics_text:
                raise RuntimeError("transcriber returned empty lyrics")
            words_text = None
            if result.get("words"):
                words_text = Path(result["words"]).read_text(encoding="utf-8")
                if not words_text.strip():
                    raise RuntimeError("transcriber returned empty words")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = Path(spec["audio_file"]).stem
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_{stem}"
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "lyrics.txt").write_text(lyrics_text + "\n", encoding="utf-8")
            kept_words = None
            if words_text is not None:
                (keep / "words.json").write_text(words_text, encoding="utf-8")
                kept_words = str(keep / "words.json")
            self.keep_artifacts(run_dir, keep, [])
            self.write_json(keep / "request.json", {
                "audio_file": spec["audio_file"],
                "gguf": spec["gguf"],
                "language": spec.get("language"),
                "max_tokens": spec["max_tokens"],
                "backend": spec.get("backend", "cuda"),
                "word_timestamps": want_words,
                "aligner_gguf": spec.get("aligner_gguf"),
                "vad_model": spec.get("vad_model"),
                "chunk_mode": "fixed" if want_words else "auto",
            })
            self.write_json(keep / "metrics.json", result["metrics"])
            return {"text": lyrics_text, "txt_path": str(keep / "lyrics.txt"),
                    "words_path": kept_words,
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
                    self._fail(f"Transcription failed: {result}")
                return
            self._lyrics_text = result.get("text", "")
            self._lyrics_path = result.get("txt_path", "")
            if not self._lyrics_text.strip():
                self._fail("Worker returned empty lyrics.")
                return
            self.outputs["lyrics"].uri = self._lyrics_path
            words_path = result.get("words_path")
            if words_path:
                self.outputs["words"].uri = words_path
                self._words_path = words_path
            else:
                # Anti-ghost: a words-less run must not leave a stale words URI.
                self.outputs["words"].uri = ""
                self._words_path = ""
            self._done_pulse = True
            self.error_msg = None
            n_chars = len(self._lyrics_text)
            self._status = "Ready"
            self._status_detail = (
                f"{Path(self._lyrics_path).name}: {n_chars} chars"
                + (" + words" if words_path else "")
            )
            self.params["last_txt"].set(self._lyrics_path)
            self.params["last_txt"].sync()
            self.params["last_words"].set(words_path or "")
            self.params["last_words"].sync()
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        # Lyrics are small text: re-link synchronously on the control thread.
        # Words JSON can be large, so only its existence is checked (no text
        # load). No done pulse: re-linked artifacts are not newly ready.
        # Old patches have no last_words key: the param keeps its "" default.
        txt = self.params["last_txt"].value if "last_txt" in self.params else ""
        words = self.params["last_words"].value if "last_words" in self.params else ""
        if words and Path(words).exists():
            self.outputs["words"].uri = words
            self._words_path = words
        else:
            self.outputs["words"].uri = ""
            self._words_path = ""
        if txt and Path(txt).exists():
            try:
                self._lyrics_text = Path(txt).read_text(encoding="utf-8")
                self._lyrics_path = txt
                self.outputs["lyrics"].uri = txt
                self._status = "Ready"
                self._status_detail = f"Re-linked {Path(txt).name}"
                self.error_msg = None
            except OSError:
                self._status = "Idle"
                self._status_detail = "Previous lyrics are missing; transcribe again"
        elif txt:
            self.outputs["lyrics"].uri = ""
            self._status = "Idle"
            self._status_detail = "Previous lyrics are missing; transcribe again"

    def get_telemetry(self) -> dict:
        preview = ""
        if self._lyrics_text:
            preview = " ".join(self._lyrics_text.split())[:160]
        detail = self._status_detail
        if self._lyrics_path:
            detail = f"{detail}\n{self._lyrics_path}"
            if preview:
                detail = f"{detail}\n{preview}"
        if self._words_path:
            detail = f"{detail}\n{self._words_path}"
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag()}
