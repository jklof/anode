"""
BSRoFormerSeparator — music stem separation via audio.cpp (Offline).

Separates a music mixture into vocals (+ derived instrumental) using the
BS-RoFormer ep368 Q8 GGUF model through the in-tree audio.cpp sidecar
(tools/audiocpp/). The minutes-long separation runs as a blocking call on
an NRT worker; the kept vocals.wav (+ instrumental.wav when produced) stay
on disk for review and wire straight into downstream consumers as URIs.

What this node does NOT do: multi-stem (drums/bass/other) separation —
only vocals + derived instrumental come out (HTDemucs covers the 4-stem
family without touching this node).

Requirements (fetched once, never auto-downloaded by the node):
  python tools/audiocpp/fetch_audiocpp.py        # native runtime (v0.8.1)
  python tools/audiocpp/fetch_bs_roformer_gguf.py # weights (ep368 Q8, ~165 MB)

Upstream checkpoint: model_bs_roformer_ep_368_sdr_12.9628.ckpt lineage
(ViperX/UVR, MIT — credit UVR and its developers); BS-RoFormer
architecture is MIT (lucidrains/BS-RoFormer). GGUF conversion by audio-cpp.
"""

import shutil
from datetime import datetime
from pathlib import Path

from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    BS_ROFORMER_DEFAULT_GGUF,
    BS_ROFORMER_DEFAULT_NUM_OVERLAP,
    BS_ROFORMER_WEIGHT_TYPES,
    AudioCppRuntime,
    AudioCppJob,
    GenerationCancelled,
    bs_roformer_fetch_specs,
    default_backend,
    run_bs_roformer,
)

INPUT_RATE = 44100
INPUT_CHANNELS = 2
AUDIO_FILTER = "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*.*)"


def load_input_wav(src_path, out_wav):
    """Decode an audio file to a 44.1 kHz stereo WAV for the separator.

    Thin wrapper over :func:`audio_io.to_engine_audio` (shared decode +
    mono-dup + NRT resample): the NRT worker calls this, tests call it
    directly. Decoding is shared with SamplePlayer (audio_io): WAV/FLAC/OGG
    via soundfile, MP3 via PyAV. Raises RuntimeError on failure.
    """
    from audio_io import to_engine_audio, write_wav_file
    audio = to_engine_audio(src_path, target_sr=INPUT_RATE,
                            target_channels=INPUT_CHANNELS,
                            label="separator input")
    return write_wav_file(out_wav, audio, INPUT_RATE)


class BSRoFormerWidget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "BSRoFormerSeparator"

    PARAM_KEYS = ("audio_file", "weight_type", "num_overlap")
    ACTION_LABEL = "Separate"
    ACTION_PARAM = "separate"
    DOWNLOAD_LABEL = "Download runtime + model (~1.0 GB)"


class BSRoFormerSeparator(AudioCppJob):
    category = "Offline"
    label = "BS-RoFormer Separator"
    description = (
        "Offline music stem separation (BS-RoFormer ep368 Q8 GGUF via the "
        "in-tree audio.cpp sidecar). A mixture becomes vocals (+ derived "
        "instrumental) WAVs for review and downstream use: wire the vocals / "
        "instrumental URI outputs onward. Publishes both URI outputs with a "
        "one-block pulse on done. Separation starts from the Separate button "
        "or a rising edge on trigger_in (e.g. wired from a done pulse); a new "
        "edge restarts separation, cancelling the previous run. Runs on a "
        "background NRT worker. Drums/bass/other stems are NOT produced (see "
        "HTDemucs). Needs tools/audiocpp/fetch_audiocpp.py and "
        "tools/audiocpp/fetch_bs_roformer_gguf.py run once (or the Download "
        "button). Checkpoint lineage ViperX/UVR, MIT."
    )

    RUN_PARAM = "separate"
    RUN_TAG = "job"
    RUNNING_STATES = ("Separating", "Downloading")
    ACTION_VERB = "Separate"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background separation "
                            "(same as the Separate button), e.g. wired from a done pulse.")
        self.add_uri_input("audio_uri",
                           help="Wired mixture path (e.g. from a File node). While connected "
                                "and non-empty it overrides audio_file; snapshots at Separate time.")
        self.add_uri_output("vocals",
                            help="Kept vocals stem path; published on completion and on patch-load relink.")
        self.add_uri_output("instrumental",
                            help="Kept derived instrumental path (empty when the CLI writes vocals only).")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when new stems are ready (wire to a trigger input).")
        self.add_file_param("audio_file", "", filter=AUDIO_FILTER,
                            help="Music mixture to separate (WAV/FLAC/OGG/MP3); converted to 44.1 kHz stereo on the background worker. "
                                 "A connected audio_uri input overrides this.")
        self.add_menu_param("weight_type", list(BS_ROFORMER_WEIGHT_TYPES), initial_idx=0,
                            help="RoFormer weight storage dtype; native = best quality.")
        self.add_int_param("num_overlap", BS_ROFORMER_DEFAULT_NUM_OVERLAP, 1, 16,
                           help="Overlapping inference windows; 4 = package default (the session option "
                                "is omitted at default to preserve stock boundary blending). Lower is faster "
                                "but can reduce separation quality.")
        self.add_string_param("model_dir", "models/BS-RoFormer-ep368-GGUF",
                              help="Model directory holding the GGUF file (repo-relative).")
        self.add_bool_param("separate", False,
                            help="Transient trigger: stages a background separation, then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads missing runtime/model in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight job.")
        # Last kept stems, for save/load re-linking. Written on completion.
        self.add_string_param("last_vocals", "",
                              help="Path of the last separated vocals stem; re-linked on patch load.")
        self.add_string_param("last_instrumental", "",
                              help="Path of the last derived instrumental; re-linked on patch load.")

        self._vocals_path = ""
        self._instrumental_path = ""
        self._done_pulse = False  # emitted as a one-block pulse on done
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No separation yet"

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
            self._request_separate()
        self._last_trig = float(trig[-1].item())

    def _request_separate(self):
        """Ask the engine thread to stage a separation (one-shot per edge).

        The audio thread must never build the job spec (file existence
        checks) or touch params directly (AGENTS.md section 5); instead it
        queues a ("param", ...) command — the unbounded command queue never
        blocks — and the engine thread runs the normal Separate path,
        including validation, cancel-previous, and instant telemetry.
        """
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "separate", True))

    def _resolve_audio(self):
        """Wired audio_uri wins over the audio_file param (both snapshot at
        Separate time). Raises ValueError with a clear message."""
        audio_in = self.inputs.get("audio_uri")
        if audio_in is not None and audio_in.connected_outputs:
            wired = audio_in.get_uri()
            if not wired:
                raise ValueError(
                    "Wired mixture is empty — publish a file first, then separate.")
            if not Path(wired).exists():
                raise ValueError(f"Wired mixture file not found: {wired}")
            return wired
        audio_file = self.params["audio_file"].value
        if not audio_file or not Path(audio_file).exists():
            raise ValueError("Pick a mixture first (audio_file is empty or missing).")
        return audio_file

    def _model_fetch_specs(self, model_dir):
        return bs_roformer_fetch_specs(model_dir)

    def _running_status(self, spec):
        return ("Separating",
                f"{Path(spec['audio_file']).name} ({spec['weight_type']}, "
                f"overlap {spec['num_overlap']})…")

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        runtime = AudioCppRuntime()
        ok, hint = runtime.check_ready()
        if not ok:
            raise ValueError(hint)
        model_dir = self._resolve_model_dir()
        if not (model_dir / BS_ROFORMER_DEFAULT_GGUF).exists():
            raise ValueError(
                f"BS-RoFormer weights {BS_ROFORMER_DEFAULT_GGUF} not found in {model_dir}. "
                "Run: python tools/audiocpp/fetch_bs_roformer_gguf.py "
                "(or press Download)"
            )
        audio_file = self._resolve_audio()
        return {
            "cli": str(runtime.cli),
            "model_dir": str(model_dir),
            "gguf": BS_ROFORMER_DEFAULT_GGUF,
            "audio_file": audio_file,
            "weight_type": BS_ROFORMER_WEIGHT_TYPES[int(self.params["weight_type"].value)],
            "num_overlap": int(self.params["num_overlap"].value),
            "backend": default_backend(),
        }

    def _run_nrt(self, cancel_event, spec):
        """Full blocking pipeline: convert input, run CLI, keep artifacts."""
        run_dir = self.new_run_dir(spec["model_dir"], "bsroformer")
        try:
            input_wav = load_input_wav(spec["audio_file"], run_dir / "input.wav")
            out_dir = run_dir / "stems"
            result = run_bs_roformer(
                spec["cli"], Path(spec["model_dir"]) / spec["gguf"],
                audio_wav=input_wav, weight_type=spec["weight_type"],
                num_overlap=spec["num_overlap"], out_dir=out_dir,
                cancel_event=cancel_event, backend=spec.get("backend"),
            )
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            stem = Path(spec["audio_file"]).stem
            keep = Path(spec["model_dir"]) / "outputs" / f"{stamp}_{stem}"
            keep.mkdir(parents=True, exist_ok=True)
            kept_vocals = keep / "vocals.wav"
            shutil.move(result["vocals"], str(kept_vocals))
            kept_instr = None
            if result.get("instrumental"):
                src = Path(result["instrumental"])
                if src.exists():
                    kept_instr = keep / "instrumental.wav"
                    shutil.move(str(src), str(kept_instr))
            shutil.rmtree(run_dir, ignore_errors=True)
            self.write_json(keep / "request.json", {
                "audio_file": spec["audio_file"],
                "gguf": spec["gguf"],
                "weight_type": spec["weight_type"],
                "num_overlap": spec["num_overlap"],
                "backend": spec.get("backend", "cuda"),
            })
            self.write_json(keep / "metrics.json", result["metrics"])
            return {"vocals_path": str(kept_vocals),
                    "instrumental_path": str(kept_instr) if kept_instr else None,
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
                    self._fail(f"Separation failed: {result}")
                return
            self._vocals_path = result.get("vocals_path", "")
            self._instrumental_path = result.get("instrumental_path") or ""
            if not self._vocals_path:
                self._fail("Worker returned no vocals stem.")
                return
            self.outputs["vocals"].uri = self._vocals_path
            # Missing instrumental is tolerated, not an error: publish empty.
            self.outputs["instrumental"].uri = self._instrumental_path
            self._done_pulse = True
            self.error_msg = None
            stems = "vocals+instrumental" if self._instrumental_path else "vocals only"
            self._status = "Ready"
            self._status_detail = (
                f"{Path(self._vocals_path).name}: {stems}"
            )
            self.params["last_vocals"].set(self._vocals_path)
            self.params["last_vocals"].sync()
            self.params["last_instrumental"].set(self._instrumental_path)
            self.params["last_instrumental"].sync()
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        # Stems are file references: re-link synchronously on the control
        # thread (no I/O beyond existence checks). No done pulse: re-linked
        # stems are not newly ready.
        vocals = self.params["last_vocals"].value if "last_vocals" in self.params else ""
        instr = self.params["last_instrumental"].value if "last_instrumental" in self.params else ""
        if vocals and Path(vocals).exists():
            self._vocals_path = vocals
            self._instrumental_path = instr if instr and Path(instr).exists() else ""
            self.outputs["vocals"].uri = vocals
            self.outputs["instrumental"].uri = self._instrumental_path
            self._status = "Ready"
            self._status_detail = f"Re-linked {Path(vocals).name}"
            self.error_msg = None
        elif vocals:
            self.outputs["vocals"].uri = ""
            self.outputs["instrumental"].uri = ""
            self._status = "Idle"
            self._status_detail = "Previous stems are missing; separate again"

    def get_telemetry(self) -> dict:
        detail = self._status_detail
        if self._vocals_path:
            detail = f"{detail}\nvocals: {self._vocals_path}"
            if self._instrumental_path:
                detail = f"{detail}\ninstrumental: {self._instrumental_path}"
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag()}
