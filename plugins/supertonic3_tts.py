"""
Supertonic3TTS — offline text-to-speech via Supertonic 3 (Offline).

Consumes inline text (or a wired text file) and a voice profile
(connected voice_style_uri, voice_style_file, or the fallback preset),
synthesizes a kept WAV on an NRT worker through a supervised child
subprocess, and publishes its path on the ``audio`` URI output with a
one-block completion pulse on ``done``.

Wire ``audio`` to SamplePlayer ``uri_in`` and ``done`` to its
``trigger_in``: SamplePlayer loads the new file on the trigger edge and
auto-plays it. Text precedence is connected ``text_uri`` over inline
``text``; profile precedence is connected ``voice_style_uri`` over
``voice_style_file`` over the ``voice`` preset. A connected-but-empty
input fails clearly instead of silently using the fallback.

Expression tags (e.g. ``<laugh>``, ``<breath>``, ``<sigh>``) are
preserved verbatim; support varies by language upstream. Synthesis runs
fully offline after the one-time model/SDK install. Model weights are
under the OpenRAIL-M license (see models/Supertonic3/LICENSE for its
terms).
"""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    AudioCppJob,
    GenerationCancelled,
    missing_specs,
    run_audiocpp_subprocess,
)
from download_util import DownloadCancelled, DownloadSpec
from supertonic3_backend import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_DIR,
    DEFAULT_SILENCE_DURATION,
    DEFAULT_SPEED,
    DEFAULT_THREADS,
    DEFAULT_TOTAL_STEPS,
    LANGUAGES,
    VOICES,
    WORKER_TIMEOUT_S,
    acquire_fetch_lock,
    build_tts_request,
    build_worker_argv,
    fetch_lock_for,
    language_menu_items,
    pick_profile_source,
    pick_text_source,
    result_request_id,
    supertonic3_fetch_specs,
    tag_exception_request_id,
    validate_tts_settings,
)

logger = logging.getLogger(__name__)

STYLE_FILTER = "Voice Style JSON (*.json);;All Files (*.*)"
WORKER_REL = Path("tools") / "supertonic3" / "worker.py"


class Supertonic3TTSWidget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "Supertonic3TTS"

    PARAM_KEYS = ("text", "language", "voice", "voice_style_file",
                  "speed", "total_steps", "silence_duration", "threads")
    ACTION_LABEL = "Generate speech"
    ACTION_PARAM = "generate"
    DOWNLOAD_LABEL = "Download model (~398 MB)"

    def _on_action_pressed(self):
        # Commit the text editor first: set_parameter only debounces, and
        # the flush preserves insertion order, so text is guaranteed to be
        # applied before generate (no stale-prompt race).
        text_widget = self.param_widgets.get("text")
        get_text = getattr(text_widget, "text", None)
        if callable(get_text):
            try:
                self.proxy.set_parameter("text", get_text())
            except Exception:
                pass
        else:
            line_edit = getattr(text_widget, "line_edit", None)
            if line_edit is not None:
                try:
                    self.proxy.set_parameter("text", line_edit.text())
                except Exception:
                    pass
            text_edit = getattr(text_widget, "text_edit", None)
            if text_edit is not None:
                try:
                    self.proxy.set_parameter("text", text_edit.toPlainText())
                except Exception:
                    pass
        self.proxy.set_parameter(self.ACTION_PARAM, True)


class Supertonic3TTS(AudioCppJob):
    category = "Offline"
    label = "Supertonic 3 TTS"
    description = (
        "Offline text-to-speech (Supertonic 3, local ONNX CPU). Type text "
        "or wire a text file to text_uri (wired input wins); pick the voice "
        "via a wired voice_style_uri (from the Voice Designer), a "
        "voice_style_file import, or the voice preset fallback. Generate "
        "synthesizes on a background NRT worker and publishes the kept WAV "
        "on the audio URI output with a one-block pulse on done — wire to "
        "SamplePlayer uri_in/trigger_in for playback. Inline expression "
        "tags such as <laugh>, <breath> and <sigh> are kept verbatim. "
        "Needs the model assets once (Download button or "
        "tools/supertonic3/fetch.py); synthesis is then fully offline. "
        "Weights are under the OpenRAIL-M license "
        "(see models/Supertonic3/LICENSE for its terms)."
    )

    RUN_PARAM = "generate"
    RUN_TAG = "gen"
    RUNNING_STATES = ("Generating", "Downloading")
    ACTION_VERB = "Generate"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background "
                            "synthesis (same as Generate speech).")
        self.add_uri_input("text_uri",
                           help="Wired text path (e.g. from a transcriber). While "
                                "connected and non-empty it overrides the text editor; "
                                "snapshots at Generate time.")
        self.add_uri_input("voice_style_uri",
                           help="Wired voice profile (e.g. from Supertonic Voice "
                                "Designer). While connected and non-empty it overrides "
                                "voice_style_file and the voice preset; snapshots at "
                                "Generate time.")
        self.add_uri_output("audio",
                            help="Kept speech WAV path; published on completion and "
                                 "on patch-load relink.")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when new speech is ready "
                                         "(wire to a trigger input).")

        self.add_string_param("text", "", multiline=True,
                              help="Speech text (UTF-8). Expression tags like <laugh> "
                                   "are preserved. A connected text_uri overrides this.")
        try:
            lang_idx = list(LANGUAGES).index(DEFAULT_LANGUAGE)
        except ValueError:
            lang_idx = 0
        self.add_menu_param("language", language_menu_items(),
                            initial_idx=lang_idx,
                            help="Synthesis language (31 codes plus Auto).")
        self.add_menu_param("voice", list(VOICES), initial_idx=0,
                            help="Fallback preset voice (M1–M5/F1–F5). A wired "
                                 "voice_style_uri or voice_style_file overrides this.")
        self.add_file_param("voice_style_file", "", filter=STYLE_FILTER,
                            help="Optional voice-style JSON overriding the preset. "
                                 "A connected voice_style_uri overrides this.")
        self.add_float_param("speed", DEFAULT_SPEED, 0.7, 2.0,
                             help="Speech speed multiplier.")
        self.add_int_param("total_steps", DEFAULT_TOTAL_STEPS, 1, 100,
                           help="Synthesis steps (more = better quality, slower).")
        self.add_float_param("silence_duration", DEFAULT_SILENCE_DURATION,
                             0.0, 5.0, unit="s",
                             help="Pause inserted between text chunks.")
        self.add_int_param("threads", DEFAULT_THREADS, 1, 8,
                           help="ONNX inference threads (CPU).")
        self.add_string_param("model_dir", DEFAULT_MODEL_DIR,
                              help="Model directory holding onnx/ and voice_styles/ "
                                   "(repo-relative).")
        self.add_bool_param("generate", False,
                            help="Transient trigger: stages a background synthesis, "
                                 "then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads the missing model assets "
                                 "in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight job.")
        self.add_string_param("last_wav", "",
                              help="Path of the last synthesized WAV; re-linked on "
                                   "patch load.")

        self._pulse = False
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No speech generated"
        self._job_seq = 0
        self._active_request_id = None
        self._cancelled_ids = set()
        self._fetch_cancelled = False

    # ------------------------------------------------------------------
    # engine/control thread
    # ------------------------------------------------------------------
    def _request_job(self):
        graph = getattr(self, "graph", None)
        engine = getattr(graph, "engine", None) if graph is not None else None
        if engine is None:
            return
        engine.push_command(("param", self.id, "generate", True))

    def start(self):
        super().start()

    def _new_request_id(self):
        self._job_seq += 1
        rid = f"{self.id}-{self._job_seq}-{uuid.uuid4().hex[:8]}"
        self._active_request_id = rid
        return rid

    def _invalidate_active(self):
        if self._active_request_id is not None:
            self._cancelled_ids.add(self._active_request_id)
            if len(self._cancelled_ids) > 64:
                self._cancelled_ids.pop()
            self._active_request_id = None

    def _on_job_cancelled(self):
        self._invalidate_active()
        self._fetch_cancelled = True

    def remove(self):
        self._invalidate_active()
        super().remove()

    def _is_current_request(self, request_id):
        return (request_id is not None
                and request_id == self._active_request_id
                and request_id not in self._cancelled_ids)

    def _resolve_text(self):
        text_in = self.inputs.get("text_uri")
        connected = (text_in is not None
                     and bool(text_in.connected_outputs))
        snapshot = text_in.get_uri() if text_in is not None else ""
        return pick_text_source(
            inline_text=self.params["text"].value,
            uri_connected=connected, uri_snapshot=snapshot)

    def _resolve_profile(self):
        style_in = self.inputs.get("voice_style_uri")
        connected = (style_in is not None
                     and bool(style_in.connected_outputs))
        snapshot = style_in.get_uri() if style_in is not None else ""
        return pick_profile_source(
            preset_idx=int(self.params["voice"].value),
            file_param=str(self.params["voice_style_file"].value or ""),
            uri_connected=connected, uri_snapshot=snapshot)

    def _build_spec(self):
        """Snapshot committed params + URI strings. Raises ValueError."""
        text_kind, text_value = self._resolve_text()
        style_kind, style_value = self._resolve_profile()
        try:
            lang_idx = int(self.params["language"].value)
            language = list(LANGUAGES)[lang_idx]
        except (TypeError, ValueError, IndexError):
            raise ValueError("unknown language selection.")
        speed = float(self.params["speed"].value)
        total_steps = int(self.params["total_steps"].value)
        silence = float(self.params["silence_duration"].value)
        threads = int(self.params["threads"].value)
        validate_tts_settings(language=language, speed=speed,
                              total_steps=total_steps,
                              silence_duration=silence, threads=threads)
        model_dir = self._resolve_model_dir()
        rid = self._new_request_id()
        return build_tts_request(
            text_kind=text_kind, text_value=text_value,
            style_kind=style_kind, style_value=style_value,
            model_dir=str(model_dir), language=language, speed=speed,
            total_steps=total_steps, silence_duration=silence,
            threads=threads, request_id=rid)

    def _model_fetch_specs(self, model_dir):
        return supertonic3_fetch_specs(model_dir)

    def _build_fetch_payload(self):
        """Model-only fetch plan (no native runtime). Raises ValueError."""
        model_dir = self._resolve_model_dir()
        # A new Download invalidates any pending Generate publication right
        # (the fetch submit bumps the NRT epoch and cancels the worker).
        self._invalidate_active()
        self._fetch_cancelled = False
        return {"runtime": [],
                "staging": str(Path(model_dir) / "_dl"),
                "bin_dir": "",
                "model_dir": str(model_dir),
                "models": [self._spec_to_dict(s)
                           for s in self._model_fetch_specs(model_dir)],
                "runtime_note": "model-only (no native runtime)"}

    def _running_status(self, spec):
        preview = str(spec.get("text_value", ""))[:40].replace("\n", " ")
        return ("Generating", f"{spec.get('language', '')} "
                              f"“{preview}…”" if preview else "Generating…")

    def _fetch_nrt(self, cancel_event, payload):
        """Download missing models under a per-destination lock. NRT only.

        The lock lives only on the NRT side (never on engine/control
        callbacks) and stays responsive to cancellation. Readiness is
        rechecked after obtaining ownership so a second worker does not
        re-download assets the first one just finished.
        """
        model_dir = Path(payload.get("model_dir") or "")
        lock = fetch_lock_for(model_dir)
        acquire_fetch_lock(lock, cancel_event)
        try:
            specs = [DownloadSpec(s["url"], Path(s["dest"]), s["size"],
                                  s["sha256"], s["label"])
                     for s in payload["models"]]
            pending = missing_specs(specs)
            if not pending:
                return {"fetched": []}
            payload = dict(payload)
            payload["models"] = [self._spec_to_dict(s) for s in pending]
            return super()._fetch_nrt(cancel_event, payload)
        finally:
            try:
                lock.release()
            except RuntimeError:
                pass

    def _on_fetch_progress(self, ok, result):
        # Stale progress (queued before Cancel) must not revert Cancelled
        # back to Downloading.
        if self._status != "Downloading":
            return
        super()._on_fetch_progress(ok, result)

    def _on_fetch_complete(self, ok, result):
        if (not ok
                and not isinstance(result, (GenerationCancelled,
                                            DownloadCancelled))
                and self._fetch_cancelled):
            self._status = "Cancelled"
            self._status_detail = "Cancelled by user"
            self.error_msg = None
            self._fetch_cancelled = False
            return
        self._fetch_cancelled = False
        super()._on_fetch_complete(ok, result)

    # ------------------------------------------------------------------
    # NRT worker
    # ------------------------------------------------------------------
    def _run_nrt(self, cancel_event, spec):
        """Blocking pipeline: request file, child synthesis, keep. NRT only."""
        import json as _json
        from supertonic3_backend import REPO_ROOT as _ROOT
        request_id = spec.get("request_id", "")
        try:
            model_dir = Path(spec["model_dir"])
            run_dir = self.new_run_dir(str(model_dir), "supertonic3")
            try:
                request_path = run_dir / "request.json"
                request_path.write_text(_json.dumps(spec, indent=2) + "\n",
                                        encoding="utf-8")
                worker_path = _ROOT / str(WORKER_REL)
                argv = build_worker_argv(str(worker_path), str(request_path))
                output_tail = run_audiocpp_subprocess(
                    argv, cancel_event=cancel_event,
                    timeout_s=WORKER_TIMEOUT_S,
                    task_label="Supertonic 3 synthesis",
                    cancel_message="synthesis cancelled",
                    prog="supertonic3")
                out_wav = run_dir / "speech.wav"
                metrics_path = run_dir / "metrics.json"
                if not out_wav.exists():
                    raise RuntimeError(
                        "supertonic3 reported success but wrote no speech.wav")
                if not metrics_path.exists():
                    raise RuntimeError(
                        "supertonic3 reported success but wrote no metrics.json")
                try:
                    metrics = _json.loads(
                        metrics_path.read_text(encoding="utf-8"))
                except ValueError as e:
                    raise RuntimeError(f"metrics.json is malformed: {e}")
                if not isinstance(metrics, dict):
                    raise RuntimeError("metrics.json is malformed")
                if (metrics.get("request_id") is not None
                        and metrics.get("request_id") != spec.get("request_id")):
                    raise RuntimeError("worker metrics request mismatch")
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled("synthesis cancelled")
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                keep = (model_dir / "outputs"
                        / f"{stamp}_{spec.get('language', 'na')}"
                          f"_{str(spec.get('request_id', ''))[-8:]}")
                self.keep_artifacts(run_dir, keep,
                                    ["speech.wav", "request.json",
                                     "metrics.json"])
                kept_wav = keep / "speech.wav"
                return {"wav": str(kept_wav), "metrics": metrics,
                        "request_id": spec.get("request_id"),
                        "duration": metrics.get("duration"),
                        "sample_rate": metrics.get("sample_rate"),
                        "tail": output_tail[-500:] if output_tail else ""}
            except BaseException:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise
        except BaseException as e:
            raise tag_exception_request_id(e, request_id)

    def _relink_nrt(self, cancel_event, wav_path):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("cancelled")
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"speech file not found: {wav_path}")
        try:
            import soundfile as sf
            info = sf.info(str(wav_path))
            frames, rate = int(info.frames), int(info.samplerate)
        except Exception as e:
            raise RuntimeError(f"cannot read speech file: {e}")
        if frames < 1 or rate < 1:
            raise RuntimeError("speech file has no audio")
        return {"wav": str(wav_path),
                "duration": float(frames) / float(rate),
                "sample_rate": rate}

    def on_nrt_complete(self, tag, ok, result):
        if tag == "gen":
            request_id = result_request_id(result)
            if not ok:
                if isinstance(result, (GenerationCancelled,
                                       DownloadCancelled)):
                    if (self._active_request_id is not None
                            and request_id is not None
                            and request_id != self._active_request_id
                            and request_id not in self._cancelled_ids):
                        return  # stale cancel; don't clobber a new job
                    self._status = "Cancelled"
                    self._status_detail = "Cancelled by user"
                    self.error_msg = None
                    self._active_request_id = None
                    return
                if request_id is not None and (
                        request_id in self._cancelled_ids
                        or request_id != self._active_request_id):
                    return
                if request_id is None and self._active_request_id is None:
                    # Post-cancel failure with no identity: retain
                    # Cancelled, previous URI, and no pulse.
                    return
                self._fail(f"Synthesis failed: {result}")
                self._active_request_id = None
                return
            if not isinstance(result, dict) or not result.get("wav"):
                self._fail("Worker returned an invalid result.")
                self._active_request_id = None
                return
            if not self._is_current_request(request_id):
                return
            self.outputs["audio"].uri = result["wav"]
            self._pulse = True
            self.error_msg = None
            secs = result.get("duration")
            try:
                detail = (f"{float(secs):.1f} s speech"
                          if secs is not None else "Speech ready")
            except (TypeError, ValueError):
                detail = "Speech ready"
            self._status = "Ready"
            self._status_detail = detail
            self.params["last_wav"].set(result["wav"])
            self.params["last_wav"].sync()
            self._active_request_id = None
        elif tag == "relink":
            if ok and isinstance(result, dict) and result.get("wav"):
                self.outputs["audio"].uri = result["wav"]
                self.error_msg = None
                secs = result.get("duration")
                try:
                    self._status_detail = (
                        f"Re-linked {float(secs):.1f} s speech")
                except (TypeError, ValueError):
                    self._status_detail = "Re-linked speech"
                self._status = "Ready"
            else:
                self.outputs["audio"].uri = ""
                self._status = "Idle"
                self._status_detail = (
                    "Previous speech is missing; generate again")
                self.error_msg = None
        elif tag == "fetch_progress":
            self._on_fetch_progress(ok, result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)

    def load_state(self, data: dict):
        super().load_state(data)
        self._pulse = False
        self._active_request_id = None
        self._fetch_cancelled = False
        # Transient triggers never restore as armed.
        for transient in ("generate", "download", "cancel"):
            if transient in self.params:
                self._restage(transient, False)
        if "language" in self.params:
            try:
                idx = int(self.params["language"].value)
            except (TypeError, ValueError):
                idx = -1
            if not 0 <= idx < len(LANGUAGES):
                self.params["language"].set(0)
                self.params["language"].sync()
        if "voice" in self.params:
            try:
                idx = int(self.params["voice"].value)
            except (TypeError, ValueError):
                idx = -1
            if not 0 <= idx < len(VOICES):
                self.params["voice"].set(0)
                self.params["voice"].sync()
        # Artifact validation always runs on NRT after graph attachment;
        # no filesystem checks on the restore boundary. No pulse on restore.
        wav = (self.params["last_wav"].value
               if "last_wav" in self.params else "")
        if wav:
            self._status = "Loading"
            self._status_detail = "Re-linking previous speech…"
            if (getattr(self, "graph", None) is not None
                    and self.graph.engine is not None):
                self.submit_job("relink", self._relink_nrt, wav)
        else:
            self.outputs["audio"].uri = ""

    def get_telemetry(self) -> dict:
        detail = self._status_detail
        uri = self.outputs.get("audio")
        path = getattr(uri, "uri", "") if uri is not None else ""
        if path:
            detail = f"{detail}\n{path}"
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag()}
