"""
Supertonic3VoiceDesigner — offline voice-profile publisher (Offline).

Chooses a shipped preset voice (M1–M5/F1–F5) or imports an existing
Supertonic 3 voice-style JSON, validates it on an NRT worker, and
publishes a unique kept ``voice_style.json`` on the ``voice_style`` URI
output with a one-block pulse on ``done``.

Connect ``voice_style`` to Supertonic3TTS ``voice_style_uri`` and
``done`` to its ``trigger_in``: pressing **Apply voice** (or a rising
edge on ``trigger_in``) prepares the profile, then the TTS node
generates the preview sentence and SamplePlayer plays it. With the
engine stopped, Apply still saves the profile but no block pulse
propagates — press Generate on the TTS node (or start the engine)
manually.

This node loads existing voice identities; it does not extract a voice
from reference audio (no cloning pipeline ships with the public
release). Imported files keep their exact values — never normalized or
blended. JSON structure alone cannot prove v3 suitability, so
compatibility of externally supplied files is the user's here.
"""

import logging
import shutil
import uuid
from pathlib import Path

from offline_job_widget import OfflineJobWidget

from audiocpp_backend import (
    AudioCppJob,
    GenerationCancelled,
    missing_specs,
)
from download_util import DownloadCancelled, DownloadSpec
from supertonic3_backend import (
    DEFAULT_MODEL_DIR,
    VOICES,
    acquire_fetch_lock,
    fetch_lock_for,
    load_and_validate_profile_file,
    prepare_profile,
    result_request_id,
    supertonic3_fetch_specs,
    tag_exception_request_id,
)

logger = logging.getLogger(__name__)

PROFILE_FILTER = "Voice Style JSON (*.json);;All Files (*.*)"


class Supertonic3DesignerWidget(OfflineJobWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "Supertonic3VoiceDesigner"

    PARAM_KEYS = ("voice", "voice_style_file", "profile_name")
    ACTION_LABEL = "Apply voice"
    ACTION_PARAM = "apply"
    DOWNLOAD_LABEL = "Download model (~398 MB)"


class Supertonic3VoiceDesigner(AudioCppJob):
    category = "Offline"
    label = "Supertonic Voice Designer"
    description = (
        "Offline voice-profile publisher for Supertonic 3 TTS. Pick a "
        "shipped preset (M1–M5/F1–F5) or import an existing Supertonic 3 "
        "voice-style JSON (voice_style_file overrides the preset); Apply "
        "validates on a background NRT worker and publishes a unique kept "
        "voice_style.json on the voice_style URI output with a one-block "
        "pulse on done. Wire voice_style to the TTS voice_style_uri input "
        "and done to its trigger_in, give the TTS node a short test "
        "sentence, and wire it to SamplePlayer: Apply then previews the "
        "selected voice when the engine runs. Imports load existing "
        "identities — this is not reference-audio cloning, and values are "
        "preserved exactly. Model weights are under the OpenRAIL-M license "
        "(see models/Supertonic3/LICENSE for its terms); fetch once via "
        "the Download button or tools/supertonic3/fetch.py."
    )

    RUN_PARAM = "apply"
    RUN_TAG = "design"
    RUNNING_STATES = ("Designing", "Downloading")
    ACTION_VERB = "Apply"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge stages a background "
                            "profile preparation (same as Apply voice).")
        self.add_uri_output("voice_style",
                            help="Kept voice_style.json path; published on completion "
                                 "and on patch-load relink.")
        self.done = self.add_output("done", channels=1,
                                    help="One-block 1.0 pulse when a new profile is "
                                         "published (wire to a trigger input).")

        self.add_menu_param("voice", list(VOICES), initial_idx=0,
                            help="Shipped preset voice (M1–M5/F1–F5). "
                                 "A voice_style_file import overrides this.")
        self.add_file_param("voice_style_file", "", filter=PROFILE_FILTER,
                            help="Optional Supertonic 3 voice-style JSON to import; "
                                 "overrides the preset above. Must contain "
                                 "style_ttl/style_dp embeddings.")
        self.add_string_param("profile_name", "My Voice",
                              help="Display name for the kept profile (used in "
                                   "file names and provenance).")
        self.add_string_param("model_dir", DEFAULT_MODEL_DIR,
                              help="Model directory holding onnx/ and voice_styles/ "
                                   "(repo-relative).")
        self.add_bool_param("apply", False,
                            help="Transient trigger: validates and publishes the "
                                 "voice profile on a background worker, then resets itself.")
        self.add_bool_param("download", False,
                            help="Transient trigger: downloads the missing model assets "
                                 "in the background (resumable), then resets itself.")
        self.add_bool_param("cancel", False,
                            help="Transient trigger: terminates the in-flight job.")
        self.add_string_param("last_profile", "",
                              help="Path of the last published voice_style.json; "
                                   "re-linked on patch load.")

        self._pulse = False
        self._last_trig = 0.0
        self._status = "Idle"
        self._status_detail = "No voice applied"
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
        engine.push_command(("param", self.id, "apply", True))

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

    def _build_spec(self):
        """Snapshot committed params into a worker spec. Raises ValueError."""
        try:
            voice_idx = int(self.params["voice"].value)
        except (TypeError, ValueError):
            raise ValueError("unknown voice preset.")
        if not 0 <= voice_idx < len(VOICES):
            raise ValueError("unknown voice preset.")
        profile_name = str(self.params["profile_name"].value or "").strip()
        model_dir = self._resolve_model_dir()
        rid = self._new_request_id()
        return {"voice_idx": voice_idx,
                "voice": VOICES[voice_idx],
                "voice_style_file": str(
                    self.params["voice_style_file"].value or ""),
                "profile_name": profile_name or VOICES[voice_idx],
                "model_dir": str(model_dir),
                "request_id": rid}

    def _model_fetch_specs(self, model_dir):
        return supertonic3_fetch_specs(model_dir)

    def _build_fetch_payload(self):
        """Model-only fetch plan (no native runtime). Raises ValueError."""
        model_dir = self._resolve_model_dir()
        # A new Download invalidates any pending Apply publication right
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
        return ("Designing", f"{spec.get('voice', '')}…")

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
        """Validate the preset/import and keep a unique profile. NRT only."""
        request_id = spec.get("request_id", "")
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("cancelled")
            model_dir = Path(spec["model_dir"])
            style_file = spec.get("voice_style_file") or ""
            if style_file:
                obj, source_hash, _validation = load_and_validate_profile_file(
                    style_file)
                source = f"file:{style_file}"
                preset_name = ""
                display = spec.get("profile_name") or Path(style_file).stem
            else:
                preset = spec.get("voice", VOICES[0])
                preset_path = model_dir / "voice_styles" / f"{preset}.json"
                try:
                    obj, source_hash, _validation = load_and_validate_profile_file(
                        preset_path)
                except ValueError as e:
                    raise ValueError(
                        f"{e} (run tools/supertonic3/fetch.py or press Download)")
                source = f"preset:{preset}"
                preset_name = preset
                display = spec.get("profile_name") or preset
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("cancelled")
            kept = prepare_profile(profile_obj=obj, source=source,
                                   source_hash=source_hash,
                                   preset_name=preset_name,
                                   display_name=display,
                                   profiles_dir=model_dir / "profiles")
            if cancel_event is not None and cancel_event.is_set():
                shutil.rmtree(Path(kept["profile_path"]).parent,
                              ignore_errors=True)
                raise GenerationCancelled("cancelled")
            kept["request_id"] = request_id
            kept["source"] = source
            return kept
        except BaseException as e:
            raise tag_exception_request_id(e, request_id)

    def _relink_nrt(self, cancel_event, profile_path):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("cancelled")
        obj, source_hash, _validation = load_and_validate_profile_file(
            profile_path)
        return {"profile": str(profile_path), "source_hash": source_hash,
                "display_name": Path(profile_path).parent.name}

    def on_nrt_complete(self, tag, ok, result):
        if tag == "design":
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
                self._fail(f"Apply failed: {result}")
                self._active_request_id = None
                return
            if not isinstance(result, dict) or not result.get("profile_path"):
                self._fail("Worker returned an invalid profile.")
                self._active_request_id = None
                return
            if not self._is_current_request(request_id):
                return
            self.outputs["voice_style"].uri = result["profile_path"]
            self._pulse = True
            self.error_msg = None
            self._status = "Ready"
            self._status_detail = (
                f"{result.get('display_name', '')}: "
                f"{Path(result['profile_path']).name}")
            self.params["last_profile"].set(result["profile_path"])
            self.params["last_profile"].sync()
            self._active_request_id = None
        elif tag == "relink":
            if ok and isinstance(result, dict) and result.get("profile"):
                self.outputs["voice_style"].uri = result["profile"]
                self.error_msg = None
                self._status = "Ready"
                self._status_detail = (
                    f"Re-linked {Path(result['profile']).name}")
            else:
                self.outputs["voice_style"].uri = ""
                self._status = "Idle"
                self._status_detail = (
                    "Previous profile is missing; apply again")
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
        for transient in ("apply", "download", "cancel"):
            if transient in self.params:
                self._restage(transient, False)
        # Artifact validation always runs on NRT after graph attachment;
        # no filesystem checks on the restore boundary. No pulse on restore.
        profile = (self.params["last_profile"].value
                   if "last_profile" in self.params else "")
        if profile:
            self._status = "Loading"
            self._status_detail = "Re-linking previous profile…"
            if (getattr(self, "graph", None) is not None
                    and self.graph.engine is not None):
                self.submit_job("relink", self._relink_nrt, profile)
        else:
            self.outputs["voice_style"].uri = ""

    def get_telemetry(self) -> dict:
        detail = self._status_detail
        uri = self.outputs.get("voice_style")
        path = getattr(uri, "uri", "") if uri is not None else ""
        if path:
            detail = f"{detail}\n{path}"
        return {"status": self._status, "audio": detail,
                "busy": self._busy_flag()}
