"""Tests for Supertonic3VoiceDesigner: lifecycle, publication, save/load.

Fake profile files only; no model/network downloads.
"""
import json
from pathlib import Path

import pytest
import torch
import tracemalloc

import plugin_system
from audiocpp_backend import GenerationCancelled
from base import BLOCK_SIZE, CHANNELS


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Supertonic3VoiceDesigner")
    assert cls is not None, "Supertonic3VoiceDesigner not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _live():
    import sys
    return sys.modules["supertonic3_voice_designer"]


def _write_profile(path, ttl_val=0.1, dp_val=0.2):
    ttl_inner = [[float(ttl_val)] * 256 for _ in range(50)]
    dp_inner = [[float(dp_val)] * 16 for _ in range(8)]
    obj = {"style_ttl": {"dims": [1, 50, 256], "data": [ttl_inner]},
           "style_dp": {"dims": [1, 8, 16], "data": [dp_inner]}}
    path.write_text(json.dumps(obj), encoding="utf-8")
    return obj


class _StubEngine:
    def __init__(self):
        import queue
        self.output_queue = queue.Queue()
        self.commands = []
        self.snapshots = 0

    def push_command(self, cmd):
        self.commands.append(cmd)
        return len(self.commands)

    def _emit_snapshot(self):
        self.snapshots += 1


def _attach(node):
    from types import SimpleNamespace
    engine = _StubEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


def _feed_trigger(node, level):
    trig = torch.full((CHANNELS, BLOCK_SIZE), level, dtype=torch.float32)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def _complete_design(node, profile="p.json", request_id="r1"):
    node.on_nrt_complete("design", True,
                         {"profile_path": profile, "display_name": "V",
                          "request_id": request_id})


# ----------------------------------------------------------------------
# registration / ports
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Offline"
    assert node_cls.label == "Supertonic Voice Designer"
    assert getattr(node_cls, "is_offline") is True
    assert node_cls.RUN_PARAM == "apply"


def test_ports_are_uri_and_pulse(node_cls):
    node = make_node(node_cls)
    assert node.outputs["voice_style"].slot_type == "uri"
    assert node.outputs["done"].slot_type == "audio"
    assert node.outputs["done"].buffer.shape == (1, BLOCK_SIZE)
    assert node.inputs["trigger_in"].slot_type == "audio"


def test_action_label_is_apply_voice():
    import sys
    mod = sys.modules.get("supertonic3_voice_designer")
    assert mod is not None
    assert mod.Supertonic3DesignerWidget.ACTION_LABEL == "Apply voice"
    assert mod.Supertonic3DesignerWidget.ACTION_PARAM == "apply"


# ----------------------------------------------------------------------
# trigger / pulse
# ----------------------------------------------------------------------
def test_trigger_edge_queues_single_apply(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    _feed_trigger(node, 0.0)
    assert engine.commands == []
    _feed_trigger(node, 1.0)
    assert engine.commands == [("param", node.id, "apply", True)]
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 1
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 2


def test_complete_publishes_uri_before_pulse(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    _complete_design(node, profile="p.json", request_id="r1")
    assert node.outputs["voice_style"].uri == "p.json"
    assert node.params["last_profile"].value == "p.json"
    assert node._status == "Ready"
    node.process()
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_pulse_cleared_on_start(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    _complete_design(node, request_id="r1")
    node.start()
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_no_net_allocation_emitting_pulse(node_cls):
    import gc
    node = make_node(node_cls)
    node._active_request_id = "r1"
    _complete_design(node, request_id="r1")
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024


# ----------------------------------------------------------------------
# specs / worker
# ----------------------------------------------------------------------
def test_build_spec_snapshots_and_unique_ids(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path))
    node.params["model_dir"].sync()
    s1 = node._build_spec()
    s2 = node._build_spec()
    assert s1["request_id"] != s2["request_id"]
    assert s1["voice"] == "M1"
    node.params["voice"].set(9)
    node.params["voice"].sync()
    assert node._build_spec()["voice"] == "F5"


def test_fetch_payload_is_model_only(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path / "Supertonic3"))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert len(payload["models"]) > 10


def test_run_nrt_preset_prepares_unique_profiles(node_cls, tmp_path):
    model_dir = tmp_path / "Supertonic3"
    (model_dir / "voice_styles").mkdir(parents=True)
    _write_profile(model_dir / "voice_styles" / "M1.json", ttl_val=0.3)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    import threading
    spec = node._build_spec()
    r1 = node._run_nrt(threading.Event(), spec)
    spec2 = node._build_spec()
    r2 = node._run_nrt(threading.Event(), spec2)
    assert r1["profile_path"] != r2["profile_path"]
    for r in (r1, r2):
        obj = json.loads(Path(r["profile_path"]).read_text(
            encoding="utf-8"))
        assert obj["style_ttl"]["dims"] == [1, 50, 256]
        assert Path(r["provenance_path"]).exists()


def test_run_nrt_file_overrides_preset(node_cls, tmp_path):
    model_dir = tmp_path / "Supertonic3"
    (model_dir / "voice_styles").mkdir(parents=True)
    _write_profile(model_dir / "voice_styles" / "M1.json", ttl_val=0.1)
    imported = tmp_path / "custom.json"
    _write_profile(imported, ttl_val=0.9)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["voice_style_file"].set(str(imported))
    node.params["voice_style_file"].sync()
    import threading
    result = node._run_nrt(threading.Event(), node._build_spec())
    kept = json.loads(Path(result["profile_path"]).read_text(
        encoding="utf-8"))
    assert kept["style_ttl"]["data"][0][0][0] == pytest.approx(0.9)
    assert result["source"].startswith("file:")


def test_run_nrt_missing_preset_hints_fetch(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path / "empty"))
    node.params["model_dir"].sync()
    import threading
    with pytest.raises(ValueError, match="fetch.py|Download"):
        node._run_nrt(threading.Event(), node._build_spec())


# ----------------------------------------------------------------------
# cancellation / failure / supersession
# ----------------------------------------------------------------------
def test_cancel_rejects_queued_result(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    node.outputs["voice_style"].uri = "old.json"
    # Cancel captures the id; the queued success must not publish.
    node._cancelled_ids.add("r1")
    node._active_request_id = None
    node._status = "Cancelled"
    node.on_nrt_complete("design", True,
                         {"profile_path": "new.json",
                          "display_name": "V", "request_id": "r1"})
    assert node.outputs["voice_style"].uri == "old.json"
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_failure_retains_previous_uri(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r9"
    node.outputs["voice_style"].uri = "old.json"
    node.on_nrt_complete("design", False, RuntimeError("boom"))
    assert node._status == "Error"
    assert node.outputs["voice_style"].uri == "old.json"
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_cancelled_completion_is_not_error(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    node.on_nrt_complete("design", False,
                         GenerationCancelled("stop"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_stale_progress_cannot_revert(node_cls):
    node = make_node(node_cls)
    # Live fetch progress still flows while Downloading…
    node._status = "Downloading"
    node.on_nrt_complete("fetch_progress", True,
                         {"label": "x", "done": 1, "total": 2,
                          "state": "downloading"})
    assert node._status == "Downloading"
    # …but progress queued before Cancel must not resurrect Downloading,
    # and a stale design result must not publish or pulse.
    node._status = "Cancelled"
    node._status_detail = "Cancelled by user"
    node.on_nrt_complete("fetch_progress", True,
                         {"label": "x", "done": 1, "total": 2,
                          "state": "downloading"})
    assert node._status == "Cancelled"
    node._status = "Ready"
    node.on_nrt_complete("design", True,
                         {"profile_path": "stale.json",
                          "display_name": "V", "request_id": "old-id"})
    assert node.outputs["voice_style"].uri == ""


# ----------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------
def test_save_load_relinks_without_pulse(node_cls, tmp_path):
    profile = tmp_path / "voice_style.json"
    _write_profile(profile)
    node = make_node(node_cls)
    node.params["last_profile"].set(str(profile))
    node.params["last_profile"].sync()
    snapshot = node.to_dict()
    node2 = make_node(node_cls)
    node2.load_state(snapshot)
    assert node2.outputs["voice_style"].uri == ""
    result = node2._relink_nrt(None, str(profile))
    node2.on_nrt_complete("relink", True, result)
    assert node2.outputs["voice_style"].uri == str(profile)
    node2.process()
    assert torch.all(node2.done.buffer == 0.0)


def test_relink_missing_goes_idle(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["last_profile"].set(str(tmp_path / "gone.json"))
    node.params["last_profile"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    node2.on_nrt_complete("relink", False, FileNotFoundError("gone"))
    assert node2._status == "Idle"
    assert node2.outputs["voice_style"].uri == ""


def test_load_state_resets_transients(node_cls, tmp_path):
    node = make_node(node_cls)
    for t in ("apply", "download", "cancel"):
        node.params[t].set(True)
        node.params[t].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    for t in ("apply", "download", "cancel"):
        assert node2.params[t].value is False


def test_load_state_delegates_artifact_check_to_nrt(node_cls, tmp_path):
    """Missing kept files must not be rejected on the restore boundary;
    the relink worker validates on NRT and reports Idle."""
    import time
    from core import Engine
    engine = Engine()
    try:
        node = make_node(node_cls)
        node.params["last_profile"].set(str(tmp_path / "gone.json"))
        node.params["last_profile"].sync()
        data = node.to_dict()
        node2 = make_node(node_cls)
        engine.graph.add_node(node2)
        node2.load_state(data)
        # Control thread did not pre-reject: relink submitted, Loading shown.
        assert node2._status == "Loading"
        deadline = time.monotonic() + 10.0
        while node2._nrt_inbox is None or node2._nrt_inbox.empty():
            assert time.monotonic() < deadline, "relink job never ran"
            time.sleep(0.02)
        time.sleep(0.2)
        engine._drain_nrt_all()
        assert node2.outputs["voice_style"].uri == ""
        assert node2._status == "Idle"
        node2.process()
        assert torch.all(node2.done.buffer == 0.0)
    finally:
        engine.nrt.shutdown(wait=True)


# ----------------------------------------------------------------------
# P1: shared-destination download coordination
# ----------------------------------------------------------------------
def test_concurrent_downloads_serialize_part_access(node_cls, tmp_path,
                                                   monkeypatch):
    """Two fetch workers for one model dir never overlap in downloads."""
    import threading
    import time
    from audiocpp_backend import AudioCppJob
    model_dir = tmp_path / "Supertonic3"
    model_dir.mkdir()
    n1, n2 = make_node(node_cls), make_node(node_cls)
    for n in (n1, n2):
        n.params["model_dir"].set(str(model_dir))
        n.params["model_dir"].sync()
    p1 = n1._build_fetch_payload()
    p2 = n2._build_fetch_payload()
    assert p1["model_dir"] == p2["model_dir"]
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def fake_fetch(self, cancel_event, payload):
        # Runs after the per-destination lock is held: any overlap here
        # means two writers own the same `.part` files simultaneously.
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            assert state["active"] == 1, "overlapping .part writers"
        try:
            time.sleep(0.05)
            return {"fetched": ["f"]}
        finally:
            with guard:
                state["active"] -= 1

    monkeypatch.setattr(AudioCppJob, "_fetch_nrt", fake_fetch)
    # Bypass the missing-asset recheck (all assets missing by design).
    monkeypatch.setattr(n1, "_model_fetch_specs", lambda d: [])
    monkeypatch.setattr(n2, "_model_fetch_specs", lambda d: [])
    barrier = threading.Barrier(2)
    results = {}

    def run1():
        barrier.wait(timeout=10)
        results["r1"] = n1._fetch_nrt(threading.Event(), p1)

    def run2():
        barrier.wait(timeout=10)
        results["r2"] = n2._fetch_nrt(threading.Event(), p2)

    t1 = threading.Thread(target=run1)
    t2 = threading.Thread(target=run2)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()
    assert results["r1"] == {"fetched": ["f"]}
    assert results["r2"] == {"fetched": ["f"]}
    assert state["max"] == 1


def test_waiting_download_cancel_does_not_damage_owner(node_cls, tmp_path):
    """A cancelled wait leaves the owner's lock and download intact."""
    import threading
    import time
    from download_util import DownloadCancelled
    import supertonic3_backend as _b
    model_dir = tmp_path / "Supertonic3"
    model_dir.mkdir()
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    lock = _b.fetch_lock_for(model_dir)
    assert lock.acquire(timeout=5)
    try:
        cancel = threading.Event()
        outcome = {}

        def waiter():
            try:
                node._fetch_nrt(cancel, payload)
                outcome["result"] = "ran"
            except DownloadCancelled:
                outcome["result"] = "cancelled"
            except Exception as e:  # pragma: no cover
                outcome["result"] = f"wrong: {e!r}"

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)  # let the waiter block in acquire_fetch_lock
        cancel.set()
        t.join(timeout=10)
        assert not t.is_alive()
        assert outcome.get("result") == "cancelled"
        assert lock.locked()  # still owned by this thread
    finally:
        lock.release()
    # The destination still verifies/works after the cancelled wait.
    assert _b.fetch_lock_for(model_dir) is lock


def test_fetch_progress_after_cancel_ignored(node_cls):
    node = make_node(node_cls)
    node._status = "Cancelled"
    node._status_detail = "Cancelled by user"
    node.on_nrt_complete("fetch_progress", True,
                         {"label": "f", "done": 5, "total": 10,
                          "state": "downloading"})
    assert node._status == "Cancelled"


def test_fetch_failure_after_cancel_stays_cancelled(node_cls):
    node = make_node(node_cls)
    node._status = "Cancelled"
    node._status_detail = "Cancelled by user"
    node._fetch_cancelled = True
    node.on_nrt_complete("fetch", False, RuntimeError("net down"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


# ----------------------------------------------------------------------
# P2: cancel/drain ordering with an ordinary queued exception
# ----------------------------------------------------------------------
def test_cancel_rejects_ordinary_queued_failure(node_cls):
    """Submit a failing job, cancel, then drain: Cancelled must stick."""
    import threading
    import time
    from core import Engine
    engine = Engine()
    try:
        node = make_node(node_cls)
        engine.graph.add_node(node)
        node.outputs["voice_style"].uri = "old.json"
        node._status = "Designing"
        node._status_detail = "M1…"
        node._active_request_id = "r-ordinary"
        node._cancelled_ids = set()

        def boom(cancel_event):
            time.sleep(0.05)
            raise RuntimeError("boom")

        node.submit_job("design", boom)
        time.sleep(0.02)
        # Engine/control-thread Cancel (mirrors the cancel param handler).
        node._cancel_job()
        node._on_job_cancelled()
        node._status = "Cancelled"
        node._status_detail = "Cancelled by user"
        deadline = time.monotonic() + 10.0
        while node._nrt_inbox is None or node._nrt_inbox.empty():
            assert time.monotonic() < deadline, "job never completed"
            time.sleep(0.02)
        time.sleep(0.1)
        engine._drain_nrt_all()
        assert node._status == "Cancelled"
        assert node.error_msg is None
        assert node.outputs["voice_style"].uri == "old.json"
        node.process()
        assert torch.all(node.done.buffer == 0.0)
    finally:
        engine.nrt.shutdown(wait=True)
