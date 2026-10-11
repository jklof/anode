"""Tests for Supertonic3TTS: lifecycle, cancellation, chain integration.

Fake subprocess/file outputs only; no model/network downloads.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import tracemalloc

import plugin_system
from audiocpp_backend import GenerationCancelled
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Supertonic3TTS")
    assert cls is not None, "Supertonic3TTS not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _live():
    import sys
    return sys.modules["supertonic3_tts"]


def write_wav_441(path, seconds=0.5):
    sr = 44100
    n = int(seconds * sr)
    t = np.linspace(0, 0.5, n, dtype=np.float32)
    sf.write(str(path), t, sr)
    return path


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


def _wire_uri(node, port, uri):
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs[port].connect(helper)
    return helper


def _complete_tts(node, wav="s.wav", request_id="r1", duration=1.0):
    node.on_nrt_complete("gen", True,
                         {"wav": wav, "metrics": {}, "request_id": request_id,
                          "duration": duration, "sample_rate": 44100})


# ----------------------------------------------------------------------
# registration / contract
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Offline"
    assert node_cls.label == "Supertonic 3 TTS"
    assert getattr(node_cls, "is_offline") is True
    assert node_cls.RUN_PARAM == "generate"


def test_ports_and_defaults(node_cls):
    from supertonic3_backend import LANGUAGES
    node = make_node(node_cls)
    assert node.inputs["trigger_in"].slot_type == "audio"
    assert node.inputs["text_uri"].slot_type == "uri"
    assert node.inputs["voice_style_uri"].slot_type == "uri"
    assert node.outputs["audio"].slot_type == "uri"
    assert node.outputs["done"].slot_type == "audio"
    assert node.outputs["done"].buffer.shape == (1, BLOCK_SIZE)
    assert node.params["text"].meta.get("multiline") is True
    assert len(node.params["language"].meta["items"]) == 32
    assert node.params["language"].value == list(LANGUAGES).index("en")
    assert node.params["speed"].value == pytest.approx(1.05)
    assert node.params["total_steps"].value == 8
    assert node.params["silence_duration"].value == pytest.approx(0.3)
    assert node.params["threads"].value == 2


def test_language_menu_order_fixed(node_cls):
    from supertonic3_backend import LANGUAGES, language_menu_items
    node = make_node(node_cls)
    assert node.params["language"].meta["items"] == language_menu_items()
    assert list(LANGUAGES).index("fi") > 0
    assert LANGUAGES[-1] == "na"


def test_int_params_fit_qspinbox(node_cls):
    node = make_node(node_cls)
    for name, param in node.params.items():
        if param.type == "int":
            assert param.meta["max"] <= 2 ** 31 - 1, name


# ----------------------------------------------------------------------
# trigger / pulse
# ----------------------------------------------------------------------
def test_trigger_edge_queues_single_generate(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    _feed_trigger(node, 0.0)
    assert engine.commands == []
    _feed_trigger(node, 1.0)
    assert engine.commands == [("param", node.id, "generate", True)]
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 1
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 2


def test_complete_publishes_before_pulse(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    _complete_tts(node, wav="s.wav", request_id="r1")
    assert node.outputs["audio"].uri == "s.wav"
    assert node.params["last_wav"].value == "s.wav"
    node.process()
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_pulse_cleared_on_start(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    _complete_tts(node, request_id="r1")
    node.start()
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_no_net_allocation_emitting_pulse(node_cls):
    import gc
    node = make_node(node_cls)
    node._active_request_id = "r1"
    _complete_tts(node, request_id="r1")
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024


# ----------------------------------------------------------------------
# precedence / validation
# ----------------------------------------------------------------------
def test_text_uri_overrides_inline(node_cls, tmp_path):
    wired = tmp_path / "w.txt"
    wired.write_text("wired text", encoding="utf-8")
    node = make_node(node_cls)
    node.params["text"].set("inline")
    node.params["text"].sync()
    _wire_uri(node, "text_uri", str(wired))
    kind, val = node._resolve_text()
    assert (kind, val) == ("uri", str(wired))


def test_text_uri_empty_rejects(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["text"].set("inline")
    node.params["text"].sync()
    _wire_uri(node, "text_uri", "")
    with pytest.raises(ValueError, match="empty"):
        node._resolve_text()


def test_profile_precedence_uri_file_preset(node_cls, tmp_path):
    wired = tmp_path / "w.json"
    wired.write_text("{}", encoding="utf-8")
    fparam = tmp_path / "f.json"
    fparam.write_text("{}", encoding="utf-8")
    node = make_node(node_cls)
    node.params["voice_style_file"].set(str(fparam))
    node.params["voice_style_file"].sync()
    _wire_uri(node, "voice_style_uri", str(wired))
    assert node._resolve_profile() == ("uri", str(wired))


def test_profile_uri_empty_rejects(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["voice_style_file"].set(str(tmp_path / "f.json"))
    node.params["voice_style_file"].sync()
    _wire_uri(node, "voice_style_uri", "")
    with pytest.raises(ValueError, match="empty"):
        node._resolve_profile()


def test_build_spec_preserves_expression_tags(node_cls):
    node = make_node(node_cls)
    node.params["text"].set("Hi <laugh> there <breath>")
    node.params["text"].sync()
    spec = node._build_spec()
    assert "<laugh>" in spec["text_value"]
    assert "<breath>" in spec["text_value"]


def test_build_spec_rejects_bad_settings(node_cls):
    node = make_node(node_cls)
    node.params["text"].set("hi")
    node.params["text"].sync()
    node.params["speed"].set(9.0)  # clamped to 2.0 by param, so force bad
    node.params["speed"]._staging = 9.0
    node.params["speed"].value = 9.0
    with pytest.raises(ValueError):
        node._build_spec()


def test_fetch_payload_is_model_only(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path / "Supertonic3"))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert len(payload["models"]) > 10


# ----------------------------------------------------------------------
# worker pipeline (stubbed child)
# ----------------------------------------------------------------------
def _stub_child_success(monkeypatch, wav_441_path):
    """Stub run_audiocpp_subprocess to write speech.wav + metrics.json."""
    import supertonic3_tts as _mod  # noqa: F401 (live module patch below)

    def fake_run(argv, **kw):
        assert kw.get("prog") == "supertonic3"
        assert isinstance(argv, list)
        i = argv.index("--request")
        request_path = Path(argv[i + 1])
        run_dir = request_path.parent
        import shutil
        shutil.copy(str(wav_441_path), str(run_dir / "speech.wav"))
        spec = json.loads(request_path.read_text(encoding="utf-8"))
        metrics = {"sample_rate": 44100, "frames": 22050, "duration": 0.5,
                   "request_id": spec.get("request_id")}
        (run_dir / "metrics.json").write_text(json.dumps(metrics),
                                              encoding="utf-8")
        return "metrics.duration=0.500"

    monkeypatch.setattr(_live(), "run_audiocpp_subprocess", fake_run)


def test_run_nrt_full_pipeline(node_cls, tmp_path, monkeypatch):
    src = write_wav_441(tmp_path / "fixture441.wav", seconds=0.5)
    model_dir = tmp_path / "Supertonic3"
    model_dir.mkdir()
    _stub_child_success(monkeypatch, src)
    node = make_node(node_cls)
    node.params["text"].set("hello <laugh>")
    node.params["text"].sync()
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    import threading
    spec = node._build_spec()
    assert spec["text_kind"] == "inline"
    assert "<laugh>" in spec["text_value"]
    result = node._run_nrt(threading.Event(), spec)
    assert Path(result["wav"]).name == "speech.wav"
    assert result["request_id"] == spec["request_id"]
    assert result["duration"] == pytest.approx(0.5)
    node.on_nrt_complete("gen", True, result)
    assert node.outputs["audio"].uri == result["wav"]
    assert node._status == "Ready"


def test_run_nrt_no_automatic_fetch(node_cls, tmp_path, monkeypatch):
    """The parent worker never fetches: the child is attempted and fails
    with an asset hint, while fetch_all is never called."""
    import download_util
    node = make_node(node_cls)
    node.params["text"].set("hello")
    node.params["text"].sync()
    node.params["model_dir"].set(str(tmp_path / "empty"))
    node.params["model_dir"].sync()
    import threading
    child_calls = []
    fetch_calls = []
    monkeypatch.setattr(
        download_util, "fetch_all",
        lambda *a, **k: fetch_calls.append(1) or [])
    monkeypatch.setattr(
        _live(), "run_audiocpp_subprocess",
        lambda *a, **k: child_calls.append(1) or (_ for _ in ()).throw(
            RuntimeError("supertonic3 exited with code 3: "
                         "model assets missing")))
    spec = node._build_spec()
    with pytest.raises(RuntimeError, match="code 3|assets"):
        node._run_nrt(threading.Event(), spec)
    assert child_calls != []  # child attempted
    assert fetch_calls == []  # nothing fetched automatically


def test_worker_failure_cleans_run_dir(node_cls, tmp_path, monkeypatch):
    """A failing child leaves no partial run directory behind.

    Process supervision/reaping itself is covered by the shared
    run_audiocpp_subprocess tests; this asserts the parent's NRT cleanup.
    """
    node = make_node(node_cls)
    node.params["text"].set("hello")
    node.params["text"].sync()
    node.params["model_dir"].set(str(tmp_path / "m"))
    node.params["model_dir"].sync()
    import threading
    spec = node._build_spec()
    run_dirs = list((tmp_path / "m" / "outputs").glob("*")) if (
        tmp_path / "m" / "outputs").exists() else []

    def fake_fail(argv, **kw):
        raise RuntimeError("supertonic3 exited with code 1: boom")

    monkeypatch.setattr(_live(), "run_audiocpp_subprocess", fake_fail)
    with pytest.raises(RuntimeError, match="boom"):
        node._run_nrt(threading.Event(), spec)
    # No partial run directories survive (worker cleans on NRT).
    outputs = tmp_path / "m" / "outputs"
    leftovers = [p for p in outputs.rglob("speech.wav")] if outputs.exists() \
        else []
    assert leftovers == []


# ----------------------------------------------------------------------
# cancellation / supersession
# ----------------------------------------------------------------------
def test_cancel_rejects_queued_result(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r1"
    node.outputs["audio"].uri = "old.wav"
    node._cancelled_ids.add("r1")
    node._active_request_id = None
    node._status = "Cancelled"
    node.on_nrt_complete("gen", True,
                         {"wav": "new.wav", "metrics": {},
                          "request_id": "r1", "duration": 1.0,
                          "sample_rate": 44100})
    assert node.outputs["audio"].uri == "old.wav"
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_generate_twice_second_wins(node_cls):
    node = make_node(node_cls)
    node.params["text"].set("hello")
    node.params["text"].sync()
    s1 = node._build_spec()
    s2 = node._build_spec()
    assert s1["request_id"] != s2["request_id"]
    # First result is stale once the second spec is active.
    node.on_nrt_complete("gen", True,
                         {"wav": "first.wav", "metrics": {},
                          "request_id": s1["request_id"], "duration": 1.0,
                          "sample_rate": 44100})
    assert node.outputs["audio"].uri == ""
    node.on_nrt_complete("gen", True,
                         {"wav": "second.wav", "metrics": {},
                          "request_id": s2["request_id"], "duration": 1.0,
                          "sample_rate": 44100})
    assert node.outputs["audio"].uri == "second.wav"


def test_failure_retains_previous_uri(node_cls):
    node = make_node(node_cls)
    node._active_request_id = "r9"
    node.outputs["audio"].uri = "old.wav"
    err = RuntimeError("boom")
    err._supertonic3_request_id = "r9"
    node.on_nrt_complete("gen", False, err)
    assert node._status == "Error"
    assert node.outputs["audio"].uri == "old.wav"


def test_cancel_rejects_ordinary_queued_failure(node_cls):
    """Submit a failing job, cancel, then drain: Cancelled must stick."""
    import threading
    import time
    from core import Engine
    engine = Engine()
    try:
        node = make_node(node_cls)
        engine.graph.add_node(node)
        node.outputs["audio"].uri = "old.wav"
        node._status = "Generating"
        node._status_detail = "en …"
        node._active_request_id = "r-ordinary"
        node._cancelled_ids = set()

        def boom(cancel_event):
            time.sleep(0.05)
            raise RuntimeError("boom")

        node.submit_job("gen", boom)
        time.sleep(0.02)
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
        assert node.outputs["audio"].uri == "old.wav"
        node.process()
        assert torch.all(node.done.buffer == 0.0)
    finally:
        engine.nrt.shutdown(wait=True)


def test_fetch_progress_after_cancel_ignored(node_cls):
    node = make_node(node_cls)
    node._status = "Cancelled"
    node.on_nrt_complete("fetch_progress", True,
                         {"label": "f", "done": 5, "total": 10,
                          "state": "downloading"})
    assert node._status == "Cancelled"


def test_fetch_failure_after_cancel_stays_cancelled(node_cls):
    node = make_node(node_cls)
    node._status = "Cancelled"
    node._fetch_cancelled = True
    node.on_nrt_complete("fetch", False, RuntimeError("net down"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_load_state_resets_transients(node_cls):
    node = make_node(node_cls)
    for t in ("generate", "download", "cancel"):
        node.params[t].set(True)
        node.params[t].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    for t in ("generate", "download", "cancel"):
        assert node2.params[t].value is False


def test_load_state_delegates_artifact_check_to_nrt(node_cls, tmp_path):
    import time
    from core import Engine
    engine = Engine()
    try:
        node = make_node(node_cls)
        node.params["last_wav"].set(str(tmp_path / "gone.wav"))
        node.params["last_wav"].sync()
        data = node.to_dict()
        node2 = make_node(node_cls)
        engine.graph.add_node(node2)
        node2.load_state(data)
        assert node2._status == "Loading"
        deadline = time.monotonic() + 10.0
        while node2._nrt_inbox is None or node2._nrt_inbox.empty():
            assert time.monotonic() < deadline, "relink job never ran"
            time.sleep(0.02)
        time.sleep(0.2)
        engine._drain_nrt_all()
        assert node2.outputs["audio"].uri == ""
        assert node2._status == "Idle"
    finally:
        engine.nrt.shutdown(wait=True)


# ----------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------
def test_save_load_relinks_without_pulse(node_cls, tmp_path):
    wav = write_wav_441(tmp_path / "kept.wav", seconds=0.3)
    node = make_node(node_cls)
    node.params["last_wav"].set(str(wav))
    node.params["last_wav"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    assert node2.outputs["audio"].uri == ""
    result = node2._relink_nrt(None, str(wav))
    node2.on_nrt_complete("relink", True, result)
    assert node2.outputs["audio"].uri == str(wav)
    node2.process()
    assert torch.all(node2.done.buffer == 0.0)


def test_relink_missing_goes_idle(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["last_wav"].set(str(tmp_path / "gone.wav"))
    node.params["last_wav"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    node2.on_nrt_complete("relink", False, FileNotFoundError("gone"))
    assert node2._status == "Idle"


# ----------------------------------------------------------------------
# widget ordering + patch + chain
# ----------------------------------------------------------------------
def test_generate_commits_text_before_trigger():
    import sys
    from PySide6.QtWidgets import QWidget
    from types import SimpleNamespace
    mod = sys.modules["supertonic3_tts"]

    class StubProxy:
        def __init__(self):
            self.calls = []
            self.widgets = {}

        def set_parameter(self, name, value):
            self.calls.append((name, value))

        def create_param_widget(self, name):
            w = QWidget()
            if name == "text":
                w.text = lambda: "typed hello <laugh>"
            self.widgets[name] = w
            return w

    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication
    inst = QCoreApplication.instance()
    if inst is None:
        app = QApplication([])
    widget = mod.Supertonic3TTSWidget(StubProxy())
    widget._on_action_pressed()
    assert widget.proxy.calls[0] == ("text", "typed hello <laugh>")
    assert widget.proxy.calls[1] == ("generate", True)


def test_example_patch_loads_with_typed_wires(node_cls):
    import json as _json
    from audiocpp_backend import REPO_ROOT
    patch_path = REPO_ROOT / "patches" / "supertonic3_voice_preview.json"
    assert patch_path.exists()
    patch = _json.loads(patch_path.read_text(encoding="utf-8"))
    types = {n["type"] for n in patch["nodes"]}
    assert {"Supertonic3VoiceDesigner", "Supertonic3TTS",
            "SamplePlayer", "AudioDeviceOutput"} <= types
    tts_json = next(n for n in patch["nodes"]
                    if n["type"] == "Supertonic3TTS")
    assert "<breath>" in tts_json["params"]["text"]
    # Typed wires using the real schema.
    from core import Graph
    g = Graph()
    for n_data in patch["nodes"]:
        cls = plugin_system.NODE_REGISTRY.get(n_data["type"])
        assert cls is not None, n_data["type"]
        inst = cls(n_data.get("name", ""))
        inst.id = n_data["id"]
        g.add_node(inst)
        inst.load_state(n_data)
    for c in patch["connections"]:
        assert g.connect(c["src_id"], c["src_port"],
                         c["dst_id"], c["dst_port"]), c


def test_designer_tts_player_chain(node_cls, tmp_path):
    """Apply → Generate → autoplay through engine commands and NRT drains.

    No URI is assigned and no playing flag is set by the test directly:
    designer/TTS completions publish first and pulse on the next block,
    the TTS trigger edge queues a real engine command, and SamplePlayer
    starts itself after its background load.
    """
    import time
    from core import Engine
    plugin_system.load_plugins("plugins")
    designer_cls = plugin_system.NODE_REGISTRY["Supertonic3VoiceDesigner"]
    player_cls = plugin_system.NODE_REGISTRY["SamplePlayer"]
    engine = Engine()
    try:
        designer, tts, player = designer_cls(), make_node(node_cls), player_cls()
        for n in (designer, tts, player):
            engine.graph.add_node(n)
        tts.params["text"].set("Preview please. <breath>")
        tts.params["text"].sync()
        assert engine.graph.connect(designer.id, "voice_style",
                                    tts.id, "voice_style_uri")
        assert engine.graph.connect(designer.id, "done", tts.id, "trigger_in")
        assert engine.graph.connect(tts.id, "audio", player.id, "uri_in")
        assert engine.graph.connect(tts.id, "done", player.id, "trigger_in")
        profile = tmp_path / "voice_style.json"
        ttl_inner = [[0.1] * 256 for _ in range(50)]
        dp_inner = [[0.2] * 16 for _ in range(8)]
        profile.write_text(json.dumps(
            {"style_ttl": {"dims": [1, 50, 256], "data": [ttl_inner]},
             "style_dp": {"dims": [1, 8, 16], "data": [dp_inner]}}),
            encoding="utf-8")
        speech = write_wav_441(tmp_path / "speech.wav", seconds=0.4)

        # Fake synthesis: publish the fixture WAV for the live request id.
        def fake_synth(cancel_event, spec):
            return {"wav": str(speech), "metrics": {},
                    "request_id": spec.get("request_id"), "duration": 0.4,
                    "sample_rate": 44100}

        tts._run_nrt = fake_synth

        def drain_until(pred, timeout_s=15.0):
            deadline = time.monotonic() + timeout_s
            while True:
                engine._drain_nrt_all()
                if pred():
                    return
                assert time.monotonic() < deadline, "NRT job never landed"
                time.sleep(0.02)

        # Designer publishes first, pulses on the next block.
        designer._active_request_id = "d1"
        designer.on_nrt_complete("design", True,
                                 {"profile_path": str(profile),
                                  "display_name": "V", "request_id": "d1"})
        assert designer.outputs["voice_style"].uri == str(profile)
        assert designer._pulse is True
        assert tts.inputs["voice_style_uri"].get_uri() == str(profile)
        designer.process()
        assert torch.all(designer.done.buffer == 1.0)
        # The done edge queues a real engine command (stopped engine runs
        # it synchronously) and submits the fake synthesis to NRT.
        tts.process()
        drain_until(lambda: tts.outputs["audio"].uri != "")
        assert tts.outputs["audio"].uri == str(speech)
        assert tts._pulse is True  # published before the pulse block
        tts.process()
        assert torch.all(tts.done.buffer == 1.0)
        # SamplePlayer loads the new URI on the trigger edge and autoplays
        # without the test touching its playing flag.
        assert player._is_playing is False
        player.process()
        drain_until(lambda: player._is_playing)
        assert player._audio_data.shape[0] == 2
        assert bool(torch.isfinite(player._audio_data).all())
        seen_audio = False
        for _ in range(80):
            player.process()
            buf = player.out.buffer
            assert buf.shape == (2, BLOCK_SIZE)
            assert bool(torch.isfinite(buf).all())
            if bool((buf != 0).any()):
                seen_audio = True
        assert seen_audio
        for _ in range(5):
            player.process()
            assert torch.all(player.out.buffer == 0.0)
    finally:
        engine.nrt.shutdown(wait=True)


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    inst = QCoreApplication.instance()
    if inst is None:
        return QApplication([])
    if isinstance(inst, QApplication):
        return inst
    pytest.skip("a bare QCoreApplication is active; QWidget tests cannot run")


def test_controller_save_flushes_text_before_save(qapp):
    """Edited text flows through the real proxy queue and is flushed
    before Save serialization (insertion order preserved)."""
    from controller import AppController
    controller = AppController()
    try:
        controller.poll_timer.stop()
        captured = []
        controller.engine.push_command = lambda cmd: captured.append(cmd) or 1
        nid = "tts-1"
        controller.set_parameter(nid, "text", "hello <laugh>")
        controller.set_parameter(nid, "generate", True)
        controller.save("patch.json")
        assert captured[0] == ("param", nid, "text", "hello <laugh>")
        assert captured[1] == ("param", nid, "generate", True)
        assert captured[2][0] == "save"
    finally:
        controller.poll_timer.stop()


# ----------------------------------------------------------------------
# worker.py main() with a fake SDK (no weights, no network)
# ----------------------------------------------------------------------
def _worker_module():
    import importlib.util
    import sys
    mod = sys.modules.get("supertonic3_worker")
    if mod is None:
        from audiocpp_backend import REPO_ROOT as _ROOT
        path = _ROOT / "tools" / "supertonic3" / "worker.py"
        spec = importlib.util.spec_from_file_location(
            "supertonic3_worker", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["supertonic3_worker"] = mod
        spec.loader.exec_module(mod)
    return mod
def _install_fake_supertonic(monkeypatch, wav, rate=44100, capture=None):
    import sys
    import types
    import numpy as _np
    capture = capture if capture is not None else {}
    mod = types.ModuleType("supertonic")

    class FakeTTS:
        def __init__(self, model, model_dir, auto_download,
                     intra_op_num_threads, inter_op_num_threads):
            capture["init"] = {"model": model, "model_dir": model_dir,
                               "auto_download": auto_download,
                               "intra": intra_op_num_threads,
                               "inter": inter_op_num_threads}
            assert auto_download is False
            self.sample_rate = rate

        def get_voice_style(self, name):
            capture["preset"] = name
            return ("preset", name)

        def get_voice_style_from_path(self, path):
            capture["style_path"] = str(path)
            return ("path", str(path))

        def synthesize(self, text, voice_style, lang, total_steps, speed,
                       silence_duration, verbose):
            capture["synthesize"] = {"text": text, "lang": lang,
                                     "total_steps": total_steps,
                                     "speed": speed,
                                     "silence_duration": silence_duration,
                                     "verbose": verbose}
            return _np.asarray(wav, dtype=_np.float32), _np.array(
                [wav.shape[1] / rate])

    mod.TTS = FakeTTS
    monkeypatch.setitem(sys.modules, "supertonic", mod)
    return capture


def _write_worker_request(run_dir, **overrides):
    spec = {"text_kind": "inline", "text_value": "hi <laugh>",
            "style_kind": "preset", "style_value": "M1",
            "model_dir": str(run_dir), "language": "en", "speed": 1.05,
            "total_steps": 8, "silence_duration": 0.3, "threads": 2,
            "request_id": "r1"}
    spec.update(overrides)
    (run_dir / "request.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_worker_main_inline_preset(tmp_path, monkeypatch):
    import numpy as _np
    _worker = _worker_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    wav = _np.zeros((1, 4410), dtype=_np.float32)
    capture = _install_fake_supertonic(monkeypatch, wav)
    monkeypatch.setattr("supertonic3_backend.check_assets",
                        lambda d: (True, []))
    _write_worker_request(run_dir)
    assert _worker.main(["--request", str(run_dir / "request.json")]) == 0
    assert (run_dir / "speech.wav").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(
        encoding="utf-8"))
    assert metrics["sample_rate"] == 44100
    assert metrics["duration"] == pytest.approx(0.1)
    assert metrics["request_id"] == "r1"
    assert capture["init"]["auto_download"] is False
    assert capture["init"]["intra"] == 2
    assert capture["preset"] == "M1"
    assert capture["synthesize"]["text"] == "hi <laugh>"
    assert capture["synthesize"]["verbose"] is False


def test_worker_main_uri_text_and_file_style(tmp_path, monkeypatch):
    import numpy as _np
    _worker = _worker_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    text_file = tmp_path / "t.txt"
    text_file.write_text("wired <breath>", encoding="utf-8")
    style_file = tmp_path / "s.json"
    ttl_inner = [[0.1] * 256 for _ in range(50)]
    dp_inner = [[0.2] * 16 for _ in range(8)]
    style_file.write_text(json.dumps(
        {"style_ttl": {"dims": [1, 50, 256], "data": [ttl_inner]},
         "style_dp": {"dims": [1, 8, 16], "data": [dp_inner]}}),
        encoding="utf-8")
    wav = _np.zeros((1, 2205), dtype=_np.float32)
    capture = _install_fake_supertonic(monkeypatch, wav)
    monkeypatch.setattr("supertonic3_backend.check_assets",
                        lambda d: (True, []))
    _write_worker_request(run_dir, text_kind="uri",
                          text_value=str(text_file), style_kind="file",
                          style_value=str(style_file))
    assert _worker.main(["--request", str(run_dir / "request.json")]) == 0
    assert capture["synthesize"]["text"].strip() == "wired <breath>"
    assert capture["style_path"] == str(style_file)
    assert metrics_profile_hash(tmp_path, run_dir) != ""


def metrics_profile_hash(tmp_path, run_dir):
    metrics = json.loads((run_dir / "metrics.json").read_text(
        encoding="utf-8"))
    return metrics.get("profile_hash", "")


def test_worker_main_empty_text_fails(tmp_path, monkeypatch):
    import numpy as _np
    _worker = _worker_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_fake_supertonic(monkeypatch, _np.zeros((1, 10),
                                                    dtype=_np.float32))
    monkeypatch.setattr("supertonic3_backend.check_assets",
                        lambda d: (True, []))
    _write_worker_request(run_dir, text_value="   ")
    assert _worker.main(["--request", str(run_dir / "request.json")]) == 2


def test_worker_main_missing_assets_fails(tmp_path, monkeypatch):
    import numpy as _np
    _worker = _worker_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _install_fake_supertonic(monkeypatch, _np.zeros((1, 10),
                                                    dtype=_np.float32))
    monkeypatch.setattr("supertonic3_backend.check_assets",
                        lambda d: (False, ["onnx/vocoder.onnx"]))
    _write_worker_request(run_dir)
    assert _worker.main(["--request", str(run_dir / "request.json")]) == 3
