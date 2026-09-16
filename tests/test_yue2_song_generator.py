"""Tests for the YuE2 song generator node and the shared audio.cpp backend.

All tests run without a GPU: the sidecar subprocess is stubbed, NRT
results are delivered by calling the completion handlers directly, and
the one GPU-gated test is skipped unless ANODE_YUE2_GPU=1.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import tracemalloc

import plugin_system
import audiocpp_backend as backend
from audiocpp_backend import (
    AudioCppRuntime,
    GenerationCancelled,
    _build_yue2_argv,
    run_yue2_gen,
)
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("YuE2SongGenerator")
    assert cls is not None, "YuE2SongGenerator not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _live():
    """Live top-level plugin module object.

    load_plugins() imports node files WITHOUT the plugins. package prefix,
    so ``import plugins.yue2_song_generator`` would bind an unpatched
    duplicate. Anything that patches module state (or relies on patched
    state) must go through this object; requires the node_cls fixture to
    have run first.
    """
    import sys
    return sys.modules["yue2_song_generator"]


def write_wav(path, seconds=1.0, sr=SAMPLE_RATE, channels=2):
    n = int(seconds * sr)
    t = np.linspace(0, 1, n, dtype=np.float32)
    data = np.stack([t * (i + 1) / channels for i in range(channels)], axis=1)
    sf.write(str(path), data, sr)
    return path


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Sources"
    assert node_cls.label == "YuE2 Song Generator"
    assert "audio.cpp" in node_cls.description


def test_int_params_fit_qspinbox(node_cls):
    """IntParamWidget is a 32-bit QSpinBox (ui_system.py); an out-of-range
    max raises OverflowError when the node UI is built."""
    node = make_node(node_cls)
    for name, param in node.params.items():
        if param.type == "int":
            assert param.meta["max"] <= 2 ** 31 - 1, name
            assert param.meta["min"] >= -(2 ** 31), name


# ----------------------------------------------------------------------
# argv builder (pure, no process)
# ----------------------------------------------------------------------
def _spec(**kw):
    spec = dict(lyrics="[Verse]\nla\n", style="pop", cot="full",
                seed=7, main_gguf="m.gguf", vae_gguf="v.gguf",
                abc_file=None)
    spec.update(kw)
    return spec


def test_argv_builds_expected_command(tmp_path):
    argv = _build_yue2_argv("cli", "models", out_wav=str(tmp_path / "o.wav"), **_spec())
    assert argv[:6] == ["cli", "--task", "gen", "--family", "yue2", "--model"]
    assert "--request-option" in argv
    assert "cot=full" in argv
    assert "yue2.model_gguf=m.gguf" in argv
    assert "--metrics" in argv


def test_argv_rejects_bad_inputs(tmp_path):
    out = str(tmp_path / "o.wav")
    with pytest.raises(ValueError):
        _build_yue2_argv("cli", "m", out_wav=out, **_spec(cot="bogus"))
    with pytest.raises(ValueError):
        _build_yue2_argv("cli", "m", out_wav=out, **_spec(seed=-1))
    with pytest.raises(ValueError):
        _build_yue2_argv("cli", "m", out_wav=out, **_spec(lyrics="  "))
    with pytest.raises(ValueError):
        _build_yue2_argv("cli", "m", out_wav=out,
                         **_spec(cot="off", abc_file="score.abc"))


# ----------------------------------------------------------------------
# run_yue2_gen with a stubbed subprocess
# ----------------------------------------------------------------------
class _StubProc:
    def __init__(self, argv, out_wav, code=0, metrics_text=""):
        self.argv = argv
        self._out = out_wav
        self._code = code
        self._text = metrics_text
        self.terminated = False

    def communicate(self, timeout=None):
        if self._code == 0 and self._out is not None:
            Path(self._out).parent.mkdir(parents=True, exist_ok=True)
            write_wav(self._out, seconds=0.5)
        return self._text, ""

    def terminate(self):
        self.terminated = True

    def poll(self):
        # A live process: run_yue2_gen only skips terminate() after the real
        # communicate() has reaped an exit code.
        return None

    @property
    def returncode(self):
        return self._code

    @property
    def stdout(self):
        return None


def _patch_popen(monkeypatch):
    seen = {}

    def fake_popen(argv, **kwargs):
        proc = _StubProc(argv, seen.get("out"), code=seen.get("code", 0),
                         metrics_text=seen.get("text", ""))
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(backend.subprocess, "Popen", fake_popen)
    return seen


def test_run_gen_success_parses_metrics(tmp_path, monkeypatch):
    out = tmp_path / "run" / "song.wav"
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(parents=True, exist_ok=True)
    (model / "m.gguf").write_text("x")
    (model / "v.gguf").write_text("x")
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    seen["text"] = "metrics.wall_ms=1000\nmetrics.audio_duration_ms=500\nmetrics.rtf=2.0\n"
    import threading
    result = run_yue2_gen(str(cli), str(model), lyrics="la", style="pop",
                          main_gguf="m.gguf", vae_gguf="v.gguf",
                          out_wav=str(out), cancel_event=threading.Event())
    assert result["metrics"]["rtf"] == pytest.approx(2.0)
    assert Path(result["wav"]).exists()
    argv = seen["proc"].argv
    assert "cot=full" in argv and "--metrics" in argv


def test_run_gen_cancel_terminates(tmp_path, monkeypatch):
    out = tmp_path / "song.wav"
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(exist_ok=True)
    (model / "m.gguf").write_text("x")
    (model / "v.gguf").write_text("x")
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        run_yue2_gen(str(cli), str(model), lyrics="la", style="pop",
                     main_gguf="m.gguf", vae_gguf="v.gguf",
                     out_wav=str(out), cancel_event=cancel)
    assert seen["proc"].terminated


def test_run_gen_missing_cli(tmp_path):
    import threading
    with pytest.raises(FileNotFoundError):
        run_yue2_gen(str(tmp_path / "nope"), str(tmp_path), lyrics="la",
                     style="pop", out_wav=str(tmp_path / "o.wav"),
                     cancel_event=threading.Event())


def test_runtime_check_ready_missing(tmp_path):
    runtime = AudioCppRuntime(root=tmp_path / "empty")
    ok, hint = runtime.check_ready()
    assert not ok and "fetch_audiocpp" in hint


# ----------------------------------------------------------------------
# node: song URI + ready pulse (generate-only; playback lives in SamplePlayer)
# ----------------------------------------------------------------------
def _complete_song(node, wav="x.wav", seconds=1.0):
    data = torch.linspace(0, 1, int(seconds * SAMPLE_RATE),
                          dtype=torch.float32).unsqueeze(0).repeat(CHANNELS, 1).contiguous()
    node.on_nrt_complete("gen", True, {"audio": data, "wav": wav,
                                       "metrics": {"rtf": 2.0},
                                       "seed": 1, "cot": "full"})
    return data


def test_ports_are_uri_and_pulse(node_cls):
    node = make_node(node_cls)
    assert node.outputs["song"].slot_type == "uri"
    assert node.outputs["ready"].slot_type == "audio"
    assert node.inputs["trigger_in"].slot_type == "audio"
    assert "out" not in node.outputs
    assert "loop" not in node.params and "gain" not in node.params


class _StubEngine:
    """Minimal engine double: telemetry queue + command capture."""

    def __init__(self):
        import queue
        self.output_queue = queue.Queue()
        self.commands = []
        self.running = False

    def push_command(self, cmd):
        self.commands.append(cmd)
        return len(self.commands)


def _attach_engine(node):
    from types import SimpleNamespace
    engine = _StubEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


def _attach(node):
    return _attach_engine(node)


def _feed_trigger(node, level):
    trig = torch.full((CHANNELS, BLOCK_SIZE), level, dtype=torch.float32)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def test_trigger_edge_queues_single_generate(node_cls):
    node = make_node(node_cls)
    engine = _attach_engine(node)
    _feed_trigger(node, 0.0)
    assert engine.commands == []
    _feed_trigger(node, 1.0)  # rising edge
    assert engine.commands == [("param", node.id, "generate", True)]
    _feed_trigger(node, 1.0)  # sustained high: no repeat
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 1
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)  # new edge after release
    assert len(engine.commands) == 2


def test_trigger_without_engine_is_noop(node_cls):
    node = make_node(node_cls)
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)  # must not raise headless
    assert node._status == "Idle"


def test_complete_publishes_uri_and_pulses_once(node_cls):
    node = make_node(node_cls)
    _complete_song(node, wav="x.wav")
    assert node.outputs["song"].uri == "x.wav"
    assert node._status == "Ready"
    assert node.error_msg is None
    assert node.params["last_wav"].value == "x.wav"
    node.process()  # first block after completion: full 1.0 pulse
    assert torch.all(node.ready.buffer == 1.0)
    node.process()  # afterwards: silence, never retriggers
    assert torch.all(node.ready.buffer == 0.0)


def test_pulse_cleared_on_start(node_cls):
    node = make_node(node_cls)
    _complete_song(node)
    node.start()
    node.process()
    assert torch.all(node.ready.buffer == 0.0)


def test_relink_sets_uri_without_pulse(node_cls):
    node = make_node(node_cls)
    data = torch.zeros((CHANNELS, 100), dtype=torch.float32)
    node.on_nrt_complete("relink", True, {"audio": data, "wav": "r.wav",
                                          "metrics": {}, "seed": None, "cot": None})
    assert node.outputs["song"].uri == "r.wav"
    assert node._status == "Ready"
    node.process()
    assert torch.all(node.ready.buffer == 0.0)


def test_relink_missing_clears_uri(node_cls):
    node = make_node(node_cls)
    node.outputs["song"].uri = "stale.wav"
    node.on_nrt_complete("relink", False, FileNotFoundError("gone"))
    assert node.outputs["song"].uri == ""
    assert node._status == "Idle"


def _wire_uri(node, uri):
    """Connect the node's abc_uri input to a scratch URI output carrying uri."""
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs["abc_uri"].connect(helper)
    return helper


def test_abc_uri_overrides_param(node_cls, tmp_path):
    wired = tmp_path / "wired.abc"
    wired.write_text("X:1\n", encoding="utf-8")
    param_score = tmp_path / "param.abc"
    param_score.write_text("X:1\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["abc_file"].set(str(param_score))
    node.params["abc_file"].sync()
    _wire_uri(node, str(wired))
    assert node._resolve_score() == str(wired)


def test_abc_uri_empty_means_not_ready(node_cls, tmp_path):
    param_score = tmp_path / "param.abc"
    param_score.write_text("X:1\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["abc_file"].set(str(param_score))
    node.params["abc_file"].sync()
    _wire_uri(node, "")
    with pytest.raises(ValueError, match="transcribe first"):
        node._resolve_score()


def test_abc_uri_missing_file_rejected(node_cls, tmp_path):
    node = make_node(node_cls)
    _wire_uri(node, str(tmp_path / "gone.abc"))
    with pytest.raises(ValueError, match="not found"):
        node._resolve_score()


def test_abc_param_used_when_unwired(node_cls, tmp_path):
    param_score = tmp_path / "param.abc"
    param_score.write_text("X:1\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["abc_file"].set(str(param_score))
    node.params["abc_file"].sync()
    assert node._resolve_score() == str(param_score)


def test_complete_failure_sets_error(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("gen", False, RuntimeError("boom"))
    assert node._status == "Error"
    assert "boom" in node.error_msg


def test_complete_cancelled_is_not_an_error(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("gen", False, GenerationCancelled("x"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_no_net_allocation_emitting_pulse(node_cls):
    node = make_node(node_cls)
    _complete_song(node)
    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"


# ----------------------------------------------------------------------
# node: spec validation + wav loading + save/load
# ----------------------------------------------------------------------
def test_build_spec_rejects_missing_backend(node_cls, tmp_path, monkeypatch):
    node = make_node(node_cls)
    monkeypatch.setattr(node, "_resolve_model_dir",
                        lambda: tmp_path / "YuE2-3B-GGUF")
    with pytest.raises(ValueError, match="fetch_"):
        node._build_spec()


def test_load_song_wav_mono_to_stereo(node_cls, tmp_path):
    p = write_wav(tmp_path / "m.wav", seconds=0.2, channels=1)
    load_song_wav = _live().load_song_wav
    t = load_song_wav(p)
    assert t.shape == (CHANNELS, int(0.2 * SAMPLE_RATE))
    assert torch.allclose(t[0], t[1])


def test_save_load_relinks(node_cls, tmp_path):
    node = make_node(node_cls)
    wav = write_wav(tmp_path / "kept.wav", seconds=0.2)
    node.params["last_wav"].set(str(wav))
    node.params["last_wav"].sync()
    snapshot = node.to_dict()

    node2 = make_node(node_cls)
    node2.load_state(snapshot)  # headless: no engine, no auto-submit
    assert node2.outputs["song"].uri == ""
    # Drive the relink path directly (as the NRT worker would).
    result = node2._relink_nrt(None, str(wav))
    node2.on_nrt_complete("relink", True, result)
    assert node2.outputs["song"].uri == str(wav)
    assert node2._status == "Ready"
    node2.process()  # relink must not pulse
    assert torch.all(node2.ready.buffer == 0.0)


def test_relink_missing_file_goes_idle(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["last_wav"].set(str(tmp_path / "gone.wav"))
    node.params["last_wav"].sync()
    snapshot = node.to_dict()
    node2 = make_node(node_cls)
    node2.load_state(snapshot)
    node2.on_nrt_complete("relink", False, FileNotFoundError("gone"))
    assert node2._status == "Idle"
    assert node2.error_msg is None


# ----------------------------------------------------------------------
# GPU-gated end to end (needs ANODE_YUE2_GPU=1 + fetched runtime/models)
# ----------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("ANODE_YUE2_GPU") != "1",
                    reason="needs local GPU + fetched audio.cpp runtime")
def test_gpu_smoke_cot_off(node_cls, tmp_path):
    import subprocess as sp
    import threading
    load_song_wav = _live().load_song_wav
    node = make_node(node_cls)
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse]\nSoft morning light.\n[Chorus]\nSing with the sunrise.\n",
                      encoding="utf-8")
    node.params["lyrics_file"].set(str(lyrics))
    node.params["lyrics_file"].sync()
    spec = node._build_spec()  # raises if runtime/models missing
    cancel = threading.Event()
    out_wav = tmp_path / "smoke.wav"
    gen = run_yue2_gen(spec["cli"], spec["model_dir"],
                       lyrics=lyrics.read_text(encoding="utf-8"),
                       style="English, indie pop", cot="off", seed=831001,
                       main_gguf=spec["main_gguf"], out_wav=str(out_wav),
                       cancel_event=cancel)
    audio = load_song_wav(gen["wav"])
    assert audio.shape[0] == CHANNELS and audio.shape[1] > SAMPLE_RATE
    peak = float(sp.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"]).decode().strip().splitlines()[0])
    assert peak < 6144, f"unexpected VRAM peak: {peak} MiB"


# ----------------------------------------------------------------------
# background fetch (no network: payloads assembled by hand)
# ----------------------------------------------------------------------
def _fetch_payload(url, dest, content: bytes):
    import hashlib
    return {
        "runtime": [],
        "staging": str(dest.parent / "staging"),
        "bin_dir": str(dest.parent / "bin"),
        "models": [{"url": url, "dest": str(dest), "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "label": dest.name}],
    }


def test_fetch_progress_updates_status(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("fetch_progress", True,
                         {"label": "f.bin", "done": 512, "total": 1024,
                          "state": "downloading"})
    assert node._status == "Downloading"
    assert "50.0%" in node._status_detail


def test_fetch_complete_goes_idle(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("fetch", True, {"fetched": ["a", "b"]})
    assert node._status == "Idle"
    assert "Generate" in node._status_detail
    assert node.error_msg is None


def test_fetch_failure_and_cancel(node_cls):
    from download_util import DownloadCancelled
    node = make_node(node_cls)
    node.on_nrt_complete("fetch", False, RuntimeError("net down"))
    assert node._status == "Error"
    assert "net down" in node.error_msg
    node.on_nrt_complete("fetch", False, DownloadCancelled("stop"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_push_telemetry_schema(node_cls):
    """The pushed message must match the engine's own telemetry schema
    (controller routes {"type": "telemetry", "node_data": ...} to widgets)."""
    node = make_node(node_cls)
    engine = _attach(node)
    node._status = "Downloading"
    node._status_detail = "f.bin: 50.0% (512 B / 1.0 KB)"
    node._push_telemetry()
    msg = engine.output_queue.get_nowait()
    assert msg["type"] == "telemetry"
    assert msg["node_data"][node.id] == node.get_telemetry()


def test_push_telemetry_no_engine_is_noop(node_cls):
    node = make_node(node_cls)
    node._push_telemetry()  # must not raise without a graph


def test_download_nothing_missing_pushes_feedback(node_cls, tmp_path, monkeypatch):
    """The reported bug: Download with everything fetched gave zero UI
    feedback while stopped (no NRT traffic -> no telemetry emission)."""
    import sys

    import download_util

    mod = sys.modules["yue2_song_generator"]

    class StubRuntime:
        def __init__(self, *a, **k):
            self.root = tmp_path / "tools" / "audiocpp"
            self.bin_dir = self.root / "bin"
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    node = make_node(node_cls)
    engine = _attach(node)
    # CLI "present" + size-correct model dir -> nothing to fetch.
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin" / "audiocpp_cli.exe").write_text("x")
    node.params["model_dir"].set(str(tmp_path / "models" / "YuE2-3B-GGUF"))
    node.params["model_dir"].sync()
    for spec in mod.yue2_fetch_specs(tmp_path / "models" / "YuE2-3B-GGUF"):
        spec.dest.parent.mkdir(parents=True, exist_ok=True)
        with open(spec.dest, "wb") as f:
            f.truncate(spec.size)
    # Size-correct placeholders: relax content hashes (tested in
    # test_download_util.py against real bytes).
    from pathlib import Path as _Path
    monkeypatch.setattr(
        download_util, "_verify",
        lambda p, s, h: _Path(p).is_file() and (s is None or _Path(p).stat().st_size == s))
    node.params["download"].set(True)
    node.params["download"].sync()
    node.on_ui_param_change("download")
    assert node.params["download"].value is False  # transient restaged
    assert node._status_detail == "Everything already downloaded"
    msg = engine.output_queue.get_nowait()
    assert msg["node_data"][node.id]["audio"] == "Everything already downloaded"


def test_fetch_worker_downloads_file(node_cls, tmp_path):
    import http.server
    import threading

    content = bytes(range(256)) * 64

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        node = make_node(node_cls)
        dest = tmp_path / "models" / "f.bin"
        payload = _fetch_payload(f"http://127.0.0.1:{port}/f.bin", dest, content)
        result = node._fetch_nrt(threading.Event(), payload)
        assert result == {"fetched": [dest.name]}
        assert dest.read_bytes() == content
        node.on_nrt_complete("fetch", True, result)
        assert node._status == "Idle"
    finally:
        httpd.shutdown()


def test_build_fetch_payload_lists_missing(node_cls, tmp_path, monkeypatch):
    import sys

    import download_util
    # NB: load_plugins() imports node files as top-level modules, so the
    # live module is sys.modules["yue2_song_generator"], not
    # plugins.yue2_song_generator (a duplicate import would not affect the
    # registered class).
    mod = sys.modules["yue2_song_generator"]
    from pathlib import Path as _Path

    class StubRuntime:
        def __init__(self, *a, **k):
            self.root = tmp_path / "tools" / "audiocpp"
            self.bin_dir = self.root / "bin"
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path / "models" / "YuE2-3B-GGUF"))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    assert len(payload["runtime"]) == 2  # CLI absent: both zips pending
    assert len(payload["models"]) == 6
    # CLI present + size-correct model dir -> nothing to fetch. (Content
    # hashes are the real pins, unfakeable here, so relax to size-only.)
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin" / "audiocpp_cli.exe").write_text("x")
    for spec in mod.yue2_fetch_specs(tmp_path / "models" / "YuE2-3B-GGUF"):
        spec.dest.parent.mkdir(parents=True, exist_ok=True)
        # Size-correct placeholder without writing gigabytes: truncate
        # extends cheaply and stat() reports the extended size.
        with open(spec.dest, "wb") as f:
            f.truncate(spec.size)
    monkeypatch.setattr(
        download_util, "_verify",
        lambda p, s, h: _Path(p).is_file() and (s is None or _Path(p).stat().st_size == s))
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert payload["models"] == []


# ----------------------------------------------------------------------
# prompt staleness + seed controls (widget-level, offscreen)
# ----------------------------------------------------------------------
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


def test_string_widget_commits_on_focus_loss(qapp):
    """Typed-but-unconfirmed text must not silently diverge: editingFinished
    (focus loss) commits like Return does."""
    from ui_system import StringParamWidget
    seen = []
    w = StringParamWidget("style", {}, "old", seen.append)
    w.line_edit.setText("heavy metal")
    assert seen == []
    w.line_edit.editingFinished.emit()
    assert seen == ["heavy metal"]
    # No-op when nothing changed.
    w.line_edit.editingFinished.emit()
    assert seen == ["heavy metal"]


class _StubProxy:
    """Records set_parameter calls; hands out stub widgets."""

    def __init__(self):
        self.calls = []
        self.widgets = {}

    def set_parameter(self, name, value):
        self.calls.append((name, value))

    def create_param_widget(self, name):
        from types import SimpleNamespace
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        if name == "style":
            w.line_edit = SimpleNamespace(text=lambda: "typed style")
        self.widgets[name] = w
        return w


def _make_widget(qapp):
    import sys
    mod = sys.modules["yue2_song_generator"]
    return mod.YuE2Widget(_StubProxy())


def test_generate_commits_style_before_trigger(qapp):
    """Clicking Generate must apply the editor text before the trigger, so
    the debounced flush can never order them stale-first."""
    widget = _make_widget(qapp)
    widget._on_generate_pressed()
    assert widget.proxy.calls[0] == ("style", "typed style")
    assert widget.proxy.calls[1] == ("generate", True)


def test_seed_button_draws_in_range(qapp):
    widget = _make_widget(qapp)
    widget._on_seed_pressed()
    assert len(widget.proxy.calls) == 1
    name, value = widget.proxy.calls[0]
    assert name == "seed" and 0 <= value < 2 ** 31


def test_widget_covers_all_control_params(qapp, node_cls):
    """Every param widget the custom UI requests must resolve to a real
    node param — this is what left auto_seed unreachable."""
    widget = _make_widget(qapp)
    node = make_node(node_cls)
    for name in widget.proxy.widgets:
        assert name in node.params, name
    assert "auto_seed" in widget.proxy.widgets


def test_auto_seed_widget_builds(qapp):
    from ui_system import ParamWidgetFactory
    widget = ParamWidgetFactory.create(
        "auto_seed", "bool", {"help": "draw fresh seed"}, False, lambda v: None)
    assert widget is not None


def test_pick_seed_manual(node_cls):
    node = make_node(node_cls)
    assert node._pick_seed() == 831001


def test_pick_seed_auto_writes_back(node_cls):
    node = make_node(node_cls)
    node.params["auto_seed"].set(True)
    node.params["auto_seed"].sync()
    seeds = {node._pick_seed() for _ in range(20)}
    assert all(0 <= s < 2 ** 31 for s in seeds)
    assert len(seeds) > 1  # random, not stuck
    assert node.params["seed"].value in seeds  # written back
