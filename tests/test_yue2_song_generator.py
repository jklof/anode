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
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


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


def feed_gate(node, level):
    trig = torch.full((CHANNELS, BLOCK_SIZE), level, dtype=DTYPE)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def trigger_rise(node):
    trig = torch.cat([torch.zeros((CHANNELS, BLOCK_SIZE // 2), dtype=DTYPE),
                      torch.ones((CHANNELS, BLOCK_SIZE // 2), dtype=DTYPE)], dim=1)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


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
# node: playback kernel
# ----------------------------------------------------------------------
def _install_song(node, seconds=1.0):
    data = torch.linspace(0, 1, int(seconds * SAMPLE_RATE),
                          dtype=DTYPE).unsqueeze(0).repeat(CHANNELS, 1).contiguous()
    node.on_nrt_complete("gen", True, {"audio": data, "wav": "x.wav",
                                       "metrics": {"rtf": 2.0},
                                       "seed": 1, "cot": "full"})
    return data


def test_idle_output_is_exact_zero(node_cls):
    node = make_node(node_cls)
    node.out.buffer.fill_(0.99)
    feed_gate(node, 0.0)
    assert torch.all(node.out.buffer == 0.0)


def test_complete_installs_without_autoplay(node_cls):
    node = make_node(node_cls)
    data = _install_song(node)
    assert node._audio_data is not None
    assert node._is_playing is False
    assert node._status == "Ready"
    assert node.error_msg is None
    assert node.params["last_wav"].value == "x.wav"
    feed_gate(node, 0.0)
    assert torch.all(node.out.buffer == 0.0)
    assert data.shape == (CHANNELS, SAMPLE_RATE)


def test_trigger_plays_and_stops_at_end(node_cls):
    node = make_node(node_cls)
    _install_song(node, seconds=0.05)  # 2400 samples < 5 blocks
    trigger_rise(node)
    assert node._is_playing
    assert torch.any(node.out.buffer != 0.0)
    for _ in range(10):
        feed_gate(node, 1.0)
    assert node._is_playing is False
    assert torch.all(node.out.buffer == 0.0)


def test_loop_wraps(node_cls):
    node = make_node(node_cls)
    _install_song(node, seconds=0.05)
    node.params["loop"].set(True)
    node.params["loop"].sync()
    trigger_rise(node)
    for _ in range(10):
        feed_gate(node, 1.0)
    assert node._is_playing is True


def test_gain_applied(node_cls):
    node = make_node(node_cls)
    data = torch.full((CHANNELS, SAMPLE_RATE), 0.5, dtype=DTYPE)
    node.on_nrt_complete("gen", True, {"audio": data, "wav": "x",
                                       "metrics": {}, "seed": 1, "cot": "off"})
    node.params["gain"].set(0.5)
    node.params["gain"].sync()
    trigger_rise(node)
    assert torch.allclose(node.out.buffer[:, :BLOCK_SIZE // 2],
                          torch.full((CHANNELS, BLOCK_SIZE // 2), 0.25))


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


def test_no_net_allocation_while_playing(node_cls):
    node = make_node(node_cls)
    _install_song(node)
    trigger_rise(node)
    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        feed_gate(node, 1.0)
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
    assert node2._audio_data is None
    # Drive the relink path directly (as the NRT worker would).
    result = node2._relink_nrt(None, str(wav))
    node2.on_nrt_complete("relink", True, result)
    assert node2._audio_data.shape == (CHANNELS, int(0.2 * SAMPLE_RATE))
    assert node2._status == "Ready"


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


class _StubEngine:
    """Minimal engine double: only what AudioCppJob._push_telemetry needs."""

    def __init__(self):
        import queue
        self.output_queue = queue.Queue()
        self.running = False


def _attach(node):
    from types import SimpleNamespace
    engine = _StubEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


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
