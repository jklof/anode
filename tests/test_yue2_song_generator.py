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
    assert node_cls.category == "Offline"
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


def test_argv_backend_defaults_to_cuda_and_overrides(tmp_path):
    """Manual installs on GPU-less platforms need --backend cpu instead of
    the hardcoded cuda that could never succeed there."""
    out = str(tmp_path / "o.wav")
    argv = _build_yue2_argv("cli", "m", out_wav=out, **_spec())
    i = argv.index("--backend")
    assert argv[i + 1] == "cuda"
    argv = _build_yue2_argv("cli", "m", out_wav=out, **_spec(), backend="cpu")
    i = argv.index("--backend")
    assert argv[i + 1] == "cpu"


def test_default_backend_follows_torch_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert backend.default_backend() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backend.default_backend() == "cpu"


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


def test_argv_weight_type_defaults_native_and_overrides(tmp_path):
    """q4_0 storage cuts NAR-phase VRAM ~15% for ~1.5x slower generation
    (measured 4016->3380 MiB peak at equal 1500-token work on 8 GB)."""
    out = str(tmp_path / "o.wav")
    argv = _build_yue2_argv("cli", "m", out_wav=out, **_spec())
    assert "yue2.model_weight_type=native" in argv
    argv = _build_yue2_argv("cli", "m", out_wav=out, **_spec(),
                            weight_type="q4_0")
    assert "yue2.model_weight_type=q4_0" in argv


def test_run_gen_forwards_weight_type(tmp_path, monkeypatch):
    out = tmp_path / "run" / "song.wav"
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(parents=True, exist_ok=True)
    (model / "m.gguf").write_text("x")
    (model / "v.gguf").write_text("x")
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    run_yue2_gen(str(cli), str(model), lyrics="la", style="pop",
                 main_gguf="m.gguf", vae_gguf="v.gguf",
                 out_wav=str(out), cancel_event=threading.Event(),
                 weight_type="q4_0")
    assert "yue2.model_weight_type=q4_0" in seen["proc"].argv


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
    """Minimal engine double: telemetry queue + command/snapshot capture."""

    def __init__(self):
        import queue
        self.output_queue = queue.Queue()
        self.commands = []
        self.snapshots = 0
        self.running = False

    def push_command(self, cmd):
        self.commands.append(cmd)
        return len(self.commands)

    def _emit_snapshot(self):
        self.snapshots += 1


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


def _wire_lyrics_uri(node, uri):
    """Connect the node's lyrics_uri input to a scratch URI output."""
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs["lyrics_uri"].connect(helper)
    return helper


def test_lyrics_uri_overrides_param(node_cls, tmp_path):
    wired = tmp_path / "wired.txt"
    wired.write_text("[Verse]\nla\n", encoding="utf-8")
    param_lyrics = tmp_path / "param.txt"
    param_lyrics.write_text("[Verse]\nla\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["lyrics_file"].set(str(param_lyrics))
    node.params["lyrics_file"].sync()
    _wire_lyrics_uri(node, str(wired))
    assert node._resolve_lyrics() == str(wired)


def test_lyrics_uri_empty_means_not_ready(node_cls, tmp_path):
    param_lyrics = tmp_path / "param.txt"
    param_lyrics.write_text("[Verse]\nla\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["lyrics_file"].set(str(param_lyrics))
    node.params["lyrics_file"].sync()
    _wire_lyrics_uri(node, "")
    with pytest.raises(ValueError, match="fit first"):
        node._resolve_lyrics()


def test_lyrics_uri_missing_file_rejected(node_cls, tmp_path):
    node = make_node(node_cls)
    _wire_lyrics_uri(node, str(tmp_path / "gone.txt"))
    with pytest.raises(ValueError, match="not found"):
        node._resolve_lyrics()


def test_lyrics_param_used_when_unwired(node_cls, tmp_path):
    param_lyrics = tmp_path / "param.txt"
    param_lyrics.write_text("[Verse]\nla\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["lyrics_file"].set(str(param_lyrics))
    node.params["lyrics_file"].sync()
    assert node._resolve_lyrics() == str(param_lyrics)


def test_fitter_to_generator_wires_up(node_cls, tmp_path):
    """LyricFitter.lyrics -> YuE2.lyrics_uri connects with matching types."""
    plugin_system.load_plugins("plugins")
    fitter_cls = plugin_system.NODE_REGISTRY.get("LyricFitter")
    assert fitter_cls is not None
    fitter, gen = fitter_cls(), make_node(node_cls)
    gen.inputs["lyrics_uri"].connect(fitter.outputs["lyrics"])
    assert gen.inputs["lyrics_uri"].get_uri() == ""


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


def test_build_spec_carries_weight_type(node_cls, tmp_path, monkeypatch):
    mod = _live()

    class StubRuntime:
        def __init__(self, *a, **k):
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

        def check_ready(self):
            return True, ""

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "yue2-3b-q4_k_m.gguf").write_text("x")
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse]\nla\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["lyrics_file"].set(str(lyrics))
    node.params["lyrics_file"].sync()
    spec = node._build_spec()
    assert spec["weight_type"] == "native"
    node.params["weight_type"].set(1)
    node.params["weight_type"].sync()
    assert node._build_spec()["weight_type"] == "q4_0"


def test_load_song_file_mono_to_stereo(node_cls, tmp_path):
    p = write_wav(tmp_path / "m.wav", seconds=0.2, channels=1)
    load_song_file = _live().load_song_file
    t = load_song_file(p)
    assert t.shape == (CHANNELS, int(0.2 * SAMPLE_RATE))
    assert torch.allclose(t[0], t[1])


def test_load_song_file_mp3(node_cls, tmp_path):
    """Relinking an MP3 keep must decode (MP3 is the default format)."""
    from audio_io import encode_mp3_file
    wav = write_wav(tmp_path / "s.wav", seconds=0.5)
    load_song_file = _live().load_song_file
    ref = load_song_file(wav)
    mp3 = tmp_path / "s.mp3"
    try:
        encode_mp3_file(mp3, ref.numpy(), SAMPLE_RATE)
    except RuntimeError as e:
        pytest.skip(f"mp3 encoder unavailable: {e}")
    t = load_song_file(mp3)
    assert t.shape[0] == CHANNELS
    # MP3 encoder padding shifts length slightly; content must correlate.
    n = min(t.shape[1], ref.shape[1])
    assert torch.nn.functional.cosine_similarity(
        t[:, :n].flatten(), ref[:, :n].flatten(), dim=0) > 0.99


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
    load_song_file = _live().load_song_file
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
    audio = load_song_file(gen["wav"])
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
    # registered class). The fetch machinery lives on the AudioCppJob base,
    # so the runtime seam is patched in audiocpp_backend, not the node module.
    mod = sys.modules["yue2_song_generator"]
    import audiocpp_backend
    from pathlib import Path as _Path

    class StubRuntime:
        def __init__(self, *a, **k):
            self.root = tmp_path / "tools" / "audiocpp"
            self.bin_dir = self.root / "bin"
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

    monkeypatch.setattr(audiocpp_backend, "AudioCppRuntime", StubRuntime)
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
    assert w.text() == "old" and w.text_edit is None
    w.line_edit.setText("heavy metal")
    assert seen == []
    w.line_edit.editingFinished.emit()
    assert seen == ["heavy metal"]
    # No-op when nothing changed.
    w.line_edit.editingFinished.emit()
    assert seen == ["heavy metal"]


def test_multiline_string_widget_commits_on_focus_loss_and_ctrl_enter(qapp):
    """Multiline mode (style prompt box): QTextEdit content commits on focus
    loss; plain Return inserts a newline, Ctrl+Enter commits."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QFocusEvent, QKeyEvent
    from ui_system import StringParamWidget
    seen = []
    w = StringParamWidget("style", {"multiline": True}, "old", seen.append)
    assert w.line_edit is None
    assert w.text() == "old"
    w.text_edit.setPlainText("line one\nline two")
    assert w.text() == "line one\nline two"
    assert seen == []
    # Plain Return: newline, no commit.
    assert w.eventFilter(
        w.text_edit, QKeyEvent(QEvent.KeyPress, Qt.Key_Return,
                               Qt.NoModifier)) is False
    assert seen == []
    # Ctrl+Enter: commit.
    assert w.eventFilter(
        w.text_edit, QKeyEvent(QEvent.KeyPress, Qt.Key_Return,
                               Qt.ControlModifier)) is True
    assert seen == ["line one\nline two"]
    # Focus loss with no change: no duplicate commit.
    assert w.eventFilter(
        w.text_edit, QFocusEvent(QEvent.FocusOut)) is False
    assert seen == ["line one\nline two"]


def test_multiline_widget_syncs_from_backend(qapp):
    from ui_system import StringParamWidget
    w = StringParamWidget("style", {"multiline": True}, "old", lambda v: None)
    w.update_from_backend("new prompt")
    assert w.text() == "new prompt"
    assert w.current_value == "new prompt"


def test_style_param_is_multiline(node_cls):
    node = make_node(node_cls)
    assert node.params["style"].meta.get("multiline") is True
    assert node.params["lyrics_file"].type == "file"  # untouched param kinds


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


def test_generate_prefers_text_accessor_over_line_edit(qapp):
    """The real multiline widget exposes text(); the line_edit fallback is
    only for stubs. A widget with both must commit via text()."""
    from types import SimpleNamespace
    widget = _make_widget(qapp)
    style_widget = widget.proxy.widgets["style"]
    style_widget.text = lambda: "from text()"
    style_widget.line_edit = SimpleNamespace(text=lambda: "from line_edit")
    widget.proxy.calls.clear()
    widget._on_generate_pressed()
    assert widget.proxy.calls[0] == ("style", "from text()")


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


def _keep_spec(tmp_path, fmt):
    return {"style": "s", "lyrics_file": "l", "abc_file": None, "cot": "full",
            "seed": 7, "main_gguf": "m.gguf", "format": fmt,
            "cli": "c", "model_dir": str(tmp_path)}


def _keep_audio(seconds=0.5):
    import numpy as np
    n = int(seconds * SAMPLE_RATE)
    t = torch.linspace(0, 1, n, dtype=torch.float32)
    return t.unsqueeze(0).repeat(CHANNELS, 1).contiguous()


def test_keep_song_wav_format(node_cls, tmp_path):
    node = make_node(node_cls)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "song.wav").write_bytes(b"cli-bytes")
    keep = tmp_path / "keep"
    kept = node._keep_song(run_dir, keep, _keep_audio(), _keep_spec(tmp_path, "wav"),
                           {"rtf": 1.0})
    assert kept == keep / "song.wav"
    assert (keep / "song.wav").read_bytes() == b"cli-bytes"
    assert not (keep / "song.mp3").exists()
    assert not run_dir.exists()  # temp cleaned
    import json
    assert json.loads((keep / "request.json").read_text())["format"] == "wav"


def test_keep_song_mp3_format_keeps_only_mp3(node_cls, tmp_path):
    node = make_node(node_cls)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "song.wav").write_bytes(b"cli-bytes")
    keep = tmp_path / "keep"
    try:
        kept = node._keep_song(run_dir, keep, _keep_audio(), _keep_spec(tmp_path, "mp3"),
                               {"rtf": 1.0})
    except RuntimeError as e:
        pytest.skip(f"mp3 encoder unavailable: {e}")
    assert kept == keep / "song.mp3"
    assert (keep / "song.mp3").stat().st_size > 1024
    assert not (keep / "song.wav").exists()  # exactly one audio file kept
    assert not run_dir.exists()
    # And the kept MP3 re-decodes for relinking.
    load_song_file = _live().load_song_file
    t = load_song_file(kept)
    assert t.shape[0] == CHANNELS and t.shape[1] > 0


def test_refresh_ui_emits_snapshot(node_cls):
    """The stale-red-box fix: error clearing must snapshot, not just push
    telemetry (headers/badges only update on snapshots)."""
    node = make_node(node_cls)
    engine = _attach_engine(node)
    node._fail("boom")
    node._refresh_ui()
    assert engine.snapshots == 1
    msg = engine.output_queue.get_nowait()
    assert msg["node_data"][node.id]["status"] == "Error"


def test_generate_validation_error_refreshes(node_cls, tmp_path, monkeypatch):
    node = make_node(node_cls)
    engine = _attach_engine(node)
    monkeypatch.setattr(node, "_resolve_model_dir",
                        lambda: tmp_path / "YuE2-3B-GGUF")
    node.params["generate"].set(True)
    node.params["generate"].sync()
    node.on_ui_param_change("generate")
    assert node._status == "Error"
    assert engine.snapshots == 1  # red box appears immediately


def test_telemetry_shows_elapsed_while_generating(node_cls):
    import time
    node = make_node(node_cls)
    node._status = "Generating"
    node._status_detail = "seed 7, full, m.gguf…"
    node._gen_t0 = time.monotonic() - 65
    telem = node.get_telemetry()
    assert "elapsed" in telem["audio"]
    node._gen_t0 = None
    assert "elapsed" not in node.get_telemetry()["audio"]


# ----------------------------------------------------------------------
# score/lyrics fit warning (pure helpers + node hook)
# ----------------------------------------------------------------------
def test_lyric_section_names():
    assert backend.lyric_section_names(
        "[Verse]\nla la\n\n[Chorus]\nna na\n") == ["Verse", "Chorus"]
    assert backend.lyric_section_names("just words, no tags\n") == []


def test_abc_section_names():
    assert backend.abc_section_names(
        "X:1\n% intro\nCDEF\n% verse\nGABc\n") == ["intro", "verse"]


def test_fit_warning_matrix():
    warn = backend.score_lyrics_fit_warning
    assert warn(abc_sections=[], lyric_sections=["Verse"]) is None
    assert warn(abc_sections=["a", "b"], lyric_sections=["Verse", "Chorus"]) is None
    assert warn(abc_sections=["a", "b"], lyric_sections=["V", "C", "B"]) is None
    msg = warn(abc_sections=[f"s{i}" for i in range(8)],
               lyric_sections=["Verse", "Chorus"])
    assert msg and "8 sections" in msg and "2" in msg
    msg = warn(abc_sections=["a", "b", "c", "d"], lyric_sections=[])
    assert msg and "no [section] tags" in msg


def _write_lyrics_and_score(tmp_path, lyrics, abc=None):
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text(lyrics, encoding="utf-8")
    abc_file = None
    if abc is not None:
        abc_file = tmp_path / "score.abc"
        abc_file.write_text(abc, encoding="utf-8")
    return lyrics_file, abc_file


def test_spec_warnings_quiet_without_score(node_cls, tmp_path):
    lyrics_file, _ = _write_lyrics_and_score(tmp_path, "[Verse]\nla\n")
    node = make_node(node_cls)
    assert node._spec_warnings({"lyrics_file": str(lyrics_file),
                                "abc_file": None}) == []


def test_spec_warnings_matching_sections(node_cls, tmp_path):
    lyrics_file, abc_file = _write_lyrics_and_score(
        tmp_path, "[Verse]\nla\n\n[Chorus]\nna\n", "X:1\n% verse\nCDEF\n% chorus\nGABc\n")
    node = make_node(node_cls)
    assert node._spec_warnings({"lyrics_file": str(lyrics_file),
                                "abc_file": str(abc_file)}) == []


def test_spec_warnings_diverged_sections(node_cls, tmp_path):
    lyrics_file, abc_file = _write_lyrics_and_score(
        tmp_path, "[Verse]\nla\n\n[Chorus]\nna\n",
        "X:1\n" + "".join(f"% part{i}\nCDEF\n" for i in range(8)))
    node = make_node(node_cls)
    warnings = node._spec_warnings({"lyrics_file": str(lyrics_file),
                                    "abc_file": str(abc_file)})
    assert len(warnings) == 1 and "garble" in warnings[0]


# ----------------------------------------------------------------------
# busy flag (canvas running-glow)
# ----------------------------------------------------------------------
def test_busy_flag_follows_running_states(node_cls):
    node = make_node(node_cls)
    for status in ("Generating", "Downloading"):
        node._status = status
        assert node._busy_flag() is True
    for status in ("Idle", "Ready", "Cancelled", "Error"):
        node._status = status
        assert node._busy_flag() is False


def test_telemetry_carries_busy_flag(node_cls):
    node = make_node(node_cls)
    node._status, node._status_detail = "Generating", "seed 7…"
    assert node.get_telemetry()["busy"] is True
    node._status, node._gen_t0 = "Ready", None
    telem = node.get_telemetry()
    assert telem["busy"] is False and telem["status"] == "Ready"


def test_busy_glow_sets_and_clears(qapp):
    from ui_system import NodeItem
    item = NodeItem({"id": "n1", "type": "YuE2SongGenerator", "name": "yue2",
                     "params": {}, "monitor_queue": None,
                     "inputs": {}, "outputs": {}}, controller=None)
    assert item._job_busy is False
    item.propagate_telemetry({"status": "Generating", "busy": True})
    assert item._job_busy is True
    # Terminal states clear the glow even with no explicit flag.
    item.propagate_telemetry({"status": "Ready"})
    assert item._job_busy is False


# ----------------------------------------------------------------------
# request.json carries full reproduction texts
# ----------------------------------------------------------------------
def test_request_json_carries_repro_texts(node_cls, tmp_path):
    node = make_node(node_cls)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "song.wav").write_bytes(b"cli-bytes")
    keep = tmp_path / "keep"
    spec = _keep_spec(tmp_path, "wav")
    kept = node._keep_song(run_dir, keep, _keep_audio(), spec, {"rtf": 1.0},
                           lyrics_text="[Verse]\nla la\n",
                           abc_text="X:1\n% verse\nCDEF\n")
    assert kept == keep / "song.wav"
    import json
    req = json.loads((keep / "request.json").read_text())
    assert req["lyrics_text"] == "[Verse]\nla la\n"
    assert req["abc_text"] == "X:1\n% verse\nCDEF\n"
    assert req["lyrics_file"] == spec["lyrics_file"]  # paths kept alongside
    assert req["seed"] == 7 and req["cot"] == "full"
    assert req["vae_gguf"] == _live().VAE_GGUF
    assert req["runtime"]["audio_cpp"] == backend.AUDIOCPP_PIN_VERSION


def test_request_json_records_backend(node_cls, tmp_path):
    node = make_node(node_cls)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "song.wav").write_bytes(b"cli-bytes")
    keep = tmp_path / "keep"
    spec = _keep_spec(tmp_path, "wav")
    spec["backend"] = "cpu"
    spec["weight_type"] = "q4_0"
    node._keep_song(run_dir, keep, _keep_audio(), spec, {"rtf": 1.0})
    import json
    req = json.loads((keep / "request.json").read_text())
    assert req["backend"] == "cpu"
    assert req["weight_type"] == "q4_0"
