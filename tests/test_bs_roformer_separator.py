"""Tests for the BS-RoFormer separator node and backend.

All tests run without a GPU: the sidecar subprocess is stubbed, NRT
results are delivered by calling the completion handlers directly.
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import plugin_system
import audiocpp_backend as backend
from audiocpp_backend import (
    BS_ROFORMER_DEFAULT_GGUF,
    BS_ROFORMER_DEFAULT_NUM_OVERLAP,
    GenerationCancelled,
    _build_bs_roformer_argv,
    run_bs_roformer,
)
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("BSRoFormerSeparator")
    assert cls is not None, "BSRoFormerSeparator not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _live():
    """Live top-level plugin module object.

    load_plugins() imports node files WITHOUT the plugins. package prefix,
    so ``import plugins.bs_roformer_separator`` would bind an unpatched
    duplicate. Anything that patches module state (or relies on patched
    state) must go through this object; requires the node_cls fixture to
    have run first.
    """
    import sys
    return sys.modules["bs_roformer_separator"]


def write_wav(path, seconds=1.0, sr=SAMPLE_RATE, channels=2, freq=440.0):
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone * (i + 1) / channels for i in range(channels)], axis=1)
    sf.write(str(path), data, sr)
    return path


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Offline"
    assert node_cls.label == "BS-RoFormer Separator"


# ----------------------------------------------------------------------
# argv builder (pure, no process)
# ----------------------------------------------------------------------
def test_argv_builds_expected_command(tmp_path):
    argv = _build_bs_roformer_argv("cli", "m.gguf",
                                   audio_wav=str(tmp_path / "in.wav"),
                                   weight_type="native",
                                   num_overlap=BS_ROFORMER_DEFAULT_NUM_OVERLAP,
                                   out_dir=str(tmp_path / "stems"))
    assert argv[:4] == ["cli", "--task", "sep", "--model"]
    assert "--audio" in argv and "--out-dir" in argv and "--metrics" in argv
    assert "--threads" in argv
    assert "bs_roformer.weight_type=native" in argv
    # Package default overlap is omitted to preserve stock behavior.
    assert not any("num_overlap" in a for a in argv)
    i = argv.index("--backend")
    assert argv[i + 1] == "cuda"  # default preserved; override below
    argv = _build_bs_roformer_argv("cli", "m.gguf",
                                   audio_wav=str(tmp_path / "in.wav"),
                                   weight_type="f16",
                                   num_overlap=BS_ROFORMER_DEFAULT_NUM_OVERLAP,
                                   out_dir=str(tmp_path / "stems"),
                                   backend="cpu")
    assert argv[argv.index("--backend") + 1] == "cpu"
    assert "bs_roformer.weight_type=f16" in argv


def test_argv_nondefault_overlap_emitted(tmp_path):
    argv = _build_bs_roformer_argv("cli", "m.gguf",
                                   audio_wav=str(tmp_path / "in.wav"),
                                   weight_type="native",
                                   num_overlap=1,
                                   out_dir=str(tmp_path / "stems"))
    assert "bs_roformer.num_overlap=1" in argv


def test_argv_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        _build_bs_roformer_argv("cli", "m.gguf", audio_wav="a",
                                weight_type="q4_0",
                                num_overlap=BS_ROFORMER_DEFAULT_NUM_OVERLAP,
                                out_dir="o")
    with pytest.raises(ValueError):
        _build_bs_roformer_argv("cli", "m.gguf", audio_wav="a",
                                weight_type="native", num_overlap=0,
                                out_dir="o")
    with pytest.raises(ValueError):
        _build_bs_roformer_argv("cli", "m.gguf", audio_wav="a",
                                weight_type="native", num_overlap=True,
                                out_dir="o")


def test_fetch_specs_single_file(tmp_path):
    specs = backend.bs_roformer_fetch_specs(tmp_path / "models")
    assert len(specs) == 1
    assert specs[0].dest.name == "bs-roformer-ep368-q8_0.gguf"
    assert specs[0].size == 172532256
    assert specs[0].sha256 == "9a55a8cad369d00f6e0fb208bb0cd87e30e25430772b8491e20a4eace6423ad2"


# ----------------------------------------------------------------------
# run_bs_roformer with a stubbed subprocess
# ----------------------------------------------------------------------
class _StubProc:
    def __init__(self, argv, code=0, metrics_text="", skip_vocals=False,
                 skip_instrumental=False):
        self.argv = argv
        self._code = code
        self._text = metrics_text
        self._skip_vocals = skip_vocals
        self._skip_instrumental = skip_instrumental
        self.terminated = False

    def _out_dir(self):
        if "--out-dir" in self.argv:
            return Path(self.argv[self.argv.index("--out-dir") + 1])
        return None

    def communicate(self, timeout=None):
        if self._code == 0:
            out_dir = self._out_dir()
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                if not self._skip_vocals:
                    (out_dir / "vocals.wav").write_bytes(b"fake-vocals")
                if not self._skip_instrumental:
                    (out_dir / "instrumental.wav").write_bytes(b"fake-instr")
        return self._text, ""

    def terminate(self):
        self.terminated = True

    def poll(self):
        # A live process: the runner only skips terminate() after the real
        # communicate() has reaped an exit code.
        return None

    @property
    def returncode(self):
        return self._code


def _patch_popen(monkeypatch):
    seen = {}

    def fake_popen(argv, **kwargs):
        proc = _StubProc(argv, code=seen.get("code", 0),
                         metrics_text=seen.get("text", ""),
                         skip_vocals=seen.get("skip_vocals", False),
                         skip_instrumental=seen.get("skip_instrumental", False))
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(backend.subprocess, "Popen", fake_popen)
    return seen


def _model_files(tmp_path):
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(parents=True, exist_ok=True)
    gguf = model / BS_ROFORMER_DEFAULT_GGUF
    gguf.write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.5)
    return cli, gguf, audio


def test_run_sep_success(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out_dir = tmp_path / "run" / "stems"
    seen = _patch_popen(monkeypatch)
    seen["text"] = "metrics.wall_ms=100\nmetrics.rtf=0.2\n"
    import threading
    result = run_bs_roformer(str(cli), str(gguf), audio_wav=str(audio),
                             out_dir=str(out_dir),
                             cancel_event=threading.Event())
    assert Path(result["vocals"]).name == "vocals.wav"
    assert Path(result["instrumental"]).name == "instrumental.wav"
    assert result["stems_dir"] == str(out_dir)
    assert result["metrics"]["rtf"] == pytest.approx(0.2)
    assert "--out-dir" in seen["proc"].argv


def test_run_sep_instrumental_absent_is_none(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out_dir = tmp_path / "run" / "stems"
    seen = _patch_popen(monkeypatch)
    seen["skip_instrumental"] = True
    import threading
    result = run_bs_roformer(str(cli), str(gguf), audio_wav=str(audio),
                             out_dir=str(out_dir),
                             cancel_event=threading.Event())
    assert Path(result["vocals"]).exists()
    assert result["instrumental"] is None


def test_run_sep_missing_stems_output(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out_dir = tmp_path / "run" / "stems"
    seen = _patch_popen(monkeypatch)
    seen["skip_vocals"] = True
    seen["skip_instrumental"] = True
    import threading
    with pytest.raises(RuntimeError, match="vocals"):
        run_bs_roformer(str(cli), str(gguf), audio_wav=str(audio),
                        out_dir=str(out_dir),
                        cancel_event=threading.Event())


def test_run_sep_cancel_terminates(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out_dir = tmp_path / "run" / "stems"
    seen = _patch_popen(monkeypatch)
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        run_bs_roformer(str(cli), str(gguf), audio_wav=str(audio),
                        out_dir=str(out_dir), cancel_event=cancel)
    assert seen["proc"].terminated


def test_run_sep_missing_inputs(tmp_path):
    import threading
    cli, gguf, audio = _model_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_bs_roformer(str(tmp_path / "nope"), str(gguf),
                        audio_wav=str(audio), out_dir=str(tmp_path / "s"),
                        cancel_event=threading.Event())
    with pytest.raises(FileNotFoundError, match="weights not downloaded"):
        run_bs_roformer(str(cli), str(tmp_path / "gone.gguf"),
                        audio_wav=str(audio), out_dir=str(tmp_path / "s"),
                        cancel_event=threading.Event())
    with pytest.raises(FileNotFoundError, match="input audio"):
        run_bs_roformer(str(cli), str(gguf),
                        audio_wav=str(tmp_path / "gone.wav"),
                        out_dir=str(tmp_path / "s"),
                        cancel_event=threading.Event())


# ----------------------------------------------------------------------
# input audio loading (pure, no GPU)
# ----------------------------------------------------------------------
def test_load_input_wav_normalizes(node_cls, tmp_path):
    load_input_wav = _live().load_input_wav
    src = write_wav(tmp_path / "src48.wav", seconds=0.5, sr=48000, channels=1)
    out = load_input_wav(src, tmp_path / "out.wav")
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == 44100
    assert data.shape == (int(0.5 * 44100), 2)


def test_load_input_wav_missing(node_cls, tmp_path):
    load_input_wav = _live().load_input_wav
    with pytest.raises(RuntimeError, match="not found"):
        load_input_wav(tmp_path / "gone.wav", tmp_path / "out.wav")


# ----------------------------------------------------------------------
# node: completion paths, validation, save/load
# ----------------------------------------------------------------------
def _install_stems(node, vocals="v.wav", instrumental="i.wav"):
    node.on_nrt_complete("job", True, {"vocals_path": vocals,
                                       "instrumental_path": instrumental,
                                       "metrics": {}})


def test_ports_are_uri_and_pulse(node_cls):
    node = make_node(node_cls)
    assert node.outputs["vocals"].slot_type == "uri"
    assert node.outputs["instrumental"].slot_type == "uri"
    assert node.outputs["done"].slot_type == "audio"
    assert node.inputs["trigger_in"].slot_type == "audio"


def test_complete_installs_stems(node_cls):
    node = make_node(node_cls)
    _install_stems(node)
    assert node._vocals_path == "v.wav"
    assert node._status == "Ready"
    assert node.error_msg is None
    assert node.params["last_vocals"].value == "v.wav"
    assert node.params["last_instrumental"].value == "i.wav"
    telem = node.get_telemetry()
    assert telem["status"] == "Ready" and "v.wav" in telem["audio"]
    assert "i.wav" in telem["audio"]
    assert telem["busy"] is False


def test_telemetry_busy_while_separating(node_cls):
    node = make_node(node_cls)
    node._status, node._status_detail = "Separating", "song.wav (native)…"
    telem = node.get_telemetry()
    assert telem["busy"] is True and telem["status"] == "Separating"


def test_complete_publishes_uris_and_pulses_once(node_cls):
    node = make_node(node_cls)
    _install_stems(node, vocals="v.wav", instrumental="i.wav")
    assert node.outputs["vocals"].uri == "v.wav"
    assert node.outputs["instrumental"].uri == "i.wav"
    node.process()
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_complete_without_instrumental_leaves_it_empty(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", True, {"vocals_path": "v.wav",
                                       "instrumental_path": None,
                                       "metrics": {}})
    assert node.outputs["vocals"].uri == "v.wav"
    assert node.outputs["instrumental"].uri == ""
    assert node._status == "Ready"


def test_relink_sets_uris_without_pulse(node_cls, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"fake")
    instr = tmp_path / "instrumental.wav"
    instr.write_bytes(b"fake")
    node = make_node(node_cls)
    node.params["last_vocals"].set(str(vocals))
    node.params["last_vocals"].sync()
    node.params["last_instrumental"].set(str(instr))
    node.params["last_instrumental"].sync()
    node.load_state(node.to_dict())
    assert node.outputs["vocals"].uri == str(vocals)
    assert node.outputs["instrumental"].uri == str(instr)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_relink_vocals_only_when_instrumental_missing(node_cls, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"fake")
    node = make_node(node_cls)
    node.params["last_vocals"].set(str(vocals))
    node.params["last_vocals"].sync()
    node.params["last_instrumental"].set(str(tmp_path / "gone.wav"))
    node.params["last_instrumental"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    assert node2.outputs["vocals"].uri == str(vocals)
    assert node2.outputs["instrumental"].uri == ""
    assert node2._status == "Ready"


def test_complete_failure_and_cancel(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", False, RuntimeError("boom"))
    assert node._status == "Error"
    assert "boom" in node.error_msg
    node.on_nrt_complete("job", False, GenerationCancelled("stop"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_separate_rejects_missing_audio(node_cls, tmp_path, monkeypatch):
    mod = _live()

    class StubRuntime:
        def __init__(self, *a, **k):
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

        def check_ready(self):
            return True, ""

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / BS_ROFORMER_DEFAULT_GGUF).write_text("x")
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    from types import SimpleNamespace
    import queue
    engine = SimpleNamespace(output_queue=queue.Queue())
    node.graph = SimpleNamespace(engine=engine)
    node.params["separate"].set(True)
    node.params["separate"].sync()
    node.on_ui_param_change("separate")
    assert node._status == "Error"
    assert "audio_file" in node._status_detail
    msg = engine.output_queue.get_nowait()  # validation errors still surface
    assert msg["node_data"][node.id]["status"] == "Error"


def test_save_load_relinks(node_cls, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"fake")
    instr = tmp_path / "instrumental.wav"
    instr.write_bytes(b"fake")
    node = make_node(node_cls)
    node.params["last_vocals"].set(str(vocals))
    node.params["last_vocals"].sync()
    node.params["last_instrumental"].set(str(instr))
    node.params["last_instrumental"].sync()
    snapshot = node.to_dict()

    node2 = make_node(node_cls)
    node2.load_state(snapshot)  # file refs: relinked synchronously
    assert node2._vocals_path == str(vocals)
    assert node2._status == "Ready"


def test_load_missing_stems_goes_idle(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["last_vocals"].set(str(tmp_path / "gone.wav"))
    node.params["last_vocals"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    assert node2._status == "Idle"


def test_build_spec_carries_options(node_cls, tmp_path, monkeypatch):
    mod = _live()

    class StubRuntime:
        def __init__(self, *a, **k):
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

        def check_ready(self):
            return True, ""

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / BS_ROFORMER_DEFAULT_GGUF).write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.2)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["audio_file"].set(str(audio))
    node.params["audio_file"].sync()
    spec = node._build_spec()
    assert spec["gguf"] == BS_ROFORMER_DEFAULT_GGUF
    assert spec["weight_type"] == "native"
    assert spec["num_overlap"] == BS_ROFORMER_DEFAULT_NUM_OVERLAP
    node.params["weight_type"].set(1)
    node.params["weight_type"].sync()
    node.params["num_overlap"].set(1)
    node.params["num_overlap"].sync()
    spec = node._build_spec()
    assert spec["weight_type"] == "f32"
    assert spec["num_overlap"] == 1


def test_build_spec_rejects_missing_files(node_cls, tmp_path, monkeypatch):
    mod = _live()

    class StubRuntime:
        def __init__(self, *a, **k):
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

        def check_ready(self):
            return True, ""

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path / "models"))
    node.params["model_dir"].sync()
    with pytest.raises(ValueError, match="weights.*not found"):
        node._build_spec()


def test_run_nrt_full_pipeline(node_cls, tmp_path, monkeypatch):
    """Stub run_bs_roformer; exercise convert -> run -> keep on the worker."""
    mod = _live()
    src = write_wav(tmp_path / "song.wav", seconds=0.5)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(cli, gguf, **kw):
        out_dir = Path(kw["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocals.wav").write_bytes(b"v")
        (out_dir / "instrumental.wav").write_bytes(b"i")
        assert kw["weight_type"] == "native"
        assert kw["num_overlap"] == BS_ROFORMER_DEFAULT_NUM_OVERLAP
        return {"stems_dir": str(out_dir),
                "vocals": str(out_dir / "vocals.wav"),
                "instrumental": str(out_dir / "instrumental.wav"),
                "metrics": {"rtf": 0.2}}

    monkeypatch.setattr(mod, "run_bs_roformer", fake_run)
    node = make_node(node_cls)
    spec = {"cli": "c", "model_dir": str(model_dir),
            "gguf": BS_ROFORMER_DEFAULT_GGUF, "audio_file": str(src),
            "weight_type": "native",
            "num_overlap": BS_ROFORMER_DEFAULT_NUM_OVERLAP, "backend": "cpu"}
    import threading
    result = node._run_nrt(threading.Event(), spec)
    assert Path(result["vocals_path"]).name == "vocals.wav"
    assert Path(result["instrumental_path"]).name == "instrumental.wav"
    import json
    req = json.loads((Path(result["vocals_path"]).parent / "request.json")
                     .read_text())
    assert req["weight_type"] == "native"
    assert req["num_overlap"] == BS_ROFORMER_DEFAULT_NUM_OVERLAP
    node.on_nrt_complete("job", True, result)
    assert node.outputs["vocals"].uri == result["vocals_path"]
    assert node.outputs["instrumental"].uri == result["instrumental_path"]
    assert node._status == "Ready"


def test_build_fetch_payload_lists_missing(node_cls, tmp_path, monkeypatch):
    import download_util
    import audiocpp_backend
    from pathlib import Path as _Path

    class StubRuntime:
        def __init__(self, *a, **k):
            self.root = tmp_path / "tools" / "audiocpp"
            self.bin_dir = self.root / "bin"
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

    monkeypatch.setattr(audiocpp_backend, "AudioCppRuntime", StubRuntime)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(tmp_path / "models" / "BS-RoFormer-ep368-GGUF"))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    assert len(payload["runtime"]) == 2  # CLI absent: both zips pending
    assert len(payload["models"]) == 1
    # CLI present + size-correct model file -> nothing to fetch.
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin" / "audiocpp_cli.exe").write_text("x")
    for spec in backend.bs_roformer_fetch_specs(
            tmp_path / "models" / "BS-RoFormer-ep368-GGUF"):
        spec.dest.parent.mkdir(parents=True, exist_ok=True)
        with open(spec.dest, "wb") as f:
            f.truncate(spec.size)
    monkeypatch.setattr(
        download_util, "_verify",
        lambda p, s, h: _Path(p).is_file() and (s is None or _Path(p).stat().st_size == s))
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert payload["models"] == []


# ----------------------------------------------------------------------
# trigger_in (rising edge stages a separation, like the transcribers)
# ----------------------------------------------------------------------
class _StubEngine:
    """Minimal engine double: command capture."""

    def __init__(self):
        self.commands = []

    def push_command(self, cmd):
        self.commands.append(cmd)
        return len(self.commands)


def _attach_engine(node):
    from types import SimpleNamespace
    engine = _StubEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


def _feed_trigger(node, level):
    trig = torch.full((CHANNELS, BLOCK_SIZE), level, dtype=torch.float32)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def test_trigger_input_is_audio(node_cls):
    node = make_node(node_cls)
    assert node.inputs["trigger_in"].slot_type == "audio"


def test_trigger_edge_queues_single_separate(node_cls):
    node = make_node(node_cls)
    engine = _attach_engine(node)
    _feed_trigger(node, 0.0)
    assert engine.commands == []
    _feed_trigger(node, 1.0)  # rising edge
    assert engine.commands == [("param", node.id, "separate", True)]
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


def test_start_resets_trigger_edge(node_cls):
    """A sustained high after start() re-fires (stale level forgotten)."""
    node = make_node(node_cls)
    engine = _attach_engine(node)
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 1
    node.start()
    _feed_trigger(node, 1.0)
    assert len(engine.commands) == 2


# ----------------------------------------------------------------------
# audio_uri (wired mixture wins over the audio_file param)
# ----------------------------------------------------------------------
def _wire_audio_uri(node, uri):
    """Connect the node's audio_uri input to a scratch URI output."""
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs["audio_uri"].connect(helper)
    return helper


def test_audio_uri_overrides_param(node_cls, tmp_path):
    wired = tmp_path / "wired.wav"
    wired.write_bytes(b"RIFF" + bytes(100))
    param_audio = tmp_path / "param.wav"
    param_audio.write_bytes(b"RIFF" + bytes(100))
    node = make_node(node_cls)
    node.params["audio_file"].set(str(param_audio))
    node.params["audio_file"].sync()
    _wire_audio_uri(node, str(wired))
    assert node._resolve_audio() == str(wired)


def test_audio_uri_empty_means_not_ready(node_cls, tmp_path):
    param_audio = tmp_path / "param.wav"
    param_audio.write_bytes(b"RIFF" + bytes(100))
    node = make_node(node_cls)
    node.params["audio_file"].set(str(param_audio))
    node.params["audio_file"].sync()
    _wire_audio_uri(node, "")
    with pytest.raises(ValueError, match="publish a file"):
        node._resolve_audio()


def test_audio_uri_missing_file_rejected(node_cls):
    node = make_node(node_cls)
    _wire_audio_uri(node, "/nonexistent/gone.wav")
    with pytest.raises(ValueError, match="not found"):
        node._resolve_audio()


def test_audio_param_used_when_unwired(node_cls, tmp_path):
    param_audio = tmp_path / "param.wav"
    param_audio.write_bytes(b"RIFF" + bytes(100))
    node = make_node(node_cls)
    node.params["audio_file"].set(str(param_audio))
    node.params["audio_file"].sync()
    assert node._resolve_audio() == str(param_audio)
