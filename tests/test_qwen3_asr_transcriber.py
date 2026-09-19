"""Tests for the Qwen3 ASR transcriber node and its audio.cpp backend.

All tests run without a GPU: the sidecar subprocess is stubbed and NRT
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
    GenerationCancelled,
    _build_qwen3_asr_argv,
    run_qwen3_asr,
)
from base import BLOCK_SIZE, CHANNELS


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Qwen3ASRTranscriber")
    assert cls is not None, "Qwen3ASRTranscriber not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _live():
    """Live top-level plugin module object (see test_yue2_song_generator)."""
    import sys
    return sys.modules["qwen3_asr_transcriber"]


def write_wav(path, seconds=1.0, sr=48000, channels=2):
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
    assert node_cls.label == "Qwen3 ASR Transcriber"
    assert "audio.cpp" in node_cls.description


# ----------------------------------------------------------------------
# argv builder (pure, no process)
# ----------------------------------------------------------------------
def _spec(**kw):
    spec = dict(audio_wav="a.wav", language=None, max_tokens=4096,
                out_txt="t.txt")
    spec.update(kw)
    return spec


def test_argv_builds_expected_command(tmp_path):
    argv = _build_qwen3_asr_argv("cli", "m.gguf", backend="cuda", **_spec())
    assert argv[:6] == ["cli", "--task", "asr", "--family", "qwen3_asr",
                        "--model"]
    assert "--audio" in argv and "--text-out" in argv
    assert "--text" in argv and "" in argv
    assert "--max-tokens" in argv and "4096" in argv
    assert "--metrics" in argv
    assert "--language" not in argv  # None = auto-detect


def test_argv_language_emitted(tmp_path):
    argv = _build_qwen3_asr_argv("cli", "m.gguf", backend="cuda",
                                 **_spec(language="en"))
    i = argv.index("--language")
    assert argv[i + 1] == "en"


def test_argv_backend_override(tmp_path):
    argv = _build_qwen3_asr_argv("cli", "m.gguf", backend="cpu", **_spec())
    i = argv.index("--backend")
    assert argv[i + 1] == "cpu"


def test_argv_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        _build_qwen3_asr_argv("cli", "m.gguf", backend="cuda",
                              **_spec(max_tokens=0))
    with pytest.raises(ValueError):
        _build_qwen3_asr_argv("cli", "m.gguf", backend="cuda",
                              **_spec(max_tokens=True))
    with pytest.raises(ValueError):
        _build_qwen3_asr_argv("cli", "m.gguf", backend="cuda",
                              **_spec(language="  "))


def test_fetch_specs_single_file(tmp_path):
    specs = backend.qwen3_asr_fetch_specs(tmp_path / "models")
    assert len(specs) == 1
    assert specs[0].dest.name == "qwen3-asr-0.6b-q8_0.gguf"
    assert specs[0].size == 1151272416
    assert specs[0].sha256 is not None


# ----------------------------------------------------------------------
# run_qwen3_asr with a stubbed subprocess
# ----------------------------------------------------------------------
class _StubProc:
    def __init__(self, argv, out_txt, code=0, metrics_text=""):
        self.argv = argv
        self._out = out_txt
        self._code = code
        self._text = metrics_text
        self.terminated = False

    def communicate(self, timeout=None):
        if self._code == 0 and self._out is not None:
            Path(self._out).parent.mkdir(parents=True, exist_ok=True)
            Path(self._out).write_text("hello world\n", encoding="utf-8")
        return self._text, ""

    def terminate(self):
        self.terminated = True

    def poll(self):
        return None

    @property
    def returncode(self):
        return self._code


def _patch_popen(monkeypatch):
    seen = {}

    def fake_popen(argv, **kwargs):
        proc = _StubProc(argv, seen.get("out"), code=seen.get("code", 0),
                         metrics_text=seen.get("text", ""))
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(backend.subprocess, "Popen", fake_popen)
    return seen


def _model_files(tmp_path):
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(parents=True, exist_ok=True)
    gguf = model / "qwen3-asr-0.6b-q8_0.gguf"
    gguf.write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.5)
    return cli, gguf, audio


def test_run_asr_success(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out = tmp_path / "run" / "transcript.txt"
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    seen["text"] = "metrics.wall_ms=1000\nmetrics.rtf=0.1\n"
    import threading
    result = run_qwen3_asr(str(cli), str(gguf), audio_wav=str(audio),
                           out_txt=str(out), cancel_event=threading.Event())
    assert Path(result["txt"]).exists()
    assert result["metrics"]["rtf"] == pytest.approx(0.1)
    assert "--text-out" in seen["proc"].argv


def test_run_asr_forwards_language_and_tokens(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out = tmp_path / "run" / "transcript.txt"
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    run_qwen3_asr(str(cli), str(gguf), audio_wav=str(audio), language="en",
                  max_tokens=512, out_txt=str(out),
                  cancel_event=threading.Event())
    argv = seen["proc"].argv
    assert "en" in argv and "512" in argv


def test_run_asr_cancel_terminates(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out = tmp_path / "t.txt"
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        run_qwen3_asr(str(cli), str(gguf), audio_wav=str(audio),
                      out_txt=str(out), cancel_event=cancel)
    assert seen["proc"].terminated


def test_run_asr_missing_inputs(tmp_path):
    import threading
    cli, gguf, audio = _model_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_qwen3_asr(str(tmp_path / "nope"), str(gguf), audio_wav=str(audio),
                      out_txt=str(tmp_path / "o.txt"),
                      cancel_event=threading.Event())
    with pytest.raises(FileNotFoundError, match="weights not downloaded"):
        run_qwen3_asr(str(cli), str(tmp_path / "gone.gguf"),
                      audio_wav=str(audio), out_txt=str(tmp_path / "o.txt"),
                      cancel_event=threading.Event())
    with pytest.raises(FileNotFoundError, match="input audio"):
        run_qwen3_asr(str(cli), str(gguf),
                      audio_wav=str(tmp_path / "gone.wav"),
                      out_txt=str(tmp_path / "o.txt"),
                      cancel_event=threading.Event())


# ----------------------------------------------------------------------
# node: lyrics URI + done pulse
# ----------------------------------------------------------------------
def _complete_lyrics(node, txt="l.txt", text="hello world"):
    node.on_nrt_complete("job", True, {"text": text, "txt_path": txt,
                                       "metrics": {"rtf": 0.1}})
    return text


def test_ports_are_uri_and_pulse(node_cls):
    node = make_node(node_cls)
    assert node.outputs["lyrics"].slot_type == "uri"
    assert node.outputs["done"].slot_type == "audio"
    assert node.inputs["trigger_in"].slot_type == "audio"


# ----------------------------------------------------------------------
# trigger_in (rising edge stages a transcription, like YuE2's Generate)
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


def test_trigger_edge_queues_single_transcribe(node_cls):
    node = make_node(node_cls)
    engine = _attach_engine(node)
    _feed_trigger(node, 0.0)
    assert engine.commands == []
    _feed_trigger(node, 1.0)  # rising edge
    assert engine.commands == [("param", node.id, "transcribe", True)]
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


def test_complete_publishes_uri_and_pulses_once(node_cls):
    node = make_node(node_cls)
    _complete_lyrics(node, txt="l.txt")
    assert node.outputs["lyrics"].uri == "l.txt"
    assert node._status == "Ready"
    assert node.error_msg is None
    assert node.params["last_txt"].value == "l.txt"
    node.process()  # first block after completion: full 1.0 pulse
    assert torch.all(node.done.buffer == 1.0)
    node.process()  # afterwards: silence, never retriggers
    assert torch.all(node.done.buffer == 0.0)


def test_pulse_cleared_on_start(node_cls):
    node = make_node(node_cls)
    _complete_lyrics(node)
    node.start()
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_complete_empty_text_is_error(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", True, {"text": "  \n", "txt_path": "l.txt",
                                       "metrics": {}})
    assert node._status == "Error"


def test_complete_failure_and_cancel(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", False, RuntimeError("boom"))
    assert node._status == "Error"
    assert "boom" in node.error_msg
    node.on_nrt_complete("job", False, GenerationCancelled("x"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_relink_sets_uri_without_pulse(node_cls):
    node = make_node(node_cls)
    node.params["last_txt"].set("r.txt")
    node.params["last_txt"].sync()
    snapshot = node.to_dict()
    node2 = make_node(node_cls)
    # Missing file: idle, no uri.
    node2.load_state(snapshot)
    assert node2.outputs["lyrics"].uri == ""
    assert node2._status == "Idle"


def test_relink_existing_file(node_cls, tmp_path):
    txt = tmp_path / "lyrics.txt"
    txt.write_text("hello world\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["last_txt"].set(str(txt))
    node.params["last_txt"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    assert node2.outputs["lyrics"].uri == str(txt)
    assert node2._status == "Ready"
    node2.process()  # relink must not pulse
    assert torch.all(node2.done.buffer == 0.0)


def test_asr_to_generator_wires_up(node_cls):
    """Qwen3ASRTranscriber.lyrics -> YuE2SongGenerator.lyrics_uri connects."""
    yue_cls = plugin_system.NODE_REGISTRY.get("YuE2SongGenerator")
    assert yue_cls is not None
    asr, gen = make_node(node_cls), yue_cls()
    gen.inputs["lyrics_uri"].connect(asr.outputs["lyrics"])
    assert gen.inputs["lyrics_uri"].get_uri() == ""


def test_build_spec_rejects_missing_backend(node_cls, tmp_path, monkeypatch):
    node = make_node(node_cls)
    monkeypatch.setattr(node, "_resolve_model_dir",
                        lambda: tmp_path / "Qwen3-ASR-GGUF")
    with pytest.raises(ValueError, match="fetch_"):
        node._build_spec()


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
    (model_dir / "qwen3-asr-0.6b-q8_0.gguf").write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.2)
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["audio_file"].set(str(audio))
    node.params["audio_file"].sync()
    spec = node._build_spec()
    assert spec["gguf"] == "qwen3-asr-0.6b-q8_0.gguf"
    assert spec["language"] is None
    assert spec["max_tokens"] == 4096
    node.params["language"].set("en")
    node.params["language"].sync()
    node.params["max_tokens"].set(512)
    node.params["max_tokens"].sync()
    spec = node._build_spec()
    assert spec["language"] == "en"
    assert spec["max_tokens"] == 512


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


def test_load_input_wav_converts_to_16k_mono(node_cls, tmp_path):
    src = write_wav(tmp_path / "s.wav", seconds=0.3, sr=48000, channels=2)
    out = _live().load_input_wav(src, tmp_path / "in.wav")
    data, sr = sf.read(str(out))
    assert sr == 16000
    assert data.ndim == 1  # mono


def test_telemetry_carries_preview(node_cls):
    node = make_node(node_cls)
    _complete_lyrics(node, text="hello brave new world")
    telem = node.get_telemetry()
    assert telem["status"] == "Ready"
    assert "hello brave new world" in telem["audio"]


def test_busy_flag_follows_running_states(node_cls):
    node = make_node(node_cls)
    for status in ("Transcribing", "Downloading"):
        node._status = status
        assert node._busy_flag() is True
    for status in ("Idle", "Ready", "Cancelled", "Error"):
        node._status = status
        assert node._busy_flag() is False


def test_run_nrt_full_pipeline(node_cls, tmp_path, monkeypatch):
    """Stub run_qwen3_asr; exercise convert -> run -> keep on the worker."""
    mod = _live()
    src = write_wav(tmp_path / "song.wav", seconds=0.5)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(cli, gguf, **kw):
        out = Path(kw["out_txt"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("la la la\n", encoding="utf-8")
        assert kw["language"] == "en"
        return {"txt": str(out), "metrics": {"rtf": 0.2}}

    monkeypatch.setattr(mod, "run_qwen3_asr", fake_run)
    node = make_node(node_cls)
    spec = {"cli": "c", "model_dir": str(model_dir),
            "gguf": "qwen3-asr-0.6b-q8_0.gguf", "audio_file": str(src),
            "language": "en", "max_tokens": 4096, "backend": "cpu"}
    import threading
    result = node._run_nrt(threading.Event(), spec)
    assert result["text"] == "la la la"
    assert Path(result["txt_path"]).name == "lyrics.txt"
    import json
    req = json.loads((Path(result["txt_path"]).parent / "request.json")
                     .read_text())
    assert req["gguf"] == "qwen3-asr-0.6b-q8_0.gguf"
    assert req["language"] == "en"
    node.on_nrt_complete("job", True, result)
    assert node.outputs["lyrics"].uri == result["txt_path"]
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
    node.params["model_dir"].set(str(tmp_path / "models" / "Qwen3-ASR-GGUF"))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    assert len(payload["runtime"]) == 2  # CLI absent: both zips pending
    assert len(payload["models"]) == 1
    # CLI present + size-correct model file -> nothing to fetch.
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin" / "audiocpp_cli.exe").write_text("x")
    for spec in backend.qwen3_asr_fetch_specs(
            tmp_path / "models" / "Qwen3-ASR-GGUF"):
        spec.dest.parent.mkdir(parents=True, exist_ok=True)
        with open(spec.dest, "wb") as f:
            f.truncate(spec.size)
    monkeypatch.setattr(
        download_util, "_verify",
        lambda p, s, h: _Path(p).is_file() and (s is None or _Path(p).stat().st_size == s))
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert payload["models"] == []
