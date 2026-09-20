"""Tests for the Qwen3 forced aligner node and its audio.cpp backend.

All tests run without a GPU: the sidecar subprocess is stubbed and NRT
results are delivered by calling the completion handlers directly.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import plugin_system
import audiocpp_backend as backend
from audiocpp_backend import (
    GenerationCancelled,
    _build_qwen3_align_argv,
    run_qwen3_align,
)
from base import BLOCK_SIZE, CHANNELS

WORDS_JSON = json.dumps([
    {"word": "hello", "start": 0.0, "end": 0.4},
    {"word": "world", "start": 0.4, "end": 0.8},
])


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Qwen3ForcedAligner")
    assert cls is not None, "Qwen3ForcedAligner not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _live():
    """Live top-level plugin module object (see test_yue2_song_generator)."""
    import sys
    return sys.modules["qwen3_forced_aligner"]


def write_wav(path, seconds=1.0, sr=48000, channels=2):
    n = int(seconds * sr)
    t = np.linspace(0, 1, n, dtype=np.float32)
    data = np.stack([t * (i + 1) / channels for i in range(channels)], axis=1)
    sf.write(str(path), data, sr)
    return path


def write_transcript(path, text="hello world"):
    Path(path).write_text(text + "\n", encoding="utf-8")
    return str(path)


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Offline"
    assert node_cls.label == "Qwen3 Forced Aligner"
    assert "audio.cpp" in node_cls.description


# ----------------------------------------------------------------------
# argv builder (pure, no process)
# ----------------------------------------------------------------------
def _spec(**kw):
    spec = dict(audio_wav="a.wav", text="hello world", language="en",
                out_words="w.json")
    spec.update(kw)
    return spec


def test_argv_builds_expected_command(tmp_path):
    argv = _build_qwen3_align_argv("cli", "m.gguf", backend="cuda", **_spec())
    assert argv[:6] == ["cli", "--task", "align", "--family",
                        "qwen3_forced_aligner", "--model"]
    assert "--audio" in argv and "--words-out" in argv
    assert "--text" in argv and "hello world" in argv
    i = argv.index("--language")
    assert argv[i + 1] == "en"
    assert "--metrics" in argv
    assert "--threads" in argv and "8" in argv
    assert "--request-option" not in argv  # clamp defaults off


def test_argv_clamp_option_emitted():
    argv = _build_qwen3_align_argv("cli", "m.gguf", backend="cuda",
                                   clamp_timestamps=True, **_spec())
    i = argv.index("--request-option")
    assert argv[i + 1] == "clamp_timestamps_to_audio"


def test_argv_backend_override():
    argv = _build_qwen3_align_argv("cli", "m.gguf", backend="cpu", **_spec())
    i = argv.index("--backend")
    assert argv[i + 1] == "cpu"


def test_argv_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _build_qwen3_align_argv("cli", "m.gguf", backend="cuda",
                                **_spec(text="   "))
    with pytest.raises(ValueError):
        _build_qwen3_align_argv("cli", "m.gguf", backend="cuda",
                                **_spec(language="  "))
    with pytest.raises(ValueError):
        _build_qwen3_align_argv("cli", "m.gguf", backend="cuda",
                                **_spec(language=None))
    with pytest.raises(ValueError):
        _build_qwen3_align_argv("cli", "m.gguf", backend="cuda",
                                **_spec(out_words="  "))


def test_argv_missing_words_out_is_signature_error():
    import inspect
    assert "out_words" in inspect.signature(_build_qwen3_align_argv).parameters
    with pytest.raises(TypeError):
        _build_qwen3_align_argv("cli", "m.gguf", backend="cuda",
                                audio_wav="a.wav", text="hi", language="en")


def test_fetch_specs_single_file(tmp_path):
    specs = backend.qwen3_align_fetch_specs(tmp_path / "models")
    assert len(specs) == 1
    assert specs[0].dest.name == "qwen3-forced-aligner-0.6b-q8_0.gguf"
    assert specs[0].size == 1129966496
    assert (specs[0].sha256
            == "75209490b11cec2b0db749ca5f4ff92266f58efd30f7fd04d9eb2a3ac9cc929f")


# ----------------------------------------------------------------------
# run_qwen3_align with a stubbed subprocess
# ----------------------------------------------------------------------
class _StubProc:
    def __init__(self, argv, out_words, code=0, metrics_text=""):
        self.argv = argv
        self._out = out_words
        self._code = code
        self._text = metrics_text
        self.terminated = False
        self.wait_calls = 0
        self.kill_calls = 0

    def communicate(self, timeout=None):
        if self._code == 0 and self._out is not None:
            Path(self._out).parent.mkdir(parents=True, exist_ok=True)
            Path(self._out).write_text(WORDS_JSON, encoding="utf-8")
        return self._text, ""

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self._code

    def kill(self):
        self.kill_calls += 1

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
    gguf = model / "qwen3-forced-aligner-0.6b-q8_0.gguf"
    gguf.write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.5)
    return cli, gguf, audio


def test_run_align_success(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out = tmp_path / "run" / "words.json"
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    seen["text"] = "metrics.wall_ms=1000\nmetrics.rtf=0.1\n"
    import threading
    result = run_qwen3_align(str(cli), str(gguf), audio_wav=str(audio),
                             text="hello world", language="en",
                             out_words=str(out),
                             cancel_event=threading.Event())
    assert Path(result["words"]).exists()
    assert result["metrics"]["rtf"] == pytest.approx(0.1)
    assert "--words-out" in seen["proc"].argv
    assert "hello world" in seen["proc"].argv


def test_run_align_forwards_clamp(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out = tmp_path / "run" / "words.json"
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    run_qwen3_align(str(cli), str(gguf), audio_wav=str(audio),
                    text="hello world", language="en", out_words=str(out),
                    clamp_timestamps=True, cancel_event=threading.Event())
    argv = seen["proc"].argv
    assert "clamp_timestamps_to_audio" in argv


def test_run_align_cancel_terminates(tmp_path, monkeypatch):
    cli, gguf, audio = _model_files(tmp_path)
    out = tmp_path / "w.json"
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        run_qwen3_align(str(cli), str(gguf), audio_wav=str(audio),
                        text="hello world", language="en",
                        out_words=str(out), cancel_event=cancel)
    assert seen["proc"].terminated


def test_run_align_missing_inputs(tmp_path):
    import threading
    cli, gguf, audio = _model_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_qwen3_align(str(tmp_path / "nope"), str(gguf),
                        audio_wav=str(audio), text="hi", language="en",
                        out_words=str(tmp_path / "o.json"),
                        cancel_event=threading.Event())
    with pytest.raises(FileNotFoundError, match="weights not downloaded"):
        run_qwen3_align(str(cli), str(tmp_path / "gone.gguf"),
                        audio_wav=str(audio), text="hi", language="en",
                        out_words=str(tmp_path / "o.json"),
                        cancel_event=threading.Event())
    with pytest.raises(FileNotFoundError, match="input audio"):
        run_qwen3_align(str(cli), str(gguf),
                        audio_wav=str(tmp_path / "gone.wav"),
                        text="hi", language="en",
                        out_words=str(tmp_path / "o.json"),
                        cancel_event=threading.Event())


# ----------------------------------------------------------------------
# node: words URI + done pulse
# ----------------------------------------------------------------------
def _complete_words(node, words="w.json", text=WORDS_JSON):
    node.on_nrt_complete("job", True, {"words_text": text,
                                       "words_path": words,
                                       "word_count": 2,
                                       "metrics": {"rtf": 0.1}})
    return text


def test_ports_are_uri_and_pulse(node_cls):
    node = make_node(node_cls)
    assert node.outputs["words"].slot_type == "uri"
    assert node.outputs["done"].slot_type == "audio"
    assert node.inputs["trigger_in"].slot_type == "audio"
    assert node.inputs["lyrics_uri"].slot_type == "uri"


# ----------------------------------------------------------------------
# trigger_in (rising edge stages an alignment, like Qwen3 ASR)
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


def test_trigger_edge_queues_single_align(node_cls):
    node = make_node(node_cls)
    engine = _attach_engine(node)
    _feed_trigger(node, 0.0)
    assert engine.commands == []
    _feed_trigger(node, 1.0)  # rising edge
    assert engine.commands == [("param", node.id, "align", True)]
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
    _complete_words(node, words="w.json")
    assert node.outputs["words"].uri == "w.json"
    assert node._status == "Ready"
    assert node.error_msg is None
    assert node.params["last_words"].value == "w.json"
    node.process()  # first block after completion: full 1.0 pulse
    assert torch.all(node.done.buffer == 1.0)
    node.process()  # afterwards: silence, never retriggers
    assert torch.all(node.done.buffer == 0.0)


def test_pulse_cleared_on_start(node_cls):
    node = make_node(node_cls)
    _complete_words(node)
    node.start()
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_complete_empty_words_is_error(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", True, {"words_text": "  \n",
                                       "words_path": "w.json",
                                       "word_count": 0, "metrics": {}})
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
    node.params["last_words"].set("r.json")
    node.params["last_words"].sync()
    snapshot = node.to_dict()
    node2 = make_node(node_cls)
    # Missing file: idle, no uri.
    node2.load_state(snapshot)
    assert node2.outputs["words"].uri == ""
    assert node2._status == "Idle"


def test_relink_existing_file(node_cls, tmp_path):
    words = tmp_path / "words.json"
    words.write_text(WORDS_JSON, encoding="utf-8")
    node = make_node(node_cls)
    node.params["last_words"].set(str(words))
    node.params["last_words"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    assert node2.outputs["words"].uri == str(words)
    assert node2._status == "Ready"
    node2.process()  # relink must not pulse
    assert torch.all(node2.done.buffer == 0.0)


def test_asr_to_aligner_wires_up(node_cls):
    """Qwen3ASRTranscriber.lyrics -> Qwen3ForcedAligner.lyrics_uri connects."""
    asr_cls = plugin_system.NODE_REGISTRY.get("Qwen3ASRTranscriber")
    assert asr_cls is not None
    asr, aligner = asr_cls(), make_node(node_cls)
    aligner.inputs["lyrics_uri"].connect(asr.outputs["lyrics"])
    assert aligner.inputs["lyrics_uri"].get_uri() == ""


def _wire_lyrics_uri(node, uri):
    """Connect the node's lyrics_uri input to a scratch URI output."""
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs["lyrics_uri"].connect(helper)
    return helper


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


def test_lyrics_uri_overrides_param(node_cls, tmp_path):
    wired = tmp_path / "wired.txt"
    wired.write_text("hello world\n", encoding="utf-8")
    param_txt = tmp_path / "param.txt"
    param_txt.write_text("other words\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["transcript_file"].set(str(param_txt))
    node.params["transcript_file"].sync()
    _wire_lyrics_uri(node, str(wired))
    assert node._resolve_transcript() == str(wired)


def test_lyrics_uri_empty_means_not_ready(node_cls, tmp_path):
    param_txt = tmp_path / "param.txt"
    param_txt.write_text("hello world\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["transcript_file"].set(str(param_txt))
    node.params["transcript_file"].sync()
    _wire_lyrics_uri(node, "")
    with pytest.raises(ValueError, match="transcribe first"):
        node._resolve_transcript()


def test_lyrics_uri_missing_file_rejected(node_cls):
    node = make_node(node_cls)
    _wire_lyrics_uri(node, "/nonexistent/gone.txt")
    with pytest.raises(ValueError, match="not found"):
        node._resolve_transcript()


def test_transcript_param_used_when_unwired(node_cls, tmp_path):
    param_txt = tmp_path / "param.txt"
    param_txt.write_text("hello world\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["transcript_file"].set(str(param_txt))
    node.params["transcript_file"].sync()
    assert node._resolve_transcript() == str(param_txt)


def test_build_spec_rejects_missing_backend(node_cls, tmp_path, monkeypatch):
    node = make_node(node_cls)
    monkeypatch.setattr(node, "_resolve_model_dir",
                        lambda: tmp_path / "Qwen3-ForcedAligner-GGUF")
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
    (model_dir / "qwen3-forced-aligner-0.6b-q8_0.gguf").write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.2)
    transcript = write_transcript(tmp_path / "t.txt")
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["audio_file"].set(str(audio))
    node.params["audio_file"].sync()
    node.params["transcript_file"].set(str(transcript))
    node.params["transcript_file"].sync()
    spec = node._build_spec()
    assert spec["gguf"] == "qwen3-forced-aligner-0.6b-q8_0.gguf"
    assert spec["language"] == "en"
    assert spec["transcript_file"] == str(transcript)
    node.params["language"].set("de")
    node.params["language"].sync()
    spec = node._build_spec()
    assert spec["language"] == "de"


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


def test_build_spec_rejects_bad_language(node_cls, tmp_path, monkeypatch):
    mod = _live()

    class StubRuntime:
        def __init__(self, *a, **k):
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

        def check_ready(self):
            return True, ""

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "qwen3-forced-aligner-0.6b-q8_0.gguf").write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.2)
    transcript = write_transcript(tmp_path / "t.txt")
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["audio_file"].set(str(audio))
    node.params["audio_file"].sync()
    node.params["transcript_file"].set(str(transcript))
    node.params["transcript_file"].sync()
    node.params["language"].set("   ")
    node.params["language"].sync()
    with pytest.raises(ValueError, match="language"):
        node._build_spec()


def test_load_input_wav_converts_to_16k_mono(node_cls, tmp_path):
    src = write_wav(tmp_path / "s.wav", seconds=0.3, sr=48000, channels=2)
    out = _live().load_input_wav(src, tmp_path / "in.wav")
    data, sr = sf.read(str(out))
    assert sr == 16000
    assert data.ndim == 1  # mono


def test_telemetry_carries_word_count(node_cls):
    node = make_node(node_cls)
    _complete_words(node)
    telem = node.get_telemetry()
    assert telem["status"] == "Ready"
    assert "2 words" in telem["audio"]


def test_busy_flag_follows_running_states(node_cls):
    node = make_node(node_cls)
    for status in ("Aligning", "Downloading"):
        node._status = status
        assert node._busy_flag() is True
    for status in ("Idle", "Ready", "Cancelled", "Error"):
        node._status = status
        assert node._busy_flag() is False


def test_run_nrt_full_pipeline(node_cls, tmp_path, monkeypatch):
    """Stub run_qwen3_align; exercise convert -> run -> keep on the worker."""
    mod = _live()
    src = write_wav(tmp_path / "song.wav", seconds=0.5)
    transcript = write_transcript(tmp_path / "t.txt", "hello world")
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(cli, gguf, **kw):
        out = Path(kw["out_words"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(WORDS_JSON, encoding="utf-8")
        assert kw["language"] == "en"
        assert kw["text"] == "hello world\n"
        return {"words": str(out), "metrics": {"rtf": 0.2}}

    monkeypatch.setattr(mod, "run_qwen3_align", fake_run)
    node = make_node(node_cls)
    spec = {"cli": "c", "model_dir": str(model_dir),
            "gguf": "qwen3-forced-aligner-0.6b-q8_0.gguf",
            "audio_file": str(src), "transcript_file": str(transcript),
            "language": "en", "backend": "cpu"}
    import threading
    result = node._run_nrt(threading.Event(), spec)
    assert result["word_count"] == 2
    assert Path(result["words_path"]).name == "words.json"
    req = json.loads((Path(result["words_path"]).parent / "request.json")
                     .read_text())
    assert req["gguf"] == "qwen3-forced-aligner-0.6b-q8_0.gguf"
    assert req["language"] == "en"
    node.on_nrt_complete("job", True, result)
    assert node.outputs["words"].uri == result["words_path"]
    assert node._status == "Ready"


def test_run_nrt_empty_words_is_error(node_cls, tmp_path, monkeypatch):
    mod = _live()
    src = write_wav(tmp_path / "song.wav", seconds=0.5)
    transcript = write_transcript(tmp_path / "t.txt", "hello world")
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(cli, gguf, **kw):
        out = Path(kw["out_words"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("  \n", encoding="utf-8")
        return {"words": str(out), "metrics": {}}

    monkeypatch.setattr(mod, "run_qwen3_align", fake_run)
    node = make_node(node_cls)
    spec = {"cli": "c", "model_dir": str(model_dir),
            "gguf": "qwen3-forced-aligner-0.6b-q8_0.gguf",
            "audio_file": str(src), "transcript_file": str(transcript),
            "language": "en", "backend": "cpu"}
    import threading
    with pytest.raises(RuntimeError, match="empty words"):
        node._run_nrt(threading.Event(), spec)


# ----------------------------------------------------------------------
# long-audio guard (single-shot alignment covers ~30 s clips)
# ----------------------------------------------------------------------
def test_audio_seconds_reads_header(node_cls, tmp_path):
    mod = _live()
    src = write_wav(tmp_path / "s.wav", seconds=2.0)
    assert mod.audio_seconds(src) == pytest.approx(2.0, abs=0.05)
    assert mod.audio_seconds(tmp_path / "gone.wav") is None


def test_long_audio_message_points_at_asr_path(node_cls):
    mod = _live()
    msg = mod.long_audio_message(240.0)
    assert "240" in msg and "word_timestamps" in msg


def _spec_node(node_cls, tmp_path, monkeypatch, seconds=0.2):
    """Node with stub runtime + model/audio/transcript present."""
    mod = _live()

    class StubRuntime:
        def __init__(self, *a, **k):
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

        def check_ready(self):
            return True, ""

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "qwen3-forced-aligner-0.6b-q8_0.gguf").write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=seconds)
    transcript = write_transcript(tmp_path / "t.txt")
    node = make_node(node_cls)
    node.params["model_dir"].set(str(model_dir))
    node.params["model_dir"].sync()
    node.params["audio_file"].set(str(audio))
    node.params["audio_file"].sync()
    node.params["transcript_file"].set(str(transcript))
    node.params["transcript_file"].sync()
    return node


def test_build_spec_rejects_long_audio(node_cls, tmp_path, monkeypatch):
    node = _spec_node(node_cls, tmp_path, monkeypatch, seconds=31.0)
    with pytest.raises(ValueError, match="word_timestamps"):
        node._build_spec()


def test_build_spec_allows_short_audio(node_cls, tmp_path, monkeypatch):
    node = _spec_node(node_cls, tmp_path, monkeypatch, seconds=0.2)
    spec = node._build_spec()
    assert spec["audio_file"].endswith("in.wav")


def test_run_nrt_translates_encoder_overflow(node_cls, tmp_path, monkeypatch):
    """MP3-style inputs skip the header guard; the CLI failure is still
    translated into the same actionable guidance."""
    mod = _live()
    src = write_wav(tmp_path / "song.wav", seconds=0.5)
    transcript = write_transcript(tmp_path / "t.txt", "hello world")
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(cli, gguf, **kw):
        raise RuntimeError(
            "audio.cpp exited with code 1: ... Qwen3 ASR audio encoder "
            "token count exceeds max_source_positions")

    monkeypatch.setattr(mod, "run_qwen3_align", fake_run)
    node = make_node(node_cls)
    spec = {"cli": "c", "model_dir": str(model_dir),
            "gguf": "qwen3-forced-aligner-0.6b-q8_0.gguf",
            "audio_file": str(src), "transcript_file": str(transcript),
            "language": "en", "backend": "cpu"}
    import threading
    with pytest.raises(RuntimeError, match="word_timestamps"):
        node._run_nrt(threading.Event(), spec)


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
    node.params["model_dir"].set(str(tmp_path / "models"
                                     / "Qwen3-ForcedAligner-GGUF"))
    node.params["model_dir"].sync()
    payload = node._build_fetch_payload()
    assert len(payload["runtime"]) == 2  # CLI absent: both zips pending
    assert len(payload["models"]) == 1
    # CLI present + size-correct model file -> nothing to fetch.
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin" / "audiocpp_cli.exe").write_text("x")
    for spec in backend.qwen3_align_fetch_specs(
            tmp_path / "models" / "Qwen3-ForcedAligner-GGUF"):
        spec.dest.parent.mkdir(parents=True, exist_ok=True)
        with open(spec.dest, "wb") as f:
            f.truncate(spec.size)
    monkeypatch.setattr(
        download_util, "_verify",
        lambda p, s, h: _Path(p).is_file() and (s is None or _Path(p).stat().st_size == s))
    payload = node._build_fetch_payload()
    assert payload["runtime"] == []
    assert payload["models"] == []
