"""Tests for the SheetSage2 transcriber node and backend.

All tests run without a GPU: the sidecar subprocess is stubbed, NRT
results are delivered by calling the completion handlers directly, and
the one GPU-gated test is skipped unless ANODE_SHEETSAGE_GPU=1.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import plugin_system
import audiocpp_backend as backend
from audiocpp_backend import (
    GenerationCancelled,
    _build_sheetsage_argv,
    run_sheetsage_transcribe,
    strip_abc_chords,
)
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SheetSage2Transcriber")
    assert cls is not None, "SheetSage2Transcriber not registered"
    return cls


def make_node(node_cls):
    return node_cls()


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    inst = QCoreApplication.instance()
    if inst is None:
        return QApplication([])
    if isinstance(inst, QApplication):
        return inst
    pytest.skip("a bare QCoreApplication is active; QWidget tests cannot run")


def _live():
    """Live top-level plugin module object.

    load_plugins() imports node files WITHOUT the plugins. package prefix,
    so ``import plugins.sheetsage_transcriber`` would bind an unpatched
    duplicate. Anything that patches module state (or relies on patched
    state) must go through this object; requires the node_cls fixture to
    have run first.
    """
    import sys
    return sys.modules["sheetsage_transcriber"]


def write_wav(path, seconds=1.0, sr=SAMPLE_RATE, channels=2, freq=440.0):
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.stack([tone * (i + 1) / channels for i in range(channels)], axis=1)
    sf.write(str(path), data, sr)
    return path


SAMPLE_ABC = """X:1
T:
M:4/4
L:1/32
Q:1/4=140
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:F#m
% intro
V: Vocal
z4"F#m"z24z4|"F#m"z32|
V: Ins
z4F4F4z16z4|F4F4F4F4F4F4F4F4|
% verse
V: Vocal
"D"z32|"Bm"z32|
w: la la
"""


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Offline"
    assert node_cls.label == "SheetSage2 Transcriber"


# ----------------------------------------------------------------------
# chord stripping + section parsing (pure)
# ----------------------------------------------------------------------
def test_strip_abc_chords_removes_only_music_line_quotes(node_cls):
    section_names = _live().section_names
    melody = strip_abc_chords(SAMPLE_ABC)
    assert '"F#m"' not in melody and '"D"' not in melody and '"Bm"' not in melody
    # Headers, comments, lyrics survive byte-identical.
    for line in ("X:1", "T:", "M:4/4", "L:1/32", "Q:1/4=140", "K:F#m",
                 "% intro", "% verse", "w: la la",
                 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"'):
        assert line in melody.splitlines()
    # Note content survives around the removed chords.
    assert "z4z24z4" in melody
    assert section_names(SAMPLE_ABC) == ["intro", "verse"]


def test_strip_abc_chords_ends_with_newline():
    assert strip_abc_chords("V: Ins\nF4G4|").endswith("\n")


# ----------------------------------------------------------------------
# argv builder (pure, no process)
# ----------------------------------------------------------------------
def test_argv_builds_expected_command(tmp_path):
    argv = _build_sheetsage_argv("cli", "models",
                                 audio_wav=str(tmp_path / "in.wav"),
                                 weight_type="native",
                                 max_tokens=5120,
                                 out_abc=str(tmp_path / "s.abc"))
    assert argv[:6] == ["cli", "--task", "midi", "--family", "sheetsage2", "--model"]
    assert "--audio" in argv and "--text-out" in argv and "--metrics" in argv
    assert "sheetsage2.weight_type=native" in argv
    assert "max_tokens=5120" in argv
    i = argv.index("--backend")
    assert argv[i + 1] == "cuda"  # default preserved; override below
    argv = _build_sheetsage_argv("cli", "models",
                                 audio_wav=str(tmp_path / "in.wav"),
                                 weight_type="native",
                                 max_tokens=5120,
                                 out_abc=str(tmp_path / "s.abc"),
                                 backend="cpu")
    assert argv[argv.index("--backend") + 1] == "cpu"


def test_argv_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        _build_sheetsage_argv("cli", "m", audio_wav="a", weight_type="q8_0",
                              max_tokens=1, out_abc="o")
    with pytest.raises(ValueError):
        _build_sheetsage_argv("cli", "m", audio_wav="a", weight_type="native",
                              max_tokens=0, out_abc="o")


# ----------------------------------------------------------------------
# run_sheetsage_transcribe with a stubbed subprocess
# ----------------------------------------------------------------------
class _StubProc:
    def __init__(self, argv, out_abc, code=0, metrics_text=""):
        self.argv = argv
        self._out = out_abc
        self._code = code
        self._text = metrics_text
        self.terminated = False

    def communicate(self, timeout=None):
        if self._code == 0 and self._out is not None:
            Path(self._out).parent.mkdir(parents=True, exist_ok=True)
            Path(self._out).write_text(SAMPLE_ABC, encoding="utf-8")
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


def test_run_transcribe_success(tmp_path, monkeypatch):
    out = tmp_path / "run" / "score.abc"
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(parents=True, exist_ok=True)
    (model / "sheetsage2-orig.gguf").write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.5)
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    seen["text"] = "metrics.wall_ms=100\nmetrics.rtf=0.8\n"
    import threading
    result = run_sheetsage_transcribe(str(cli), str(model), audio_wav=str(audio),
                                      out_abc=str(out),
                                      cancel_event=threading.Event())
    assert Path(result["abc"]).read_text(encoding="utf-8") == SAMPLE_ABC
    assert result["metrics"]["rtf"] == pytest.approx(0.8)


def test_run_transcribe_cancel_terminates(tmp_path, monkeypatch):
    out = tmp_path / "score.abc"
    cli = tmp_path / "cli"
    cli.write_text("x")
    model = tmp_path / "models"
    model.mkdir(exist_ok=True)
    (model / "sheetsage2-orig.gguf").write_text("x")
    audio = write_wav(tmp_path / "in.wav", seconds=0.5)
    seen = _patch_popen(monkeypatch)
    seen["out"] = str(out)
    import threading
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled):
        run_sheetsage_transcribe(str(cli), str(model), audio_wav=str(audio),
                                 out_abc=str(out), cancel_event=cancel)
    assert seen["proc"].terminated


def test_run_transcribe_missing_inputs(tmp_path):
    import threading
    with pytest.raises(FileNotFoundError):
        run_sheetsage_transcribe(str(tmp_path / "nope"), str(tmp_path),
                                 audio_wav=str(tmp_path / "a.wav"),
                                 out_abc=str(tmp_path / "o.abc"),
                                 cancel_event=threading.Event())


# ----------------------------------------------------------------------
# input audio loading (pure, no GPU)
# ----------------------------------------------------------------------
def test_load_input_wav_normalizes(node_cls, tmp_path):
    load_input_wav = _live().load_input_wav
    src = write_wav(tmp_path / "src44.wav", seconds=0.5, sr=44100, channels=1)
    out = load_input_wav(src, tmp_path / "out.wav")
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE
    assert data.shape == (int(0.5 * SAMPLE_RATE), CHANNELS)


def test_load_input_wav_mp3(node_cls, tmp_path):
    pytest.importorskip("av")
    from audio_io import encode_mp3_file
    load_input_wav = _live().load_input_wav
    mp3 = tmp_path / "src.mp3"
    try:
        encode_mp3_file(mp3, np.zeros((2, 4410), dtype=np.float32), 44100)
    except RuntimeError as e:
        pytest.skip(f"mp3 encoder unavailable: {e}")
    out = load_input_wav(mp3, tmp_path / "out.wav")
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE and data.shape[1] == CHANNELS


def test_load_input_wav_missing(node_cls, tmp_path):
    load_input_wav = _live().load_input_wav
    with pytest.raises(RuntimeError, match="not found"):
        load_input_wav(tmp_path / "gone.wav", tmp_path / "out.wav")


def test_load_input_wav_mixed_frame_layouts(node_cls, tmp_path, monkeypatch):
    # node_cls fixture first: it loads the plugins.
    # Regression: PyAV frame layouts are not guaranteed uniform (mixed
    # planar/packed or trailing mono flush frames broke a bulk
    # concatenate with 'all input array dimensions ... must match').
    import types
    load_input_wav = _live().load_input_wav

    stereo = np.zeros((1000, 2), dtype=np.float32)   # (samples, channels)
    planar = np.zeros((2, 500), dtype=np.float32)    # (channels, samples)
    mono_flush = np.zeros(250, dtype=np.float32)     # 1-D trailing frame

    class FakeFrame:
        def __init__(self, arr):
            self._arr = arr

        def to_ndarray(self):
            return self._arr

    class FakeContainer:
        def __init__(self):
            from types import SimpleNamespace
            self.streams = SimpleNamespace(audio=[SimpleNamespace(sample_rate=48000)])

        def decode(self, audio=0):
            return iter([FakeFrame(stereo), FakeFrame(planar), FakeFrame(mono_flush)])

    fake_av = types.SimpleNamespace(open=lambda *a, **k: FakeContainer())
    # Decoding lives in audio_io now; patch its decoder handle.
    import audio_io
    monkeypatch.setattr(audio_io, "av", fake_av, raising=False)
    src = tmp_path / "mixed.mp3"
    src.write_bytes(b"fake")
    out = load_input_wav(src, tmp_path / "out.wav")
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE
    assert data.shape == (1000 + 500 + 250, CHANNELS)


# ----------------------------------------------------------------------
# node: completion paths, validation, save/load
# ----------------------------------------------------------------------
def _install_score(node, path="s.abc"):
    node.on_nrt_complete("job", True, {"abc": SAMPLE_ABC, "abc_path": path,
                                       "melody_path": None, "metrics": {},
                                       "sections": ["intro", "verse"]})


def test_complete_installs_score(node_cls):
    node = make_node(node_cls)
    _install_score(node)
    assert node._abc_text == SAMPLE_ABC
    assert node._status == "Ready"
    assert "2 sections" in node._status_detail
    assert node.error_msg is None
    assert node.params["last_abc"].value == "s.abc"
    telem = node.get_telemetry()
    assert telem["status"] == "Ready" and "s.abc" in telem["audio"]
    assert telem["busy"] is False


def test_telemetry_busy_while_transcribing(node_cls):
    node = make_node(node_cls)
    node._status, node._status_detail = "Transcribing", "song.wav (native)…"
    telem = node.get_telemetry()
    assert telem["busy"] is True and telem["status"] == "Transcribing"


# ----------------------------------------------------------------------
# score viewer (widget-level, offscreen, read-only)
# ----------------------------------------------------------------------
class _StubProxy:
    def __init__(self):
        self.calls = []

    def set_parameter(self, name, value):
        self.calls.append((name, value))

    def create_param_widget(self, name):
        from PySide6.QtWidgets import QWidget
        return QWidget()


def test_viewer_lists_sections_and_shows_text(qapp):
    widget = _live().SheetSageWidget(_StubProxy())
    assert widget.section_list.count() == 0
    widget.on_telemetry({"status": "Ready", "audio": "s.abc",
                         "score_text": SAMPLE_ABC})
    assert [widget.section_list.item(i).text() for i in
            range(widget.section_list.count())] == ["intro", "verse"]
    assert "K:F#m" in widget.score_browser.toPlainText()
    # Same text twice: no rebuild, no crash.
    widget.on_telemetry({"status": "Ready", "audio": "s.abc",
                         "score_text": SAMPLE_ABC})
    assert widget.section_list.count() == 2


def test_viewer_empty_score_clears(qapp):
    widget = _live().SheetSageWidget(_StubProxy())
    widget.on_telemetry({"status": "Ready", "audio": "", "score_text": SAMPLE_ABC})
    assert widget.section_list.count() == 2
    widget.on_telemetry({"status": "Idle", "audio": "", "score_text": ""})
    assert widget.section_list.count() == 0
    assert widget.score_browser.toPlainText() == ""


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


def test_file_node_feeds_audio_uri(node_cls, tmp_path):
    """The generic File node output wires into audio_uri with matching types."""
    plugin_system.load_plugins("plugins")
    file_cls = plugin_system.NODE_REGISTRY.get("TextFileSource")
    assert file_cls is not None
    src, node = file_cls(), make_node(node_cls)
    node.inputs["audio_uri"].connect(src.outputs["text"])
    assert node.inputs["audio_uri"].get_uri() == ""


def test_complete_publishes_uris_and_pulses_once(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", True, {"abc": SAMPLE_ABC, "abc_path": "s.abc",
                                       "melody_path": "m.abc", "metrics": {},
                                       "sections": ["intro"]})
    assert node.outputs["score"].uri == "s.abc"
    assert node.outputs["melody"].uri == "m.abc"
    node.process()
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_complete_without_melody_leaves_melody_empty(node_cls):
    node = make_node(node_cls)
    _install_score(node)  # melody_path None
    assert node.outputs["score"].uri == "s.abc"
    assert node.outputs["melody"].uri == ""


def test_relink_sets_uris_without_pulse(node_cls, tmp_path):
    score = tmp_path / "kept.abc"
    score.write_text(SAMPLE_ABC, encoding="utf-8")
    melody = tmp_path / "kept-melody.abc"
    melody.write_text("X:1\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["last_abc"].set(str(score))
    node.params["last_abc"].sync()
    node.params["last_melody"].set(str(melody))
    node.params["last_melody"].sync()
    node.load_state(node.to_dict())
    assert node.outputs["score"].uri == str(score)
    assert node.outputs["melody"].uri == str(melody)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_complete_failure_and_cancel(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("job", False, RuntimeError("boom"))
    assert node._status == "Error"
    assert "boom" in node.error_msg
    node.on_nrt_complete("job", False, GenerationCancelled("stop"))
    assert node._status == "Cancelled"
    assert node.error_msg is None


def test_transcribe_rejects_missing_audio(node_cls, tmp_path, monkeypatch):
    node = make_node(node_cls)
    from types import SimpleNamespace
    import queue
    engine = SimpleNamespace(output_queue=queue.Queue())
    node.graph = SimpleNamespace(engine=engine)
    node.params["transcribe"].set(True)
    node.params["transcribe"].sync()
    node.on_ui_param_change("transcribe")
    assert node._status == "Error"
    assert "audio_file" in node._status_detail
    msg = engine.output_queue.get_nowait()  # validation errors still surface
    assert msg["node_data"][node.id]["status"] == "Error"


def test_save_load_relinks(node_cls, tmp_path):
    abc = tmp_path / "kept.abc"
    abc.write_text(SAMPLE_ABC, encoding="utf-8")
    node = make_node(node_cls)
    node.params["last_abc"].set(str(abc))
    node.params["last_abc"].sync()
    snapshot = node.to_dict()

    node2 = make_node(node_cls)
    node2.load_state(snapshot)  # small text: relinked synchronously
    assert node2._abc_text == SAMPLE_ABC
    assert node2._status == "Ready"


def test_load_missing_score_goes_idle(node_cls, tmp_path):
    node = make_node(node_cls)
    node.params["last_abc"].set(str(tmp_path / "gone.abc"))
    node.params["last_abc"].sync()
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())
    assert node2._status == "Idle"


def test_download_nothing_missing_pushes_feedback(node_cls, tmp_path, monkeypatch):
    import sys
    mod = sys.modules["sheetsage_transcriber"]

    class StubRuntime:
        def __init__(self, *a, **k):
            self.root = tmp_path / "tools" / "audiocpp"
            self.bin_dir = self.root / "bin"
            self.cli = tmp_path / "bin" / "audiocpp_cli.exe"

    monkeypatch.setattr(mod, "AudioCppRuntime", StubRuntime)
    node = make_node(node_cls)
    from types import SimpleNamespace
    import queue
    engine = SimpleNamespace(output_queue=queue.Queue())
    node.graph = SimpleNamespace(engine=engine)
    (tmp_path / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bin" / "audiocpp_cli.exe").write_text("x")
    node.params["model_dir"].set(str(tmp_path / "models" / "SheetSage2-GGUF"))
    node.params["model_dir"].sync()
    for spec in mod.sheetsage_fetch_specs(tmp_path / "models" / "SheetSage2-GGUF"):
        spec.dest.parent.mkdir(parents=True, exist_ok=True)
        with open(spec.dest, "wb") as f:
            f.truncate(spec.size)
    import download_util
    from pathlib import Path as _Path
    monkeypatch.setattr(
        download_util, "_verify",
        lambda p, s, h: _Path(p).is_file() and (s is None or _Path(p).stat().st_size == s))
    node.params["download"].set(True)
    node.params["download"].sync()
    node.on_ui_param_change("download")
    assert node.params["download"].value is False
    assert node._status_detail == "Everything already downloaded"
    msg = engine.output_queue.get_nowait()
    assert msg["node_data"][node.id]["audio"] == "Everything already downloaded"


# ----------------------------------------------------------------------
# GPU-gated end to end (needs ANODE_SHEETSAGE_GPU=1 + fetched runtime/model)
# ----------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("ANODE_SHEETSAGE_GPU") != "1",
                    reason="needs local GPU + fetched SheetSage2 runtime")
def test_gpu_smoke_transcribe(node_cls, tmp_path):
    import threading
    load_input_wav = _live().load_input_wav
    node = make_node(node_cls)
    audio = write_wav(tmp_path / "tone.wav", seconds=10.0)
    node.params["audio_file"].set(str(audio))
    node.params["audio_file"].sync()
    spec = node._build_spec()  # raises if runtime/model missing
    cancel = threading.Event()
    converted = load_input_wav(spec["audio_file"], tmp_path / "in.wav")
    out_abc = tmp_path / "score.abc"
    result = run_sheetsage_transcribe(
        spec["cli"], spec["model_dir"], audio_wav=str(converted),
        weight_type=spec["weight_type"], max_tokens=512,
        out_abc=str(out_abc), cancel_event=cancel)
    text = Path(result["abc"]).read_text(encoding="utf-8")
    assert text.startswith("X:")
    assert len(text.strip().splitlines()) > 5


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


def test_trigger_input_is_audio(node_cls):
    node = make_node(node_cls)
    assert node.inputs["trigger_in"].slot_type == "audio"


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
