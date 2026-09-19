"""Tests for the LyricFitter node (no GPU; worker driven by hand)."""
import pytest

import plugin_system


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("LyricFitter")
    assert cls is not None, "LyricFitter not registered"
    return cls


def make_node(node_cls):
    return node_cls()


ABC = "X:1\nL:1/4\nQ:120\nK:C\n% verse\nC2 D2|z4 E2|\n% chorus\nA2 A2|\n"
LYRICS = "[Verse]\nla la\nlo\n\n[Chorus]\nna na\n"


def _write_inputs(tmp_path, lyrics=LYRICS, abc=ABC):
    lyrics_file = tmp_path / "words.txt"
    lyrics_file.write_text(lyrics, encoding="utf-8")
    abc_file = tmp_path / "score.abc"
    abc_file.write_text(abc, encoding="utf-8")
    return str(lyrics_file), str(abc_file)


def _set_params(node, lyrics_file, abc_file):
    node.params["lyrics_file"].set(lyrics_file)
    node.params["lyrics_file"].sync()
    node.params["abc_file"].set(abc_file)
    node.params["abc_file"].sync()


def test_registration(node_cls):
    assert node_cls.category == "Offline"
    assert node_cls.is_offline is True
    assert node_cls().inputs["abc_uri"].slot_type == "uri"
    assert node_cls().outputs["lyrics"].slot_type == "uri"


def test_fit_worker_writes_keep(node_cls, tmp_path, monkeypatch):
    import lyric_fitter
    node = make_node(node_cls)
    lyrics_file, abc_file = _write_inputs(tmp_path)
    keep_root = tmp_path / "outputs"
    monkeypatch.setattr(lyric_fitter, "REPO_ROOT", tmp_path)
    result = node._fit_nrt({"lyrics_file": lyrics_file, "abc_file": abc_file})
    assert result["coverage"] == pytest.approx(1.0)
    fitted = keep_root / "lyric_fits"
    assert fitted.is_dir()
    kept = next(fitted.rglob("fitted.txt"))
    assert "[chorus]\nna na\n" in kept.read_text(encoding="utf-8")
    assert (kept.parent / "report.txt").exists()
    assert (kept.parent / "request.json").exists()
    assert result["path"] == str(kept)


def test_complete_publishes_uri_and_pulses_once(node_cls, tmp_path, monkeypatch):
    import lyric_fitter
    node = make_node(node_cls)
    monkeypatch.setattr(lyric_fitter, "REPO_ROOT", tmp_path)
    lyrics_file, abc_file = _write_inputs(tmp_path)
    result = node._fit_nrt({"lyrics_file": lyrics_file, "abc_file": abc_file})
    node.on_nrt_complete("fit", True, result)
    assert node.outputs["lyrics"].uri == result["path"]
    assert node._status == "Ready" and "100%" in node._status_detail
    assert node.params["last_fit"].value == result["path"]
    node.process()
    import torch
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_complete_failure_sets_error(node_cls):
    node = make_node(node_cls)
    node.on_nrt_complete("fit", False, RuntimeError("boom"))
    assert node._status == "Error"
    assert "boom" in node.error_msg


def _wire_trigger(node, level=1.0):
    from base import Node

    class _Src(Node):
        def __init__(self):
            super().__init__()
            self.add_output("out")

        def process(self):
            pass

    src = _Src()
    src.outputs["out"].buffer.fill_(level)
    node.inputs["trigger_in"].connect(src.outputs["out"])
    return src


class _FakeEngine:
    def __init__(self):
        self.commands = []

    def push_command(self, cmd):
        self.commands.append(cmd)


def _attach_engine(node):
    from types import SimpleNamespace
    engine = _FakeEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


def test_trigger_edge_stages_fit(node_cls):
    node = make_node(node_cls)
    engine = _attach_engine(node)
    _wire_trigger(node)
    node.process()  # rising edge
    assert engine.commands == [("param", node.id, "fit", True)]
    node.process()  # held high: no duplicate command
    assert len(engine.commands) == 1


def test_trigger_without_engine_is_safe(node_cls):
    node = make_node(node_cls)
    assert getattr(node, "graph", None) is None
    _wire_trigger(node)
    node.process()  # no graph/engine: must not raise


def test_build_spec_rejects_missing_inputs(node_cls, tmp_path):
    node = make_node(node_cls)
    with pytest.raises(ValueError, match="score"):
        node._build_spec()
    lyrics_file, _ = _write_inputs(tmp_path)
    node.params["lyrics_file"].set(lyrics_file)
    node.params["lyrics_file"].sync()
    with pytest.raises(ValueError, match="score"):
        node._build_spec()
    abc_file = tmp_path / "score.abc"
    abc_file.write_text(ABC, encoding="utf-8")
    node.params["abc_file"].set(str(abc_file))
    node.params["abc_file"].sync()
    spec = node._build_spec()
    assert spec["lyrics_file"] == lyrics_file
    assert spec["source_path"] == lyrics_file


def test_relink_sets_uri_without_pulse(node_cls, tmp_path):
    keep = tmp_path / "keep"
    keep.mkdir()
    fitted = keep / "fitted.txt"
    fitted.write_text("[verse]\nla\n", encoding="utf-8")
    (keep / "report.txt").write_text("coverage: 1/1\n", encoding="utf-8")
    node = make_node(node_cls)
    node.params["last_fit"].set(str(fitted))
    node.params["last_fit"].sync()
    node.load_state({"params": {}})
    assert node.outputs["lyrics"].uri == str(fitted)
    assert "Re-linked" in node._status_detail
    node.process()
    import torch
    assert torch.all(node.done.buffer == 0.0)  # no pulse on relink


def test_telemetry_carries_report(node_cls):
    node = make_node(node_cls)
    node._status, node._status_detail, node._report = (
        "Ready", "100% phrases covered", "coverage: 3/3")
    telem = node.get_telemetry()
    assert telem["status"] == "Ready" and telem["report"] == "coverage: 3/3"


# ----------------------------------------------------------------------
# widget (offscreen)
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


class _StubProxy:
    def __init__(self):
        self.calls = []
        self.created = []

    def set_parameter(self, name, value):
        self.calls.append((name, value))

    def create_param_widget(self, name):
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        w.setObjectName(name)
        self.created.append(name)
        return w


def test_widget_embeds_params_and_fit_button(qapp):
    import sys
    widget = sys.modules["lyric_fitter"].LyricFitterWidget(_StubProxy())
    assert widget.proxy.created == ["words_file", "lyrics_file", "abc_file"]
    widget.btn_fit.click()
    assert widget.proxy.calls == [("fit", True)]


def test_widget_shows_report(qapp):
    import sys
    widget = sys.modules["lyric_fitter"].LyricFitterWidget(_StubProxy())
    widget.on_telemetry({"status": "Ready", "audio": "100%",
                         "report": "coverage: 3/3"})
    assert widget.lbl_status.text() == "Ready"
    assert "coverage: 3/3" in widget.report_browser.toPlainText()
    # Same report twice: no rebuild, no crash.
    widget.on_telemetry({"status": "Ready", "audio": "100%",
                         "report": "coverage: 3/3"})


def test_words_uri_overrides_lyrics_file(node_cls, tmp_path):
    from base import Node

    class _WordSrc(Node):
        def __init__(self, uri):
            super().__init__()
            self.add_uri_output("words")
            self.outputs["words"].uri = uri

        def process(self):
            pass

    words = tmp_path / "words.json"
    words.write_text('[{"word": "la", "start": 0.0, "end": 0.4}]', encoding="utf-8")
    lyrics_file, abc_file = _write_inputs(tmp_path)
    node = make_node(node_cls)
    node.inputs["words_uri"].connect(_WordSrc(str(words)).outputs["words"])
    node.params["lyrics_file"].set(lyrics_file)
    node.params["lyrics_file"].sync()
    node.params["abc_file"].set(abc_file)
    node.params["abc_file"].sync()
    spec = node._build_spec()
    assert spec["source_path"] == str(words)
    assert spec["source_kind"] == "words"
    assert spec["lyrics_file"] == str(words)


def test_spec_backward_compatibility_with_legacy_keys(node_cls, tmp_path, monkeypatch):
    import lyric_fitter
    node = make_node(node_cls)
    lyrics_file, abc_file = _write_inputs(tmp_path)
    monkeypatch.setattr(lyric_fitter, "REPO_ROOT", tmp_path)
    result = node._fit_nrt({"lyrics_file": lyrics_file, "abc_file": abc_file})
    assert result["coverage"] == 1.0
    assert result["timed"] is False


def test_timed_words_fit_marks_timed(node_cls, tmp_path, monkeypatch):
    import json
    import lyric_fitter
    node = make_node(node_cls)
    abc = "X:1\nL:1/4\nQ:1/4=120\nK:C\n% verse\nC2 D2|z4 E2|\n"
    abc_file = tmp_path / "score.abc"
    abc_file.write_text(abc, encoding="utf-8")
    words = [{"word": "la", "start": 0.5, "end": 0.7},
             {"word": "lo", "start": 3.0, "end": 3.2}]
    words_file = tmp_path / "words.json"
    words_file.write_text(json.dumps(words), encoding="utf-8")
    monkeypatch.setattr(lyric_fitter, "REPO_ROOT", tmp_path)
    result = node._fit_nrt({"lyrics_file": str(words_file),
                            "source_path": str(words_file),
                            "abc_file": str(abc_file)})
    assert result["timed"] is True
    assert result["coverage"] == 1.0


def test_validation_error_refreshes_ui_when_stopped(node_cls):
    node = make_node(node_cls)
    seen = {}

    class _FakeEngine:
        def __init__(self):
            from queue import Queue
            self.output_queue = Queue()

        def _emit_snapshot(self):
            seen["emitted"] = True

    engine = _FakeEngine()
    node.graph = type("G", (), {"engine": engine})()
    node.params["fit"].set(True)
    node.params["fit"].sync()
    node.on_ui_param_change("fit")
    assert node._status == "Error"
    assert seen.get("emitted") is True
    telem = node.get_telemetry()
    assert telem["busy"] is False


def test_widget_displays_error_detail(qapp):
    import sys
    widget = sys.modules["lyric_fitter"].LyricFitterWidget(_StubProxy())
    widget.on_telemetry({"status": "Error", "audio": "Pick lyrics first",
                         "report": ""})
    assert widget.lbl_detail.text() == "Pick lyrics first"
    widget.on_telemetry({"status": "Ready", "audio": "100% phrases covered",
                         "report": ""})
    assert widget.lbl_detail.text() == "100% phrases covered"
