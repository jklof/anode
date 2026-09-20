"""Tests for TextNote / ABCNote nodes and ui_size persistence (no GUI)."""
import json

import pytest
import torch

import plugin_system


@pytest.fixture(scope="module")
def registry():
    plugin_system.load_plugins("plugins")
    assert "TextNote" in plugin_system.NODE_REGISTRY
    assert "ABCNote" in plugin_system.NODE_REGISTRY
    return plugin_system.NODE_REGISTRY


def test_abstract_base_not_registered(registry):
    assert "_NoteNodeBase" not in registry
    assert registry["TextNote"].is_abstract is False
    assert registry["ABCNote"].is_abstract is False


def test_ports_and_offline_contract(registry):
    for name, uri_in, uri_out, text_param in (
        ("TextNote", "text_uri", "text", "text"),
        ("ABCNote", "abc_uri", "abc", "abc"),
    ):
        node = registry[name]()
        assert node.is_offline is True
        assert node.inputs["text_uri" if name == "TextNote" else "abc_uri"].slot_type == "uri"
        assert node.inputs["trigger_in"].slot_type == "audio"
        assert node.outputs[uri_out].slot_type == "uri"
        assert node.done.buffer.shape[1] == 512
        assert node.params[text_param].type == "string"
        assert node.params["refresh"].type == "bool"


def test_widgets_registered_resizable(registry):
    ui_text = plugin_system.get_ui_class("TextNote")
    ui_abc = plugin_system.get_ui_class("ABCNote")
    assert ui_text is not None and ui_abc is not None
    for ui in (ui_text, ui_abc):
        assert getattr(ui, "IS_RESIZABLE", False) is True
        assert len(ui.MIN_SIZE) == 2 and len(ui.DEFAULT_SIZE) == 2


def test_text_publish_worker_roundtrip(registry, tmp_path, monkeypatch):
    import hashlib
    import text_notes
    monkeypatch.setattr(text_notes, "REPO_ROOT", tmp_path)
    node = registry["TextNote"]()
    node.params["text"].set("hello\nworld")
    node.params["text"].sync()
    spec = {"op": "publish", "text": node.params["text"].value, "epoch": 1}
    node._io_epoch = 1
    result = node._io_nrt(spec)
    expected = hashlib.sha256("hello\nworld".encode()).hexdigest() + ".txt"
    assert result["path"] == str(tmp_path / "outputs" / "text_notes" / expected)
    assert open(result["path"], encoding="utf-8").read() == "hello\nworld"
    # identical content re-publishes to the same path (dedup, byte-stable)
    again = node._io_nrt({"op": "publish", "text": "hello\nworld", "epoch": 2})
    assert again["path"] == result["path"]
    assert sorted(p.name for p in (tmp_path / "outputs" / "text_notes").iterdir()) == [expected]
    node.on_nrt_complete("io", True, result)
    assert node.outputs["text"].uri == result["path"]
    assert node._status == "Ready"
    assert node.params["last_path"].value == result["path"]
    # one-block done pulse, then silence (anti-ghost)
    node.process()
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_text_load_installs_text_and_uri(registry, tmp_path, monkeypatch):
    import text_notes
    monkeypatch.setattr(text_notes, "REPO_ROOT", tmp_path)
    src = tmp_path / "words.txt"
    src.write_text("wired words", encoding="utf-8")
    node = registry["TextNote"]()
    node._io_epoch = 3
    result = node._io_nrt({"op": "load", "path": str(src), "epoch": 3})
    assert result["text"] == "wired words"
    # loads normalize into the CAS: the published path is immutable, and the
    # mutable source file is left alone
    assert result["path"] != str(src)
    assert result["path"].endswith(".txt")
    assert open(result["path"], encoding="utf-8").read() == "wired words"
    node.on_nrt_complete("io", True, result)
    assert node.params["text"].value == "wired words"
    assert node.outputs["text"].uri == result["path"]


def test_different_content_different_paths(registry, tmp_path, monkeypatch):
    import text_notes
    monkeypatch.setattr(text_notes, "REPO_ROOT", tmp_path)
    node = registry["ABCNote"]()
    node._io_epoch = 1
    r1 = node._io_nrt({"op": "publish", "abc": "X:1\nK:C\nC D E|\n", "epoch": 1})
    r2 = node._io_nrt({"op": "publish", "abc": "X:1\nK:C\nG A B|\n", "epoch": 2})
    assert r1["path"] != r2["path"]
    assert r1["path"].endswith(".abc") and r2["path"].endswith(".abc")


def test_stale_epoch_ignored(registry):
    node = registry["TextNote"]()
    node._io_epoch = 5  # newer submit superseded epoch 4
    node.on_nrt_complete("io", True, {"op": "publish", "epoch": 4,
                                      "path": "/tmp/stale.txt", "summary": "x"})
    assert node.outputs["text"].uri == ""
    assert node._status != "Ready"


def test_abc_validation(registry):
    node = registry["ABCNote"]()
    summary = node.validate_text("X:1\nK:C\nC D E F|G4 z2|\n")
    assert "notes" in summary
    with pytest.raises(ValueError):
        node.validate_text("X:1\nK:C\n% no notes here\n")


def test_abc_publish_rejects_empty_score(registry, tmp_path, monkeypatch):
    import text_notes
    monkeypatch.setattr(text_notes, "REPO_ROOT", tmp_path)
    node = registry["ABCNote"]()
    node._io_epoch = 1
    with pytest.raises(ValueError):
        node._io_nrt({"op": "publish", "abc": "X:1\nK:C\n| / . |\n", "epoch": 1})


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


def _attach_fake_engine(node):
    from types import SimpleNamespace

    class _FakeEngine:
        def __init__(self):
            self.commands = []

        def push_command(self, cmd):
            self.commands.append(cmd)

    engine = _FakeEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


def test_trigger_edge_stages_refresh(registry):
    node = registry["TextNote"]()
    engine = _attach_fake_engine(node)
    _wire_trigger(node)
    node.process()  # rising edge
    assert engine.commands == [("param", node.id, "refresh", True)]
    node.process()  # held high: no duplicate command
    assert len(engine.commands) == 1


def test_ui_size_to_dict_load_state_roundtrip(registry):
    node = registry["TextNote"]()
    assert node.ui_size is None
    assert node.to_dict()["ui_size"] is None
    node.ui_size = (500, 400)
    d = node.to_dict()
    assert d["ui_size"] == [500, 400]
    fresh = registry["TextNote"]()
    fresh.load_state(d)
    assert fresh.ui_size == (500, 400)
    # old patches without the key fall back to None
    legacy = dict(d)
    del legacy["ui_size"]
    fresh2 = registry["TextNote"]()
    fresh2.load_state(legacy)
    assert fresh2.ui_size is None
    # malformed values never crash load
    fresh3 = registry["TextNote"]()
    fresh3.load_state({"pos": (0, 0), "ui_size": "bogus", "params": {}})
    assert fresh3.ui_size is None


# ----------------------------------------------------------------------
# widget + frame (offscreen)
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
    def __init__(self, params):
        from types import SimpleNamespace
        self.calls = []
        self.node_item = SimpleNamespace(params=params)

    def set_parameter(self, name, value):
        self.calls.append((name, value))


def _widget_params(text):
    return {"text": {"value": text}, "abc": {"value": text}}


def test_text_widget_apply_and_telemetry(qapp, registry):
    import sys
    node = registry["TextNote"]()
    proxy = _StubProxy(_widget_params(node.params["text"].value))
    widget = sys.modules["text_notes"].TextNoteWidget(proxy)
    widget.editor.setPlainText("edited")
    widget.btn_apply.click()
    assert proxy.calls == [("text", "edited")]
    widget.on_telemetry({"status": "Ready", "audio": "note.txt"})
    assert "Ready" in widget.lbl_status.text()
    widget.on_telemetry({"status": "Error", "audio": "boom"})
    assert "boom" in widget.lbl_status.text()


def test_text_widget_backend_sync_respects_focus(qapp, registry):
    import sys
    node = registry["TextNote"]()
    proxy = _StubProxy(_widget_params(node.params["text"].value))
    widget = sys.modules["text_notes"].TextNoteWidget(proxy)
    widget.editor.setPlainText("user draft")
    # Focused editor is not clobbered... (offscreen has no focus; simulate by
    # marking focus via QWidget focus emulation is unreliable, so verify the
    # unfocused path applies and identical text is a no-op.)
    widget.update_from_params({"text": "remote text"})
    assert widget.editor.toPlainText() == "remote text"
    widget.update_from_params({"text": "remote text"})  # no-op, no crash


def test_file_source_widget_publish_and_warning(qapp, registry):
    import sys

    class _FileProxy(_StubProxy):
        def create_param_widget(self, name):
            from PySide6.QtWidgets import QWidget

            w = QWidget()
            w.setObjectName(name)

            def _update(value):
                w.setProperty("backend_value", value)

            w.update_from_backend = _update
            return w

    proxy = _FileProxy({})
    widget = sys.modules["text_notes"].TextFileSourceWidget(proxy)
    widget.btn_publish.click()
    assert proxy.calls == [("refresh", True)]
    widget.update_from_params({"file": "/tmp/x.txt"})
    widget.on_telemetry({"status": "Warning", "audio": "Missing: /tmp/x.txt"})
    assert "Missing" in widget.lbl_status.text()
    widget.on_telemetry({"status": "Ready", "audio": "x.txt"})
    assert "Ready" in widget.lbl_status.text()


def test_abc_widget_sections_and_load_button(qapp, registry):
    import sys
    node = registry["ABCNote"]()
    proxy = _StubProxy(_widget_params(node.params["abc"].value))
    widget = sys.modules["text_notes"].ABCNoteWidget(proxy)
    names = [widget.section_list.item(i).text() for i in range(widget.section_list.count())]
    assert names == ["Verse"]
    widget.btn_load.click()
    assert proxy.calls == [("refresh", True)]
    widget.editor.setPlainText("X:1\nK:C\n% Solo\nC D E|\n")
    names = [widget.section_list.item(i).text() for i in range(widget.section_list.count())]
    assert names == ["Solo"]


def test_abc_widget_section_click_finds_no_space_header(qapp, registry):
    import sys
    node = registry["ABCNote"]()
    proxy = _StubProxy(_widget_params(node.params["abc"].value))
    widget = sys.modules["text_notes"].ABCNoteWidget(proxy)
    widget.editor.setPlainText("X:1\nK:C\n%intro\nC D E|\n")
    names = [widget.section_list.item(i).text() for i in range(widget.section_list.count())]
    assert names == ["intro"]
    widget._on_section_clicked(widget.section_list.item(0))
    cursor = widget.editor.textCursor()
    assert not cursor.isNull()
    assert "%intro" in cursor.block().text()


def test_abc_widget_empty_section_click_is_safe_noop(qapp, registry):
    import sys
    from PySide6.QtWidgets import QListWidgetItem
    node = registry["ABCNote"]()
    proxy = _StubProxy(_widget_params(node.params["abc"].value))
    widget = sys.modules["text_notes"].ABCNoteWidget(proxy)
    widget.editor.setPlainText("X:1\nK:C\n%intro\nC D E|\n")
    cursor = widget.editor.textCursor()
    cursor.setPosition(0)
    widget.editor.setTextCursor(cursor)
    before = widget.editor.textCursor().position()
    widget._on_section_clicked(QListWidgetItem("—"))
    widget._on_section_clicked(QListWidgetItem(""))
    assert widget.editor.textCursor().position() == before


def _snapshot_for(node):
    return {
        "id": node.id, "name": node.name, "type": node.__class__.__name__,
        "pos": node.pos,
        "ui_size": list(node.ui_size) if node.ui_size else None,
        "inputs": list(node.inputs.keys()), "outputs": list(node.outputs.keys()),
        "input_types": {k: v.slot_type for k, v in node.inputs.items()},
        "output_types": {k: v.slot_type for k, v in node.outputs.items()},
        "params": {k: {"value": p.value, "type": p.type, "meta": p.meta}
                   for k, p in node.params.items()},
        "monitor_queue": None, "can_be_master": False, "is_master": False,
        "error": None,
    }


def test_nodeitem_default_frame_and_clamp(qapp, registry):
    from ui_system import NodeItem
    node = registry["TextNote"]()
    item = NodeItem(_snapshot_for(node), controller=None)
    item.build_ui()
    assert (item.width, item.height) == tuple(
        sys_modules_default_size())
    # clamp: below minimum snaps up to MIN_SIZE-based frame
    item._apply_frame_size(50, 50)
    assert item.width >= 300 + 20
    assert item.height >= item._socket_area_height() + 200 + 10
    # outputs track width, proxy fills the frame below the sockets
    for out in item.output_items.values():
        assert out.x() == item.width
    assert item.proxy.size().width() == item.width - 20


def sys_modules_default_size():
    import sys
    return sys.modules["text_notes"].TextNoteWidget.DEFAULT_SIZE


def test_nodeitem_applies_snapshot_size(qapp, registry):
    from ui_system import NodeItem
    node = registry["TextNote"]()
    item = NodeItem(_snapshot_for(node), controller=None)
    item.build_ui()
    node.ui_size = (600, 420)
    item.update_from_snapshot(_snapshot_for(node))
    assert (item.width, item.height) == (600, 420)


def test_resize_command_and_save_json_roundtrip(registry):
    from core import Engine
    eng = Engine()
    node = registry["TextNote"]()
    eng.push_command(("add", node, "t1", (10, 20), None))
    eng.push_command(("resize", "t1", 520, 360))
    assert eng.graph.node_map["t1"].ui_size == (520, 360)
    snap = eng.graph.get_snapshot()
    entry = next(n for n in snap["nodes"] if n["id"] == "t1")
    assert entry["ui_size"] == [520, 360]
    data = json.loads(eng.graph.to_json())
    entry = next(n for n in data["nodes"] if n["id"] == "t1")
    assert entry["ui_size"] == [520, 360]
    # malformed resize is ignored, never crashes
    eng.push_command(("resize", "t1", "huge", None))
    assert eng.graph.node_map["t1"].ui_size == (520, 360)


# ----------------------------------------------------------------------
# file sources (no NRT: publishing is a synchronous string assignment)
# ----------------------------------------------------------------------
def test_file_source_registration(registry):
    assert "_FileSourceBase" not in registry
    node = registry["TextFileSource"]()
    assert node.is_abstract is False
    assert node.is_offline is True
    assert node.outputs["file"].slot_type == "uri"
    assert node.params["file"].type == "file"
    assert plugin_system.get_ui_class("TextFileSource") is not None


def test_file_source_publish_and_pulse(registry, tmp_path):
    node = registry["TextFileSource"]()
    f = tmp_path / "lyrics.txt"
    f.write_text("la", encoding="utf-8")
    node.params["file"].set(str(f))
    node.params["file"].sync()
    node.on_ui_param_change("file")
    assert node.outputs["file"].uri == str(f)
    assert node._status == "Ready"
    node.process()
    assert torch.all(node.done.buffer == 1.0)
    node.process()
    assert torch.all(node.done.buffer == 0.0)


def test_file_source_is_generic(registry, tmp_path):
    """The File node publishes any file kind, not just text: audio files
    pass straight through for wiring into audio_uri consumer inputs."""
    node = registry["TextFileSource"]()
    assert node.label == "File"
    f = tmp_path / "song.wav"
    f.write_bytes(b"RIFF" + bytes(100))
    node.params["file"].set(str(f))
    node.params["file"].sync()
    node.on_ui_param_change("file")
    assert node.outputs["file"].uri == str(f)
    assert node._status == "Ready"


def test_file_source_trigger_edge_stages_refresh(registry):
    node = registry["TextFileSource"]()
    engine = _attach_fake_engine(node)
    _wire_trigger(node)
    node.process()
    assert engine.commands == [("param", node.id, "refresh", True)]


def test_complete_pushes_telemetry_and_snapshot(registry, tmp_path, monkeypatch):
    """A trigger-staged load must notify the UI immediately: direct
    set()+sync() emits no param_update and no snapshot, so without an
    explicit push the editor would refresh only on the next unrelated
    graph change."""
    import queue
    import text_notes
    monkeypatch.setattr(text_notes, "REPO_ROOT", tmp_path)
    src = tmp_path / "plan.abc"
    src.write_text("X:1\nK:C\nC D|\n", encoding="utf-8")
    node = registry["ABCNote"]()
    out_q = queue.SimpleQueue()
    snapshots = []
    from types import SimpleNamespace
    node.graph = SimpleNamespace(
        engine=SimpleNamespace(output_queue=out_q,
                               _emit_snapshot=lambda: snapshots.append(True)))
    node._io_epoch = 1
    result = node._io_nrt({"op": "load", "path": str(src), "epoch": 1})
    node.on_nrt_complete("io", True, result)
    assert snapshots == [True]
    msg = out_q.get_nowait()
    assert msg["type"] == "telemetry"
    assert msg["node_data"][node.id]["status"] == "Ready"


def test_fail_pushes_telemetry_and_snapshot(registry, tmp_path):
    """Engine-stopped fail paths (wired-file-missing, worker exceptions) must
    push telemetry + snapshot now, not leave the widget showing stale status.
    Follows the stub pattern of test_complete_pushes_telemetry_and_snapshot."""
    import queue
    from types import SimpleNamespace
    node = registry["TextNote"]()
    out_q = queue.SimpleQueue()
    snapshots = []
    node.graph = SimpleNamespace(
        engine=SimpleNamespace(output_queue=out_q,
                               _emit_snapshot=lambda: snapshots.append(True)))
    # worker-exception path via on_nrt_complete(ok=False)
    node.on_nrt_complete("io", False, "disk exploded")
    assert snapshots == [True]
    msg = out_q.get_nowait()
    assert msg["type"] == "telemetry"
    data = msg["node_data"][node.id]
    assert data["status"] == "Error"
    assert data["audio"] == "I/O failed: disk exploded"
    assert data["busy"] is False
    # direct _fail with no engine attached must not crash (guard handles it)
    node.graph = None
    node._fail("no engine here")
    assert node._status == "Error"


def test_note_busy_flag_tracks_in_flight_work(registry, monkeypatch):
    """get_telemetry carries 'busy' (appended last): True while NRT work is
    in flight (Publishing/Loading), False in terminal states."""
    node = registry["TextNote"]()
    # drive the real staging path: committed text stages NRT publish work
    # (stub out the pool submit; the status transition is what matters)
    monkeypatch.setattr(node, "submit_nrt", lambda fn, spec, tag=None: None)
    node.params["text"].set("fresh words")
    node.params["text"].sync()
    node.on_ui_param_change("text")
    assert node._status == "Publishing"
    data = node.get_telemetry()
    assert list(data) == ["status", "audio", "busy"]  # key order stable
    assert data["busy"] is True
    node._status = "Loading"
    assert node.get_telemetry()["busy"] is True
    for terminal in ("Ready", "Idle", "Error", "Warning"):
        node._status = terminal
        assert node.get_telemetry()["busy"] is False, terminal


def test_file_source_publish_pushes_telemetry_with_busy(registry, tmp_path):
    """Engine-stopped pick/publish must push telemetry + snapshot now (the
    publish is synchronous, so no NRT traffic would ever deliver it).
    Sources never have NRT in flight, so 'busy' is always present but False."""
    import queue
    from types import SimpleNamespace
    node = registry["TextFileSource"]()
    out_q = queue.SimpleQueue()
    snapshots = []
    node.graph = SimpleNamespace(
        engine=SimpleNamespace(output_queue=out_q,
                               _emit_snapshot=lambda: snapshots.append(True)))
    f = tmp_path / "picked.txt"
    f.write_text("picked", encoding="utf-8")
    node.params["file"].set(str(f))
    node.params["file"].sync()
    node.on_ui_param_change("file")
    assert snapshots == [True]
    msg = out_q.get_nowait()
    assert msg["type"] == "telemetry"
    data = msg["node_data"][node.id]
    assert data == {"status": "Ready", "audio": "picked.txt", "busy": False}
    # missing file still pushes (Warning), and clearing pushes Idle
    node.params["file"].set(str(tmp_path / "gone.txt"))
    node.params["file"].sync()
    node.on_ui_param_change("file")
    data = out_q.get_nowait()["node_data"][node.id]
    assert data["status"] == "Warning" and data["busy"] is False
    node.params["file"].set("")
    node.params["file"].sync()
    node.on_ui_param_change("file")
    data = out_q.get_nowait()["node_data"][node.id]
    assert data == {"status": "Idle", "audio": "No file selected",
                    "busy": False}


def test_focused_editor_holds_then_applies_on_focus_out(qapp, registry):
    import sys
    from PySide6.QtCore import QEvent
    node = registry["TextNote"]()
    proxy = _StubProxy(_widget_params(node.params["text"].value))
    widget = sys.modules["text_notes"].TextNoteWidget(proxy)
    widget.editor.setPlainText("user draft")
    widget.editor.hasFocus = lambda: True  # simulate in-progress typing
    widget.update_from_params({"text": "remote text"})
    assert widget.editor.toPlainText() == "user draft"  # not clobbered...
    assert widget._pending_text == "remote text"  # ...nor dropped
    widget.editor.hasFocus = lambda: False
    widget.eventFilter(widget.editor, QEvent(QEvent.FocusOut))
    assert widget.editor.toPlainText() == "remote text"
    assert widget._pending_text is None
