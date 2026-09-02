"""Regression tests for UI/telemetry bugs found in the codebase review:

- ``PianoWidget._note_at_pos()`` must return ``-1`` (not ``None``) when the
  click lands outside the keybed, so ``mousePressEvent``'s ``if note >= 0``
  guard cannot crash with a ``TypeError``.
- ``MIDIKeyboardNode.get_telemetry()`` must not carry a stray unreachable
  ``return -1`` statement.
- ``MIDIOutputWidget.on_telemetry()`` must not reference a nonexistent
  ``_status`` attribute (previously it crashed on every telemetry delivery).
- ``NodeProxy.push_custom_event()`` routes UI-originated events to the
  engine-side node's SPSC queue without widgets traversing
  ``controller.engine.graph`` directly.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


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
    # A bare QCoreApplication created by an earlier test module is still
    # alive; Qt cannot attach GUI support to it afterwards, and constructing
    # a second app would abort. Skip rather than kill the whole suite.
    pytest.skip("a bare QCoreApplication is active; QWidget tests cannot run")


# ---------------------------------------------------------------------------
# PianoWidget hit-testing
# ---------------------------------------------------------------------------


def test_piano_note_at_pos_returns_int_in_keybed(qapp):
    from plugins.midi_keyboard import PianoWidget

    w = PianoWidget()
    w.resize(280, 90)  # 2 octaves * 7 white keys = 14 white keys -> width 20 px
    note = w._note_at_pos(5.0, 45.0)
    assert isinstance(note, int)
    assert note >= 0
    assert note == 48  # first white key of C major at start_note=48


def test_piano_note_at_pos_out_of_range_x_returns_minus_one(qapp):
    from plugins.midi_keyboard import PianoWidget

    w = PianoWidget()
    w.resize(280, 90)

    # x beyond the right edge used to fall off the method and return None,
    # crashing mousePressEvent with "'>=' not supported between instances of
    # 'NoneType' and 'int'".
    assert w._note_at_pos(float(w.width()), 45.0) == -1   # exact right edge
    assert w._note_at_pos(float(w.width() + 50), 45.0) == -1
    assert w._note_at_pos(-10.0, 45.0) == -1              # left of the keybed


def test_piano_mouse_press_outside_keybed_does_not_crash(qapp):
    from PySide6.QtCore import QPointF

    from plugins.midi_keyboard import PianoWidget

    w = PianoWidget()
    w.resize(280, 90)

    class _FakeEvent:
        def position(self):
            return QPointF(w.width() + 50, 45.0)

    pressed = []
    w.noteOn.connect(lambda note, vel: pressed.append((note, vel)))
    w.mousePressEvent(_FakeEvent())  # must not raise TypeError
    assert pressed == []


def test_piano_mouse_press_inside_keybed_emits_note(qapp):
    from PySide6.QtCore import QPointF

    from plugins.midi_keyboard import PianoWidget

    w = PianoWidget()
    w.resize(280, 90)

    class _FakeEvent:
        def position(self):
            return QPointF(5.0, 45.0)

    pressed = []
    w.noteOn.connect(lambda note, vel: pressed.append((note, vel)))
    w.mousePressEvent(_FakeEvent())
    assert pressed == [(48, 100)]


# ---------------------------------------------------------------------------
# MIDIKeyboardNode.get_telemetry (stray return cleanup)
# ---------------------------------------------------------------------------


def test_midi_keyboard_node_get_telemetry_clean():
    from plugins.midi_keyboard import MIDIKeyboardNode

    node = MIDIKeyboardNode()
    assert node.get_telemetry() == {}
    node.monitor_queue.push({"active_notes": [60, 64]})
    assert node.get_telemetry() == {"active_notes": [60, 64]}


# ---------------------------------------------------------------------------
# MIDIOutputWidget.on_telemetry (AttributeError fix)
# ---------------------------------------------------------------------------


def test_midi_output_widget_on_telemetry_no_attribute_error(qapp):
    from plugins.midi_devices import MIDIOutputWidget

    class _FakeItem:
        params = {"device_name": {"value": ""}}

    class _FakeProxy:
        node_id = "midi_out_1"
        node_item = _FakeItem()

    w = MIDIOutputWidget(_FakeProxy())
    # Previously returned {"status": self._status} -> AttributeError on every
    # telemetry delivery (ui_system.NodeItem.propagate_telemetry has no guard).
    w.on_telemetry({"status": "Active"})
    assert w.lbl_status.text() == "Active"
    w.on_telemetry({})  # missing status key must be a no-op, not a crash


# ---------------------------------------------------------------------------
# NodeProxy.push_custom_event encapsulation
# ---------------------------------------------------------------------------


def test_node_proxy_push_custom_event_routes_to_engine_node(qapp):
    from ui_system import NodeProxy
    from plugins.midi_keyboard import MIDIKeyboardNode

    class _FakeItem:
        pass

    node = MIDIKeyboardNode()
    node.id = "kb1"

    class _FakeGraph:
        node_map = {"kb1": node}

    class _FakeEngine:
        graph = _FakeGraph()

    class _FakeController:
        engine = _FakeEngine()

    proxy = NodeProxy("kb1", _FakeController(), None, _FakeItem())
    assert proxy.push_custom_event(("note_on", 60, 100)) is True

    item, ok = node._ui_queue.try_pop()
    assert ok and item == ("note_on", 60, 100)

    # A missing node must report False, never raise.
    missing = NodeProxy("does_not_exist", _FakeController(), None, _FakeItem())
    assert missing.push_custom_event(("note_off", 60, 0)) is False


def test_midi_keyboard_telemetry_pushes_only_on_change():
    """Telemetry (dict+list) must be pushed only when the active-note set
    changes — not on every audio block (93.75 timed pushes/sec was constant
    heap churn)."""
    from plugins.midi_keyboard import MIDIKeyboardNode

    node = MIDIKeyboardNode()

    # Steady state (no events): nothing pushed.
    node.process()
    frame, ok = node.monitor_queue.try_pop()
    assert not ok

    # UI note-on event -> exactly one telemetry push.
    node._ui_queue.try_push(("note_on", 60, 100))
    node.process()
    frame, ok = node.monitor_queue.try_pop()
    assert ok and frame == {"active_notes": [60]}

    # Further quiet blocks: no additional pushes.
    node.process()
    frame, ok = node.monitor_queue.try_pop()
    assert not ok

    # Note-off -> push announces the empty set so the UI clears the keys.
    node._ui_queue.try_push(("note_off", 60, 0))
    node.process()
    frame, ok = node.monitor_queue.try_pop()
    assert ok and frame == {"active_notes": []}