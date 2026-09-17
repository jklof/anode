"""Tests for the ABCPlayer score-to-MIDI node (no GPU, no Qt).

Scheduling is driven by calling process() directly with slot-level wiring
(no engine needed); the NRT worker is driven by hand like the other
offline-load tests.
"""
import pytest

import plugin_system
from base import Node


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("ABCPlayer")
    assert cls is not None, "ABCPlayer not registered"
    return cls


def make_node(node_cls):
    return node_cls()


class _Src(Node):
    category = "Utilities"

    def __init__(self):
        super().__init__()
        self.add_output("out")
        self.add_uri_output("song")

    def process(self):
        pass


def _wire_trigger(node, level=1.0):
    src = _Src()
    src.outputs["out"].buffer.fill_(level)
    node.inputs["trigger_in"].connect(src.outputs["out"])
    return src


def _wire_uri(node, uri):
    src = _Src()
    src.outputs["song"].uri = uri
    node.inputs["uri_in"].connect(src.outputs["song"])
    return src


def _load_text(node, tmp_path, text="X:1\nL:1/4\nQ:120\nK:C\nC2 D2|\n",
               name="s.abc"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    score = node._load_score_nrt(str(path))
    node.on_nrt_complete("load", True, score)
    return score


def test_registration(node_cls):
    assert node_cls.category == "MIDI"
    assert node_cls.label == "ABC Score Player"


def test_ports_typed(node_cls):
    node = make_node(node_cls)
    assert node.inputs["trigger_in"].slot_type == "audio"
    assert node.inputs["uri_in"].slot_type == "uri"
    assert node.outputs["midi_out"].slot_type == "midi"


def test_load_worker_rejects_empty(node_cls, tmp_path):
    node = make_node(node_cls)
    path = tmp_path / "e.abc"
    path.write_text("X:1\nK:C\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no playable notes"):
        node._load_score_nrt(str(path))
    node.on_nrt_complete("load", False, RuntimeError("boom"))
    assert node._score is None and node.error_msg is not None


def _drain(node, nblocks):
    """Run nblocks process() calls, collecting all MIDI messages."""
    pytest.importorskip("mido")
    out = []
    for _ in range(nblocks):
        node.process()
        out.extend((msg.type, msg.note, off)
                   for off, msg in node.midi_out.packet.messages)
    return out


def test_trigger_plays_scale_then_stops(node_cls, tmp_path):
    pytest.importorskip("mido")
    node = make_node(node_cls)
    _load_text(node, tmp_path)  # C(60) beats 0-2, D(62) beats 2-4 @120bpm
    _wire_trigger(node)  # held high: first block is the rising edge
    events = _drain(node, 401)  # ~8.5 s > 2 s score
    assert node._is_playing is False  # no loop: stops at the end
    ons = [(t, n) for t, n, _ in events if t == "note_on"]
    offs = [(t, n) for t, n, _ in events if t == "note_off"]
    assert [n for _, n in ons] == [60, 62]
    assert sorted(n for _, n in offs) == [60, 62]
    # Packet anti-ghosting: idle blocks emit nothing.
    node.process()
    assert node.midi_out.packet.messages == []


def test_loop_replays(node_cls, tmp_path):
    pytest.importorskip("mido")
    node = make_node(node_cls)
    _load_text(node, tmp_path)
    node.params["loop"].set(True)
    node.params["loop"].sync()
    _wire_trigger(node)  # held high: first block is the rising edge
    events = _drain(node, 401)
    assert node._is_playing is True
    assert [n for t, n, _ in events if t == "note_on"].count(60) >= 2


def test_uri_override_loads_on_trigger(node_cls, tmp_path):
    node = make_node(node_cls)
    wired = tmp_path / "wired.abc"
    wired.write_text("X:1\nL:1/4\nK:C\nE2|\n", encoding="utf-8")
    _wire_uri(node, str(wired))
    _wire_trigger(node)
    node.process()  # edge with no graph/engine: submit is a safe no-op
    assert node._pending_restart is True
    # Engine would now run the worker; drive it by hand.
    score = node._load_score_nrt(str(wired))
    node.on_nrt_complete("load", True, score)
    assert node._is_playing is True
    assert node.outputs["midi_out"] is not None


def test_param_load_waits_for_trigger(node_cls, tmp_path):
    node = make_node(node_cls)
    _load_text(node, tmp_path)
    node._pending_restart = False
    node.on_nrt_complete("load", True, node._score)
    assert node._is_playing is False


def test_telemetry_reports_position(node_cls, tmp_path):
    pytest.importorskip("mido")
    node = make_node(node_cls)
    _load_text(node, tmp_path, "X:1\nL:1/4\nK:C\n% verse\nC2|\n")
    _wire_trigger(node)  # held high: first block is the rising edge
    node.process()
    telem = node.get_telemetry()
    assert telem["status"] == "Playing"
    assert "verse" in telem["audio"]
