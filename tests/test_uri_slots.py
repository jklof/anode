"""Tests for the uri slot type: creation, pull semantics, graph rules.

URI ports carry file-path strings (.uri) with no per-block data. Consumers
pull the current value via InputSlot.get_uri() (e.g. on a trigger edge);
all file I/O stays on NRT workers.
"""
import pytest
import torch

import plugin_system
from base import InputSlot, Node, OutputSlot, apply_bypass
from core import Graph


class _UriNode(Node):
    category = "Utilities"

    def __init__(self, name=""):
        super().__init__(name)
        self.add_uri_input("uri_in")
        self.add_uri_output("uri_out")
        self.add_input("audio_in")
        self.add_output("audio_out")

    def process(self):
        pass


def test_uri_slots_carry_no_buffers():
    node = _UriNode()
    out = node.outputs["uri_out"]
    assert out.uri == ""
    assert not hasattr(out, "buffer")
    assert not hasattr(out, "packet")
    assert not hasattr(node.inputs["uri_in"], "_scratch")


def test_unknown_slot_type_rejected():
    node = _UriNode()
    with pytest.raises(ValueError):
        OutputSlot("x", node, slot_type="bogus")
    with pytest.raises(ValueError):
        InputSlot("x", node, slot_type="bogus")


def test_get_uri_pull_semantics():
    src, dst = _UriNode(), _UriNode()
    assert dst.inputs["uri_in"].get_uri() == ""  # unconnected
    dst.inputs["uri_in"].connect(src.outputs["uri_out"])
    assert dst.inputs["uri_in"].get_uri() == ""  # nothing published yet
    src.outputs["uri_out"].uri = "/tmp/song.wav"
    assert dst.inputs["uri_in"].get_uri() == "/tmp/song.wav"
    # Non-URI slots never yield paths, even when (mis)wired in tests.
    assert dst.inputs["audio_in"].get_uri() == ""


def test_get_uri_first_connection_wins():
    """Tiebreak for programmatic double-wiring (Graph.connect rejects the
    second wire; direct slot connects bypass that gate)."""
    a, b, dst = _UriNode(), _UriNode(), _UriNode()
    dst.inputs["uri_in"].connect(a.outputs["uri_out"])
    dst.inputs["uri_in"].connect(b.outputs["uri_out"])
    a.outputs["uri_out"].uri = "/tmp/a.wav"
    b.outputs["uri_out"].uri = "/tmp/b.wav"
    assert dst.inputs["uri_in"].get_uri() == "/tmp/a.wav"


def _graph_two():
    g = Graph()
    a, b = _UriNode("a"), _UriNode("b")
    g.add_node(a)
    g.add_node(b)
    return g, a, b


def test_connect_uri_to_uri():
    g, a, b = _graph_two()
    assert g.connect(a.id, "uri_out", b.id, "uri_in") is True
    # Execution order follows the URI dependency.
    order = [n.id for n in g.execution_order]
    assert order.index(a.id) < order.index(b.id)


def test_connect_rejects_type_mismatches():
    g, a, b = _graph_two()
    assert g.connect(a.id, "audio_out", b.id, "uri_in") is False
    assert g.connect(a.id, "uri_out", b.id, "audio_in") is False
    # Control: same-type audio wiring still works.
    assert g.connect(a.id, "audio_out", b.id, "audio_in") is True


def test_connect_rejects_uri_cycle_and_self_loop():
    g, a, b = _graph_two()
    assert g.connect(a.id, "uri_out", b.id, "uri_in") is True
    assert g.connect(b.id, "uri_out", a.id, "uri_in") is False
    assert g.connect(a.id, "uri_out", a.id, "uri_in") is False


def test_second_uri_wire_rejected():
    """A URI input carries one file reference; unlike audio (summed) or
    MIDI (aggregated) a second wire has no merge and would sit dead."""
    g, a, b = _graph_two()
    assert g.connect(a.id, "uri_out", b.id, "uri_in") is True
    c = _UriNode("c")
    g.add_node(c)
    assert g.connect(c.id, "uri_out", b.id, "uri_in") is False
    assert b.inputs["uri_in"].get_uri() == ""  # first wire untouched
    # Re-dragging the identical edge stays a silent no-op success.
    assert g.connect(a.id, "uri_out", b.id, "uri_in") is True
    # After disconnecting, a different source may take the slot.
    b.inputs["uri_in"].disconnect(a.outputs["uri_out"])
    assert g.connect(c.id, "uri_out", b.id, "uri_in") is True


def test_bypass_leaves_uri_intact():
    node = _UriNode()
    node.outputs["uri_out"].uri = "/tmp/keep.wav"
    node.outputs["audio_out"].buffer.fill_(0.5)
    node.params["enabled"].set(False)
    node.params["enabled"].sync()
    apply_bypass(node)
    assert torch.all(node.outputs["audio_out"].buffer == 0.0)
    assert node.outputs["uri_out"].uri == "/tmp/keep.wav"


def test_uri_node_documentation():
    plugin_system.load_plugins("plugins")
    doc = plugin_system.get_node_documentation("YuE2SongGenerator")
    assert doc["outputs"]["song"]["slot_type"] == "uri"
    assert doc["outputs"]["ready"]["slot_type"] == "audio"


def test_full_cover_chain_wires_up():
    """SheetSage.melody -> YuE2.abc_uri, SheetSage.done -> YuE2.trigger_in,
    YuE2.song -> SamplePlayer.uri_in, YuE2.ready -> SamplePlayer.trigger_in:
    the whole cover/generate/play loop connects with matching types and a
    valid execution order."""
    plugin_system.load_plugins("plugins")
    g = Graph()
    nodes = {}
    for key, cls_name in (("sage", "SheetSage2Transcriber"),
                          ("yue2", "YuE2SongGenerator"),
                          ("player", "SamplePlayer")):
        cls = plugin_system.NODE_REGISTRY.get(cls_name)
        assert cls is not None, cls_name
        node = cls()
        g.add_node(node)
        nodes[key] = node
    wires = [(("sage", "melody"), ("yue2", "abc_uri")),
             (("sage", "done"), ("yue2", "trigger_in")),
             (("yue2", "song"), ("player", "uri_in")),
             (("yue2", "ready"), ("player", "trigger_in"))]
    for (src, sport), (dst, dport) in wires:
        assert g.connect(nodes[src].id, sport, nodes[dst].id, dport) is True
    order = [n.id for n in g.execution_order]
    assert order.index(nodes["sage"].id) < order.index(nodes["yue2"].id)
    assert order.index(nodes["yue2"].id) < order.index(nodes["player"].id)
