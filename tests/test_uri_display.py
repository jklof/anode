"""Tests for UriDisplay: trigger tap, URI latch, copy widget.

No file I/O, no NRT: the node only forwards path strings.
"""
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("UriDisplay")
    assert cls is not None, "UriDisplay not registered"
    return cls


@pytest.fixture(scope="module")
def widget_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.UI_REGISTRY.get("UriDisplay")
    assert cls is not None, "UriDisplay widget not registered"
    return cls


def make_node(node_cls):
    return node_cls()


def _wire_uri(node, uri):
    """Connect the node's uri_in to a scratch URI output carrying uri."""
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs["uri_in"].connect(helper)
    return helper


def _feed_trigger(node, level, channels=CHANNELS):
    trig = torch.full((channels, BLOCK_SIZE), level, dtype=torch.float32)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


# ----------------------------------------------------------------------
# registration / ports
# ----------------------------------------------------------------------
def test_registration(node_cls):
    assert node_cls.category == "Visual & Analysis"
    assert node_cls.label == "URI Display"


def test_ports(node_cls):
    node = make_node(node_cls)
    assert node.inputs["uri_in"].slot_type == "uri"
    assert node.inputs["trigger_in"].slot_type == "audio"
    assert node.outputs["uri_out"].slot_type == "uri"
    assert node.outputs["trigger_out"].slot_type == "audio"
    assert node.outputs["trigger_out"].buffer.shape == (1, BLOCK_SIZE)


# ----------------------------------------------------------------------
# trigger pass-through + edge latch
# ----------------------------------------------------------------------
def test_trigger_passes_through_unchanged(node_cls):
    node = make_node(node_cls)
    _feed_trigger(node, 0.0)
    assert torch.all(node.trigger_out.buffer == 0.0)
    _feed_trigger(node, 1.0)
    assert torch.all(node.trigger_out.buffer == 1.0)
    assert node.trigger_out.buffer.shape == (1, BLOCK_SIZE)


def test_trigger_out_shape_survives_stereo_trigger(node_cls):
    node = make_node(node_cls)
    _feed_trigger(node, 0.5, channels=2)
    assert node.trigger_out.buffer.shape == (1, BLOCK_SIZE)
    assert torch.all(node.trigger_out.buffer == 0.5)


def test_latch_on_rising_edge_only(node_cls):
    node = make_node(node_cls)
    helper = _wire_uri(node, "a.txt")
    assert node.outputs["uri_out"].uri == ""
    _feed_trigger(node, 0.0)
    assert node.outputs["uri_out"].uri == ""  # no edge, no latch
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == "a.txt"
    helper.uri = "b.txt"  # wire changes mid-high: no re-latch
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == "a.txt"
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == "b.txt"


def test_latch_empty_uri_clears(node_cls):
    node = make_node(node_cls)
    helper = _wire_uri(node, "a.txt")
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == "a.txt"
    helper.uri = ""
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == ""


def test_start_resets_edge_state(node_cls):
    """A sustained high after start() re-fires (stale level forgotten)."""
    node = make_node(node_cls)
    _wire_uri(node, "a.txt")
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == "a.txt"
    assert node.get_telemetry()["snapshots"] == 1
    node.start()
    _feed_trigger(node, 1.0)
    assert node.get_telemetry()["snapshots"] == 2


def test_telemetry_reports_latched_and_live(node_cls):
    node = make_node(node_cls)
    _wire_uri(node, "live.txt")
    telem = node.get_telemetry()
    assert telem["uri"] == ""
    assert telem["live_uri"] == "live.txt"
    assert telem["status"] == "Idle"
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    telem = node.get_telemetry()
    assert telem["uri"] == "live.txt"
    assert telem["status"] == "Ready"


def test_no_net_allocation_in_process(node_cls):
    import gc
    node = make_node(node_cls)
    _wire_uri(node, "a.txt")
    _feed_trigger(node, 1.0)
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"


# ----------------------------------------------------------------------
# wiring + save/load
# ----------------------------------------------------------------------
def test_typed_wires_to_sample_player(node_cls):
    """uri_out -> SamplePlayer.uri_in and trigger_out -> trigger_in."""
    from core import Graph
    plugin_system.load_plugins("plugins")
    player_cls = plugin_system.NODE_REGISTRY.get("SamplePlayer")
    assert player_cls is not None
    g = Graph()
    display, player = make_node(node_cls), player_cls()
    g.add_node(display)
    g.add_node(player)
    assert g.connect(display.id, "uri_out", player.id, "uri_in")
    assert g.connect(display.id, "trigger_out", player.id, "trigger_in")
    # Cross-type wires still rejected.
    assert not g.connect(display.id, "trigger_out", player.id, "uri_in")


def test_save_load_keeps_no_stale_uri(node_cls):
    node = make_node(node_cls)
    _wire_uri(node, "a.txt")
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node.outputs["uri_out"].uri == "a.txt"
    node2 = make_node(node_cls)
    node2.load_state(node.to_dict())  # latch is runtime state: not saved
    assert node2.outputs["uri_out"].uri == ""
    assert node2.get_telemetry()["uri"] == ""


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
    def set_parameter(self, name, value):
        raise AssertionError("display widget never sets parameters")

    def create_param_widget(self, name):
        raise AssertionError("display widget has no param widgets")


def test_widget_shows_empty_state(qapp, widget_cls):
    widget = widget_cls(_StubProxy())
    widget.on_telemetry({"uri": "", "live_uri": ""})
    assert widget.btn_copy.text() == "—"
    assert not widget.btn_copy.isEnabled()


def test_widget_shows_uri_on_button(qapp, widget_cls):
    widget = widget_cls(_StubProxy())
    widget.on_telemetry({"uri": "C:/audio/take.wav",
                         "live_uri": "C:/audio/take.wav"})
    assert widget.btn_copy.text() == "C:/audio/take.wav"
    assert widget.btn_copy.toolTip() == "C:/audio/take.wav"
    assert widget.btn_copy.isEnabled()


def test_widget_previews_live_wire_before_trigger(qapp, widget_cls):
    widget = widget_cls(_StubProxy())
    widget.on_telemetry({"uri": "", "live_uri": "C:/audio/live.wav"})
    assert widget.btn_copy.text() == "C:/audio/live.wav"
    assert widget.btn_copy.isEnabled()


def test_widget_copy_button_writes_clipboard(qapp, widget_cls):
    from PySide6.QtWidgets import QApplication
    widget = widget_cls(_StubProxy())
    widget.on_telemetry({"uri": "C:/audio/take.wav", "live_uri": ""})
    widget.btn_copy.click()
    assert QApplication.clipboard().text() == "C:/audio/take.wav"
    assert widget.btn_copy.text() == "Copied!"
    widget._restore_text()
    assert widget.btn_copy.text() == "C:/audio/take.wav"


def test_widget_copy_empty_is_noop(qapp, widget_cls):
    from PySide6.QtWidgets import QApplication
    QApplication.clipboard().setText("sentinel")
    widget = widget_cls(_StubProxy())
    widget.on_telemetry({"uri": "", "live_uri": ""})
    widget._on_copy_pressed()  # button disabled, but direct call is safe
    assert QApplication.clipboard().text() == "sentinel"
