"""Tests for MediaPlayer URI input + trigger start (no devices needed).

The streaming worker itself is not exercised here; these tests cover the
audio-thread edge handoff (pending URI + engine command, never worker
lifecycle) and the engine-side switch in on_ui_param_change("playing").
"""
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS


@pytest.fixture(scope="module")
def node_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("MediaPlayerNode")
    assert cls is not None, "MediaPlayerNode not registered"
    return cls


def make_node(node_cls):
    return node_cls()


class _StubNRT:
    def __init__(self):
        self.submitted = []

    def submit(self, node, fn, args, tag=None):
        self.submitted.append((tag, args))


class _StubEngine:
    def __init__(self):
        self.commands = []
        self.nrt = _StubNRT()

    def push_command(self, cmd):
        self.commands.append(cmd)
        return len(self.commands)


def _attach(node):
    from types import SimpleNamespace
    engine = _StubEngine()
    node.graph = SimpleNamespace(engine=engine)
    return engine


def _feed_trigger(node, level):
    trig = torch.full((CHANNELS, BLOCK_SIZE), level, dtype=torch.float32)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def _wire_uri(node, uri):
    from base import OutputSlot
    helper = OutputSlot("helper", node, slot_type="uri")
    helper.uri = uri
    node.inputs["uri_in"].connect(helper)
    return helper


def test_registration(node_cls):
    assert node_cls.category == "I/O"
    assert node_cls.label == "Media Player"
    node = make_node(node_cls)
    assert node.inputs["uri_in"].slot_type == "uri"
    assert node.inputs["trigger_in"].slot_type == "audio"


def test_edge_with_new_uri_requests_switch(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    node.current_path = "/tmp/old.wav"
    _wire_uri(node, "/tmp/new.wav")
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node._pending_uri == "/tmp/new.wav"
    assert engine.commands == [("param", node.id, "playing", True)]
    _feed_trigger(node, 1.0)  # sustained high: no repeat
    assert len(engine.commands) == 1


def test_edge_same_source_only_resumes(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    node.current_path = "/tmp/same.wav"
    _wire_uri(node, "/tmp/same.wav")
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node._pending_uri is None
    assert engine.commands == [("param", node.id, "playing", True)]


def test_edge_empty_uri_requests_no_switch(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    node.current_path = "/tmp/old.wav"
    _wire_uri(node, "")
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node._pending_uri is None
    assert engine.commands == [("param", node.id, "playing", True)]


def test_edge_unwired_resumes(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)
    assert node._pending_uri is None
    assert engine.commands == [("param", node.id, "playing", True)]


def test_edge_without_engine_is_noop(node_cls):
    node = make_node(node_cls)
    _feed_trigger(node, 0.0)
    _feed_trigger(node, 1.0)  # must not raise headless
    assert node._pending_uri is None


def test_playing_handler_switches_to_pending_uri(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    node.current_path = "/tmp/old.wav"
    node._pending_uri = "/tmp/new.wav"
    node.params["playing"].set(True)
    node.params["playing"].sync()
    node.on_ui_param_change("playing")
    assert node._pending_uri is None
    assert node.current_path == "/tmp/new.wav"
    assert engine.nrt.submitted
    tag, args = engine.nrt.submitted[-1]
    assert tag == "restart" and args[0] == "/tmp/new.wav"


def test_playing_handler_no_pending_no_restart(node_cls):
    node = make_node(node_cls)
    engine = _attach(node)
    node.params["playing"].set(True)
    node.params["playing"].sync()
    node.on_ui_param_change("playing")
    assert engine.nrt.submitted == []  # idle node, nothing to do
