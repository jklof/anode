import pytest
import json
from unittest.mock import patch
from core import Graph
import plugin_system


def test_save_load():
    # Load plugins to populate NODE_REGISTRY
    plugin_system.load_plugins("plugins")

    # Create a graph, add two nodes, connect them
    graph = Graph()

    # Add Gain node
    gain_cls = plugin_system.NODE_REGISTRY.get("Gain")
    gain = gain_cls()
    gain.id = "gain1"
    graph.add_node(gain)

    # Add SineOscillator node
    sine_cls = plugin_system.NODE_REGISTRY.get("SineOscillator")
    sine = sine_cls()
    sine.id = "sine1"
    graph.add_node(sine)

    # Connect sine output to gain input
    graph.connect("sine1", "signal", "gain1", "in")

    # Call to_json()
    json_str = graph.to_json()

    # Create a fresh Graph and load using Engine equivalent logic
    fresh_graph = Graph()
    data = json.loads(json_str)

    for n_data in data["nodes"]:
        cls = plugin_system.NODE_REGISTRY.get(n_data["type"])
        if cls:
            node = cls(n_data["name"])
            node.id = n_data["id"]
            node.load_state(n_data)
            fresh_graph.add_node(node)

    for c in data["connections"]:
        if c["src_id"] in fresh_graph.node_map and c["dst_id"] in fresh_graph.node_map:
            fresh_graph.connect(c["src_id"], c["src_port"], c["dst_id"], c["dst_port"])

    if data.get("clock_id") and data["clock_id"] in fresh_graph.node_map:
        fresh_graph.set_master_clock(fresh_graph.node_map[data["clock_id"]])

    # Assert the new graph has 2 nodes and 1 connection
    assert len(fresh_graph.nodes) == 2
    assert len(fresh_graph.node_map) == 2
    assert "gain1" in fresh_graph.node_map
    assert "sine1" in fresh_graph.node_map

    # Check connections: gain1 in inp should have one connection from sine1 out
    gain_node = fresh_graph.node_map["gain1"]
    assert len(gain_node.inp.connected_outputs) == 1
    assert gain_node.inp.connected_outputs[0].parent.id == "sine1"
    assert gain_node.inp.connected_outputs[0].name == "signal"


def test_undo_metadata_preserved_in_snapshot():
    """Regression: undo/redo parameter snapshots must preserve help/unit
    metadata inside meta so the help system never loses documentation after
    a restore round trip."""
    from core import Graph
    import plugin_system

    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Gain")
    node = cls()
    node.id = "g1"
    graph = Graph()
    graph.add_node(node)

    metadata = node.params["vol"].meta
    assert "help" in metadata and metadata["help"]
    assert "unit" in metadata and metadata["unit"] == "x"

    # Snapshot-style dict restore (as produced by undo/redo) keeps meta intact
    snapshot = {
        "id": node.id,
        "type": "Gain",
        "name": node.name,
        "params": {k: {"value": v.value, "type": v.type, "meta": v.meta} for k, v in node.params.items()},
        "pos": node.pos,
    }
    node.load_state(snapshot)
    restored_meta = node.params["vol"].meta
    assert restored_meta.get("help") == metadata["help"]
    assert restored_meta.get("unit") == "x"


def test_delete_undo_attaches_graph_before_load_state():
    """Regression: DeleteNodeCommand.undo() used to call node.load_state()
    BEFORE the engine attached node.graph, so nodes that submit background
    (NRT) work from load_state() dropped their tasks. The bare node is now
    handed to the engine and load_state runs after graph attachment."""
    import plugin_system
    from core import Engine
    from base import Node as BaseNode
    from commands import DeleteNodeCommand

    events = []
    PROBE = "_GraphAttachProbe"

    class _GraphAttachProbe(BaseNode):
        def __init__(self, name=""):
            super().__init__(name)
            self.add_input("in")

        def load_state(self, data):
            events.append(getattr(self, "graph", None) is not None)

    ProbeNode = _GraphAttachProbe
    plugin_system.NODE_REGISTRY[PROBE] = ProbeNode
    try:
        eng = Engine()
        node = ProbeNode("probe")
        node.id = "p1"
        eng.push_command(("add", node, "p1", (0, 0), None))

        class Ctl:
            engine = eng

        cmd = DeleteNodeCommand(Ctl(), "p1")
        cmd.execute()
        assert "p1" not in eng.graph.node_map

        cmd.undo()
        assert "p1" in eng.graph.node_map
    finally:
        plugin_system.NODE_REGISTRY.pop(PROBE, None)

    assert events == [True], "load_state must run with node.graph attached"


def test_engine_load_respects_clockless_patch():
    """A patch saved without a clock_id must load with no master clock:
    Graph.add_node() auto-assigns the first clock provider, and the load path
    must clear that auto-assignment when the saved clock_id is null."""
    from base import Node, IClockProvider
    from core import Engine

    class _TestClockNode(Node, IClockProvider):
        def __init__(self, name=""):
            Node.__init__(self, name)
            IClockProvider.__init__(self)

        def start_clock(self, tick_callback):
            pass

        def stop_clock(self):
            pass

    plugin_system.NODE_REGISTRY["_TestClockNode"] = _TestClockNode
    try:
        engine = Engine()
        payload = {
            "nodes": [{"type": "_TestClockNode", "id": "clk", "name": "clk"}],
            "connections": [],
            "clock_id": None,
        }
        engine.push_command(("load", json.dumps(payload)))
        assert "clk" in engine.graph.node_map
        assert engine.graph.clock_source is None
        assert engine.graph.node_map["clk"].is_master is False

        # Positive control: a saved clock_id is honored.
        payload["clock_id"] = "clk"
        engine.push_command(("load", json.dumps(payload)))
        assert engine.graph.clock_source is engine.graph.node_map["clk"]
        assert engine.graph.node_map["clk"].is_master is True
    finally:
        plugin_system.NODE_REGISTRY.pop("_TestClockNode", None)



def test_parameter_set_stores_native_python_scalars():
    """Parameter.set() must store native Python scalars, not numpy scalars
    (np.clip() returned np.float64 / np.int64 which leaked into snapshots,
    telemetry and UI state). Values must remain JSON-serializable with the
    plain stdlib encoder."""
    import numpy as np
    from base import Parameter

    p = Parameter(0.5, "float", min=0.0, max=1.0)
    p.set(np.float64(0.7))
    p.sync()
    assert type(p.value) is float and p.value == 0.7
    json.dumps({"v": p.value})

    # Clamping still applies (and clamped results stay native floats)
    p.set(2.5)
    p.sync()
    assert type(p.value) is float and p.value == 1.0
    json.dumps({"v": p.value})

    q = Parameter(5, "int", min=0, max=10)
    q.set(7.9)
    q.sync()
    assert type(q.value) is int and q.value == 7
    json.dumps({"v": q.value})

