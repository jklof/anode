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
