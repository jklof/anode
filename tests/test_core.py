import pytest
from core import Graph
from base import Node, IClockProvider


class MockNode:
    def __init__(self, name="MockNode"):
        self.id = name  # simplified for test, usually UUID
        self.name = name
        self.pos = (0, 0)
        self.error_msg = None
        self.inputs = {"in": MockInputSlot("in", self)}
        self.outputs = {"out": MockOutputSlot("out", self)}
        self.params = {}
        self.monitor_queue = None

    def sync(self):
        pass

    def process(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def on_ui_param_change(self, param_name):
        pass

    def to_dict(self):
        return {"id": self.id, "type": "MockNode", "name": self.name, "params": {}, "pos": self.pos}


class MockInputSlot:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.param_name = None
        self.connected_outputs = []

    def connect(self, output):
        self.connected_outputs.append(output)

    def disconnect(self, target=None):
        if target is None:
            self.connected_outputs = []
        else:
            if target in self.connected_outputs:
                self.connected_outputs.remove(target)


class MockOutputSlot:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent


def test_add_node():
    graph = Graph()
    node = MockNode("test_node")
    graph.add_node(node)
    assert len(graph.nodes) == 1
    assert graph.node_map["test_node"] is node
    assert graph.execution_order == [node]


def test_remove_node():
    graph = Graph()
    node = MockNode("test_node")
    graph.add_node(node)
    graph.remove_node("test_node")
    assert len(graph.nodes) == 0
    assert "test_node" not in graph.node_map
    assert graph.execution_order == []


def test_connect():
    graph = Graph()
    node1 = MockNode("node1")
    node2 = MockNode("node2")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.connect("node1", "out", "node2", "in")
    assert len(node2.inputs["in"].connected_outputs) == 1
    assert node2.inputs["in"].connected_outputs[0] is node1.outputs["out"]


def test_disconnect():
    graph = Graph()
    node1 = MockNode("node1")
    node2 = MockNode("node2")
    graph.add_node(node1)
    graph.add_node(node2)
    graph.connect("node1", "out", "node2", "in")
    graph.disconnect("node1", "out", "node2", "in")
    assert len(node2.inputs["in"].connected_outputs) == 0


def test_recalculate_order_chain():
    graph = Graph()
    nodeA = MockNode("A")
    nodeB = MockNode("B")
    nodeC = MockNode("C")
    graph.add_node(nodeA)
    graph.add_node(nodeB)
    graph.add_node(nodeC)
    graph.connect("A", "out", "B", "in")
    graph.connect("B", "out", "C", "in")

    order_ids = [n.id for n in graph.execution_order]
    does_A_come_before_B = order_ids.index("A") < order_ids.index("B")
    does_B_come_before_C = order_ids.index("B") < order_ids.index("C")
    assert does_A_come_before_B and does_B_come_before_C


def test_cycle_detection():
    graph = Graph()
    nodeA = MockNode("A")
    nodeB = MockNode("B")
    graph.add_node(nodeA)
    graph.add_node(nodeB)
    graph.connect("A", "out", "B", "in")

    # Try to create cycle A -> B -> A
    graph.connect("B", "out", "A", "in")

    # Should have logged warning but not crash
    # For the test, just ensure it doesn't crash and graph is still operational
    assert len(graph.nodes) == 2
    assert len(graph.execution_order) == 2  # Even in cycle, order should be calculated


def test_clock_switching():
    graph = Graph()

    # Mock two clock providers
    class ClockNode(Node, IClockProvider):
        def __init__(self, name):
            Node.__init__(self, name)
            IClockProvider.__init__(self)

        def start_clock(self):
            pass

        def stop_clock(self):
            pass

        def wait_for_sync(self):
            pass

    c1 = ClockNode("Clock 1")
    c2 = ClockNode("Clock 2")

    graph.add_node(c1)
    graph.add_node(c2)

    # First added node usually becomes default
    assert graph.clock_source == c1
    assert c1.is_master
    assert not c2.is_master

    # Switch to C2
    graph.set_master_clock(c2)

    assert graph.clock_source == c2
    assert c2.is_master
    assert not c1.is_master

    # Switch back
    graph.set_master_clock(c1)
    assert graph.clock_source == c1
