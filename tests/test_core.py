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
    """Test cycle detection in graph - cycles rejected at connection time."""
    graph = Graph()
    nodeA = MockNode("A")
    nodeB = MockNode("B")
    graph.add_node(nodeA)
    graph.add_node(nodeB)
    # First connection succeeds
    assert graph.connect("A", "out", "B", "in") is True
    
    # Second connection (creating cycle A -> B -> A) should be rejected
    assert graph.connect("B", "out", "A", "in") is False
    
    # Graph should still have 2 nodes and valid execution order
    assert len(graph.nodes) == 2
    assert len(graph.execution_order) == 2  # Both nodes in valid DAG order
    
    # Test self-loop rejection
    nodeC = MockNode("C")
    graph.add_node(nodeC)
    assert graph.connect("C", "out", "C", "in") is False


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


# --- Undo/Redo Tests for Engine Restore Command ---
from unittest.mock import patch


class MockEngine:
    """Mock engine for testing restore command without audio dependencies"""

    def __init__(self):
        import queue
        self.graph = Graph()
        self.graph.engine = self
        self.nrt = None
        self.running = False
        self._apply_command_called = False
        self._last_command = None
        self._stats_buffer = {}
        self.output_queue = queue.Queue()
        self._active_plan = self.graph.compile_execution_plan()

    def _drain_nrt_all(self):
        pass

    def push_command(self, cmd):
        self._apply_command(cmd)

    def _apply_command(self, cmd):
        self._apply_command_called = True
        self._last_command = cmd
        # Import here to avoid circular imports in test
        from core import Engine

        # Call the actual _apply_command method
        Engine._apply_command(self, cmd)


def test_restore_command_valid_node():
    """Test restore command with valid node data"""
    engine = MockEngine()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)
            self.add_float_param("test_param", 1.0, 0.0, 10.0)
            self.add_int_param("int_param", 5, 0, 100)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Test node data
        node_data = {
            "id": "test-node-123",
            "name": "Test Node",
            "type": "TestNode",
            "pos": (100, 200),
            "params": {"test_param": 7.5, "int_param": 42},
        }

        # Execute restore command
        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify node was added to graph
        assert "test-node-123" in engine.graph.node_map
        node = engine.graph.node_map["test-node-123"]

        # Verify node properties
        assert node.id == "test-node-123"
        assert node.name == "Test Node"
        assert node.__class__.__name__ == "TestNode"
        assert node.pos == (100, 200)
        assert len(node.params) == 2

        # Verify parameters were restored
        assert "test_param" in node.params
        assert node.params["test_param"].get_staging_safe() == 7.5
        assert "int_param" in node.params
        assert node.params["int_param"].get_staging_safe() == 42


def test_restore_command_invalid_node_type():
    """Test restore command with invalid node type"""
    engine = MockEngine()

    # Mock plugin system registry (empty)
    with patch("plugin_system.NODE_REGISTRY", {}):
        # Test node data with invalid type
        node_data = {
            "id": "invalid-node-456",
            "name": "Invalid Node",
            "type": "NonExistentNodeType",
            "pos": (300, 400),
            "params": {},
        }

        # Execute restore command
        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify node was NOT added to graph
        assert "invalid-node-456" not in engine.graph.node_map


def test_restore_command_missing_type_field():
    """Test restore command with missing type field"""
    engine = MockEngine()

    # Test node data missing type field
    node_data = {
        "id": "incomplete-node-789",
        "name": "Incomplete Node",
        # Missing "type" field
        "pos": (500, 600),
        "params": {},
    }

    # Execute restore command
    restore_cmd = ("restore", node_data)
    engine.push_command(restore_cmd)

    # Verify node was NOT added to graph
    assert "incomplete-node-789" not in engine.graph.node_map


def test_restore_command_missing_id_field():
    """Test restore command with missing id field"""
    engine = MockEngine()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Test node data missing id field
        node_data = {"name": "Test Node", "type": "TestNode", "pos": (100, 200), "params": {}}

        # Execute restore command
        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify no nodes were added to graph (should fail gracefully)
        assert len(engine.graph.node_map) == 0


def test_restore_command_empty_params():
    """Test restore command with empty parameters"""
    engine = MockEngine()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)
            self.add_float_param("default_param", 1.0, 0.0, 10.0)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Test node data with empty params
        node_data = {
            "id": "empty-params-node",
            "name": "Empty Params Node",
            "type": "TestNode",
            "pos": (150, 250),
            "params": {},
        }

        # Execute restore command
        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify node was added
        assert "empty-params-node" in engine.graph.node_map
        node = engine.graph.node_map["empty-params-node"]

        # Verify default parameters are still present
        assert "default_param" in node.params
        assert node.params["default_param"].get_staging_safe() == 1.0


def test_restore_command_partial_params():
    """Test restore command with partial parameter restoration"""
    engine = MockEngine()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)
            self.add_float_param("param1", 1.0, 0.0, 10.0)
            self.add_int_param("param2", 5, 0, 100)
            self.add_bool_param("param3", True)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Test node data with only some parameters
        node_data = {
            "id": "partial-params-node",
            "name": "Partial Params Node",
            "type": "TestNode",
            "pos": (200, 300),
            "params": {
                "param1": 3.14,
                "param3": False,
                # param2 not included
            },
        }

        # Execute restore command
        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify node was added
        assert "partial-params-node" in engine.graph.node_map
        node = engine.graph.node_map["partial-params-node"]

        # Verify restored parameters
        assert node.params["param1"].get_staging_safe() == 3.14
        assert node.params["param3"].get_staging_safe() == False

        # Verify non-specified parameters remain at defaults
        assert node.params["param2"].get_staging_safe() == 5


def test_restore_command_position_restoration():
    """Test that node position is correctly restored"""
    engine = MockEngine()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Test node data with specific position
        node_data = {
            "id": "position-test-node",
            "name": "Position Test Node",
            "type": "TestNode",
            "pos": (450, 720),  # Non-default position
            "params": {},
        }

        # Execute restore command
        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify node was added with correct position
        assert "position-test-node" in engine.graph.node_map
        node = engine.graph.node_map["position-test-node"]
        assert node.pos == (450, 720)


def test_restore_command_integration_with_existing_nodes():
    """Test restore command works when graph already has nodes"""
    engine = MockEngine()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # First, add an existing node
        existing_node = TestNode("Existing Node")
        existing_node.id = "existing-node"
        engine.graph.add_node(existing_node)

        # Verify initial state
        assert len(engine.graph.node_map) == 1
        assert "existing-node" in engine.graph.node_map

        # Now restore another node
        node_data = {
            "id": "restored-node",
            "name": "Restored Node",
            "type": "TestNode",
            "pos": (100, 100),
            "params": {},
        }

        restore_cmd = ("restore", node_data)
        engine.push_command(restore_cmd)

        # Verify both nodes exist
        assert len(engine.graph.node_map) == 2
        assert "existing-node" in engine.graph.node_map
        assert "restored-node" in engine.graph.node_map

        # Verify the restored node has correct properties
        restored_node = engine.graph.node_map["restored-node"]
        assert restored_node.name == "Restored Node"
        assert restored_node.pos == (100, 100)


# --- Tests for Updated DeleteNodeCommand with Restore Opcode ---


def test_delete_node_command_with_restore():
    """Test DeleteNodeCommand uses restore opcode for undo"""
    from commands import DeleteNodeCommand

    # Mock controller
    class MockController:
        def __init__(self):
            self.engine = MockEngine()
            self._snapshot_connections = []

        def get_connections_from_snapshot(self):
            return self._snapshot_connections

    controller = MockController()

    # Mock plugin system registry
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)
            self.add_float_param("test_param", 1.0, 0.0, 10.0)

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Create a node and add it to the engine
        node_data = {
            "id": "test-node-for-delete",
            "name": "Test Node for Delete",
            "type": "TestNode",
            "pos": (100, 200),
            "params": {"test_param": 5.0},
        }

        restore_cmd = ("restore", node_data)
        controller.engine.push_command(restore_cmd)

        # Verify node was added
        assert "test-node-for-delete" in controller.engine.graph.node_map

        # Set up mock connections for testing
        controller._snapshot_connections = [
            {"src_id": "test-node-for-delete", "src_port": "out", "dst_id": "other-node", "dst_port": "in"},
            {"src_id": "another-node", "src_port": "out", "dst_id": "test-node-for-delete", "dst_port": "in"},
        ]

        # Create DeleteNodeCommand (now captures state from engine graph)
        delete_cmd = DeleteNodeCommand(controller, "test-node-for-delete")

        # Execute delete
        delete_cmd.execute()

        # Verify node was deleted
        assert "test-node-for-delete" not in controller.engine.graph.node_map

        # Undo delete using restore opcode
        delete_cmd.undo()

        # Verify node was restored using restore command
        assert "test-node-for-delete" in controller.engine.graph.node_map
        restored_node = controller.engine.graph.node_map["test-node-for-delete"]

        # Verify node properties were restored
        assert restored_node.name == "Test Node for Delete"
        assert restored_node.pos == (100, 200)
        assert restored_node.params["test_param"].get_staging_safe() == 5.0


def test_delete_node_command_with_missing_snapshot_data():
    """Test DeleteNodeCommand handles missing node gracefully"""
    from commands import DeleteNodeCommand

    # Mock controller
    class MockController:
        def __init__(self):
            self.engine = MockEngine()
            self._snapshot_connections = []

        def get_connections_from_snapshot(self):
            return self._snapshot_connections

    controller = MockController()

    # Create DeleteNodeCommand for non-existent node
    delete_cmd = DeleteNodeCommand(controller, "nonexistent-node")

    # Execute delete (should not crash)
    delete_cmd.execute()

    # Undo delete with missing data (should not crash)
    delete_cmd.undo()

    # Verify no nodes were added (graceful failure)
    assert len(controller.engine.graph.node_map) == 0


def test_delete_node_command_connection_restoration():
    """Test that DeleteNodeCommand properly restores connections in a real connected graph (NodeA -> NodeB -> NodeC)."""
    from commands import DeleteNodeCommand
    from unittest.mock import patch
    from base import Node

    class TestNode(Node):
        def __init__(self, name=""):
            super().__init__(name)
            self.add_output("out")
            self.add_input("in")

        def process(self):
            pass

    class MockController:
        def __init__(self, engine):
            self.engine = engine

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        engine = MockEngine()
        controller = MockController(engine)

        # 1. Construct real connected graph: NodeA -> NodeB -> NodeC
        node_a = TestNode("NodeA")
        node_a.id = "node-a"
        node_b = TestNode("NodeB")
        node_b.id = "node-b"
        node_c = TestNode("NodeC")
        node_c.id = "node-c"

        engine.push_command(("restore", (node_a.to_dict(), node_a)))
        engine.push_command(("restore", (node_b.to_dict(), node_b)))
        engine.push_command(("restore", (node_c.to_dict(), node_c)))

        engine.push_command(("conn", "node-a", "out", "node-b", "in"))
        engine.push_command(("conn", "node-b", "out", "node-c", "in"))

        # Verify initial connections
        assert len(engine.graph.node_map["node-b"].inputs["in"].connected_outputs) == 1
        assert engine.graph.node_map["node-b"].inputs["in"].connected_outputs[0].parent.id == "node-a"
        assert len(engine.graph.node_map["node-c"].inputs["in"].connected_outputs) == 1
        assert engine.graph.node_map["node-c"].inputs["in"].connected_outputs[0].parent.id == "node-b"

        # 2. Create and execute delete command on NodeB
        delete_cmd = DeleteNodeCommand(controller, "node-b")
        delete_cmd.execute()

        # 3. Assert NodeB and its connections are gone
        assert "node-b" not in engine.graph.node_map
        assert len(engine.graph.node_map["node-c"].inputs["in"].connected_outputs) == 0

        # 4. Undo delete
        delete_cmd.undo()

        # 5. Assert NodeB is restored AND both connections are verified present in engine.graph
        assert "node-b" in engine.graph.node_map
        restored_b = engine.graph.node_map["node-b"]
        current_c = engine.graph.node_map["node-c"]

        assert len(restored_b.inputs["in"].connected_outputs) == 1
        assert restored_b.inputs["in"].connected_outputs[0].parent.id == "node-a"
        assert len(current_c.inputs["in"].connected_outputs) == 1
        assert current_c.inputs["in"].connected_outputs[0].parent.id == "node-b"


def test_lazy_evaluation_topological_sort():
    """Verify that execution_order calculation is lazily evaluated."""
    graph = Graph()
    node = MockNode("A")
    graph.add_node(node)

    assert graph._order_dirty is True
    # Reading execution_order triggers calculation and clears dirty flag
    order = graph.execution_order
    assert len(order) == 1
    assert graph._order_dirty is False

    # Mutating graph sets dirty flag back to True
    graph.recalculate_order()
    assert graph._order_dirty is True

    # Reading execution_order again clears it
    assert graph.execution_order == [node]
    assert graph._order_dirty is False


def test_parameter_sync_tensor_and_numpy_array():
    """Verify that Parameter.sync handles tensors and numpy arrays without crashing."""
    import torch
    import numpy as np
    from base import Parameter

    # Test PyTorch Tensor
    t_param = Parameter(1.0, "float")
    t_param._staging = torch.tensor([2.0, 3.0])
    # Should sync successfully without ValueError
    t_param.sync()
    assert isinstance(t_param.value, torch.Tensor)

    # Test Numpy Array
    np_param = Parameter(1.0, "float")
    np_param._staging = np.array([4.0, 5.0])
    # Should sync successfully without ValueError
    np_param.sync()
    assert isinstance(np_param.value, np.ndarray)


def test_app_controller_delete_node_single_unselected_node():
    """Regression: AppController.delete_node() checked cmd.node_data BEFORE
    execute(), but the memento holder is only filled by the engine when the
    delete is processed — so single-node deletion silently failed."""
    import plugin_system
    from controller import AppController
    from PySide6.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        _app = QCoreApplication([])

    plugin_system.load_plugins("plugins")
    gain_cls = plugin_system.NODE_REGISTRY.get("Gain")

    ctl = AppController()
    try:
        node = gain_cls()
        node.id = "n1"
        ctl.engine.push_command(("add", node, "n1", (0, 0), None))
        ctl._latest_snapshot = {"nodes": [{"id": "n1"}]}

        ctl.delete_node("n1")
        assert "n1" not in ctl.engine.graph.node_map
        assert ctl._latest_snapshot.get("nodes") == []
        assert len(ctl.history.undo_stack) == 1

        # Unknown node: no crash, no phantom history entry
        ctl.delete_node("does-not-exist")
        assert len(ctl.history.undo_stack) == 1

        # Undo restores the node (with its authoritative memento)
        ctl.undo()
        assert "n1" in ctl.engine.graph.node_map
    finally:
        ctl.poll_timer.stop()


def test_telemetry_dict_ring_buffer_no_stale_keys():
    """Regression: push() must clear the reused slot dict before updating it,
    otherwise keys from earlier frames leak into the frame the consumer pops
    when the pushed dicts vary in keys."""
    from base import TelemetryDictRingBuffer

    rb = TelemetryDictRingBuffer(capacity=4)
    assert rb.push({"rms": 0.5, "peak": 0.9})
    assert rb.push({"rms": 0.3})

    latest = rb.pop_latest()
    assert latest == {"rms": 0.3}, f"stale keys leaked: {latest}"

    # Empty dict must fully replace the previous frame's keys.
    assert rb.push({})
    assert rb.pop_latest() == {}

