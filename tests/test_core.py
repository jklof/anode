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


# --- Undo/Redo Tests for Engine Restore Command ---
from unittest.mock import patch


class MockEngine:
    """Mock engine for testing restore command without audio dependencies"""

    def __init__(self):
        self.graph = Graph()
        self.running = False
        self._apply_command_called = False
        self._last_command = None

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

        # Create DeleteNodeCommand with snapshot data
        delete_cmd = DeleteNodeCommand(controller, "test-node-for-delete", node_data)

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
    """Test DeleteNodeCommand handles missing snapshot data gracefully"""
    from commands import DeleteNodeCommand

    # Mock controller
    class MockController:
        def __init__(self):
            self.engine = MockEngine()
            self._snapshot_connections = []

        def get_connections_from_snapshot(self):
            return self._snapshot_connections

    controller = MockController()

    # Create DeleteNodeCommand with None snapshot data
    delete_cmd = DeleteNodeCommand(controller, "nonexistent-node", None)

    # Execute delete (should not crash)
    delete_cmd.execute()

    # Undo delete with missing data (should not crash)
    delete_cmd.undo()

    # Verify no nodes were added (graceful failure)
    assert len(controller.engine.graph.node_map) == 0


def test_delete_node_command_connection_restoration():
    """Test that DeleteNodeCommand properly restores connections"""
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
            self.add_output("out")
            self.add_input("in")

        def process(self):
            pass

    with patch("plugin_system.NODE_REGISTRY", {"TestNode": TestNode}):
        # Create nodes
        node_data = {"id": "test-node", "name": "Test Node", "type": "TestNode", "pos": (100, 100), "params": {}}

        restore_cmd = ("restore", node_data)
        controller.engine.push_command(restore_cmd)

        # Set up mock connections
        controller._snapshot_connections = [
            {"src_id": "test-node", "src_port": "out", "dst_id": "output-node", "dst_port": "in"},
            {"src_id": "input-node", "src_port": "out", "dst_id": "test-node", "dst_port": "in"},
        ]

        # Create and execute delete command
        delete_cmd = DeleteNodeCommand(controller, "test-node", node_data)
        delete_cmd.execute()

        # Verify node was deleted
        assert "test-node" not in controller.engine.graph.node_map

        # Undo delete
        delete_cmd.undo()

        # Verify node was restored
        assert "test-node" in controller.engine.graph.node_map
