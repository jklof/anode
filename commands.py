"""
Command Pattern implementation for topology changes (undo/redo support).
Updated to support Compound Commands (Macros) and Batch operations.
"""

from abc import ABC, abstractmethod
import uuid


class ICommand(ABC):
    """Abstract base class for all commands."""

    @abstractmethod
    def execute(self):
        """Execute the command."""
        pass

    @abstractmethod
    def undo(self):
        """Undo the command."""
        pass


class CompoundCommand(ICommand):
    """Executes a list of commands as a single atomic unit (Macro)."""

    def __init__(self, name="Macro"):
        self.name = name
        self.commands = []

    def add(self, command):
        self.commands.append(command)

    def execute(self):
        for cmd in self.commands:
            cmd.execute()

    def undo(self):
        # Undo in reverse order
        for cmd in reversed(self.commands):
            cmd.undo()


class AddNodeCommand(ICommand):
    """
    Command to add a node to the graph.
    Note: params dictionary is expected in the snapshot format:
    {"param_name": {"value": val, "type": ptype, "meta": pmeta}}
    """

    def __init__(self, controller, node_type, pos, node_id=None, params=None):
        self.controller = controller
        self.node_type = node_type
        self.pos = pos
        self.node_id = node_id if node_id is not None else str(uuid.uuid4())
        self.params = params

    def execute(self):
        import plugin_system

        cls = plugin_system.NODE_REGISTRY.get(self.node_type)
        if cls:
            # We instantiate the Node on the UI thread to perform any heavy initialization
            # (e.g. PyTorch tensors, C++ ctypes load_library) prior to graph insertion.
            # Passing this distinct, non-shared object across the thread queue is safe in CPython
            # because no other thread holds a reference to it yet.
            node = cls()
            self.controller.engine.push_command(("add", node, self.node_id, self.pos, self.params))

    def undo(self):
        self.controller.engine.push_command(("del", self.node_id))


class DeleteNodeCommand(ICommand):
    """
    Command to delete a node.
    Captures full authoritative state from the engine graph at creation time.
    Does NOT rely on the asynchronous UI snapshot cache (_latest_snapshot).
    """

    def __init__(self, controller, node_id):
        self.controller = controller
        self.node_id = node_id
        self.node_data = None
        self.connections = []

        # Capture authoritative state directly from engine graph (synchronous)
        # This works whether engine is running or stopped since we're on the UI thread
        # and the engine graph is only mutated on the engine thread via commands.
        # When engine is running, we read from the graph which reflects all applied commands.
        engine = controller.engine
        graph = engine.graph
        node = graph.node_map.get(node_id)
        if node:
            # Capture complete node state including params, connections
            self.node_data = node.to_dict()
            # Also capture full param snapshot (type, meta) for exact restoration
            for k, p in node.params.items():
                if k in self.node_data.get("params", {}):
                    self.node_data["params"][k] = {
                        "value": p.get_staging_safe(),
                        "type": p.type,
                        "meta": p.meta
                    }

            # Capture connections from authoritative graph
            for dst in graph.nodes:
                for d_port, inp in dst.inputs.items():
                    for out in inp.connected_outputs:
                        if out.parent.id == node_id or dst.id == node_id:
                            self.connections.append({
                                "src_id": out.parent.id,
                                "src_port": out.name,
                                "dst_id": dst.id,
                                "dst_port": d_port
                            })

    def execute(self):
        self.controller.engine.push_command(("del", self.node_id))

    def undo(self):
        if not self.node_data:
            return

        # 1. Restore the Node using the robust 'restore' opcode
        import plugin_system

        cls = plugin_system.NODE_REGISTRY.get(self.node_data["type"])
        if cls:
            node = cls(self.node_data["name"])
            node.id = self.node_data.get("id", str(uuid.uuid4()))
            node.load_state(self.node_data)
            self.controller.engine.push_command(("restore", (self.node_data, node)))

        # 2. Restore the connections that were implicitly removed
        for c in self.connections:
            self.controller.engine.push_command(("conn", c["src_id"], c["src_port"], c["dst_id"], c["dst_port"]))


class MultiMoveNodeCommand(ICommand):
    """
    Command to move multiple nodes at once.
    moves_dict: { node_id: (new_pos, old_pos) }
    """

    def __init__(self, controller, moves_dict):
        self.controller = controller
        self.moves_dict = moves_dict

    def execute(self):
        for node_id, (new_pos, _) in self.moves_dict.items():
            self.controller.engine.push_command(("move", node_id, new_pos[0], new_pos[1]))

    def undo(self):
        for node_id, (_, old_pos) in self.moves_dict.items():
            self.controller.engine.push_command(("move", node_id, old_pos[0], old_pos[1]))


class ConnectCommand(ICommand):
    """Command to connect two nodes."""

    def __init__(self, controller, src_id, src_port, dst_id, dst_port):
        self.controller = controller
        self.src_id = src_id
        self.src_port = src_port
        self.dst_id = dst_id
        self.dst_port = dst_port

    def execute(self):
        self.controller.engine.push_command(("conn", self.src_id, self.src_port, self.dst_id, self.dst_port))

    def undo(self):
        self.controller.engine.push_command(("disconn", self.src_id, self.src_port, self.dst_id, self.dst_port))


class DisconnectCommand(ICommand):
    """Command to disconnect two nodes."""

    def __init__(self, controller, src_id, src_port, dst_id, dst_port):
        self.controller = controller
        self.src_id = src_id
        self.src_port = src_port
        self.dst_id = dst_id
        self.dst_port = dst_port

    def execute(self):
        self.controller.engine.push_command(("disconn", self.src_id, self.src_port, self.dst_id, self.dst_port))

    def undo(self):
        self.controller.engine.push_command(("conn", self.src_id, self.src_port, self.dst_id, self.dst_port))
