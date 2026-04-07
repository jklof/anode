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
    """Command to add a node to the graph."""

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
            node = cls()
            self.controller.engine.push_command(("add", node, self.node_id, self.pos, self.params))

    def undo(self):
        self.controller.engine.push_command(("del", self.node_id))


class DeleteNodeCommand(ICommand):
    """
    Command to delete a node.
    Captures full state immediately upon creation to prevent race conditions.
    """

    def __init__(self, controller, node_id, snapshot_data):
        self.controller = controller
        self.node_id = node_id
        # We pass the data in via constructor so it is captured
        # AT THE MOMENT of user action, not 30ms later in execute()
        self.node_data = snapshot_data
        self.connections = []

        # Find connections associated with this node in the snapshot
        # This handles the "Implicit Disconnect" restoration
        if self.node_data:
            snapshot_conns = self.controller.get_connections_from_snapshot()
            for c in snapshot_conns:
                if c["src_id"] == node_id or c["dst_id"] == node_id:
                    self.connections.append(c)

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
