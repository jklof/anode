"""
Command Pattern implementation for topology changes (undo/redo support).
Only topology changes are undoable: adding, deleting, moving, and connecting nodes.
Parameter changes are NOT undoable.
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


class AddNodeCommand(ICommand):
    """Command to add a node to the graph."""

    def __init__(self, controller, node_type, pos, node_id=None, params=None):
        self.controller = controller
        self.node_type = node_type
        self.pos = pos
        self.node_id = node_id if node_id is not None else str(uuid.uuid4())
        self.params = params

    def execute(self):
        self.controller.engine.push_command(("add", self.node_type, self.node_id, self.pos, self.params))

    def undo(self):
        self.controller.engine.push_command(("del", self.node_id))


class DeleteNodeCommand(ICommand):
    """Command to delete a node from the graph. Implements Memento Pattern for restoration."""

    def __init__(self, controller, node_id):
        self.controller = controller
        self.node_id = node_id
        self.memento = None

    def execute(self):
        # Capture state before deletion
        self.memento = self.controller.create_node_memento(self.node_id)
        self.controller.engine.push_command(("del", self.node_id))

    def undo(self):
        if self.memento is None:
            return

        node_data = self.memento["node_data"]
        connections = self.memento["connections"]

        # Restore the node
        self.controller.engine.push_command(
            ("add", node_data["type"], node_data["id"], node_data["pos"], node_data.get("params"))
        )

        # Restore connections
        for conn in connections:
            self.controller.engine.push_command(
                ("conn", conn["src_id"], conn["src_port"], conn["dst_id"], conn["dst_port"])
            )


class MoveNodeCommand(ICommand):
    """Command to move a node to a new position."""

    def __init__(self, controller, node_id, new_pos, old_pos):
        self.controller = controller
        self.node_id = node_id
        self.new_pos = new_pos
        self.old_pos = old_pos

    def execute(self):
        self.controller.engine.push_command(("move", self.node_id, self.new_pos[0], self.new_pos[1]))

    def undo(self):
        self.controller.engine.push_command(("move", self.node_id, self.old_pos[0], self.old_pos[1]))


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
