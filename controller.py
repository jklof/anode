from PySide6.QtCore import QObject, Signal, QTimer
import logging
from core import Engine
from commands import (
    AddNodeCommand,
    DeleteNodeCommand,
    MultiMoveNodeCommand,
    ConnectCommand,
    DisconnectCommand,
    CompoundCommand,
)


class CommandHistory:
    """Handles undo/redo functionality using the Command Pattern."""

    def __init__(self, max_length=50):
        self.undo_stack = []
        self.redo_stack = []
        self.max_length = max_length

    def push(self, cmd):
        """Add a command to the undo stack and clear redo stack."""
        self.redo_stack.clear()
        self.undo_stack.append(cmd)

        if len(self.undo_stack) > self.max_length:
            self.undo_stack.pop(0)

    def undo(self):
        """Undo the last command."""
        if not self.undo_stack:
            return

        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)

    def redo(self):
        """Redo the last undone command."""
        if not self.redo_stack:
            return

        cmd = self.redo_stack.pop()
        cmd.execute()
        self.undo_stack.append(cmd)


class AppController(QObject):
    graphUpdated = Signal(dict)
    telemetryUpdated = Signal(dict)
    parameterUpdated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.engine = Engine()
        self.history = CommandHistory()
        self._latest_snapshot = {}

        self.poll_timer = QTimer()
        self.poll_timer.interval = 30
        self.poll_timer.timeout.connect(self.check_engine_messages)
        self.poll_timer.start()

    def check_engine_messages(self):
        while not self.engine.output_queue.empty():
            try:
                msg = self.engine.output_queue.get_nowait()
                if msg.get("type") == "telemetry":
                    self.telemetryUpdated.emit(msg)
                elif msg.get("type") == "param_update":
                    self.parameterUpdated.emit(msg)
                else:
                    self._latest_snapshot = msg  # cache latest snapshot
                    self.graphUpdated.emit(msg)
            except Exception:
                logging.exception("Error processing engine message")

    def start_audio(self):
        self.engine.start()

    def stop_audio(self):
        self.engine.stop()

    def create_node_memento(self, node_id):
        """
        Create a memento from the cached UI snapshot.
        SAFE: No audio thread access, no blocking.
        """
        snapshot = self._latest_snapshot

        # Find node in cached snapshot
        node_data = None
        for n in snapshot.get("nodes", []):
            if n["id"] == node_id:
                node_data = n.copy()
                break

        if not node_data:
            return None

        # Find all connections involving this node
        connections = []
        for conn in snapshot.get("connections", []):
            if conn["src_id"] == node_id or conn["dst_id"] == node_id:
                connections.append(conn.copy())

        return {
            "node_data": node_data,
            "connections": connections,
        }

    # -------------------------------------------------------------------------
    # Topology Methods (Undoable)
    # -------------------------------------------------------------------------

    def add_node(self, node_type, pos, node_id=None, params=None):
        """Add a node to the graph."""
        cmd = AddNodeCommand(self, node_type, pos, node_id=node_id, params=params)
        cmd.execute()
        self.history.push(cmd)
        return cmd.node_id

    def delete_node(self, node_id):
        """Delete a node from the graph."""
        cmd = DeleteNodeCommand(self, node_id)
        cmd.execute()
        self.history.push(cmd)

    def move_nodes(self, moves_dict):
        """
        Move multiple nodes in one undo step.
        moves_dict: { node_id: (new_pos, old_pos) }
        """
        if not moves_dict:
            return
        cmd = MultiMoveNodeCommand(self, moves_dict)
        cmd.execute()
        self.history.push(cmd)

    def delete_selection(self, node_ids, connection_tuples):
        """
        Deletes a list of nodes and specific connections atomically.
        connection_tuples: list of (sid, sp, did, dp)
        """
        macro = CompoundCommand("Delete Selection")

        # 1. Explicitly disconnect selected wires first
        for c_data in connection_tuples:
            macro.add(DisconnectCommand(self, *c_data))

        # 2. Delete nodes (Engine handles implicit disconnection,
        # but DeleteNodeCommand memento handles restoration)
        for nid in node_ids:
            macro.add(DeleteNodeCommand(self, nid))

        if macro.commands:
            macro.execute()
            self.history.push(macro)

    def paste_structure(self, nodes_data, connections_data):
        """
        Pastes a set of nodes and connections atomically.
        """
        macro = CompoundCommand("Paste")

        # 1. Add Nodes
        for n in nodes_data:
            macro.add(AddNodeCommand(self, n["type"], n["pos"], node_id=n["id"], params=n["params"]))

        # 2. Add Connections
        for c in connections_data:
            macro.add(ConnectCommand(self, c["src_id"], c["src_port"], c["dst_id"], c["dst_port"]))

        if macro.commands:
            macro.execute()
            self.history.push(macro)

    def connect_nodes(self, src_id, src_port, dst_id, dst_port):
        """Connect two nodes."""
        cmd = ConnectCommand(self, src_id, src_port, dst_id, dst_port)
        cmd.execute()
        self.history.push(cmd)

    def disconnect_nodes(self, src_id, src_port, dst_id, dst_port):
        """Disconnect two nodes."""
        cmd = DisconnectCommand(self, src_id, src_port, dst_id, dst_port)
        cmd.execute()
        self.history.push(cmd)

    # -------------------------------------------------------------------------
    # Non-Undoable Methods
    # -------------------------------------------------------------------------

    def set_master_clock(self, node_id):
        """Set the master clock node (not undoable)."""
        self.engine.push_command(("clock", node_id))

    def set_parameter(self, node_id, param_name, value):
        """Set a parameter value (not undoable)."""
        self.engine.push_command(("param", node_id, param_name, value))

    def save(self, filename):
        """Save the graph to a file."""
        if not filename:
            return
        self.engine.push_command(("save", filename))

    def load(self, filename):
        """Load a graph from a file."""
        if not filename:
            return
        try:
            with open(filename, "r") as f:
                json_str = f.read()
                self.engine.push_command(("load", json_str))
                self.history = CommandHistory()
        except Exception as e:
            print(f"Controller Load Error: {e}")

    def clear(self):
        """Clear the graph."""
        self.engine.push_command(("clear",))
        self.history = CommandHistory()

    def reload_plugins(self):
        """Reload all plugins."""
        self.engine.push_command(("reload",))

    # -------------------------------------------------------------------------
    # Undo/Redo
    # -------------------------------------------------------------------------

    def undo(self):
        """Undo the last command."""
        self.history.undo()

    def redo(self):
        """Redo the last undone command."""
        self.history.redo()
