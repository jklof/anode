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
        # Instantiate the node OFF the real-time thread. When the engine is
        # running, commands are queued to the audio thread; passing a type
        # name would make the audio thread run cls() (ctypes.CDLL loads,
        # FIR design, etc.). Pre-instantiating here (UI/control thread) keeps
        # engine-side graph insertion O(1).
        import plugin_system

        cls = plugin_system.NODE_REGISTRY.get(self.node_type)
        node = cls() if cls else None
        self.cmd_id = self.controller.engine.push_command(("add", node, self.node_id, self.pos, self.params))

    def undo(self):
        self.controller.engine.push_command(("del", self.node_id))


class DeleteNodeCommand(ICommand):
    """
    Command to delete a node.

    The undo memento is NOT captured at construction time (which would read
    the live graph from the UI thread and can race with queued commands).
    Instead, execute() passes a holder dict with the command; the authoritative
    command executor (engine/control side) fills it at the exact moment the
    delete is processed, so undo restores state as of that point.
    Does NOT rely on the asynchronous UI snapshot cache (_latest_snapshot).
    """

    def __init__(self, controller, node_id):
        self.controller = controller
        self.node_id = node_id
        self.cmd_id = None
        self._holder = {}
        # Pre-capture the immutable type/name metadata now, on the UI/control
        # thread, from the controller's snapshot. This lets undo()
        # pre-instantiate the bare node OFF the audio thread (AGENTS.md §3)
        # even when Undo is invoked before the queued 'del' has been
        # processed and the holder is still empty. The mutable memento
        # (params/state/connections) still comes from the authoritative
        # holder, filled by the engine at the exact moment the delete is
        # applied.
        snapshot = None
        get_data = getattr(controller, "get_node_data", None)
        if callable(get_data):
            try:
                snapshot = get_data(node_id)
            except Exception:
                snapshot = None
        if snapshot is None:
            # Fallback: read the controller's latest UI snapshot cache
            # directly (read-only; the authoritative memento still comes
            # from the engine-side holder at delete time).
            latest = getattr(controller, "_latest_snapshot", None)
            if latest:
                for n in latest.get("nodes", []):
                    if n.get("id") == node_id:
                        snapshot = n
                        break
        node_data = snapshot or {}
        self.node_type = node_data.get("type")
        self.node_name = node_data.get("name", "")
        if not self.node_type:
            # Last-resort fallback for headless/mock controllers with neither
            # get_node_data() nor a snapshot cache: read the live node object
            # (class name is the plugin registry key). Read-only.
            engine = getattr(controller, "engine", None)
            graph = getattr(engine, "graph", None)
            node_map = getattr(graph, "node_map", None)
            if node_map and node_id in node_map:
                node = node_map[node_id]
                self.node_type = node.__class__.__name__
                self.node_name = getattr(node, "name", "")

    @property
    def node_data(self):
        """Authoritative node memento (filled by the engine at delete time)."""
        return self._holder.get("node")

    @property
    def connections(self):
        return self._holder.get("connections", [])

    def execute(self):
        self.cmd_id = self.controller.engine.push_command(("del", self.node_id, self._holder))

    def undo(self):
        # NOTE: no early return when the holder is still empty. When the
        # engine is running, execute() queued ('del', id, holder) and the
        # holder is only filled when the audio thread applies it. Undo may
        # run before that — but commands are applied strictly in FIFO
        # order, so the 'restore' pushed below is ALWAYS applied after the
        # 'del' has populated the holder. The restore handler therefore
        # reads the authoritative memento at the right time.
        #
        # The bare node is pre-instantiated HERE (UI/control thread) using
        # the metadata captured in __init__, never on the audio thread
        # (AGENTS.md §3: cls() may load native libraries via ctypes.CDLL,
        # design FIR filters, or allocate large structures).
        import plugin_system

        holder_node = self._holder.get("node") or {}
        node_type = self.node_type or holder_node.get("type")
        node_name = self.node_name or holder_node.get("name", "")
        if not node_type:
            return

        cls = plugin_system.NODE_REGISTRY.get(node_type)
        if cls:
            node = cls(node_name)
            node.id = self.node_id
            # Pass the holder itself as the payload; the 'restore' handler
            # unwraps holder["node"] / holder["connections"] once the 'del'
            # has filled it.
            self.controller.engine.push_command(("restore", (self._holder, node)))


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
        self.cmd_id = None

    def execute(self):
        self.cmd_id = self.controller.engine.push_command(
            ("conn", self.src_id, self.src_port, self.dst_id, self.dst_port)
        )

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
        self.cmd_id = None

    def execute(self):
        self.cmd_id = self.controller.engine.push_command(
            ("disconn", self.src_id, self.src_port, self.dst_id, self.dst_port)
        )

    def undo(self):
        self.controller.engine.push_command(("conn", self.src_id, self.src_port, self.dst_id, self.dst_port))
