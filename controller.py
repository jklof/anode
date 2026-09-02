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
        import collections

        self.undo_stack = collections.deque(maxlen=max_length)
        self.redo_stack = collections.deque(maxlen=max_length)
        self.max_length = max_length

    def push(self, cmd):
        """Add a command to the undo stack and clear redo stack."""
        self.redo_stack.clear()
        self.undo_stack.append(cmd)

    def remove_by_cmd_id(self, cmd_id):
        """Remove the command whose engine-assigned identity matches cmd_id.
        Used when the engine reports a rejection/failed result so history does
        not depend on 'the last command happens to be this one'."""
        if cmd_id is None:
            return False
        for i in range(len(self.undo_stack) - 1, -1, -1):
            if getattr(self.undo_stack[i], "cmd_id", None) == cmd_id:
                del self.undo_stack[i]
                return True
        return False

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
    nodeMoved = Signal(str, tuple)

    def __init__(self):
        super().__init__()
        self.engine = Engine()
        self.history = CommandHistory()
        self._latest_snapshot = {}
        self._pending_params = {}

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(30)
        self.poll_timer.timeout.connect(self.check_engine_messages)
        self.poll_timer.start()

    # Helper to get safe access to snapshot data
    def get_node_data(self, node_id):
        for n in self._latest_snapshot.get("nodes", []):
            if n["id"] == node_id:
                return n.copy()
        return None

    def get_connections_from_snapshot(self):
        return self._latest_snapshot.get("connections", [])

    def check_engine_messages(self):
        # Flush debounced parameters
        for (nid, pname), val in self._pending_params.items():
            self.engine.push_command(("param", nid, pname, val))
        self._pending_params.clear()

        # If the engine is stopped, completed background NRT tasks (like file
        # loads) are only drained by the engine's real-time loop; we poll and
        # drain them here (via Engine._drain_nrt_all) so on_nrt_complete()
        # still fires while the engine is stopped (AGENTS.md §6).
        if not self.engine.running:
            nrt_completed = False
            for node in self.engine.graph.nodes:
                inbox = getattr(node, "_nrt_inbox", None)
                if inbox and not inbox.empty():
                    nrt_completed = True
            
            if nrt_completed:
                self.engine._drain_nrt_all()
                self.engine._emit_snapshot()
                node_data = {}
                for node in self.engine.graph.nodes:
                    telemetry = node.get_telemetry()
                    if telemetry:
                        node_data[node.id] = telemetry
                self.engine._emit_telemetry(0.0, node_data)

        if self.engine.output_queue.empty():
            return

        # Snapshot consistency: Work on a shallow copy of the top-level dict
        # and ensure nested lists are copied before mutation (Copy-on-write).
        ws = self._latest_snapshot.copy()
        graph_changed = False
        
        # We collect non-structural messages (telemetry, param_update) to process 
        # AFTER structural changes are applied. This guarantees UI elements exist.
        non_structural_msgs = []

        while not self.engine.output_queue.empty():
            try:
                msg = self.engine.output_queue.get_nowait()
                m_type = msg.get("type")
                if m_type in ("telemetry", "param_update"):
                    non_structural_msgs.append(msg)
                elif m_type == "graph_update":
                    ws = msg.copy()
                    graph_changed = True
                elif m_type == "node_added":
                    if "nodes" not in ws:
                        ws["nodes"] = []
                    else:
                        ws["nodes"] = ws["nodes"].copy()
                    ws["nodes"].append(msg["node"])
                    graph_changed = True
                elif m_type == "node_removed":
                    nid = msg["node_id"]
                    if "nodes" in ws:
                        ws["nodes"] = [n for n in ws["nodes"] if n["id"] != nid]
                    if "connections" in ws:
                        ws["connections"] = [c for c in ws["connections"] if c["src_id"] != nid and c["dst_id"] != nid]
                    graph_changed = True
                elif m_type == "connected":
                    if "connections" not in ws:
                        ws["connections"] = []
                    else:
                        ws["connections"] = ws["connections"].copy()
                    conn_dict = {
                        "src_id": msg["src_id"],
                        "src_port": msg["src_port"],
                        "dst_id": msg["dst_id"],
                        "dst_port": msg["dst_port"],
                    }
                    # Defensive dedup: the engine no longer re-announces
                    # no-op re-connects, but never let the snapshot drift
                    # into duplicate connection records.
                    if conn_dict not in ws["connections"]:
                        ws["connections"].append(conn_dict)
                        graph_changed = True
                elif m_type == "connect_rejected":
                    # Associate the rejection with the exact originating
                    # command via its engine-assigned id.
                    self.history.remove_by_cmd_id(msg.get("cmd_id"))
                elif m_type == "disconnected":
                    if "connections" in ws:
                        c_list = ws["connections"]
                        ws["connections"] = [
                            c
                            for c in c_list
                            if not (
                                c["src_id"] == msg["src_id"]
                                and c["src_port"] == msg["src_port"]
                                and c["dst_id"] == msg["dst_id"]
                                and c["dst_port"] == msg["dst_port"]
                            )
                        ]
                    graph_changed = True
                elif m_type == "node_moved":
                    if "nodes" in ws:
                        # Copy-on-write pattern to prevent mutating shared state
                        ws["nodes"] = ws["nodes"].copy()
                        for i, n in enumerate(ws["nodes"]):
                            if n["id"] == msg["node_id"]:
                                n_new = n.copy()
                                n_new["pos"] = msg["pos"]
                                ws["nodes"][i] = n_new
                                break
                    self.nodeMoved.emit(msg["node_id"], msg["pos"])
                elif m_type == "clock_changed":
                    ws["clock_id"] = msg["node_id"]
                    if "nodes" in ws:
                        ws["nodes"] = ws["nodes"].copy()
                        for i, n in enumerate(ws["nodes"]):
                            # Optimization: only copy if state actually changes
                            was_master = n.get("is_master", False)
                            is_now_master = n["id"] == msg["node_id"]
                            if was_master != is_now_master:
                                n_new = n.copy()
                                n_new["is_master"] = is_now_master
                                ws["nodes"][i] = n_new
                    graph_changed = True
            except Exception:
                logging.exception("Error processing engine message")

        # 1. Apply structural changes first
        self._latest_snapshot = ws
        if graph_changed:
            self.graphUpdated.emit(self._latest_snapshot)

        # 2. Process non-structural updates now that UI nodes are guaranteed to exist
        for msg in non_structural_msgs:
            m_type = msg.get("type")
            if m_type == "telemetry":
                self.telemetryUpdated.emit(msg)
            elif m_type == "param_update":
                if "nodes" in self._latest_snapshot:
                    self._latest_snapshot["nodes"] = self._latest_snapshot["nodes"].copy()
                    for i, n in enumerate(self._latest_snapshot["nodes"]):
                        if n["id"] == msg["node_id"] and "params" in n:
                            if msg["param"] in n["params"]:
                                n_new = n.copy()
                                n_new["params"] = n["params"].copy()
                                n_new["params"][msg["param"]] = n["params"][msg["param"]].copy()
                                n_new["params"][msg["param"]]["value"] = msg["value"]
                                self._latest_snapshot["nodes"][i] = n_new
                                break
                self.parameterUpdated.emit(msg)

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
        cmd = AddNodeCommand(self, node_type, pos, node_id=node_id, params=params)
        cmd.execute()
        self.history.push(cmd)

        # Optimistic Update: Add to local snapshot so immediate subsequent actions see it
        # (Simplified: we wait for engine update for 'add', but 'move'/'del' need patching)
        return cmd.node_id

    def delete_node(self, node_id):
        # DeleteNodeCommand captures its authoritative memento inside
        # execute() (the engine fills it when the delete is processed), so we
        # must validate node existence against the latest snapshot instead of
        # cmd.node_data, which is always None before execute() runs.
        if "nodes" in self._latest_snapshot:
            if not any(n["id"] == node_id for n in self._latest_snapshot["nodes"]):
                logging.warning(f"Attempted to delete unknown node {node_id}")
                return

        cmd = DeleteNodeCommand(self, node_id)
        cmd.execute()
        self.history.push(cmd)

        # Optimistic Update: Remove from local snapshot
        if "nodes" in self._latest_snapshot:
            self._latest_snapshot["nodes"] = [n for n in self._latest_snapshot["nodes"] if n["id"] != node_id]

    def move_nodes(self, moves_dict):
        """
        moves_dict: { node_id: (new_pos, old_pos) }
        """
        if not moves_dict:
            return
        cmd = MultiMoveNodeCommand(self, moves_dict)
        cmd.execute()
        self.history.push(cmd)

        # CRITICAL FIX: Optimistic Update
        # Update local snapshot immediately so if we delete right after moving,
        # the DeleteCommand captures the NEW position, not the old one.
        if "nodes" in self._latest_snapshot:
            for n in self._latest_snapshot["nodes"]:
                if n["id"] in moves_dict:
                    n["pos"] = moves_dict[n["id"]][0]

    def delete_selection(self, node_ids, connection_tuples):
        """
        Deletes a list of nodes and specific connections atomically.
        """
        macro = CompoundCommand("Delete Selection")

        # 1. Delete Explicitly Selected Wires
        # (Only those NOT connected to nodes we are about to delete,
        # to avoid double-restoration logic, or rely on set logic in graph)
        # However, for simplicity: deleting a node implicitly deletes wires.
        # We only strictly need explicit Disconnect commands for wires where
        # NEITHER end is being deleted (i.e., user selected just a wire).

        nodes_set = set(node_ids)

        for c_data in connection_tuples:
            sid, sp, did, dp = c_data
            # Only add explicit disconnect if we AREN'T deleting the attached nodes
            # (The Engine cleans up wires attached to deleted nodes automatically)
            if sid not in nodes_set and did not in nodes_set:
                macro.add(DisconnectCommand(self, *c_data))

        # DeleteNodeCommand now captures authoritative state directly
        for nid in node_ids:
            macro.add(DeleteNodeCommand(self, nid))

        # Perform optimistic snapshot updates AFTER building all commands
        for nid in node_ids:
            if "nodes" in self._latest_snapshot:
                self._latest_snapshot["nodes"] = [n for n in self._latest_snapshot["nodes"] if n["id"] != nid]

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
        if not self.engine.running:
            src = self.engine.graph.node_map.get(src_id)
            dst = self.engine.graph.node_map.get(dst_id)
            connected = False
            if src and dst and dst_port in dst.inputs and src_port in src.outputs:
                connected = src.outputs[src_port] in dst.inputs[dst_port].connected_outputs
            if connected:
                self.history.push(cmd)
        else:
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
        """Set a parameter value (not undoable). Values are debounced/batched."""
        self._pending_params[(node_id, param_name)] = value

    def save(self, filename):
        """Save the graph to a file."""
        if not filename:
            return
        # Make Save a coherent synchronization point: commit any pending UI
        # parameter changes to the authoritative state BEFORE the save command
        # is queued. Commands run in FIFO order, so serialization (which
        # happens on the engine thread inside the save handler) sees the
        # latest values. The background writer then only receives the
        # immutable serialized JSON string.
        for (nid, pname), val in self._pending_params.items():
            self.engine.push_command(("param", nid, pname, val))
        self._pending_params.clear()
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
            logging.error(f"Controller Load Error: {e}")

    def clear(self):
        """Clear the graph."""
        self.engine.push_command(("clear",))
        self.history = CommandHistory()

    def reload_plugins(self):
        """Reload all plugins.

        Stops the engine first and leaves it stopped: reloading swaps node
        classes out from under running instances, and rebuilding the device
        widgets re-probes ALSA devices that are still held by the live
        stream (PortAudio prints C-level 'AlsaOpen failed' spam to stderr).
        Hot reload is a developer feature — seamless audio across it is not
        worth the churn; press Start afterwards if needed."""
        self.stop_audio()
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
