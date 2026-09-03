import collections
import gc
import torch
import threading
import time
import queue
import json
import logging
import concurrent.futures
from typing import Dict, List, Optional, Tuple
import plugin_system
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE, IClockProvider, Node

_STRUCTURAL_OPS = frozenset({"add", "del", "conn", "disconn", "restore", "clear", "load", "reload", "clock"})


class ExecutionPlan:
    __slots__ = ("nodes", "clock_source")

    def __init__(self, nodes, clock_source=None):
        self.nodes = tuple(nodes)
        self.clock_source = clock_source


class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.node_map: Dict[str, Node] = {}
        self._execution_order: List[Node] = []
        self._order_dirty = True
        self.structure_dirty = False
        self.clock_source: Optional[IClockProvider] = None
        self.engine = None

    def mark_dirty(self):
        self._order_dirty = True
        self.structure_dirty = True

    def compile_execution_plan(self) -> ExecutionPlan:
        return ExecutionPlan(self.execution_order, self.clock_source)

    @property
    def execution_order(self) -> List[Node]:
        if self._order_dirty:
            self._recalculate_order()
            self._order_dirty = False
        return self._execution_order

    def _get_upstream_nodes(self, node: Node) -> List[Node]:
        upstream = []
        for inp in node.inputs.values():
            for out in inp.connected_outputs:
                upstream.append(out.parent)
        return upstream

    def _get_downstream_nodes(self, node: Node) -> List[Node]:
        """Get nodes that this node connects to (downstream)."""
        downstream = []
        for out_slot in node.outputs.values():
            # We need to find all InputSlots connected to this OutputSlot
            for other_node in self.nodes:
                for inp in other_node.inputs.values():
                    for conn_out in inp.connected_outputs:
                        if conn_out is out_slot:
                            downstream.append(other_node)
        return downstream

    def add_node(self, node: Node):
        node.graph = self
        self.nodes.append(node)
        self.node_map[node.id] = node
        if isinstance(node, IClockProvider) and self.clock_source is None:
            self.set_master_clock(node)
        self.recalculate_order()

    def remove_node(self, node_id):
        if node_id not in self.node_map:
            return
        node = self.node_map[node_id]
        if self.clock_source == node:
            self.clock_source = None
        for inp in node.inputs.values():
            inp.disconnect()
        for other in self.nodes:
            for inp in other.inputs.values():
                for conn_out in list(inp.connected_outputs):
                    if conn_out.parent == node:
                        inp.disconnect(conn_out)
        self.nodes.remove(node)
        del self.node_map[node_id]
        node.graph = None
        if self.clock_source is None:
            for n in self.nodes:
                if isinstance(n, IClockProvider):
                    self.set_master_clock(n)
                    break
        self.recalculate_order()

    def capture_node_state(self, node_id):
        """Capture a complete, authoritative memento of a node and all of its
        connections. Called by the command executor (engine/control thread) at
        the moment the delete is actually processed, so undo restores state as
        of that point rather than as of when the delete was requested."""
        node = self.node_map.get(node_id)
        if node is None:
            return None
        node_data = node.to_dict()
        # Full param snapshot (type + meta) for exact restoration
        for k, p in node.params.items():
            if k in node_data.get("params", {}):
                node_data["params"][k] = {
                    "value": p.get_staging_safe(),
                    "type": p.type,
                    "meta": p.meta,
                }
        connections = []
        for dst in self.nodes:
            for d_port, inp in dst.inputs.items():
                for out in inp.connected_outputs:
                    if out.parent.id == node_id or dst.id == node_id:
                        connections.append({
                            "src_id": out.parent.id,
                            "src_port": out.name,
                            "dst_id": dst.id,
                            "dst_port": d_port,
                        })
        return {"node": node_data, "connections": connections}

    def connect(self, src_id, src_port, dst_id, dst_port):
        """Connect two nodes, rejecting cycles and impossible channel
        configurations at connection time.

        Channel policy: an output declares a fixed, positive channel count;
        inputs adapt dynamically (mono -> stereo duplication is the supported
        case, handled by InputSlot.get_tensor / node adapters). Any output
        channel count >= 1 is connectable; a non-positive count is invalid and
        rejected here so the problem surfaces immediately rather than at
        runtime.

        Returns True if connection was made, False if rejected.
        """
        src = self.node_map.get(src_id)
        dst = self.node_map.get(dst_id)
        if not (src and dst and src_port in src.outputs and dst_port in dst.inputs):
            return False

        # Reject self-loops
        if src_id == dst_id:
            return False

        # Reject impossible channel configurations. Only outputs that carry a
        # real buffer declare a channel count; if present it must be >= 1.
        src_slot = src.outputs[src_port]
        buf = getattr(src_slot, "buffer", None)

        # Enforce slot type compatibility (audio to audio, midi to midi). A
        # MIDI stream must never be wired into an audio input or vice versa.
        src_type = getattr(src_slot, "slot_type", "audio")
        dst_type = getattr(dst.inputs[dst_port], "slot_type", "audio")
        if src_type != dst_type:
            logging.warning(
                f"Connection rejected: Type mismatch between {src.name}.{src_port} "
                f"({src_type}) and {dst.name}.{dst_port} ({dst_type})"
            )
            return False

        # Reject impossible channel configurations. Only outputs that carry a
        # real buffer declare a channel count; if present it must be >= 1 and
        # within the engine's global channel format (InputSlot scratch buffers
        # are sized for CHANNELS; anything wider is unsupported).
        if buf is not None and (buf.shape[0] < 1 or buf.shape[0] > CHANNELS):
            if buf.shape[0] > CHANNELS:
                logging.warning(
                    f"Connection rejected: {src.name}.{src_port} declares "
                    f"{buf.shape[0]} channels; the engine format supports at "
                    f"most {CHANNELS}."
                )
            return False

        # Check if adding this connection would create a cycle:
        # If dst can already reach src, adding src->dst creates a cycle.
        if self._can_reach(dst_id, src_id):
            return False

        # No cycle - safe to connect
        dst.inputs[dst_port].connect(src.outputs[src_port])
        self.recalculate_order()
        return True

    def _can_reach(self, start_id: str, target_id: str) -> bool:
        """DFS reachability check: can start_id reach target_id via existing connections?"""
        visited = set()
        stack = [start_id]
        
        while stack:
            current_id = stack.pop()
            if current_id == target_id:
                return True
            if current_id in visited:
                continue
            visited.add(current_id)
            
            current_node = self.node_map.get(current_id)
            if not current_node:
                continue
                
            # Traverse downstream: nodes that current_node connects to
            for downstream in self._get_downstream_nodes(current_node):
                if downstream.id not in visited:
                    stack.append(downstream.id)
        
        return False

    def disconnect(self, src_id, src_port, dst_id, dst_port):
        src_node = self.node_map.get(src_id)
        dst_node = self.node_map.get(dst_id)
        if src_node and dst_node and src_port in src_node.outputs and dst_port in dst_node.inputs:
            output_slot = src_node.outputs[src_port]
            dst_node.inputs[dst_port].disconnect(target=output_slot)
            # Topology changed, not just execution order: mark both flags so
            # plan recompilation and snapshot invalidation trigger (AGENTS.md §8).
            self.mark_dirty()

    def set_master_clock(self, node: Node):
        if not isinstance(node, IClockProvider):
            return
        self.clock_source = node
        for n in self.nodes:
            if isinstance(n, IClockProvider):
                n.set_master(n == node)

    def clear_master_clock(self):
        """Detach any master clock (e.g. when loading a patch saved without a
        clock_id) so add_node()'s first-provider auto-assignment is not
        silently retained for patches that were saved clockless."""
        self.clock_source = None
        for n in self.nodes:
            if isinstance(n, IClockProvider):
                n.set_master(False)

    def recalculate_order(self):
        self._order_dirty = True

    def _recalculate_order(self):
        in_degree = {n.id: 0 for n in self.nodes}
        adj = {n.id: [] for n in self.nodes}

        for n in self.nodes:
            upstream_ids = set()
            for u in self._get_upstream_nodes(n):
                if u.id in in_degree:
                    upstream_ids.add(u.id)

            in_degree[n.id] = len(upstream_ids)
            for u_id in upstream_ids:
                adj[u_id].append(n.id)

        queue = collections.deque([n.id for n in self.nodes if in_degree[n.id] == 0])
        order = []

        while queue:
            curr_id = queue.popleft()
            curr_node = self.node_map.get(curr_id)
            if curr_node:
                order.append(curr_node)

            for neighbor_id in adj[curr_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(order) != len(self.nodes):
            logging.warning("Cycle detected in graph! Cyclic nodes omitted from execution.")

        self._execution_order = order

    def _get_node_data(self, n: Node) -> dict:
        p_data = {}
        for k, p in n.params.items():
            p_data[k] = {"value": p.get_staging_safe(), "type": p.type, "meta": p.meta}
        mon_q = getattr(n, "monitor_queue", None)

        is_clock_provider = isinstance(n, IClockProvider)
        is_current_master = n == self.clock_source

        return {
            "id": n.id,
            "name": n.name,
            "type": n.__class__.__name__,
            "pos": n.pos,
            "error": n.error_msg,
            "inputs": list(n.inputs.keys()),
            "outputs": list(n.outputs.keys()),
            "input_types": {k: getattr(v, "slot_type", "audio") for k, v in n.inputs.items()},
            "output_types": {k: getattr(v, "slot_type", "audio") for k, v in n.outputs.items()},
            "params": p_data,
            "monitor_queue": mon_q,
            "can_be_master": is_clock_provider,
            "is_master": is_current_master,
        }

    def get_snapshot(self) -> dict:
        data = {
            "type": "graph_update",
            "clock_id": self.clock_source.id if self.clock_source else None,
            "nodes": [],
            "connections": [],
        }
        for n in self.nodes:
            data["nodes"].append(self._get_node_data(n))
        for dst in self.nodes:
            for d_port, inp in dst.inputs.items():
                for out in inp.connected_outputs:
                    data["connections"].append(
                        {"src_id": out.parent.id, "src_port": out.name, "dst_id": dst.id, "dst_port": d_port}
                    )
        return data

    def to_json(self) -> str:
        data = {
            "clock_id": self.clock_source.id if self.clock_source else None,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [],
        }
        for dst in self.nodes:
            for d_port, inp in dst.inputs.items():
                for out in inp.connected_outputs:
                    data["connections"].append(
                        {"src_id": out.parent.id, "src_port": out.name, "dst_id": dst.id, "dst_port": d_port}
                    )
        return json.dumps(data, indent=2)


class NRTExecutor:
    """
    Centralized non-real-time task runner. Replaces the ad hoc background-thread
    patterns scattered across plugins (action queues, loader threads, mgmt threads).

    Two APIs:
      - submit(): bounded pool, for one-shot jobs (model/IR loads, device queries,
        stream open/close). Results are delivered to node.on_nrt_complete() on a
        later tick via drain(), never blocking the caller.
      - spawn_stream()/stop_stream(): for genuinely long-running producers (e.g.
        media playback) that shouldn't occupy a pool slot indefinitely.

    Per-node epoch/inbox state lives on the Node instance itself, not in a shared
    dict here, so no lock is needed: submit()/drain() for a given node are always
    invoked from whichever thread is currently applying engine commands, and that
    is already serialized by Engine (audio thread when running, UI thread when
    stopped — never both at once).
    """

    def __init__(self, max_workers=6):
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="anode-nrt"
        )
        # Nodes removed from the graph that may still have in-flight or
        # undelivered NRT results. Without tracking, their inboxes are never
        # drained by _drain_nrt_all() (it only walks graph.nodes), so
        # on_nrt_discarded() never runs and native handles / descriptors /
        # stream threads leak. These are strong references held only until
        # the node is quiescent (bounded).
        self._discarded_nodes = set()
        # Node -> count of submitted jobs that have not yet been drained.
        # Only mutated from submit()/drain()/drain_discarded(), all of which
        # run on the engine/control thread (see the class docstring), so no
        # lock is needed. An empty inbox alone does NOT imply quiescence:
        # a job still executing in the pool has not put its result yet.
        self._in_flight = {}

    def submit(self, node, fn, args, tag=None):
        node._nrt_epoch += 1
        epoch = node._nrt_epoch
        if node._nrt_inbox is None:
            node._nrt_inbox = queue.SimpleQueue()
        inbox = node._nrt_inbox

        self._in_flight[node] = self._in_flight.get(node, 0) + 1

        def _run():
            try:
                inbox.put((epoch, tag, True, fn(*args)))
            except Exception as e:
                inbox.put((epoch, tag, False, e))

        self._pool.submit(_run)

    def drain(self, node):
        inbox = node._nrt_inbox
        if inbox is None:
            return
        while True:
            try:
                epoch, tag, ok, payload = inbox.get_nowait()
            except queue.Empty:
                return
            if node in self._in_flight:
                self._in_flight[node] -= 1
                if self._in_flight[node] <= 0:
                    del self._in_flight[node]
            if epoch != node._nrt_epoch:
                # Superseded by a newer submit(): the payload is never
                # delivered to on_nrt_complete(), so give the node a chance
                # to release any resources it carries (native DSP handles,
                # open streams, worker bundles) before discarding it.
                try:
                    node.on_nrt_discarded(tag, ok, payload)
                except Exception as e:
                    logging.error(f"NRT discard handler failed for {node.name}: {e}")
                continue
            node.on_nrt_complete(tag, ok, payload)

    def discard(self, node):
        """Invalidate any in-flight results for this node. O(1), never blocks.
        Call this when a node is deleted."""
        node._nrt_epoch += 1
        # Only track nodes that can still produce a result. Nodes with no
        # pending work need no further draining.
        if self._in_flight.get(node, 0) > 0 or (
            node._nrt_inbox is not None and not node._nrt_inbox.empty()
        ):
            self._discarded_nodes.add(node)

    def drain_discarded(self):
        """Drain results for nodes already removed from the graph, and retire
        tracking once a node is quiescent: ALL in-flight pool jobs have
        returned AND its inbox is empty. Never blocks; runs on the
        engine/control thread between blocks."""
        quiescent = []
        for node in list(self._discarded_nodes):
            self.drain(node)
            if self._in_flight.get(node, 0) == 0 and (
                node._nrt_inbox is None or node._nrt_inbox.empty()
            ):
                quiescent.append(node)
        for node in quiescent:
            self._discarded_nodes.discard(node)
            self._in_flight.pop(node, None)

    def spawn_stream(self, target, *args):
        """For long-running producers that shouldn't occupy a pool slot."""
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        return t

    def stop_stream(self, node, stop_fn, thread, timeout=2.0):
        """Non-blocking stream teardown: stop+join happens on a pool thread,
        not the caller."""
        if thread is None:
            return

        def _shutdown():
            try:
                stop_fn()
            finally:
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logging.warning(
                        f"NRT stream thread for node {node.id} did not exit "
                        f"within {timeout}s; abandoning it."
                    )

        self.submit(node, _shutdown, ())

    def shutdown(self, wait=True):
        self._pool.shutdown(wait=wait)


class Engine:
    def __init__(self):
        self.graph = Graph()
        self.graph.engine = self
        self.nrt = NRTExecutor()
        self.reload_version = 0
        self.running = False
        self.abort_flag = False
        self.command_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=200)
        self.thread = None
        self._tick_semaphore = None
        self.max_buffered_frames = 4
        # Per-node processing stats. Lives on the Engine (not _worker locals) so
        # entries can be pruned when nodes are deleted.
        self._stats_buffer = {}
        self._active_plan = self.graph.compile_execution_plan()
        # Monotonic command identity: every pushed command gets a unique id so
        # results (e.g. connect_rejected) can be associated with the exact
        # originating request.
        self._next_cmd_id = 0

    def _drain_nrt_all(self):
        if self.nrt:
            for node in self.graph.nodes:
                self.nrt.drain(node)
            # Nodes removed from the graph with in-flight/undelivered results
            # still need their on_nrt_discarded() callback so native handles
            # and streams are released (AGENTS.md §6).
            self.nrt.drain_discarded()

    def tick(self):
        self._tick_semaphore.release()

    def push_command(self, cmd: Tuple):
        """
        Pushes a command to the engine.
        Dual-Path Design:
        - When the engine is running, the command is queued to the real-time audio thread
          to prevent audio glitches, and the snapshot/telemetry are emitted asynchronously.
        - When the engine is stopped, the command is executed synchronously in the UI thread
          via _apply_command, and a snapshot is immediately emitted for structural operations
          to keep the UI in sync. Parameter and position changes use their own side-channels.

        Exception: "save" - serialization happens on engine thread, file I/O on background thread.
        """
        self._next_cmd_id += 1
        cmd_id = self._next_cmd_id
        if self.running:
            if cmd[0] == "save":
                # Queue save command for engine thread to serialize, then background write
                self.command_queue.put((cmd_id, cmd))
                return cmd_id
            self.command_queue.put((cmd_id, cmd))
        else:
            self._apply_command(cmd, cmd_id)
            if cmd[0] in _STRUCTURAL_OPS or self.graph.structure_dirty:
                self.graph.structure_dirty = False
                self._emit_snapshot()
        return cmd_id

    def _save_graph(self, filename):
        """Deprecated: kept as a thin wrapper. Serialization happens in
        _apply_command at the coherent command boundary (so all queued
        parameter changes are already applied); the file write is delegated
        to a background stream via _write_background()."""
        self._write_background(self.graph.to_json(), filename)

    def _write_background(self, json_str, filename):
        """Write an already-serialized patch to disk on a background stream
        thread (never the audio/engine loop, never a raw ad-hoc thread)."""
        def _write_job(json_data: str, fname: str):
            try:
                with open(fname, "w") as f:
                    f.write(json_data)
                logging.info(f"Saved patch to {fname}")
            except Exception as e:
                logging.error(f"Save write error: {e}")

        if self.running and self.nrt:
            self.nrt.spawn_stream(_write_job, json_str, filename)
        else:
            _write_job(json_str, filename)

    def _gc_deferred(self):
        """Run a full collection off-thread. Full gc.collect() calls can stall for
        hundreds of milliseconds and must never execute on the RT loop."""
        threading.Thread(target=gc.collect, daemon=True, name="anode-gc").start()

    def _emit_snapshot(self):
        snap = self.graph.get_snapshot()
        snap["is_running"] = self.running
        snap["reload_version"] = self.reload_version
        try:
            self.output_queue.put_nowait(snap)
        except queue.Full:
            # Consistent with rest of codebase: if full, drop it.
            # UI will sync on next timer tick / snapshot command.
            pass

    def _emit_telemetry(self, cpu_load, node_data):
        if not self.output_queue.full():
            self.output_queue.put({"type": "telemetry", "cpu_load": cpu_load, "node_data": node_data})

    def _apply_command(self, cmd, cmd_id=None):
        try:
            op = cmd[0]
            if op == "add":
                # Support atomic node creation with initial parameters
                # cmd format: ("add", type_name, nid, pos, initial_params) where initial_params can be None
                _, type_name_or_node, nid, pos, initial_params = cmd
                if isinstance(type_name_or_node, str):
                    cls = plugin_system.NODE_REGISTRY.get(type_name_or_node)
                    if cls:
                        node = cls()
                        node.id = nid
                        node.pos = pos
                    else:
                        node = None
                else:
                    node = type_name_or_node
                    # Critical Fix: Set ID and POS for pre-instantiated nodes
                    # Note: if node's __init__ had any side effects using self.id,
                    # those would have run with a random UUID since this is assigned afterwards.
                    if node is not None:
                        node.id = nid
                        node.pos = pos

                if node:
                    # Fix: Add node to graph first so node.graph is valid for parameter change callbacks
                    self.graph.add_node(node)

                    # Apply initial parameters AFTER adding to graph (atomic creation)
                    if initial_params:
                        for param_name, param_data in initial_params.items():
                            if param_name in node.params:
                                # Support both dictionary format: {"value": actual_value} and raw values
                                if isinstance(param_data, dict) and "value" in param_data:
                                    val = param_data["value"]
                                else:
                                    # Handle raw values (e.g., from Node.to_dict() for Undo functionality)
                                    val = param_data
                                node.params[param_name].set(val)
                                # Commit staged value before notifying the node so
                                # on_ui_param_change sees a synchronized parameter.
                                node.params[param_name].sync()
                                node.on_ui_param_change(param_name)
                    if self.running:
                        try:
                            node.start()
                        except Exception as e:
                            logging.exception(f"Error starting node {node.name} on add")
                            node.error_msg = f"Start Error: {e}"
                    try:
                        self.output_queue.put_nowait({"type": "node_added", "node": self.graph._get_node_data(node)})
                    except Exception:
                        pass
            elif op == "del":
                nid = cmd[1]
                # Optional memento holder: filled by the authoritative command
                # executor right here, at the point the delete is actually
                # processed. Guarantees undo state is never stale.
                holder = cmd[2] if len(cmd) > 2 else None
                if holder is not None:
                    memento = self.graph.capture_node_state(nid)
                    if memento is not None:
                        holder.update(memento)
                n = self.graph.node_map.get(nid)
                if n:
                    if self.nrt:
                        self.nrt.discard(n)
                    if self.running:
                        n.stop()
                    # Clean up C++ handles or other resources
                    n.remove()

                self.graph.remove_node(nid)
                # Prune stale telemetry so dead node ids don't accumulate or
                # skew the global CPU average.
                self._stats_buffer.pop(nid, None)
                try:
                    self.output_queue.put_nowait({"type": "node_removed", "node_id": nid})
                except Exception:
                    pass
            elif op == "conn":
                _, sid, sp, did, dp = cmd
                # A re-connect of an existing edge is a no-op in Graph.connect
                # (InputSlot.connect dedups); do not re-announce it, otherwise
                # the UI snapshot accumulates duplicate connection records.
                src_n = self.graph.node_map.get(sid)
                dst_n = self.graph.node_map.get(did)
                already = (
                    src_n is not None and dst_n is not None
                    and sp in src_n.outputs and dp in dst_n.inputs
                    and src_n.outputs[sp] in dst_n.inputs[dp].connected_outputs
                )
                success = self.graph.connect(sid, sp, did, dp)
                if success and not already:
                    try:
                        self.output_queue.put_nowait(
                            {"type": "connected", "src_id": sid, "src_port": sp, "dst_id": did, "dst_port": dp}
                        )
                    except Exception:
                        pass
                else:
                    try:
                        self.output_queue.put_nowait(
                            {"type": "connect_rejected", "src_id": sid, "src_port": sp,
                             "dst_id": did, "dst_port": dp, "cmd_id": cmd_id}
                        )
                    except Exception:
                        pass
            elif op == "disconn":
                _, sid, sp, did, dp = cmd
                self.graph.disconnect(sid, sp, did, dp)
                try:
                    self.output_queue.put_nowait(
                        {"type": "disconnected", "src_id": sid, "src_port": sp, "dst_id": did, "dst_port": dp}
                    )
                except Exception:
                    pass
            elif op == "param":
                _, nid, p, val = cmd
                node = self.graph.node_map.get(nid)
                if node and p in node.params:
                    node.params[p].set(val)
                    # Commit the staged value BEFORE notifying the node so
                    # on_ui_param_change() observes a synchronized parameter
                    # (param.value == staged value, caches updated).
                    node.params[p].sync()
                    node.on_ui_param_change(p)

                    # If this parameter change triggered a structural update, we skip the side-channel
                    # message because a full snapshot update is being emitted instead.
                    if self.graph.structure_dirty:
                        pass
                    else:
                        # Push side-channel parameter update message
                        msg = {"type": "param_update", "node_id": nid, "param": p, "value": val}
                        try:
                            self.output_queue.put_nowait(msg)
                        except Exception:
                            pass  # UI queue full; drop the update, UI will sync on next snapshot
            elif op == "clock":
                _, nid = cmd
                node = self.graph.node_map.get(nid)
                if node:
                    self.graph.set_master_clock(node)
                    try:
                        self.output_queue.put_nowait({"type": "clock_changed", "node_id": nid})
                    except Exception:
                        pass
            elif op == "move":
                _, nid, x, y = cmd
                node = self.graph.node_map.get(nid)
                if node:
                    node.pos = (x, y)
                    try:
                        self.output_queue.put_nowait({"type": "node_moved", "node_id": nid, "pos": (x, y)})
                    except Exception:
                        pass

            # --- Restore Command for robust Undo ---
            elif op == "restore":
                _, n_data_payload = cmd
                if isinstance(n_data_payload, tuple):
                    node_data, node_instance = n_data_payload
                else:
                    node_data, node_instance = n_data_payload, None

                # DeleteNodeCommand.undo() passes the authoritative holder
                # dict ({"node": ..., "connections": [...]}) instead of a
                # bare node memento; FIFO ordering guarantees the 'del' has
                # populated it by the time this runs. Unwrap it here so the
                # implicit connections are restored atomically with the node.
                connections_to_restore = []
                if isinstance(node_data, dict) and "node" in node_data:
                    connections_to_restore = node_data.get("connections", [])
                    node_data = node_data.get("node")

                if node_data:
                    cls = plugin_system.NODE_REGISTRY.get(node_data["type"])
                    if cls:
                        node = node_instance if node_instance else cls(node_data["name"])
                        node.id = node_data["id"]
                        # Add node first so load_state has a valid graph reference
                        # to submit background tasks
                        self.graph.add_node(node)
                        # This restores everything: pos, params, internal meta.
                        # Always call load_state here (AFTER graph attachment) —
                        # pre-instantiated undo/restore nodes arrive bare and nodes
                        # that spawn NRT work in load_state() need self.graph set.
                        node.load_state(node_data)
                        if self.running:
                            try:
                                node.start()
                            except Exception as e:
                                logging.exception(f"Error starting restored node {node.name}")
                                node.error_msg = f"Start Error: {e}"
                        try:
                            self.output_queue.put_nowait({"type": "node_added", "node": self.graph._get_node_data(node)})
                        except Exception:
                            pass
                        for c in connections_to_restore:
                            if self.graph.connect(c["src_id"], c["src_port"], c["dst_id"], c["dst_port"]):
                                # Announce the wire to the UI (mirrors the
                                # "conn" opcode): when running, no full
                                # snapshot follows this command, so without
                                # this event the restored node comes back
                                # with invisible wires until an unrelated
                                # action triggers a snapshot.
                                try:
                                    self.output_queue.put_nowait(
                                        {"type": "connected", "src_id": c["src_id"], "src_port": c["src_port"], "dst_id": c["dst_id"], "dst_port": c["dst_port"]}
                                    )
                                except Exception:
                                    pass
            # --------------------------------------------

            elif op == "clear":
                # Invalidate NRT epochs for all nodes before destroying them
                for n in self.graph.nodes:
                    self.nrt.discard(n)
                    n.stop()
                    n.remove()
                self.graph = Graph()
                self.graph.engine = self
                self._gc_deferred()
                self._emit_snapshot()

            elif op == "save":
                # Serialize at the coherent command boundary (all queued
                # commands before this one have been applied), then delegate
                # the file I/O to a background stream — never on the audio loop.
                filename = cmd[1]
                if filename:
                    try:
                        json_str = self.graph.to_json()
                    except Exception as e:
                        logging.error(f"Save serialization error: {e}")
                    else:
                        self._write_background(json_str, filename)

            elif op == "load":
                _, json_str = cmd
                # Invalidate NRT epochs for all current nodes before replacing graph
                for n in self.graph.nodes:
                    self.nrt.discard(n)
                    n.stop()
                    n.remove()
                self.graph = Graph()
                try:
                    data = json.loads(json_str)
                    if not isinstance(data, dict):
                        raise ValueError("Loaded data is not a valid JSON object.")

                    new_graph = Graph()
                    new_graph.engine = self  # Fix: Set engine reference before loading nodes so submit_nrt works
                    for n_data in data.get("nodes", []):
                        if not isinstance(n_data, dict):
                            continue
                        cls = plugin_system.NODE_REGISTRY.get(n_data.get("type"))
                        if cls:
                            node = cls(n_data.get("name", ""))
                            if "id" in n_data:
                                node.id = n_data["id"]
                            # Fix: Add node to graph first so load_state has graph reference
                            new_graph.add_node(node)
                            node.load_state(n_data)
                    for c in data.get("connections", []):
                        if not isinstance(c, dict):
                            continue
                        src_id = c.get("src_id")
                        dst_id = c.get("dst_id")
                        if src_id in new_graph.node_map and dst_id in new_graph.node_map:
                            new_graph.connect(src_id, c.get("src_port"), dst_id, c.get("dst_port"))
                    if data.get("clock_id") and data["clock_id"] in new_graph.node_map:
                        new_graph.set_master_clock(new_graph.node_map[data["clock_id"]])
                    else:
                        new_graph.clear_master_clock()
                    self.graph = new_graph
                    self.graph.engine = self
                    if self.running:
                        for n in self.graph.nodes:
                            if n == self.graph.clock_source:
                                n.start_clock(self.tick)
                            else:
                                n.start()
                    self._emit_snapshot()
                    self._gc_deferred()
                except Exception as e:
                    logging.error(f"Load Failed: {e}")

            elif op == "reload":
                logging.info("Engine: Reloading plugins...")
                current_json = self.graph.to_json()
                # Invalidate NRT epochs for all current nodes before replacing graph
                for n in self.graph.nodes:
                    self.nrt.discard(n)
                    n.stop()
                    n.remove()
                self.graph = Graph()
                self.reload_version += 1
                try:
                    plugin_system.load_plugins()
                except Exception as e:
                    logging.error(f"Engine: Reload failed: {e}")
                    return
                try:
                    data = json.loads(current_json)
                    new_graph = Graph()
                    new_graph.engine = self  # Fix: Set engine reference before loading nodes so submit_nrt works
                    for n_data in data["nodes"]:
                        cls = plugin_system.NODE_REGISTRY.get(n_data["type"])
                        if cls:
                            node = cls(n_data["name"])
                            node.id = n_data["id"]
                            # Fix: Add node to graph first so load_state has graph reference
                            new_graph.add_node(node)
                            node.load_state(n_data)
                    for c in data["connections"]:
                        if c["src_id"] in new_graph.node_map and c["dst_id"] in new_graph.node_map:
                            new_graph.connect(c["src_id"], c["src_port"], c["dst_id"], c["dst_port"])
                    if data.get("clock_id") and data["clock_id"] in new_graph.node_map:
                        new_graph.set_master_clock(new_graph.node_map[data["clock_id"]])
                    else:
                        new_graph.clear_master_clock()
                    self.graph = new_graph
                    self.graph.engine = self
                    if self.running:
                        for n in self.graph.nodes:
                            if n == self.graph.clock_source:
                                n.start_clock(self.tick)
                            else:
                                n.start()
                    self._emit_snapshot()
                    logging.info("Engine: Hot reload complete.")
                    self._gc_deferred()
                except Exception as e:
                    logging.error(f"Engine: Restore failed after reload: {e}")

            elif op == "snapshot":
                self._emit_snapshot()

            if op in _STRUCTURAL_OPS or self.graph.structure_dirty:
                self.graph.structure_dirty = False
                self._active_plan = self.graph.compile_execution_plan()
            self._drain_nrt_all()

        except Exception:
            logging.exception("Cmd Error")

    def _reset_audio_buffers(self):
        """Zero all audio output buffers and input scratch buffers to prevent
        stuck notes / stale audio on transport start. MIDI slots carry a packet
        (not a tensor buffer) and are skipped."""
        for node in self.graph.nodes:
            for out_slot in node.outputs.values():
                if getattr(out_slot, "slot_type", "audio") == "audio":
                    out_slot.buffer.zero_()
            for inp_slot in node.inputs.values():
                if getattr(inp_slot, "slot_type", "audio") == "audio":
                    inp_slot._scratch.zero_()

    def _worker(self):
        logging.info("Engine: Started")
        gc.disable()
        with torch.no_grad():

            # --- STARTUP CLEANUP ---
            # 1. Zero out all buffers to prevent "stuck notes" or stale audio glitches
            self._reset_audio_buffers()

            # 2. Start nodes safely
            for node in self.graph.nodes:
                try:
                    if node == self.graph.clock_source:
                        node.start_clock(self.tick)
                    else:
                        node.start()
                except Exception as e:
                    logging.exception(f"Error starting node {node.name}")
                    node.error_msg = f"Start Error: {e}"

            block_duration_sec = BLOCK_SIZE / SAMPLE_RATE
            telemetry_interval = 0.1
            next_telemetry_time = time.perf_counter() + telemetry_interval

            while self.running:
                while not self.command_queue.empty():
                    entry = self.command_queue.get_nowait()
                    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], int):
                        cmd_id, cmd = entry
                    else:
                        # Backwards compatibility with raw commands
                        cmd_id, cmd = None, entry
                    # Forward the engine-assigned command identity so result
                    # messages (e.g. connect_rejected) can be associated with
                    # the exact originating request (AGENTS.md §8).
                    self._apply_command(cmd, cmd_id)

                plan = self._active_plan

                if plan.clock_source:
                    # Step A (Non-blocking check)
                    acquired = self._tick_semaphore.acquire(blocking=False)
                    if not acquired:
                        # Step B: bounded wait for the hardware tick. A dead
                        # or stalled device callback must not hang this
                        # thread forever — command draining and engine
                        # shutdown depend on this loop staying responsive.
                        if not self._tick_semaphore.acquire(blocking=True, timeout=0.2):
                            if not self.running or self.abort_flag:
                                break
                            # Clock provider stalled; fall back to a timed
                            # sleep so stop/abort and queued commands are
                            # still processed while the failure persists.
                            time.sleep(BLOCK_SIZE / SAMPLE_RATE)
                    # If acquired, we proceed directly (already decremented)
                else:
                    # Fallback sleep if no clock source is defined to prevent 100% CPU usage
                    time.sleep(BLOCK_SIZE / SAMPLE_RATE)

                for node in plan.nodes:
                    node.sync()

                for node in plan.nodes:
                    try:
                        t0 = time.perf_counter()
                        node.process()
                        node.error_msg = None
                        dt = time.perf_counter() - t0
                        self._stats_buffer[node.id] = (dt / block_duration_sec) * 100.0
                    except Exception as e:
                        logging.exception(f"Error processing node {node.name} (id: {node.id}): {e}")
                        node.error_msg = str(e)

                # Check if any node marked structure as dirty during the current block processing
                if self.graph.structure_dirty:
                    self.graph.structure_dirty = False
                    self._active_plan = self.graph.compile_execution_plan()
                    self._emit_snapshot()

                now = time.perf_counter()
                if now >= next_telemetry_time:
                    self._drain_nrt_all()
                    stats_buffer = self._stats_buffer
                    global_cpu = sum(stats_buffer.values()) / len(stats_buffer) if stats_buffer else 0.0
                    node_data = {"__cpu__": stats_buffer.copy()}
                    for node in plan.nodes:
                        try:
                            telemetry = node.get_telemetry()
                            if telemetry:
                                node_data[node.id] = telemetry
                        except Exception as e:
                            logging.exception(f"Telemetry fetch failed for node {node.name} ({node.id}): {e}")
                    self._emit_telemetry(global_cpu, node_data)
                    next_telemetry_time = now + telemetry_interval

        gc.enable()
        for n in self.graph.nodes:
            n.stop()
        if self.graph.clock_source:
            self.graph.clock_source.stop_clock()
        self._emit_snapshot()
        logging.info("Engine: Stopped")

    def start(self):
        if self.running:
            return
        self.running = True
        self.abort_flag = False
        if self.graph.clock_source:
            self.graph.clock_source.abort_flag = False
        self._emit_snapshot()
        self._tick_semaphore = threading.Semaphore(self.max_buffered_frames)
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.abort_flag = True
        if self.graph.clock_source:
            self.graph.clock_source.abort_flag = True
        if self._tick_semaphore:
            # Release multiple times to satisfy all possible pending acquires
            # in the worker thread, avoiding deadlocks when stopping.
            for _ in range(self.max_buffered_frames + 1):
                self._tick_semaphore.release()
        if self.thread:
            self.thread.join()
            self.thread = None
            self._tick_semaphore = None
        self._emit_snapshot()
