import collections
import gc
import torch
import threading
import time
import queue
import json
import logging
from typing import Dict, List, Optional, Tuple
import plugin_system
from base import BLOCK_SIZE, SAMPLE_RATE, IClockProvider, Node


class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.node_map: Dict[str, Node] = {}
        self.execution_order: List[Node] = []
        self.clock_source: Optional[IClockProvider] = None

    def _get_upstream_nodes(self, node: Node) -> List[Node]:
        upstream = []
        for inp in node.inputs.values():
            for out in inp.connected_outputs:
                upstream.append(out.parent)
        return upstream

    def add_node(self, node: Node):
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
        if self.clock_source is None:
            for n in self.nodes:
                if isinstance(n, IClockProvider):
                    self.set_master_clock(n)
                    break
        self.recalculate_order()

    def connect(self, src_id, src_port, dst_id, dst_port):
        src = self.node_map.get(src_id)
        dst = self.node_map.get(dst_id)
        if src and dst and src_port in src.outputs and dst_port in dst.inputs:
            dst.inputs[dst_port].connect(src.outputs[src_port])
            self.recalculate_order()

    def disconnect(self, src_id, src_port, dst_id, dst_port):
        src_node = self.node_map.get(src_id)
        dst_node = self.node_map.get(dst_id)
        if src_node and dst_node and src_port in src_node.outputs and dst_port in dst_node.inputs:
            output_slot = src_node.outputs[src_port]
            dst_node.inputs[dst_port].disconnect(target=output_slot)
            self.recalculate_order()

    def set_master_clock(self, node: Node):
        if not isinstance(node, IClockProvider):
            return
        self.clock_source = node
        for n in self.nodes:
            if isinstance(n, IClockProvider):
                n.set_master(n == node)

    def recalculate_order(self):
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

        self.execution_order = order

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


class Engine:
    def __init__(self):
        self.graph = Graph()
        self.reload_version = 0
        self.running = False
        self.abort_flag = False
        self.command_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=100)
        self.thread = None
        self._tick_semaphore = None
        self.max_buffered_frames = 4

    def tick(self):
        self._tick_semaphore.release()

    def push_command(self, cmd: Tuple):
        if self.running:
            self.command_queue.put(cmd)
        else:
            self._apply_command(cmd)
            # Always emit snapshot when engine is stopped to ensure UI updates
            # This is especially important for parameter changes when audio is off
            self._emit_snapshot()

    def _emit_snapshot(self):
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except Exception:
                logging.exception("Error getting message from output queue")

        snap = self.graph.get_snapshot()
        snap["is_running"] = self.running
        snap["reload_version"] = self.reload_version
        self.output_queue.put(snap)

    def _emit_telemetry(self, cpu_load, node_data):
        if not self.output_queue.full():
            self.output_queue.put({"type": "telemetry", "cpu_load": cpu_load, "node_data": node_data})

    def _apply_command(self, cmd):
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
                    node.id = nid
                    node.pos = pos

                if node:

                    # Apply initial parameters BEFORE starting the node (atomic creation)
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
                                node.on_ui_param_change(param_name)

                    self.graph.add_node(node)
                    if self.running:
                        try:
                            node.start()
                        except Exception as e:
                            logging.exception(f"Error starting node {node.name} on add")
                            node.error_msg = f"Start Error: {e}"
                    try:
                        self.output_queue.put_nowait({"type": "node_added", "node": self.graph._get_node_data(node)})
                    except Exception: pass
            elif op == "del":
                _, nid = cmd
                if self.running:
                    n = self.graph.node_map.get(nid)
                    if n:
                        n.stop()

                # --- MEMORY CLEANUP ---
                n = self.graph.node_map.get(nid)
                if n:
                    # Clean up C++ handles or other resources
                    n.remove()

                self.graph.remove_node(nid)
                try:
                    self.output_queue.put_nowait({"type": "node_removed", "node_id": nid})
                except Exception: pass
            elif op == "conn":
                _, sid, sp, did, dp = cmd
                self.graph.connect(sid, sp, did, dp)
                try:
                    self.output_queue.put_nowait({"type": "connected", "src_id": sid, "src_port": sp, "dst_id": did, "dst_port": dp})
                except Exception: pass
            elif op == "disconn":
                _, sid, sp, did, dp = cmd
                self.graph.disconnect(sid, sp, did, dp)
                try:
                    self.output_queue.put_nowait({"type": "disconnected", "src_id": sid, "src_port": sp, "dst_id": did, "dst_port": dp})
                except Exception: pass
            elif op == "param":
                _, nid, p, val = cmd
                node = self.graph.node_map.get(nid)
                if node and p in node.params:
                    node.params[p].set(val)
                    node.on_ui_param_change(p)
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
                    except Exception: pass
            elif op == "move":
                _, nid, x, y = cmd
                node = self.graph.node_map.get(nid)
                if node:
                    node.pos = (x, y)
                    try:
                        self.output_queue.put_nowait({"type": "node_moved", "node_id": nid, "pos": (x, y)})
                    except Exception: pass

            # --- NEW: Restore Command for robust Undo ---
            elif op == "restore":
                _, n_data_payload = cmd
                if isinstance(n_data_payload, tuple):
                    node_data, node_instance = n_data_payload
                else:
                    node_data, node_instance = n_data_payload, None

                cls = plugin_system.NODE_REGISTRY.get(node_data["type"])
                if cls:
                    node = node_instance if node_instance else cls(node_data["name"])
                    node.id = node_data["id"]
                    # This restores everything: pos, params, internal meta
                    if not node_instance:
                        node.load_state(node_data)
                    self.graph.add_node(node)
                    if self.running:
                        try:
                            node.start()
                        except Exception as e:
                            logging.exception(f"Error starting restored node {node.name}")
                            node.error_msg = f"Start Error: {e}"
                    try:
                        self.output_queue.put_nowait({"type": "node_added", "node": self.graph._get_node_data(node)})
                    except Exception: pass
            # --------------------------------------------

            elif op == "clear":
                for n in self.graph.nodes:
                    n.stop()
                    n.remove()
                self.graph = Graph()
                gc.collect()
                self._emit_snapshot()

            elif op == "save":
                _, filename = cmd
                try:
                    json_str = self.graph.to_json()
                    with open(filename, "w") as f:
                        f.write(json_str)
                    print(f"Saved patch to {filename}")
                except Exception as e:
                    print(f"Save Error: {e}")

            elif op == "load":
                _, json_str = cmd
                for n in self.graph.nodes:
                    n.stop()
                    n.remove()
                self.graph = Graph()
                try:
                    data = json.loads(json_str)
                    if not isinstance(data, dict):
                        raise ValueError("Loaded data is not a valid JSON object.")
                        
                    new_graph = Graph()
                    for n_data in data.get("nodes", []):
                        if not isinstance(n_data, dict):
                            continue
                        cls = plugin_system.NODE_REGISTRY.get(n_data.get("type"))
                        if cls:
                            node = cls(n_data.get("name", ""))
                            if "id" in n_data:
                                node.id = n_data["id"]
                            node.load_state(n_data)
                            new_graph.add_node(node)
                    for c in data.get("connections", []):
                        if not isinstance(c, dict):
                            continue
                        src_id = c.get("src_id")
                        dst_id = c.get("dst_id")
                        if src_id in new_graph.node_map and dst_id in new_graph.node_map:
                            new_graph.connect(src_id, c.get("src_port"), dst_id, c.get("dst_port"))
                    if data.get("clock_id") and data["clock_id"] in new_graph.node_map:
                        new_graph.set_master_clock(new_graph.node_map[data["clock_id"]])
                    self.graph = new_graph
                    if self.running:
                        for n in self.graph.nodes:
                            if n == self.graph.clock_source:
                                n.start_clock(self.tick)
                            else:
                                n.start()
                    self._emit_snapshot()
                    gc.collect()
                except Exception as e:
                    print(f"Load Failed: {e}")

            elif op == "reload":
                print("Engine: Reloading plugins...")
                current_json = self.graph.to_json()
                for n in self.graph.nodes:
                    n.stop()
                    n.remove()
                self.graph = Graph()
                self.reload_version += 1
                try:
                    plugin_system.load_plugins()
                except Exception as e:
                    print(f"Engine: Reload failed: {e}")
                    return
                try:
                    data = json.loads(current_json)
                    new_graph = Graph()
                    for n_data in data["nodes"]:
                        cls = plugin_system.NODE_REGISTRY.get(n_data["type"])
                        if cls:
                            node = cls(n_data["name"])
                            node.id = n_data["id"]
                            node.load_state(n_data)
                            new_graph.add_node(node)
                    for c in data["connections"]:
                        if c["src_id"] in new_graph.node_map and c["dst_id"] in new_graph.node_map:
                            new_graph.connect(c["src_id"], c["src_port"], c["dst_id"], c["dst_port"])
                    if data.get("clock_id") and data["clock_id"] in new_graph.node_map:
                        new_graph.set_master_clock(new_graph.node_map[data["clock_id"]])
                    self.graph = new_graph
                    if self.running:
                        for n in self.graph.nodes:
                            if n == self.graph.clock_source:
                                n.start_clock(self.tick)
                            else:
                                n.start()
                    self._emit_snapshot()
                    print("Engine: Hot reload complete.")
                    gc.collect()
                except Exception as e:
                    print(f"Engine: Restore failed after reload: {e}")

            elif op == "snapshot":
                self._emit_snapshot()

        except Exception as e:
            print(f"Cmd Error: {e}")

    def _worker(self):
        print("Engine: Started")
        gc.disable()
        with torch.no_grad():

            # --- STARTUP CLEANUP ---
            # 1. Zero out all buffers to prevent "stuck notes" or stale audio glitches
            for node in self.graph.nodes:
                for out_slot in node.outputs.values():
                    out_slot.buffer.zero_()
                for inp_slot in node.inputs.values():
                    inp_slot._scratch.zero_()

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
            stats_buffer = {}
            last_gc_time = time.time()
            GC_INTERVAL = 5.0

            while self.running:
                while not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    self._apply_command(cmd)

                if self.graph.clock_source:
                    # Step A (Non-blocking check)
                    acquired = self._tick_semaphore.acquire(blocking=False)
                    # Step B (If NOT acquired)
                    if not acquired:
                        # This means the semaphore count is 0. The engine has filled all buffers and is waiting on hardware. This is our "Safety Window".
                        if time.time() - last_gc_time > GC_INTERVAL:
                            gc.collect(0)  # Only generation 0
                            last_gc_time = time.time()
                        # Now call self._tick_semaphore.acquire(blocking=True) to wait for the actual hardware tick.
                        self._tick_semaphore.acquire(blocking=True)
                    # If acquired, we proceed directly (already decremented)
                else:
                    # Fallback sleep if no clock source is defined to prevent 100% CPU usage
                    time.sleep(BLOCK_SIZE / SAMPLE_RATE)

                for node in self.graph.nodes:
                    node.sync()

                # Snapshot to avoid race conditions with graph deletion
                current_order = list(self.graph.execution_order)
                for node in current_order:
                    try:
                        t0 = time.perf_counter()
                        node.process()
                        node.error_msg = None
                        dt = time.perf_counter() - t0
                        stats_buffer[node.id] = (dt / block_duration_sec) * 100.0
                    except Exception as e:
                        logging.exception(f"Error processing node {node.name} (id: {node.id}): {e}")
                        node.error_msg = str(e)

                now = time.perf_counter()
                if now >= next_telemetry_time:
                    global_cpu = sum(stats_buffer.values()) / len(stats_buffer) if stats_buffer else 0.0
                    node_data = {"__cpu__": stats_buffer.copy()}
                    for node in self.graph.nodes:
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
        print("Engine: Stopped")

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
            self._tick_semaphore.release()
        if self.thread:
            self.thread.join()
            self.thread = None
            self._tick_semaphore = None
        self._emit_snapshot()
