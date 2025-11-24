import torch
import threading
import time
import queue
import json
import logging
from typing import Dict, List, Optional, Tuple
import logging
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
        state = {n.id: 0 for n in self.nodes}
        order = []

        for root_node in self.nodes:
            if state[root_node.id] != 0:
                continue

            stack = [(root_node, self._get_upstream_nodes(root_node))]

            while stack:
                parent, children = stack[-1]

                if state[parent.id] == 0:
                    state[parent.id] = 1

                found_unvisited_child = False
                while children:
                    child = children.pop(0)
                    if state[child.id] == 1:
                        logging.warning(f"Cycle detected involving node {child.name}")
                        continue
                    if state[child.id] == 0:
                        stack.append((child, self._get_upstream_nodes(child)))
                        found_unvisited_child = True
                        break

                if not found_unvisited_child:
                    stack.pop()
                    state[parent.id] = 2
                    order.append(parent)

        self.execution_order = order

    def get_snapshot(self) -> dict:
        data = {
            "type": "graph_update",
            "clock_id": self.clock_source.id if self.clock_source else None,
            "nodes": [],
            "connections": [],
        }
        for n in self.nodes:
            p_data = {}
            for k, p in n.params.items():
                p_data[k] = {"value": p._staging, "type": p.type, "meta": p.meta}
            mon_q = getattr(n, "monitor_queue", None)

            is_clock_provider = isinstance(n, IClockProvider)
            is_current_master = n == self.clock_source

            data["nodes"].append(
                {
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
            )
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
        self.command_queue = queue.Queue()
        self.output_queue = queue.Queue(maxsize=5)
        self.thread = None

    def push_command(self, cmd: Tuple):
        if self.running:
            self.command_queue.put(cmd)
        else:
            self._apply_command(cmd)
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

    def _emit_stats(self, stats_data):
        if not self.output_queue.full():
            self.output_queue.put({"type": "stats", "data": stats_data})

    def _apply_command(self, cmd):
        try:
            op = cmd[0]
            if op == "add":
                _, type_name, nid, pos = cmd
                cls = plugin_system.NODE_REGISTRY.get(type_name)
                if cls:
                    node = cls()
                    node.id = nid
                    node.pos = pos
                    self.graph.add_node(node)
                    if self.running:
                        try:
                            node.start()
                        except:
                            pass
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
            elif op == "conn":
                _, sid, sp, did, dp = cmd
                self.graph.connect(sid, sp, did, dp)
            elif op == "disconn":
                _, sid, sp, did, dp = cmd
                self.graph.disconnect(sid, sp, did, dp)
            elif op == "param":
                _, nid, p, val = cmd
                node = self.graph.node_map.get(nid)
                if node and p in node.params:
                    node.params[p].set(val)
                    node.on_ui_param_change(p)
            elif op == "clock":
                _, nid = cmd
                node = self.graph.node_map.get(nid)
                if node:
                    self.graph.set_master_clock(node)
            elif op == "move":
                _, nid, x, y = cmd
                node = self.graph.node_map.get(nid)
                if node:
                    node.pos = (x, y)
            elif op == "clear":
                for n in self.graph.nodes:
                    n.stop()
                    n.remove()
                self.graph = Graph()

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
                try:
                    data = json.loads(json_str)
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
                            n.start()
                    self._emit_snapshot()
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
                            n.start()
                    self._emit_snapshot()
                    print("Engine: Hot reload complete.")
                except Exception as e:
                    print(f"Engine: Restore failed after reload: {e}")

        except Exception as e:
            print(f"Cmd Error: {e}")

    def _worker(self):
        print("Engine: Started")
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
                    node.start()
                except Exception as e:
                    logging.exception(f"Error starting node {node.name}")
                    node.error_msg = f"Start Error: {e}"

            block_duration_sec = BLOCK_SIZE / SAMPLE_RATE
            stats_interval = 0.1
            next_stats_time = time.perf_counter() + stats_interval
            stats_buffer = {}

            while self.running:
                dirty = False
                while not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    self._apply_command(cmd)
                    if cmd[0] in ["add", "del", "conn", "disconn", "clear", "load", "reload"]:
                        dirty = True

                if dirty:
                    self._emit_snapshot()

                if self.graph.clock_source:
                    self.graph.clock_source.wait_for_sync()
                else:
                    time.sleep(0.001)

                for node in self.graph.nodes:
                    node.sync()

                for node in self.graph.execution_order:
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
                if now >= next_stats_time:
                    self._emit_stats(stats_buffer.copy())
                    next_stats_time = now + stats_interval

        for n in self.graph.nodes:
            n.stop()
        self._emit_snapshot()
        print("Engine: Stopped")

    def start(self):
        if self.running:
            return
        self.running = True
        self._emit_snapshot()
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self._emit_snapshot()
