import torch
import numpy as np
import threading
import time
import queue
import uuid
import json
import abc
from typing import Dict, List, Optional, Any, Tuple

# --- Configuration ---
BLOCK_SIZE = 512
SAMPLE_RATE = 44100
CHANNELS = 2
DTYPE = torch.float32


# --- Interfaces ---
class IClockProvider(abc.ABC):
    def __init__(self):
        self._is_master_clock = False

    def set_master(self, is_master: bool):
        self._is_master_clock = is_master

    @property
    def is_master(self) -> bool:
        return self._is_master_clock

    @abc.abstractmethod
    def start_clock(self):
        pass

    @abc.abstractmethod
    def stop_clock(self):
        pass

    @abc.abstractmethod
    def wait_for_sync(self):
        pass


# --- Data Structures ---


class OutputSlot:
    def __init__(self, name: str, parent: "Node"):
        self.name = name
        self.parent = parent
        self.buffer = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)


class InputSlot:
    def __init__(self, name: str, parent: "Node", param_name: str = None):
        self.name = name
        self.parent = parent
        self.param_name = param_name
        self.connected_outputs: List[OutputSlot] = []
        self._scratch = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    def connect(self, target: OutputSlot):
        if target not in self.connected_outputs:
            self.connected_outputs.append(target)

    def disconnect(self, target=None):
        if target is None:
            self.connected_outputs = []
        else:
            if target in self.connected_outputs:
                self.connected_outputs.remove(target)

    def get_tensor(self) -> torch.Tensor:
        if self.connected_outputs:
            self._scratch.zero_()
            for out in self.connected_outputs:
                self._scratch += out.buffer
            return self._scratch
        if self.param_name and self.param_name in self.parent.params:
            return self.parent.params[self.param_name].get_tensor_cache()
        self._scratch.zero_()
        return self._scratch


class Parameter:
    def __init__(self, value: Any, param_type: str, **kwargs):
        self.value = value
        self._staging = value
        self.type = param_type
        self.meta = kwargs
        self._tensor_cache = torch.tensor([0.0], dtype=DTYPE).expand(CHANNELS, BLOCK_SIZE).clone()
        self._update_cache()

    def set(self, val: Any):
        if self.type == "float":
            self._staging = np.clip(float(val), self.meta.get("min", 0.0), self.meta.get("max", 1.0))
        elif self.type == "int":
            self._staging = int(np.clip(val, self.meta.get("min", 0), self.meta.get("max", 100)))
        elif self.type == "bool":
            self._staging = bool(val)
        elif self.type == "menu":
            self._staging = int(val)
        else:
            self._staging = val

    def sync(self):
        if self.value != self._staging:
            self.value = self._staging
            self._update_cache()

    def _update_cache(self):
        if self.type in ["float", "int", "bool"]:
            try:
                v = float(self.value)
                self._tensor_cache.fill_(v)
            except:
                pass

    def get_tensor_cache(self):
        return self._tensor_cache


class Node:
    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.pos = (0, 0)
        self.inputs: Dict[str, InputSlot] = {}
        self.outputs: Dict[str, OutputSlot] = {}
        self.params: Dict[str, Parameter] = {}

    def add_input(self, name: str, param_name: str = None) -> InputSlot:
        slot = InputSlot(name, self, param_name)
        self.inputs[name] = slot
        return slot

    def add_output(self, name: str) -> OutputSlot:
        slot = OutputSlot(name, self)
        self.outputs[name] = slot
        return slot

    def add_float_param(self, name: str, val: float, min_v=0.0, max_v=1.0):
        self.params[name] = Parameter(val, "float", min=min_v, max=max_v)

    def add_int_param(self, name: str, val: int, min_v=0, max_v=100):
        self.params[name] = Parameter(val, "int", min=min_v, max=max_v)

    def add_bool_param(self, name: str, val: bool):
        self.params[name] = Parameter(val, "bool")

    def add_string_param(self, name: str, val: str):
        self.params[name] = Parameter(val, "string")

    def add_menu_param(self, name: str, items: List[str], initial_idx=0):
        self.params[name] = Parameter(initial_idx, "menu", items=items)

    def sync(self):
        for p in self.params.values():
            p.sync()

    def process(self):
        raise NotImplementedError

    def start(self):
        pass

    def stop(self):
        pass

    def on_ui_param_change(self, param_name: str):
        pass

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "name": self.name,
            "params": {k: v._staging for k, v in self.params.items()},
            "pos": self.pos,
        }

    def load_state(self, data: dict):
        self.pos = tuple(data.get("pos", (0, 0)))
        if "params" in data:
            for k, v in data["params"].items():
                if k in self.params:
                    self.params[k].set(v)
                    self.params[k].sync()


class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.node_map: Dict[str, Node] = {}
        self.execution_order: List[Node] = []
        self.clock_source: Optional[IClockProvider] = None

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

        def visit(n):
            if state[n.id] == 1:
                return
            if state[n.id] == 2:
                return
            state[n.id] = 1
            for inp in n.inputs.values():
                for out in inp.connected_outputs:
                    visit(out.parent)
            state[n.id] = 2
            order.append(n)

        for n in self.nodes:
            if state[n.id] == 0:
                visit(n)
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
            data["nodes"].append(
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.__class__.__name__,
                    "pos": n.pos,
                    "inputs": list(n.inputs.keys()),
                    "outputs": list(n.outputs.keys()),
                    "params": p_data,
                    "monitor_queue": mon_q,
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
            except:
                pass

        snap = self.graph.get_snapshot()
        snap["is_running"] = self.running
        self.output_queue.put(snap)

    def _emit_stats(self, stats_data):
        if not self.output_queue.full():
            self.output_queue.put({"type": "stats", "data": stats_data})

    def _apply_command(self, cmd):
        import plugin_system

        try:
            op = cmd[0]
            if op == "add":
                _, type_name, nid, pos = cmd
                cls = plugin_system.NODE_REGISTRY.get(type_name)
                if cls:
                    node = cls(name=type_name)
                    node.id = nid
                    node.pos = pos
                    self.graph.add_node(node)
                    if self.running:
                        node.start()
            elif op == "del":
                _, nid = cmd
                if self.running:
                    n = self.graph.node_map.get(nid)
                    if n:
                        n.stop()
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
                self.graph = Graph()
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
            for node in self.graph.nodes:
                node.start()

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
                        dt = time.perf_counter() - t0
                        stats_buffer[node.id] = (dt / block_duration_sec) * 100.0
                    except:
                        pass

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
