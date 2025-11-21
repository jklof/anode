import torch
import numpy as np
import uuid
import json
import abc
import threading
from typing import Dict, List, Optional, Type

# --- Configuration ---
BLOCK_SIZE = 512
SAMPLE_RATE = 44100
CHANNELS = 2
DTYPE = torch.float32

# --- Registry ---
NODE_REGISTRY: Dict[str, Type["Node"]] = {}


def register_node(cls):
    NODE_REGISTRY[cls.__name__] = cls
    return cls


# --- Interfaces ---
class IClockProvider(abc.ABC):
    """Mixin for nodes that are capable of regulating the Engine's speed."""

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


# --- Data Structures ---
class OutputSlot:
    def __init__(self, name: str, parent: "Node"):
        self.name = name
        self.parent = parent
        self.buffer = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)


class Parameter:
    def __init__(self, value: float, min_v=0.0, max_v=1.0):
        self.value = value
        self._staging = value
        self.min, self.max = min_v, max_v

    def set(self, val: float):
        """Thread-safe setter."""
        self._staging = np.clip(val, self.min, self.max)

    def sync(self):
        """Sync staging to active value."""
        self.value = self._staging


class InputSlot:
    def __init__(self, name: str, parent: "Node", param_name: str = None):
        self.name = name
        self.parent = parent
        self.param_name = param_name
        self.connected_output: Optional[OutputSlot] = None
        self._scratch = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    def connect(self, target: OutputSlot):
        self.connected_output = target

    def get_tensor(self) -> torch.Tensor:
        """Hot-path data retrieval."""
        if self.connected_output:
            return self.connected_output.buffer
        if self.param_name:
            val = self.parent.params[self.param_name].value
            self._scratch.fill_(val)
            return self._scratch
        self._scratch.zero_()
        return self._scratch


class Node:
    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
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

    def add_param(self, name: str, val: float, min_v=0.0, max_v=1.0):
        self.params[name] = Parameter(val, min_v, max_v)

    def sync(self):
        for p in self.params.values():
            p.sync()

    def process(self):
        raise NotImplementedError

    def start(self):
        pass

    def stop(self):
        pass

    # Serialization
    def to_dict(self):
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "name": self.name,
            "params": {k: v._staging for k, v in self.params.items()},
        }

    def load_state(self, data):
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

    def connect(self, src_node, src_port, dst_node, dst_port):
        out_slot = src_node.outputs[src_port]
        dst_node.inputs[dst_port].connect(out_slot)
        self.recalculate_order()

    def set_master_clock(self, node: Node):
        if not isinstance(node, IClockProvider):
            raise ValueError(f"Node '{node.name}' cannot act as a Clock.")

        if self.clock_source:
            self.clock_source.set_master(False)

        self.clock_source = node
        self.clock_source.set_master(True)
        print(f"Graph: Master Clock set to '{node.name}'")

    def recalculate_order(self):
        visited, order = set(), []

        def visit(n):
            if n in visited:
                return
            for inp in n.inputs.values():
                if inp.connected_output:
                    visit(inp.connected_output.parent)
            visited.add(n)
            order.append(n)

        for n in self.nodes:
            visit(n)
        self.execution_order = order

    def to_json(self):
        data = {
            "clock_id": self.clock_source.id if self.clock_source else None,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [],
        }
        for dst in self.nodes:
            for d_port, inp in dst.inputs.items():
                if inp.connected_output:
                    out = inp.connected_output
                    data["connections"].append(
                        {"src_id": out.parent.id, "src_port": out.name, "dst_id": dst.id, "dst_port": d_port}
                    )
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str):
        d = json.loads(json_str)
        g = Graph()
        for n in d["nodes"]:
            if n["type"] in NODE_REGISTRY:
                o = NODE_REGISTRY[n["type"]](n["name"])
                o.id = n["id"]
                o.load_state(n)
                g.add_node(o)
        for c in d["connections"]:
            src = g.node_map.get(c["src_id"])
            dst = g.node_map.get(c["dst_id"])
            if src and dst:
                g.connect(src, c["src_port"], dst, c["dst_port"])
        if d.get("clock_id") and d["clock_id"] in g.node_map:
            g.set_master_clock(g.node_map[d["clock_id"]])
        return g


class Engine:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def _worker(self):
        print("Engine: Started")
        with self.lock:
            for node in self.graph.nodes:
                node.start()

        while self.running:
            with self.lock:
                # Double check running under lock to avoid starting a process cycle during stop
                if not self.running:
                    break

                for node in self.graph.nodes:
                    node.sync()

                with torch.no_grad():
                    try:
                        for node in self.graph.execution_order:
                            node.process()
                    except Exception as e:
                        print(f"Engine Process Error: {e}")

        print("Engine: Worker Loop Finished")

    def start(self):
        if not self.graph.clock_source:
            print("WARNING: No Master Clock set.")
        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def stop(self):
        self.running = False

        # Explicitly stop nodes from Main Thread to ensure immediate stream shutdown.
        # This prevents the "U" (Underrun) errors by ensuring streams are aborted
        # before the worker thread logic finishes.
        with self.lock:
            for node in self.graph.nodes:
                node.stop()

        if self.thread:
            self.thread.join()
        print("Engine: Stopped")
