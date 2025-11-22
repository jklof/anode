import torch
import numpy as np
import uuid
import json
import abc
import threading
import time
import queue
from typing import Dict, List, Optional, Type, Any, Callable

# --- Configuration ---
BLOCK_SIZE = 512
SAMPLE_RATE = 44100
CHANNELS = 2
DTYPE = torch.float32

# --- Registry ---
NODE_REGISTRY: Dict[str, Type['Node']] = {}

def register_node(cls):
    NODE_REGISTRY[cls.__name__] = cls
    return cls

# --- Interfaces ---
class IClockProvider(abc.ABC):
    """
    Interface for nodes that can regulate the Engine's speed (e.g. Audio Output).
    """
    def __init__(self):
        self._is_master_clock = False

    def set_master(self, is_master: bool):
        self._is_master_clock = is_master

    @property
    def is_master(self) -> bool:
        return self._is_master_clock

    @abc.abstractmethod
    def start_clock(self): pass
    
    @abc.abstractmethod
    def stop_clock(self): pass

    @abc.abstractmethod
    def wait_for_sync(self):
        """
        Called by Engine OUTSIDE the lock.
        Implementations should block here until ready to process next frame.
        """
        pass

# --- Data Structures ---

class OutputSlot:
    def __init__(self, name: str, parent: 'Node'):
        self.name = name
        self.parent = parent
        self.buffer = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

class InputSlot:
    def __init__(self, name: str, parent: 'Node', param_name: str = None):
        self.name = name
        self.parent = parent
        self.param_name = param_name
        self.connected_output: Optional[OutputSlot] = None
        self._scratch = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    def connect(self, target: OutputSlot):
        self.connected_output = target

    def disconnect(self):
        self.connected_output = None

    def get_tensor(self) -> torch.Tensor:
        """Hot-path data retrieval."""
        if self.connected_output:
            return self.connected_output.buffer
        
        if self.param_name and self.param_name in self.parent.params:
            p = self.parent.params[self.param_name]
            if p.type == 'float' or p.type == 'int':
                self._scratch.fill_(float(p.value))
                return self._scratch
                
        self._scratch.zero_()
        return self._scratch

class Parameter:
    def __init__(self, value: Any, param_type: str, **kwargs):
        self.value = value      # Active value (Audio Thread)
        self._staging = value   # Staging value (UI Thread)
        self.type = param_type  # 'float', 'int', 'bool', 'string', 'menu'
        self.meta = kwargs      # dict for min, max, items, etc.

    def set(self, val: Any):
        """Called from UI/Main Thread. Handles type casting/clipping."""
        if self.type == 'float':
            self._staging = np.clip(float(val), self.meta.get('min', 0.0), self.meta.get('max', 1.0))
        elif self.type == 'int':
            self._staging = int(np.clip(val, self.meta.get('min', 0), self.meta.get('max', 100)))
        elif self.type == 'bool':
            self._staging = bool(val)
        elif self.type == 'menu':
            self._staging = int(val)
        else:
            self._staging = val

    def sync(self):
        """Called from Audio Thread."""
        self.value = self._staging

class Node:
    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.pos = (0, 0)
        self.inputs: Dict[str, InputSlot] = {}
        self.outputs: Dict[str, OutputSlot] = {}
        self.params: Dict[str, Parameter] = {}

    # --- Setup Helpers ---
    def add_input(self, name: str, param_name: str = None) -> InputSlot:
        slot = InputSlot(name, self, param_name)
        self.inputs[name] = slot
        return slot

    def add_output(self, name: str) -> OutputSlot:
        slot = OutputSlot(name, self)
        self.outputs[name] = slot
        return slot

    def add_float_param(self, name: str, val: float, min_v=0.0, max_v=1.0):
        self.params[name] = Parameter(val, 'float', min=min_v, max=max_v)

    def add_int_param(self, name: str, val: int, min_v=0, max_v=100):
        self.params[name] = Parameter(val, 'int', min=min_v, max=max_v)

    def add_bool_param(self, name: str, val: bool):
        self.params[name] = Parameter(val, 'bool')

    def add_string_param(self, name: str, val: str):
        self.params[name] = Parameter(val, 'string')
        
    def add_menu_param(self, name: str, items: List[str], initial_idx=0):
        self.params[name] = Parameter(initial_idx, 'menu', items=items)

    # --- Runtime ---
    def sync(self):
        for p in self.params.values(): p.sync()

    def process(self): 
        raise NotImplementedError
    
    def start(self): pass
    def stop(self): pass

    def on_ui_param_change(self, param_name: str):
        """Hook called by Controller on Main Thread when a param changes."""
        pass

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            'id': self.id, 
            'type': self.__class__.__name__, 
            'name': self.name,
            'params': {k: v._staging for k,v in self.params.items()}, 
            'pos': self.pos
        }
    
    def load_state(self, data: dict):
        self.pos = tuple(data.get('pos', (0,0)))
        if 'params' in data:
            for k,v in data['params'].items():
                if k in self.params:
                    self.params[k].set(v)
                    self.params[k].sync()
                    try: self.on_ui_param_change(k)
                    except Exception as e: print(f"Error restoring param {k}: {e}")

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
        if node_id not in self.node_map: return
        node = self.node_map[node_id]
        
        for inp in node.inputs.values(): inp.disconnect()
        for other in self.nodes:
            for inp in other.inputs.values():
                if inp.connected_output and inp.connected_output.parent == node:
                    inp.disconnect()
        
        if self.clock_source == node:
            print(f"Graph: Master Clock '{node.name}' removed.")
            self.clock_source = None
            for n in self.nodes:
                if isinstance(n, IClockProvider) and n != node:
                    self.set_master_clock(n)
                    break
        
        self.nodes.remove(node)
        del self.node_map[node_id]
        self.recalculate_order()

    def connect(self, src_id, src_port, dst_id, dst_port):
        src = self.node_map.get(src_id)
        dst = self.node_map.get(dst_id)
        if src and dst and src_port in src.outputs and dst_port in dst.inputs:
            dst.inputs[dst_port].disconnect()
            dst.inputs[dst_port].connect(src.outputs[src_port])
            self.recalculate_order()

    def disconnect(self, dst_id, dst_port):
        dst = self.node_map.get(dst_id)
        if dst and dst_port in dst.inputs:
            dst.inputs[dst_port].disconnect()
            self.recalculate_order()

    def set_master_clock(self, node: Node):
        if not isinstance(node, IClockProvider):
            raise ValueError(f"Node '{node.name}' cannot act as a Clock.")
        
        self.clock_source = node
        for n in self.nodes:
            if isinstance(n, IClockProvider):
                n.set_master(n == node)
        
        print(f"Graph: Master Clock set to '{node.name}'")

    def recalculate_order(self):
        state = {n.id: 0 for n in self.nodes} 
        order = []
        def visit(n):
            if state[n.id] == 1: return 
            if state[n.id] == 2: return 
            state[n.id] = 1
            for inp in n.inputs.values():
                if inp.connected_output: visit(inp.connected_output.parent)
            state[n.id] = 2
            order.append(n)
        for n in self.nodes:
            if state[n.id] == 0: visit(n)
        self.execution_order = order

    def to_json(self) -> str:
        data = {
            'clock_id': self.clock_source.id if self.clock_source else None,
            'nodes': [n.to_dict() for n in self.nodes],
            'connections': []
        }
        for dst in self.nodes:
            for d_port, inp in dst.inputs.items():
                if inp.connected_output:
                    out = inp.connected_output
                    data['connections'].append({
                        'src_id': out.parent.id, 'src_port': out.name,
                        'dst_id': dst.id, 'dst_port': d_port
                    })
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> 'Graph':
        data = json.loads(json_str)
        g = Graph()
        for n_data in data['nodes']:
            if n_data['type'] in NODE_REGISTRY:
                node = NODE_REGISTRY[n_data['type']](n_data['name'])
                node.id = n_data['id']
                node.load_state(n_data)
                g.add_node(node)
        for c in data['connections']:
            src = g.node_map.get(c['src_id'])
            dst = g.node_map.get(c['dst_id'])
            if src and dst: g.connect(c['src_id'], c['src_port'], c['dst_id'], c['dst_port'])
        if data.get('clock_id') and data['clock_id'] in g.node_map:
            g.set_master_clock(g.node_map[data['clock_id']])
        return g

class Engine:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.command_queue = queue.Queue()

    def push_command(self, func: Callable):
        self.command_queue.put(func)

    def _worker(self):
        print("Engine: Started")
        with self.lock:
            for node in self.graph.nodes: node.start()

        while self.running:
            # 1. Process Commands
            while not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                    with self.lock: cmd()
                except Exception as e: print(f"Cmd Error: {e}")

            # 2. Wait for Sync (Backpressure)
            if self.graph.clock_source:
                self.graph.clock_source.wait_for_sync()
            else:
                time.sleep(0.001)

            # 3. Process DSP
            with self.lock:
                if not self.running: break
                for node in self.graph.nodes: node.sync()
                with torch.no_grad():
                    try:
                        for node in self.graph.execution_order: node.process()
                    except Exception: pass
        
        with self.lock:
            for node in self.graph.nodes: node.stop()
        print("Engine: Stopped")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()

    def stop(self):
        self.running = False
        with self.lock:
            for node in self.graph.nodes: node.stop()
        if self.thread: self.thread.join()