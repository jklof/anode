import torch
import numpy as np
import uuid
import abc
import threading
from typing import Dict, List, Any, Optional

# --- Configuration ---
BLOCK_SIZE = 512
SAMPLE_RATE = 48000
CHANNELS = 2
DTYPE = torch.float32


class TelemetryRingBuffer:
    """
    Lock-free Single-Producer Single-Consumer (SPSC) pre-allocated ring buffer
    for passing telemetry arrays from the audio thread to UI widgets.

    Zero steady-state heap allocation:
    - Pre-allocates a fixed array of slots (capacity, *shape).
    - Audio thread writes into storage[head] via np.copyto (no allocations).
    - Producer never contends with consumer.
    """
    __slots__ = ("capacity", "shape", "dtype", "storage", "head", "tail")

    def __init__(self, capacity: int, shape: tuple, dtype=np.float32):
        self.capacity = max(2, capacity)
        self.shape = shape
        self.dtype = dtype
        self.storage = np.zeros((self.capacity, *shape), dtype=dtype)
        self.head = 0  # Written only by audio thread
        self.tail = 0  # Written only by UI thread

    def push(self, data: np.ndarray) -> bool:
        """Audio thread: copies data into head slot. Never blocks, zero allocation."""
        next_head = (self.head + 1) % self.capacity
        if next_head == self.tail:
            return False  # Overflow -> drop frame cleanly
        np.copyto(self.storage[self.head], data)
        self.head = next_head
        return True

    def pop_latest(self) -> Optional[np.ndarray]:
        """UI thread: returns the latest available frame as an isolated array snapshot."""
        h = self.head
        t = self.tail
        if h == t:
            return None
        latest_idx = (h - 1) % self.capacity
        result = self.storage[latest_idx].copy()
        self.tail = h
        return result

    def pop_all(self) -> List[np.ndarray]:
        """UI thread: returns all frames in chronological order since last pop."""
        h = self.head
        t = self.tail
        if h == t:
            return []
        items = []
        curr = t
        while curr != h:
            items.append(self.storage[curr].copy())
            curr = (curr + 1) % self.capacity
        self.tail = h
        return items


class TelemetryDictRingBuffer:
    """Pre-allocated ring of dictionaries for DataDisplayNode HUD statistics."""
    __slots__ = ("capacity", "slots", "head", "tail")

    def __init__(self, capacity: int = 4):
        self.capacity = max(2, capacity)
        self.slots = [{} for _ in range(self.capacity)]
        self.head = 0
        self.tail = 0

    def push(self, data_dict: dict) -> bool:
        next_head = (self.head + 1) % self.capacity
        if next_head == self.tail:
            return False
        self.slots[self.head].update(data_dict)
        self.head = next_head
        return True

    def pop_latest(self) -> Optional[dict]:
        h = self.head
        t = self.tail
        if h == t:
            return None
        latest_idx = (h - 1) % self.capacity
        self.tail = h
        return dict(self.slots[latest_idx])

    def try_pop(self):
        """Consumer thread: attempt to pop. Returns (item, True) or (None, False)."""
        h = self.head
        t = self.tail
        if h == t:
            return None, False
        latest_idx = (h - 1) % self.capacity
        self.tail = h
        return dict(self.slots[latest_idx]), True


class SPSCRingBuffer:
    """
    Generic lock-free Single-Producer Single-Consumer ring buffer for
    passing arbitrary Python objects between threads.
    Used by FileRecorder for block indices (not NumPy arrays).
    """
    __slots__ = ("_buf", "_cap", "_head", "_tail")

    def __init__(self, capacity: int = 2):
        if capacity < 2:
            raise ValueError("SPSCRingBuffer capacity must be >= 2")
        self._cap = capacity
        self._buf = [None] * capacity
        self._head = 0
        self._tail = 0

    def try_push(self, item) -> bool:
        """Audio thread: attempt to push. Returns False if full (item dropped)."""
        next_head = (self._head + 1) % self._cap
        if next_head == self._tail:
            return False
        self._buf[self._head] = item
        self._head = next_head
        return True

    def try_pop(self):
        """UI/writer thread: attempt to pop. Returns (item, True) or (None, False)."""
        h = self._head
        t = self._tail
        if h == t:
            return None, False
        item = self._buf[t]
        self._buf[t] = None
        self._tail = (t + 1) % self._cap
        return item, True


# --- Interfaces ---
class IClockProvider(abc.ABC):
    def __init__(self):
        self._is_master_clock = False
        self.abort_flag = False

    def set_master(self, is_master: bool):
        self._is_master_clock = is_master

    @property
    def is_master(self) -> bool:
        return self._is_master_clock

    @abc.abstractmethod
    def start_clock(self, tick_callback):
        pass

    @abc.abstractmethod
    def stop_clock(self):
        pass


# --- Data Structures ---


class OutputSlot:
    def __init__(self, name: str, parent: "Node", channels: int = CHANNELS):
        self.name = name
        self.parent = parent
        self.buffer = torch.zeros((channels, BLOCK_SIZE), dtype=DTYPE)


class InputSlot:
    def __init__(self, name: str, parent: "Node", param_name: str = None):
        self.name = name
        self.parent = parent
        self.param_name = param_name
        self.connected_outputs: List[OutputSlot] = []
        # Allocate max channels (Stereo) but we will slice it dynamically
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
        """
        Retrieves the input audio buffer.
        """
        if self.connected_outputs:
            # SAFETY CHANGE: Removed Zero-Copy Optimization.
            # We always copy to self._scratch to ensure downstream nodes cannot
            # corrupt the upstream buffer via in-place operations.
            # Since self._scratch is pre-allocated, this avoids GC overhead.

            # Determine max channels
            max_channels = 1
            for out in self.connected_outputs:
                if out.buffer.shape[0] > max_channels:
                    max_channels = out.buffer.shape[0]

            # Create a view of the scratch buffer (no memory allocation)
            target = self._scratch[:max_channels]

            # Directly copy the buffer from the first connected output into target
            target.copy_(self.connected_outputs[0].buffer[:max_channels])

            # Then add the remaining connected outputs
            for out in self.connected_outputs[1:]:
                target.add_(out.buffer[:max_channels])

            # Ghosting Fix: Zero out unused channels in scratch buffer
            if max_channels < CHANNELS:
                self._scratch[max_channels:].zero_()

            return target

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
        try:
            if isinstance(self.value, (np.ndarray, torch.Tensor)) or isinstance(
                self._staging, (np.ndarray, torch.Tensor)
            ):
                changed = True
            else:
                changed = bool(self.value != self._staging)
        except Exception:
            changed = True
        if changed:
            self.value = self._staging
            self._update_cache()

    def _update_cache(self):
        if self.type in ["float", "int", "bool"]:
            try:
                v = float(self.value)
                self._tensor_cache.fill_(v)
            except (ValueError, TypeError):
                pass

    def get_tensor_cache(self):
        return self._tensor_cache

    def get_staging_safe(self):
        return self._staging


class Node:
    category: str = "Uncategorized"
    label: str = ""

    def __init__(self, name: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name if name else (getattr(self, "label", "") or self.__class__.__name__)
        self.pos = (0, 0)
        self.error_msg = None
        self.inputs: Dict[str, InputSlot] = {}
        self.outputs: Dict[str, OutputSlot] = {}
        self.params: Dict[str, Parameter] = {}
        self._nrt_epoch = 0
        self._nrt_inbox = None

    def add_input(self, name: str, param_name: str = None) -> InputSlot:
        slot = InputSlot(name, self, param_name)
        self.inputs[name] = slot
        return slot

    def add_output(self, name: str, channels: int = CHANNELS) -> OutputSlot:
        slot = OutputSlot(name, self, channels)
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

    def add_file_param(self, name: str, val: str, filter: str = "All Files (*.*)", mode: str = "open"):
        self.params[name] = Parameter(val, "file", filter=filter, mode=mode)

    def submit_nrt(self, fn, *args, tag=None):
        """Schedule fn(*args) on the engine's background pool. Never blocks.
        Override on_nrt_complete() to receive the result on a later tick."""
        if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
            self.graph.engine.nrt.submit(self, fn, args, tag)

    def on_nrt_complete(self, tag, ok, result):
        """Override in subclasses to receive background task results. Called
        from sync(), so it's safe to mutate node state here."""
        pass

    def sync(self):
        for p in self.params.values():
            p.sync()

    def process(self):
        raise NotImplementedError

    def start(self):
        """Called when the audio engine starts."""
        pass

    def stop(self):
        """Called when the audio engine stops."""
        pass

    def remove(self):
        """Called when the node is deleted from the graph."""
        pass

    def get_telemetry(self) -> dict:
        return {}

    def on_ui_param_change(self, param_name: str):
        pass

    def request_graph_rebuild(self):
        """
        Signals to the parent graph that this node's internal structure (such as
        inputs, outputs, or custom layout) has changed and needs to be resynchronized with the UI.
        """
        if hasattr(self, "graph") and self.graph:
            self.graph.mark_dirty()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "name": self.name,
            "params": {k: v.get_staging_safe() for k, v in self.params.items()},
            "pos": self.pos,
        }

    def load_state(self, data: dict):
        self.pos = tuple(data.get("pos", (0, 0)))
        if "params" in data:
            for k, v in data["params"].items():
                if k in self.params:
                    # FIX: Handle full snapshot dicts (from Undo/Restore) vs simple values (from Load/Save)
                    # Snapshot format: {"value": 0.5, "type": "float", ...}
                    # Simple format: 0.5
                    val = v
                    if isinstance(v, dict) and "value" in v:
                        val = v["value"]

                    self.params[k].set(val)
                    self.params[k].sync()
