import torch
import numpy as np
import uuid
import abc
from typing import Dict, List, Any

# --- Configuration ---
BLOCK_SIZE = 512
SAMPLE_RATE = 48000
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
        if self.connected_outputs:
            # Determine if we need Stereo or Mono output
            max_channels = 1
            for out in self.connected_outputs:
                if out.buffer.shape[0] > max_channels:
                    max_channels = out.buffer.shape[0]

            # Create a view of the scratch buffer (no memory allocation)
            # If max_channels is 1, this is shape (1, 512)
            # If max_channels is 2, this is shape (2, 512)
            target = self._scratch[:max_channels]

            target.zero_()
            for out in self.connected_outputs:
                # PyTorch add_ handles broadcasting automatically:
                # (2, N) + (1, N) -> (2, N)
                # (1, N) + (1, N) -> (1, N)
                target.add_(out.buffer)
            return target

        if self.param_name and self.param_name in self.parent.params:
            return self.parent.params[self.param_name].get_tensor_cache()

        # Default case: return silence.
        # We return the full buffer (Stereo) to be safe for uninitialized inputs,
        # or we could return mono silence. Let's default to full scratch.
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
    def __init__(self, name: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name if name else self.__class__.__name__
        self.pos = (0, 0)
        self.error_msg = None
        self.inputs: Dict[str, InputSlot] = {}
        self.outputs: Dict[str, OutputSlot] = {}
        self.params: Dict[str, Parameter] = {}

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
        """
        mode: 'open' or 'save'
        filter: e.g. "WAV Files (*.wav);;All Files (*.*)"
        """
        self.params[name] = Parameter(val, "file", filter=filter, mode=mode)

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
