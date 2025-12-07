import ctypes
import os
import sys
import torch
import logging
from base import Node, BLOCK_SIZE, CHANNELS

logger = logging.getLogger(__name__)


class FFINode(Node):
    """
    A generic base class for C++ nodes.
    Assumes the C++ library implements the Standard ANode C-ABI.
    """

    # Subclasses define these
    LIB_NAME: str = ""  # Name of the .dll/.so file (without extension)
    PARAM_MAP: dict = {}  # Map param name -> C++ integer ID: {"vol": 0, "freq": 1}

    def __init__(self, name: str):
        super().__init__(name)
        self.lib = None
        self.dsp_handle = None
        # Pre-allocate persistent scratch buffer for zero-allocation copying
        self._ffi_in_buffer = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
        self._load_library()

        # Initialize C++ object
        if self.lib:
            self.dsp_handle = self.lib.create()
            if not self.dsp_handle:
                logger.error(f"[{self.name}] Failed to create C++ instance.")
                self.error_msg = "C++ Init Failed"

    def _load_library(self):
        if not self.LIB_NAME:
            return

        # Determine extension
        ext = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
        lib_filename = f"{self.LIB_NAME}{ext}"

        # Look in the same folder as the defining python file
        # This handles the case where plugins are in subfolders
        module_path = sys.modules[self.__class__.__module__].__file__
        folder = os.path.dirname(os.path.abspath(module_path))
        path = os.path.join(folder, lib_filename)

        try:
            self.lib = ctypes.CDLL(path)
            self._bind_functions()
        except OSError as e:
            logger.error(f"[{self.name}] Could not load library at {path}: {e}")
            self.error_msg = f"Missing {lib_filename}"

    def _bind_functions(self):
        """Bind the standard C-ABI functions."""
        # void* create()
        self.lib.create.restype = ctypes.c_void_p
        self.lib.create.argtypes = []

        # void destroy(void* handle)
        self.lib.destroy.restype = None
        self.lib.destroy.argtypes = [ctypes.c_void_p]

        # void process(void* handle, float* in, float* out, int channels, int frames)
        self.lib.process.restype = None
        self.lib.process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]

        # void set_param(void* handle, int param_id, float value)
        self.lib.set_param.restype = None
        self.lib.set_param.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

    def _preprocess_input(self, in_tensor: torch.Tensor, scratch_buffer: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to modify input tensor before C++ processing. Default pass-through."""
        return in_tensor

    def on_ui_param_change(self, param_name: str):
        """Automatically pushes UI changes to C++."""
        super().on_ui_param_change(param_name)
        if self.dsp_handle and param_name in self.PARAM_MAP:
            self.params[param_name].sync()
            param_id = self.PARAM_MAP[param_name]
            val = self.params[param_name].value
            # Convert bool/int to float for simplicity
            self.lib.set_param(self.dsp_handle, param_id, float(val))

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # 1. Get Raw Tensor from Input Slot
        if "in" in self.inputs:
            raw_tensor = self.inputs["in"].get_tensor()
        else:
            # Default to silence if disconnected
            self._ffi_in_buffer.zero_()
            raw_tensor = self._ffi_in_buffer

        # Allow subclasses to preprocess (e.g., apply gain)
        processed_tensor = self._preprocess_input(raw_tensor, self._ffi_in_buffer)

        # 2. Determine Actual Dimensions
        in_channels = processed_tensor.shape[0]

        out_slot = self.outputs.get("out")
        if not out_slot:
            return
        out_tensor = out_slot.buffer
        out_channels = out_tensor.shape[0]

        # Check output tensor contiguity
        if not out_tensor.is_contiguous():
            raise RuntimeError(f"Output tensor is not contiguous. Node: {self.name}")

        # 3. Ensure Contiguity (Critical for C pointers)
        # Use zero-allocation strategy: pre-allocated scratch buffer for copying non-contiguous tensors
        if processed_tensor.is_contiguous():
            processing_tensor = processed_tensor
        else:
            self._ffi_in_buffer.copy_(processed_tensor)
            processing_tensor = self._ffi_in_buffer

        # 4. Calculate Safe Processable Channels
        # We process whichever is smaller: input available or output capacity.
        process_channels = min(in_channels, out_channels)

        # 5. Anti-Ghosting: Zero out unused output channels
        # If input is Mono (1) and Output is Stereo (2), C++ only writes channel 0.
        # We must zero channel 1 to remove stale data from previous frames.
        if process_channels < out_channels:
            out_tensor[process_channels:].zero_()

        # 6. Get Pointers
        in_ptr = ctypes.cast(processing_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # 7. Call C++ with ACTUAL channel count
        self.lib.process(self.dsp_handle, in_ptr, out_ptr, process_channels, BLOCK_SIZE)

    def stop(self):
        # CHANGED: Do NOT destroy C++ object on transport stop.
        # We want the plugin state to persist (like a VST) even if the audio engine stops.
        pass

    def remove(self):
        # CHANGED: Destroy C++ object ONLY when the node is deleted from graph.
        if self.lib and self.dsp_handle:
            self.lib.destroy(self.dsp_handle)
            self.dsp_handle = None
