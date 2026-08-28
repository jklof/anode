import ctypes
import os
import sys
import torch
import logging
from base import Node, BLOCK_SIZE, CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


class FFINode(Node):
    """
    A generic base class for C++ nodes.
    Assumes the C++ library implements the Standard ANode C-ABI.

    Parameter Synchronization (Canonical Path):
    - UI/Script writes to Parameter._staging
    - Engine thread calls Parameter.sync() -> Parameter.value
    - FFINode._sync_params_to_cpp() called ONCE before native process()
    - Native set_param() updates DSP state
    - Native process() executes with synchronized parameters

    Audio-Rate Modulation Exception:
    - If an input slot explicitly modulates a parameter per block
      (e.g., BiquadFilter.in_mod modulating cutoff), process() may
      calculate modulation and call lib.set_param() directly AFTER
      staged parameters have synced.
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
        # Parameter sync state: dirty flag — set when a parameter value changes,
        # consumed (and cleared) by _sync_params_to_cpp() on the engine thread.
        self._native_params_dirty = True
        self._load_library()

        # Initialize C++ object
        if self.lib:
            self.dsp_handle = self.lib.create()
            if not self.dsp_handle:
                logger.error(f"[{self.name}] Failed to create C++ instance.")
                self.error_msg = "C++ Init Failed"
            elif hasattr(self.lib, "set_samplerate"):
                # Optional part of the standard ABI: propagate the engine rate.
                try:
                    self.lib.set_samplerate(self.dsp_handle, float(SAMPLE_RATE))
                except Exception as e:
                    logger.error(f"[{self.name}] set_samplerate failed: {e}")

    def _load_library(self):
        if not self.LIB_NAME:
            return

        # Determine full name with extension
        if sys.platform == "win32":
            lib_filename = f"{self.LIB_NAME}.dll"
        elif sys.platform == "darwin":
            lib_filename = f"lib{self.LIB_NAME}.dylib"
        else:
            lib_filename = f"lib{self.LIB_NAME}.so"

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

        # void set_samplerate(void* handle, float samplerate) — optional
        if hasattr(self.lib, "set_samplerate"):
            self.lib.set_samplerate.restype = None
            self.lib.set_samplerate.argtypes = [ctypes.c_void_p, ctypes.c_float]

        # void reset(void* handle) — optional
        if hasattr(self.lib, "reset"):
            self.lib.reset.restype = None
            self.lib.reset.argtypes = [ctypes.c_void_p]

    def _call_reset(self):
        """Call native reset() if available. Called on engine thread in start()."""
        if self.lib and self.dsp_handle and hasattr(self.lib, "reset"):
            try:
                self.lib.reset(self.dsp_handle)
            except Exception as e:
                logger.error(f"[{self.name}] reset failed: {e}")

    def _sync_params_to_cpp(self):
        """
        Synchronize staged parameters to native DSP.
        Called ONCE per block, BEFORE native process().
        Dirty-flag based: parameters are only pushed when something changed,
        so unchanged parameters cause no repeated set_param() calls.
        """
        if not (self.lib and self.dsp_handle):
            return
        if not self.PARAM_MAP:
            return
        if not self._native_params_dirty:
            return

        for name, pid in self.PARAM_MAP.items():
            self.lib.set_param(self.dsp_handle, pid, float(self.params[name].value))
        self._native_params_dirty = False

    def _preprocess_input(self, in_tensor: torch.Tensor, scratch_buffer: torch.Tensor) -> torch.Tensor:
        """Hook for subclasses to modify input tensor before C++ processing. Default pass-through."""
        return in_tensor

    def on_ui_param_change(self, param_name: str):
        """UI parameter change - stages value only. Native sync happens on engine thread."""
        super().on_ui_param_change(param_name)

    def _mark_param_dirty(self):
        # Any committed parameter change (set()+sync(), engine 'param'
        # command, load_state) marks native params dirty for the next block.
        self._native_params_dirty = True

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # 1. Synchronize staged parameters to native DSP (CANONICAL PATH)
        self._sync_params_to_cpp()

        # 2. Get Raw Tensor from Input Slot
        if "in" in self.inputs:
            raw_tensor = self.inputs["in"].get_tensor()
        else:
            # Default to silence if disconnected
            self._ffi_in_buffer.zero_()
            raw_tensor = self._ffi_in_buffer

        # Allow subclasses to preprocess (e.g., apply gain)
        processed_tensor = self._preprocess_input(raw_tensor, self._ffi_in_buffer)

        # 3. Determine Actual Dimensions
        in_channels = processed_tensor.shape[0]

        out_slot = self.outputs.get("out")
        if not out_slot:
            return
        out_tensor = out_slot.buffer
        out_channels = out_tensor.shape[0]

        # Check output tensor contiguity
        if not out_tensor.is_contiguous():
            raise RuntimeError(f"Output tensor is not contiguous. Node: {self.name}")

        # 4. Ensure Contiguity & Safety (Critical for C pointers)
        # Verify device is CPU
        if processed_tensor.device.type != "cpu":
            processed_tensor = processed_tensor.cpu()

        # Use zero-allocation strategy: pre-allocated scratch buffer for copying non-contiguous tensors
        if processed_tensor.is_contiguous():
            processing_tensor = processed_tensor
        else:
            self._ffi_in_buffer.copy_(processed_tensor)
            processing_tensor = self._ffi_in_buffer

        # 5. Calculate Safe Processable Channels
        # Channel adaptation policy: a mono (1-channel) input into a stereo
        # node is DUPLICATED to both channels before native processing — the
        # C++ side only writes `channels` output channels, so processing with
        # process_channels=1 would mute the right channel.
        if in_channels == 1 and out_channels == 2:
            self._ffi_in_buffer[0].copy_(processed_tensor[0])
            self._ffi_in_buffer[1].copy_(processed_tensor[0])
            processing_tensor = self._ffi_in_buffer
            process_channels = 2
        else:
            process_channels = min(in_channels, out_channels)
            # Anti-Ghosting: Zero out unused output channels
            if process_channels < out_channels:
                out_tensor[process_channels:].zero_()

        # 7. Get Pointers
        in_ptr = ctypes.cast(processing_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # 8. Call C++ with ACTUAL channel count
        self.lib.process(self.dsp_handle, in_ptr, out_ptr, process_channels, BLOCK_SIZE)

    def start(self):
        # Reset native DSP state and force parameter re-push on next block
        self._call_reset()
        self._native_params_dirty = True

    def load_state(self, data: dict):
        super().load_state(data)
        # Force native parameter sync on next block
        self._native_params_dirty = True

    def stop(self):
        # CHANGED: Do NOT destroy C++ object on transport stop.
        # We want the plugin state to persist (like a VST) even if the audio engine stops.
        pass

    def remove(self):
        # CHANGED: Destroy C++ object ONLY when the node is deleted from graph.
        if self.lib and self.dsp_handle:
            self.lib.destroy(self.dsp_handle)
            self.dsp_handle = None
