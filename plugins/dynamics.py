import ctypes
import torch
from ffi_base import FFINode
from base import CHANNELS, BLOCK_SIZE


class Compressor(FFINode):
    LIB_NAME = "compressor"
    category = "Effects"
    label = "Compressor"

    # Map params to C++ switch-case IDs
    PARAM_MAP = {"thresh": 0, "ratio": 1, "attack": 2, "release": 3, "knee": 4, "makeup": 5}

    def __init__(self, name=""):
        super().__init__(name)

        # Audio Ports
        self.add_input("in")
        self.add_input("sidechain")  # Optional input
        self.add_output("out")

        # Parameters
        self.add_float_param("thresh", -20.0, -60.0, 0.0)
        self.add_float_param("ratio", 4.0, 1.0, 20.0)
        self.add_float_param("knee", 6.0, 0.0, 24.0)
        self.add_float_param("attack", 10.0, 0.1, 200.0)
        self.add_float_param("release", 100.0, 10.0, 1000.0)
        self.add_float_param("makeup", 0.0, 0.0, 24.0)

        # Pre-allocate buffer for sidechain alignment
        self._sc_buffer = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)

        # Bind the extended C API
        if self.lib:
            try:
                self.lib.process_with_sidechain.restype = None
                self.lib.process_with_sidechain.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),  # In
                    ctypes.POINTER(ctypes.c_float),  # Sidechain
                    ctypes.POINTER(ctypes.c_float),  # Out
                    ctypes.c_int,
                    ctypes.c_int,
                ]

                self.lib.get_gain_reduction.restype = ctypes.c_float
                self.lib.get_gain_reduction.argtypes = [ctypes.c_void_p]
            except Exception as e:
                print(f"Compressor Bind Error: {e}")

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # 1. Main Input
        in_tensor = self.inputs["in"].get_tensor()

        # 2. Sidechain Input
        # If not connected, we pass None to C++ (which handles it by using main input)
        sc_ptr = None
        if self.inputs["sidechain"].connected_outputs:
            sc_tensor = self.inputs["sidechain"].get_tensor()

            # Ensure CPU & Contiguous
            if sc_tensor.device.type != "cpu":
                sc_tensor = sc_tensor.cpu()

            if sc_tensor.is_contiguous():
                sc_ptr = ctypes.cast(sc_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
            else:
                self._sc_buffer.copy_(sc_tensor)
                sc_ptr = ctypes.cast(self._sc_buffer.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # 3. Prepare Input Buffer (Generic handling from FFI logic)
        # We manually handle contiguity here similar to base class but for local tensors
        if in_tensor.device.type != "cpu":
            in_tensor = in_tensor.cpu()

        if not in_tensor.is_contiguous():
            self._ffi_in_buffer.copy_(in_tensor)
            in_tensor = self._ffi_in_buffer

        # 4. Prepare Output
        out_slot = self.outputs.get("out")
        out_tensor = out_slot.buffer

        # 5. Channel Logic
        in_channels = in_tensor.shape[0]
        out_channels = out_tensor.shape[0]
        process_channels = min(in_channels, out_channels)

        if process_channels < out_channels:
            out_tensor[process_channels:].zero_()

        in_ptr = ctypes.cast(in_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # 6. Execute DSP
        if sc_ptr:
            self.lib.process_with_sidechain(self.dsp_handle, in_ptr, sc_ptr, out_ptr, process_channels, BLOCK_SIZE)
        else:
            # Use base standard process which passes nullptr for SC
            self.lib.process(self.dsp_handle, in_ptr, out_ptr, process_channels, BLOCK_SIZE)

    def get_telemetry(self):
        # Optional: Report Gain Reduction to UI
        gr = 1.0
        if self.lib and self.dsp_handle:
            gr = self.lib.get_gain_reduction(self.dsp_handle)

        return {"gr": gr}
