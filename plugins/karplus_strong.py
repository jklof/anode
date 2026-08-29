import ctypes
import logging
import torch

from base import Node, BLOCK_SIZE, CHANNELS
from ffi_base import FFINode

logger = logging.getLogger(__name__)


class KarplusStrong(FFINode):
    category = "Sources"
    label = "Karplus-Strong String"
    description = (
        "Native C++ Karplus-Strong physical modeling string synthesizer. "
        "Audio-rate trigger inputs excite a delay line with filtered noise plucks. "
        "Compensated fractional delay and internal damping provide accurate musical "
        "pitch tuning and natural string decay."
    )

    LIB_NAME = "karplus"
    PARAM_MAP = {
        "freq": 0,
        "damping": 1,
        "brightness": 2,
        "decay": 3,
    }

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger", help="Impulse or gate signal; rising edge triggers a string pluck.")
        self.add_input("freq_in", "freq", help="Audio-rate frequency modulation (Hz). Unconnected: uses 'freq' param.")
        self.add_output("out", channels=CHANNELS, help="Stereo physical modeling string output.")

        self.add_float_param("freq", 220.0, 20.0, 2000.0, unit="Hz", help="Fundamental frequency of the string.")
        self.add_float_param("damping", 0.5, 0.0, 0.99, help="High-frequency absorption in the feedback loop.")
        self.add_float_param("brightness", 0.8, 0.0, 1.0, help="Spectral brightness of the pluck excitation.")
        self.add_float_param("decay", 0.99, 0.8, 1.0, help="Sustain factor of the string resonance.")

        self._was_freq_mod_connected = False

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # 1. Sync staged parameters (CANONICAL PATH)
        self._sync_params_to_cpp()

        # 2. Audio-rate frequency modulation:
        if self.inputs["freq_in"].connected_outputs:
            freq_tensor = self.inputs["freq_in"].get_tensor()
            eff_freq = float(freq_tensor[0].mean().item())
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["freq"], eff_freq)
            self._was_freq_mod_connected = True
        elif self._was_freq_mod_connected:
            # Re-push staged frequency value when freq_in is disconnected
            self.lib.set_param(self.dsp_handle, self.PARAM_MAP["freq"], float(self.params["freq"].value))
            self._was_freq_mod_connected = False

        # 3. Handle Trigger Input (mono-to-stereo scratch expansion)
        raw_trig = self.inputs["trigger"].get_tensor()
        if raw_trig.device.type != "cpu":
            raw_trig = raw_trig.cpu()

        if raw_trig.shape[0] == 1:
            self._ffi_in_buffer[0].copy_(raw_trig[0])
            self._ffi_in_buffer[1].copy_(raw_trig[0])
            trig_tensor = self._ffi_in_buffer
        elif not raw_trig.is_contiguous():
            self._ffi_in_buffer.copy_(raw_trig)
            trig_tensor = self._ffi_in_buffer
        else:
            trig_tensor = raw_trig

        out_tensor = self.outputs["out"].buffer
        if not out_tensor.is_contiguous():
            raise RuntimeError(f"Output tensor is not contiguous in {self.name}")

        in_ptr = ctypes.cast(trig_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(out_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # 4. Native DSP invocation
        self.lib.process(self.dsp_handle, in_ptr, out_ptr, CHANNELS, BLOCK_SIZE)