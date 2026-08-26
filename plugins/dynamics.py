import ctypes
import logging
import math
import torch
from base import Node, CHANNELS, BLOCK_SIZE, DTYPE, SAMPLE_RATE
from ffi_base import FFINode

logger = logging.getLogger(__name__)


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
                    ctypes.c_int,  # Sidechannel channel count
                ]

                self.lib.get_gain_reduction.restype = ctypes.c_float
                self.lib.get_gain_reduction.argtypes = [ctypes.c_void_p]

                # Bind and call set_samplerate
                self.lib.set_samplerate.restype = None
                self.lib.set_samplerate.argtypes = [ctypes.c_void_p, ctypes.c_float]

                from base import SAMPLE_RATE

                self.lib.set_samplerate(self.dsp_handle, float(SAMPLE_RATE))
            except Exception as e:
                logger.error(f"Compressor Bind Error: {e}")

    def process(self):
        if not self.lib or not self.dsp_handle:
            return

        # 1. Main Input
        in_tensor = self.inputs["in"].get_tensor()

        # 2. Sidechain Input
        # If not connected, we pass None to C++ (which handles it by using main input)
        sc_ptr = None
        sc_channels = 0
        if self.inputs["sidechain"].connected_outputs:
            sc_tensor = self.inputs["sidechain"].get_tensor()

            # Ensure CPU & Contiguous
            if sc_tensor.device.type != "cpu":
                sc_tensor = sc_tensor.cpu()

            if sc_tensor.is_contiguous():
                sc_buffer = sc_tensor
            else:
                self._sc_buffer.copy_(sc_tensor)
                sc_buffer = self._sc_buffer

            sc_channels = sc_buffer.shape[0]
            # The C++ side replicates channel 0 when sc_channels < main
            # channels, so a mono sidechain into a stereo compressor stays
            # in-bounds and uses the mono signal for detection.
            sc_ptr = ctypes.cast(sc_buffer.data_ptr(), ctypes.POINTER(ctypes.c_float))

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
            self.lib.process_with_sidechain(self.dsp_handle, in_ptr, sc_ptr, out_ptr, process_channels, BLOCK_SIZE,
                                            sc_channels)
        else:
            # Use base standard process which passes nullptr for SC
            self.lib.process(self.dsp_handle, in_ptr, out_ptr, process_channels, BLOCK_SIZE)

    def get_telemetry(self):
        # Optional: Report Gain Reduction to UI
        gr = 1.0
        if self.lib and self.dsp_handle:
            gr = self.lib.get_gain_reduction(self.dsp_handle)

        return {"gr": gr}


class NoiseGate(Node):
    category = "Effects"
    label = "Noise Gate"

    """Downward expander with lookahead and hold.

    Architecture per docs/new_node_specifications.md §4:
    - Block-rate detector: one RMS level per block over a /16 decimated
      sidechain slice. The only transient is the tiny pow(2) temp on the
      32-sample slice (no out= variant for slice-reduce; same documented
      exception as the FFT nodes).
    - LOOKAHEAD (64 samples) delay ring aligns the smoothed gain trajectory
      ahead of transients.
    - Ballistics are one-pole-per-block toward the target gain reduction,
      rendered as a linear gain ramp across the block (linspace out=).
    - Sidechain falls back to the main input when unconnected; mono
      sidechains are reduced across all dims, never indexed by CHANNELS.
    """

    LOOKAHEAD = 64
    DECIM = 16

    PARAM_ORDER = ("thresh", "ratio", "attack", "hold", "release", "range")

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.sc = self.add_input("sidechain")
        self.out = self.add_output("out", channels=CHANNELS)

        self.add_float_param("thresh", -40.0, -80.0, 0.0)
        self.add_float_param("ratio", 10.0, 1.0, 50.0)
        self.add_float_param("attack", 1.0, 0.1, 50.0)
        self.add_float_param("hold", 50.0, 0.0, 500.0)
        self.add_float_param("release", 100.0, 5.0, 1000.0)
        self.add_float_param("range", 60.0, 0.0, 90.0)

        self.gr_db = 0.0
        self.hold_left = 0
        self._coeffs = (0.5, 0.1, 2400)   # att_c, rel_c, hold_samples
        self._param_state = None

        self._ring = torch.zeros((CHANNELS, self.LOOKAHEAD), dtype=DTYPE)
        self._delayed = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)

    def start(self):
        self.gr_db = 0.0
        self.hold_left = 0
        self._ring.zero_()

    def process(self):
        sig = self.inp.get_tensor()
        sc_slot = self.sc if self.sc.connected_outputs else self.inp
        sc = sc_slot.get_tensor()

        state = tuple(self.params[k].value for k in self.PARAM_ORDER)
        if state != self._param_state:
            self._param_state = state
            thresh, ratio, att_ms, hold_ms, rel_ms, _range = state
            block_dur = BLOCK_SIZE / SAMPLE_RATE
            att_c = 1.0 - math.exp(-block_dur / max(att_ms, 1e-4) * 1000.0)
            rel_c = 1.0 - math.exp(-block_dur / max(rel_ms, 1e-4) * 1000.0)
            self._coeffs = (att_c, rel_c, int(hold_ms * SAMPLE_RATE / 1000.0))
            self._thresh_db = thresh
            self._slope = ratio - 1.0
            self._range_db = _range

        # --- Detector: RMS dB of the decimated sidechain ---
        # Small documented transient: pow(2) temp on the (ch, 32) slice.
        level_lin = float(torch.mean(sc[:, ::self.DECIM].pow(2)))
        level_db = 10.0 * math.log10(level_lin + 1e-9)

        # --- Target gain reduction + hold state machine ---
        open_now = level_db >= self._thresh_db
        target_gr = 0.0
        if not open_now:
            target_gr = (level_db - self._thresh_db) * self._slope
            if target_gr < -self._range_db:
                target_gr = -self._range_db
        if open_now:
            self.hold_left = self._coeffs[2]
        elif self.hold_left > 0:
            self.hold_left -= BLOCK_SIZE
            target_gr = 0.0

        att_c, rel_c, _ = self._coeffs
        coeff = att_c if target_gr > self.gr_db else rel_c
        prev_lin = 10.0 ** (self.gr_db / 20.0)
        self.gr_db += coeff * (target_gr - self.gr_db)
        end_lin = 10.0 ** (self.gr_db / 20.0)
        torch.linspace(prev_lin, end_lin, BLOCK_SIZE, out=self._ramp)

        # --- Lookahead delay + gain ramp application ---
        la = self.LOOKAHEAD
        self._delayed[:, :la].copy_(self._ring)
        self._delayed[:, la:].copy_(sig[:, :BLOCK_SIZE - la])
        self._ring.copy_(sig[:, BLOCK_SIZE - la:])
        self.out.buffer.copy_(self._delayed).mul_(self._ramp)
