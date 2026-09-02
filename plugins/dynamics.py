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
    description = (
        "Native C++ feed-forward dynamics compressor with soft-knee, attack/"
        "release ballistics and makeup gain. Supports an optional sidechain "
        "input for external keying; without a sidechain the main input is used "
        "for detection. Mono sidechains are replicated across channels."
    )

    # Map params to C++ switch-case IDs
    PARAM_MAP = {"thresh": 0, "ratio": 1, "attack": 2, "release": 3, "knee": 4, "makeup": 5}

    def __init__(self, name=""):
        super().__init__(name)

        # Audio Ports
        self.add_input("in", help="Signal to compress.")
        self.add_input("sidechain", help="Optional detector signal. Unconnected: the main input drives gain reduction.")  # Optional input
        self.add_output("out", help="Compressed signal with makeup gain applied.")

        # Parameters
        self.add_float_param("thresh", -20.0, -60.0, 0.0, unit="dB",
                             help="Level above which gain reduction is applied.")
        self.add_float_param("ratio", 4.0, 1.0, 20.0, unit=":1",
                             help="Compression ratio above the threshold.")
        self.add_float_param("knee", 6.0, 0.0, 24.0, unit="dB",
                             help="Soft-knee width around the threshold.")
        self.add_float_param("attack", 10.0, 0.1, 200.0, unit="ms",
                             help="Time constant for gain reduction onset.")
        self.add_float_param("release", 100.0, 10.0, 1000.0, unit="ms",
                             help="Time constant for gain recovery.")
        self.add_float_param("makeup", 0.0, 0.0, 24.0, unit="dB",
                             help="Makeup gain applied after compression.")

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

        # MANDATORY: Sync parameters before native processing
        self._sync_params_to_cpp()

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
        # Mono -> stereo duplication (AGENTS.md §2): a mono source into a
        # stereo node must broadcast to BOTH output channels, not mute the
        # right channel. Mirrors FFINode.process()'s adaptation policy.
        if in_channels == 1 and out_channels == 2:
            self._ffi_in_buffer[0].copy_(in_tensor[0])
            self._ffi_in_buffer[1].copy_(in_tensor[0])
            in_tensor = self._ffi_in_buffer
            process_channels = 2
        else:
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
    description = (
        "Downward expander / noise gate with attack, hold, and release stages. "
        "Signal below the threshold is attenuated by up to 'range' dB according "
        "to the ratio. An optional sidechain input lets another signal control "
        "the gating of the main input."
    )

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
        self.inp = self.add_input("in", help="Signal to gate.")
        self.sc = self.add_input("sidechain", help="Optional detector signal. Unconnected: the main input drives the gate.")
        self.out = self.add_output("out", channels=CHANNELS, help="Gated stereo output.")

        self.add_float_param("thresh", -40.0, -80.0, 0.0, unit="dB",
                             help="Level below which the gate closes.")
        self.add_float_param("ratio", 10.0, 1.0, 50.0, unit=":1",
                             help="Downward expansion ratio applied under the threshold.")
        self.add_float_param("attack", 1.0, 0.1, 50.0, unit="ms",
                             help="Time constant for gate opening.")
        self.add_float_param("hold", 50.0, 0.0, 500.0, unit="ms",
                             help="Time the gate stays open after the signal drops below the threshold.")
        self.add_float_param("release", 100.0, 5.0, 1000.0, unit="ms",
                             help="Time constant for gate closing.")
        self.add_float_param("range", 60.0, 0.0, 90.0, unit="dB",
                             help="Maximum attenuation applied when the gate is fully closed.")

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


class BrickwallLimiter(Node):
    category = "Effects"
    label = "Brickwall Limiter"
    description = (
        "Peak limiter with a 240-sample lookahead delay and program-dependent "
        "release: smoothly ramps gain down before peaks cross the threshold "
        "and recovers over the release time. Ceiling and threshold are in dBFS; threshold can be "
        "audio-rate modulated via a parameter-bound input."
    )

    LOOKAHEAD = 240

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in", help="Signal to limit; mono inputs are expanded to stereo.")
        self.add_input("thresh_in", "threshold",
                       help="Audio-rate threshold modulation (linear amplitude). Unconnected: uses 'threshold' parameter.")
        self.add_output("out", channels=CHANNELS, help="Limited stereo output clamped to the ceiling.")

        self.add_float_param("threshold", -0.1, -40.0, 0.0, unit="dBFS",
                             help="Level where gain reduction begins.")
        self.add_float_param("ceiling", -0.1, -20.0, 0.0, unit="dBFS",
                             help="Maximum output level; peaks never exceed this value.")
        self.add_float_param("release", 50.0, 1.0, 1000.0, unit="ms",
                             help="Time for gain to recover after a peak.")

        self._ring = torch.zeros((CHANNELS, self.LOOKAHEAD), dtype=DTYPE)
        self._delayed = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._mono_peak = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._mono_indices = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
        self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._current_gain = 1.0
        self._prev_gain = 1.0

    def start(self):
        self._ring.zero_()
        self._current_gain = 1.0
        self._prev_gain = 1.0

    def process(self):
        sig = self.inputs["in"].get_tensor()
        # Anti-ghosting: ensure input is CHANNELS (expand mono if needed)
        if sig.shape[0] == 1 and CHANNELS == 2:
            sig = sig.expand(2, BLOCK_SIZE)

        thresh_db = float(self.inputs["thresh_in"].get_tensor()[0, 0].item())
        ceiling_db = self.params["ceiling"].value
        release_ms = max(1.0, self.params["release"].value)

        thresh_lin = 10.0 ** (thresh_db / 20.0)
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)

        torch.abs(sig, out=self._delayed)
        torch.max(self._delayed, dim=0, out=(self._mono_peak, self._mono_indices))

        max_peak = max(float(self._mono_peak.max().item()), 1e-9)
        if max_peak > thresh_lin:
            target_gain = min(1.0, ceiling_lin / max_peak)
        else:
            target_gain = 1.0

        if target_gain < self._current_gain:
            self._current_gain = target_gain
        else:
            alpha_rel = 1.0 - math.exp(-(BLOCK_SIZE / SAMPLE_RATE) / (release_ms / 1000.0))
            alpha_rel = min(1.0, alpha_rel)
            self._current_gain += alpha_rel * (target_gain - self._current_gain)

        torch.linspace(self._prev_gain, self._current_gain, BLOCK_SIZE, out=self._ramp)
        self._prev_gain = self._current_gain

        la = self.LOOKAHEAD
        self._delayed[:, :la].copy_(self._ring)
        self._delayed[:, la:].copy_(sig[:, :BLOCK_SIZE - la])
        self._ring.copy_(sig[:, BLOCK_SIZE - la:])

        out = self.outputs["out"].buffer
        out.copy_(self._delayed)
        out.mul_(self._ramp)
        out.clamp_(-ceiling_lin, ceiling_lin)


class TransientShaper(Node):
    category = "Effects"
    label = "Transient Shaper"
    description = (
        "Two-band transient designer: separates the fast attack portion of the "
        "envelope from the sustain portion and boosts or cuts each "
        "independently (-1..+2 attack, -1..+1 sustain). Both sections are "
        "audio-rate modulatable via parameter-bound inputs."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in", help="Signal to process; mono inputs are expanded to stereo.")
        self.add_input("attack_mod", "attack",
                       help="Audio-rate attack amount modulation. Unconnected: uses 'attack' parameter.")
        self.add_input("sustain_mod", "sustain",
                       help="Audio-rate sustain amount modulation. Unconnected: uses 'sustain' parameter.")
        self.add_output("out", channels=CHANNELS, help="Transient-shaped stereo output.")

        self.add_float_param("attack", 0.0, -1.0, 2.0,
                             help="Attack emphasis: negative dulls transients, positive sharpens them.")
        self.add_float_param("sustain", 0.0, -1.0, 1.0,
                             help="Sustain emphasis: negative tightens decay, positive extends it.")
        self.add_float_param("output_gain_db", 0.0, -18.0, 18.0, unit="dB",
                             help="Output trim applied after shaping.")

        self._e_fast = 0.0
        self._e_slow = 0.0
        self._prev_gain = 1.0
        self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._mono = torch.zeros(BLOCK_SIZE, dtype=DTYPE)

    def start(self):
        self._e_fast = 0.0
        self._e_slow = 0.0
        self._prev_gain = 1.0

    def process(self):
        sig = self.inputs["in"].get_tensor()
        # Anti-shrink: expand mono input before any out= op. Functional ops
        # with out= would otherwise resize the (CHANNELS, BLOCK_SIZE) output
        # buffer down to (1, BLOCK_SIZE) for mono inputs.
        if sig.shape[0] == 1 and CHANNELS == 2:
            sig = sig.expand(CHANNELS, BLOCK_SIZE)
        attack_val = float(self.inputs["attack_mod"].get_tensor()[0, 0].item())
        sustain_val = float(self.inputs["sustain_mod"].get_tensor()[0, 0].item())
        out_gain = 10.0 ** (self.params["output_gain_db"].value / 20.0)

        out_buf = self.outputs["out"].buffer
        torch.abs(sig, out=out_buf)
        torch.mean(out_buf, dim=0, out=self._mono)
        peak = float(torch.max(self._mono).item())

        dt_block = BLOCK_SIZE / SAMPLE_RATE
        a_f = dt_block / 0.001 if peak > self._e_fast else dt_block / 0.020
        a_s = dt_block / 0.025 if peak > self._e_slow else dt_block / 0.200
        a_f = min(1.0, a_f)
        a_s = min(1.0, a_s)

        self._e_fast += a_f * (peak - self._e_fast)
        self._e_slow += a_s * (peak - self._e_slow)

        delta = self._e_fast - self._e_slow
        target_gain = ((delta * (1.0 + attack_val)) + (self._e_slow * (1.0 + sustain_val))) / (self._e_fast + 1e-6)
        target_gain = max(0.0, min(4.0, target_gain))

        torch.linspace(self._prev_gain, target_gain, BLOCK_SIZE, out=self._ramp)
        self._prev_gain = target_gain

        out_buf.copy_(sig)
        out_buf.mul_(self._ramp).mul_(out_gain)


class AutoGain(Node):
    category = "Utilities"
    label = "Auto Gain / Leveler"
    description = (
        "Slow automatic leveler: measures the RMS level over a sliding window "
        "(0.2-10 s) and applies a smoothed, gain-limited correction toward the "
        "target level. A silence gate freezes the gain during quiet passages to "
        "avoid boosting noise."
    )

    MAX_BLOCKS = int(10.0 * SAMPLE_RATE / BLOCK_SIZE)

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("in", help="Signal to level; mono inputs are expanded to stereo.")
        self.add_output("out", channels=CHANNELS, help="Gain-smoothed stereo output.")

        self.add_float_param("target_db", -14.0, -40.0, 0.0, unit="dB",
                             help="RMS level the node aims to converge to.")
        self.add_float_param("window_s", 2.0, 0.2, 10.0, unit="s",
                             help="Length of the RMS averaging window.")
        self.add_float_param("max_gain_db", 18.0, 0.0, 36.0, unit="dB",
                             help="Maximum correction applied in either direction.")
        self.add_float_param("silence_gate_db", -50.0, -80.0, -20.0, unit="dB",
                             help="RMS level below which gain adjustment is frozen.")

        self._rms_history = torch.zeros(self.MAX_BLOCKS, dtype=DTYPE)
        self._hist_ptr = 0
        self._hist_count = 0
        self._current_gain_lin = 1.0
        self._prev_gain_lin = 1.0
        self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._mono = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        # Pre-allocated scratch buffer for power computation
        self._pow_scratch = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    def start(self):
        self._rms_history.zero_()
        self._hist_ptr = 0
        self._hist_count = 0
        self._current_gain_lin = 1.0
        self._prev_gain_lin = 1.0

    def process(self):
        sig = self.inputs["in"].get_tensor()
        # Anti-shrink: expand mono input before any out= op (see TransientShaper).
        if sig.shape[0] == 1 and CHANNELS == 2:
            sig = sig.expand(CHANNELS, BLOCK_SIZE)
        target_db = self.params["target_db"].value
        window_s = self.params["window_s"].value
        max_gain_db = self.params["max_gain_db"].value
        silence_gate_db = self.params["silence_gate_db"].value

        out = self.outputs["out"].buffer

        # Use pre-allocated scratch buffer for power computation
        torch.pow(sig, 2.0, out=self._pow_scratch)
        torch.mean(self._pow_scratch, dim=0, out=self._mono)
        rms = float(torch.sqrt(torch.mean(self._mono)).item())
        rms_db = 20.0 * math.log10(max(rms, 1e-9))

        if rms_db < silence_gate_db:
            torch.linspace(self._prev_gain_lin, self._current_gain_lin, BLOCK_SIZE, out=self._ramp)
            self._prev_gain_lin = self._current_gain_lin
            out.copy_(sig)
            out.mul_(self._ramp)
            return

        N = int(window_s * SAMPLE_RATE / BLOCK_SIZE)
        N = max(1, min(N, self.MAX_BLOCKS))
        self._rms_history[self._hist_ptr] = rms
        self._hist_ptr = (self._hist_ptr + 1) % self.MAX_BLOCKS
        self._hist_count = min(self._hist_count + 1, N)

        # The history is a true ring: once _hist_ptr passes the window length,
        # fresh samples land past the linear prefix, so reading the linear
        # prefix _rms_history[:_hist_count] would average stale data. Read the
        # last _hist_count entries ending at _hist_ptr - 1 instead, as one
        # contiguous slice or two wrapped slices.
        K = self._hist_count
        P = self._hist_ptr
        if P >= K:
            hist_slice = self._rms_history[P - K:P]
            mean_sq = float(torch.mean(torch.pow(hist_slice, 2.0)).item())
        else:
            s1 = self._rms_history[self.MAX_BLOCKS - (K - P):]
            s2 = self._rms_history[:P]
            sum_sq = (float(torch.sum(torch.pow(s1, 2.0)).item())
                      + float(torch.sum(torch.pow(s2, 2.0)).item()))
            mean_sq = sum_sq / float(K)

        long_rms = math.sqrt(max(0.0, mean_sq))
        long_rms_db = 20.0 * math.log10(max(long_rms, 1e-9))

        gain_db = target_db - long_rms_db
        gain_db = max(-max_gain_db, min(max_gain_db, gain_db))
        target_gain = 10.0 ** (gain_db / 20.0)

        alpha = 1.0 - math.exp(-(BLOCK_SIZE / SAMPLE_RATE) / 0.5)
        alpha = min(1.0, alpha)
        self._current_gain_lin += alpha * (target_gain - self._current_gain_lin)

        torch.linspace(self._prev_gain_lin, self._current_gain_lin, BLOCK_SIZE, out=self._ramp)
        self._prev_gain_lin = self._current_gain_lin

        out.copy_(sig)
        out.mul_(self._ramp)
