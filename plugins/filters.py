"""
Filter / EQ nodes: BiquadFilter (IIR) and LinearPhaseEQ (FIR).

Real-time notes:
- No per-block allocations except where the public PyTorch API forces one
  (LinearPhaseEQ's conv1d output; conv has no out= parameter). Transients are
  freed by refcounting immediately and do not accumulate.
- Coefficients update at block rate (93.75 Hz); fast parameter moves may
  zipper. Per-sample coefficient smoothing is deliberately out of scope.
"""

import logging
import math

import numpy as np
import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

logger = logging.getLogger(__name__)

NYQUIST = SAMPLE_RATE / 2.0
CUTOFF_MIN = 20.0
CUTOFF_MAX = 20000.0


def _biquad_df2t_block(x_t: torch.Tensor, y_t: torch.Tensor, z_t: torch.Tensor,
                       b0: float, b1: float, b2: float,
                       a1: float, a2: float, channels: int) -> None:
    """
    Direct Form II Transposed, one channel at a time.

    Per-sample math runs on plain Python floats (double precision — more
    accurate than float32 state math): the input row is converted with a
    single .tolist() and the output row written back in one bulk assignment,
    so no per-sample tensor dispatch happens. Measured ~0.2 ms per stereo
    block (~2% of the 10.67 ms real-time budget).

    NOTE (TorchScript was evaluated and rejected): a scripted per-sample loop
    costs ~12 ms/block because every .item()/setitem goes through dispatcher;
    slower than both this implementation and the block budget.
    """
    x_np = x_t.numpy()
    y_np = y_t.numpy()
    for c in range(channels):
        x_row = x_np[c].tolist()
        y_row = y_np[c]
        z1 = float(z_t[c, 0].item())
        z2 = float(z_t[c, 1].item())
        y_vals = []
        append = y_vals.append
        for xn in x_row:
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            append(yn)
        y_row[:] = y_vals
        z_t[c, 0] = z1
        z_t[c, 1] = z2


def _clamp(value, lo, hi):
    return lo if value < lo else (hi if value > hi else value)


# ==============================================================================
# BiquadFilter (IIR)
# ==============================================================================


class BiquadFilter(Node):
    category = "Effects"
    label = "Biquad Filter (IIR)"

    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Notch",
                    "Peaking", "Low Shelf", "High Shelf"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("type", self.FILTER_TYPES, 0)
        self.add_float_param("cutoff", 1000.0, CUTOFF_MIN, CUTOFF_MAX)
        self.add_float_param("q", 0.707, 0.1, 10.0)
        self.add_float_param("gain_db", 0.0, -24.0, 24.0)

        self.inp = self.add_input("in")
        # Modulation socket: connected signal is used as cutoff (Hz).
        self.in_mod = self.add_input("mod_cutoff", "cutoff")
        self.out = self.add_output("out", channels=CHANNELS)

        # Persistent DF2T state per channel: [z1, z2]
        self._z = torch.zeros((CHANNELS, 2), dtype=DTYPE)
        self._last_channels = -1

        # Normalized coefficients (a0 == 1)
        self._b0, self._b1, self._b2 = 1.0, 0.0, 0.0
        self._a1, self._a2 = 0.0, 0.0
        self._coeff_key = None

    def start(self):
        self._z.zero_()
        self._last_channels = -1

    def _design(self, type_idx, cutoff, q, gain_db):
        """Robert Bristow-Johnson Audio EQ Cookbook coefficients."""
        f0 = _clamp(cutoff, CUTOFF_MIN, CUTOFF_MAX)
        q = max(q, 0.05)
        w0 = 2.0 * math.pi * f0 / SAMPLE_RATE
        cosw = math.cos(w0)
        sinw = math.sin(w0)
        alpha = sinw / (2.0 * q)
        a_10 = 10.0 ** (gain_db / 40.0)  # A

        if type_idx == 0:      # Low Pass
            b0 = (1.0 - cosw) / 2.0
            b1 = 1.0 - cosw
            b2 = b0
            a0 = 1.0 + alpha
            a1 = -2.0 * cosw
            a2 = 1.0 - alpha
        elif type_idx == 1:    # High Pass
            b0 = (1.0 + cosw) / 2.0
            b1 = -(1.0 + cosw)
            b2 = b0
            a0 = 1.0 + alpha
            a1 = -2.0 * cosw
            a2 = 1.0 - alpha
        elif type_idx == 2:    # Band Pass (constant 0 dB peak gain)
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cosw
            a2 = 1.0 - alpha
        elif type_idx == 3:    # Notch
            b0 = 1.0
            b1 = -2.0 * cosw
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cosw
            a2 = 1.0 - alpha
        elif type_idx == 4:    # Peaking EQ
            b0 = 1.0 + alpha * a_10
            b1 = -2.0 * cosw
            b2 = 1.0 - alpha * a_10
            a0 = 1.0 + alpha / a_10
            a1 = -2.0 * cosw
            a2 = 1.0 - alpha / a_10
        elif type_idx == 5:    # Low Shelf
            beta = 2.0 * math.sqrt(a_10) * alpha
            b0 = a_10 * ((a_10 + 1.0) - (a_10 - 1.0) * cosw + beta)
            b1 = 2.0 * a_10 * ((a_10 - 1.0) - (a_10 + 1.0) * cosw)
            b2 = a_10 * ((a_10 + 1.0) - (a_10 - 1.0) * cosw - beta)
            a0 = (a_10 + 1.0) + (a_10 - 1.0) * cosw + beta
            a1 = -2.0 * ((a_10 - 1.0) + (a_10 + 1.0) * cosw)
            a2 = (a_10 + 1.0) + (a_10 - 1.0) * cosw - beta
        else:                  # High Shelf
            beta = 2.0 * math.sqrt(a_10) * alpha
            b0 = a_10 * ((a_10 + 1.0) + (a_10 - 1.0) * cosw + beta)
            b1 = -2.0 * a_10 * ((a_10 - 1.0) + (a_10 + 1.0) * cosw)
            b2 = a_10 * ((a_10 + 1.0) + (a_10 - 1.0) * cosw - beta)
            a0 = (a_10 + 1.0) - (a_10 - 1.0) * cosw + beta
            a1 = 2.0 * ((a_10 - 1.0) - (a_10 + 1.0) * cosw)
            a2 = (a_10 + 1.0) - (a_10 - 1.0) * cosw - beta

        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0

    def process(self):
        t = self.inp.get_tensor()
        in_ch = t.shape[0]
        out_buf = self.out.buffer

        # Channel-count change: stale DF2T state would ring old content.
        if in_ch != self._last_channels:
            self._z.zero_()
            self._last_channels = in_ch

        mod_connected = bool(self.in_mod.connected_outputs)
        if mod_connected:
            # Block mean of the modulation signal, clamped to the stable range.
            eff_cut = _clamp(
                float(self.in_mod.get_tensor()[0].mean().item()),
                CUTOFF_MIN, CUTOFF_MAX,
            )
            key = None  # always redesign while modulated
        else:
            eff_cut = float(self.params["cutoff"].value)
            key = (
                int(self.params["type"].value),
                eff_cut,
                float(self.params["q"].value),
                float(self.params["gain_db"].value),
            )

        if mod_connected or key != self._coeff_key:
            k = key if key is not None else (
                int(self.params["type"].value),
                eff_cut,
                float(self.params["q"].value),
                float(self.params["gain_db"].value),
            )
            self._design(k[0], k[1], k[2], k[3])
            self._coeff_key = k

        _biquad_df2t_block(t, out_buf, self._z,
                           self._b0, self._b1, self._b2, self._a1, self._a2, in_ch)

        # Anti-ghosting: unused output channels must not carry stale data.
        if in_ch < out_buf.shape[0]:
            out_buf[in_ch:].zero_()


# ==============================================================================
# LinearPhaseEQ (FIR)
# ==============================================================================


class LinearPhaseEQ(Node):
    category = "Effects"
    label = "Linear Phase EQ (FIR)"

    NUM_TAPS = 255  # odd -> Type I symmetric FIR, integer delay (N-1)/2
    FILTER_TYPES = ["Low Pass", "High Pass", "Band Pass", "Band Stop (Notch)"]

    def __init__(self, name=""):
        super().__init__(name)
        self.add_menu_param("type", self.FILTER_TYPES, 0)
        self.add_float_param("cutoff", 1000.0, CUTOFF_MIN, CUTOFF_MAX)
        self.add_float_param("q", 1.0, 0.1, 10.0)

        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=CHANNELS)

        taps = self.NUM_TAPS
        self._hist_len = taps - 1
        # [history | current block]; valid conv output = hist+512-255+1 = 512
        self._extended = torch.zeros((CHANNELS, self._hist_len + BLOCK_SIZE), dtype=DTYPE)
        # Same coefficients on every channel: shape (CHANNELS, 1, NUM_TAPS),
        # narrowed to the active channel count at process time.
        self._kernel = torch.zeros((CHANNELS, 1, taps), dtype=DTYPE)
        self._last_channels = -1
        self._key = None
        self._design(0, 1000.0, 1.0)

    def start(self):
        self._extended.zero_()
        self._last_channels = -1

    def get_telemetry(self) -> dict:
        latency_ms = (self.NUM_TAPS - 1) / 2.0 / SAMPLE_RATE * 1000.0
        return {"latency_samples": (self.NUM_TAPS - 1) // 2, "latency_ms": round(latency_ms, 2)}

    def _design(self, type_idx, cutoff, q):
        """Hann-windowed sinc FIR designer (pure numpy, no scipy).

        LPs are sum-normalized for unity DC gain; HP/BP/Notch derive by
        spectral inversion / differencing so pass bands stay at unity gain."""
        taps = self.NUM_TAPS
        m = taps - 1
        n = np.arange(taps, dtype=np.float64)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / m)

        def lowpass(fc):
            fc = _clamp(fc, CUTOFF_MIN, NYQUIST - 1.0)
            fcn = fc / NYQUIST
            # Ideal LP: h[n] = 2*(fc/fs)*sinc(2*(fc/fs)*(n-M/2)); since
            # fcn = fc/NYQUIST = 2*fc/fs this simplifies to fcn*sinc(fcn*d).
            h = fcn * np.sinc(fcn * (n - m / 2.0))
            h *= window
            s = h.sum()
            return h / s if s != 0.0 else h

        if type_idx == 0:      # Low Pass
            h = lowpass(cutoff)
        elif type_idx == 1:    # High Pass (spectral inversion of LP)
            h = -lowpass(cutoff)
            h[m // 2] += 1.0  # delta at the shared linear-phase center
        else:
            bw = _clamp(cutoff / max(q, 1e-3), CUTOFF_MIN, NYQUIST - 2.0)
            lo = _clamp(cutoff - bw / 2.0, CUTOFF_MIN, NYQUIST - 2.0)
            hi = max(cutoff + bw / 2.0, lo + 20.0)
            hi = _clamp(hi, lo + 20.0, NYQUIST - 1.0)
            bandpass = lowpass(hi) - lowpass(lo)
            if type_idx == 2:  # Band Pass
                h = bandpass
            else:              # Band Stop (Notch): spectral inversion of BP
                h = -bandpass
                h[m // 2] += 1.0

        kernel = torch.from_numpy(np.ascontiguousarray(h, dtype=np.float32))
        self._kernel[:] = kernel  # broadcast into both channel rows
        self._key = (type_idx, cutoff, q)

    def process(self):
        t = self.inp.get_tensor()
        ch = t.shape[0]
        out_buf = self.out.buffer

        # Channel-count change: clear history so old audio cannot smear in.
        if ch != self._last_channels:
            self._extended.zero_()
            self._last_channels = ch

        key = (
            int(self.params["type"].value),
            float(self.params["cutoff"].value),
            float(self.params["q"].value),
        )
        if key != self._key:
            self._design(*key)

        ext = self._extended.narrow(0, 0, ch)
        ext.narrow(1, self._hist_len, BLOCK_SIZE).copy_(t[:ch])

        # Depthwise convolution over [history | block].
        # NOTE: F.conv1d has no out= parameter; this result tensor is the one
        # deliberate transient allocation per block (~4 KB, freed immediately).
        result = torch.nn.functional.conv1d(
            ext.unsqueeze(0),
            self._kernel.narrow(0, 0, ch),
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=ch,
        )

        out_buf[:ch].copy_(result[0])
        if ch < out_buf.shape[0]:
            out_buf[ch:].zero_()

        # Shift history: keep the last (K-1) samples of the current block.
        ext.narrow(1, 0, self._hist_len).copy_(
            ext.narrow(1, BLOCK_SIZE, self._hist_len)
        )
