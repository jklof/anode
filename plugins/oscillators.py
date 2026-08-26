"""
WaveformOscillator — anti-aliased PolyBLEP multi-wave generator (Sources).

Real-time notes:
- Phase accumulation and waveform shaping are fully vectorized; the only
  per-block scalar sync is the phase carry (.item()).
- Modulation inputs are param-bound ("freq_in" -> "freq",
  "pw_in" -> "pulse_width"), so unconnected slots fall back to the
  parameter constant tensor cache instead of zeroing the oscillator.
- All scratch buffers pre-allocated; the PolyBLEP residual uses
  pre-allocated boolean masks (comparisons support out=).
"""

import numpy as np
import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, DTYPE, CHANNELS


class WaveformOscillator(Node):
    category = "Sources"
    label = "Waveform Oscillator"

    def __init__(self, name=""):
        super().__init__(name)
        self.in_freq = self.add_input("freq_in", "freq")
        self.in_pw = self.add_input("pw_in", "pulse_width")
        self.out_sig = self.add_output("signal", channels=1)

        self.add_menu_param("waveform", ["Sine", "Triangle", "Sawtooth", "Square"], 0)
        self.add_float_param("freq", 440.0, 1.0, 20000.0)
        self.add_float_param("amp", 0.5, 0.0, 1.0)
        self.add_float_param("pulse_width", 0.5, 0.01, 0.99)

        self.phase = 0.0
        self._dt = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._phase_buf = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._naive = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._blep = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._blep_a = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._blep_b = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._temp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._mask_a = torch.zeros(BLOCK_SIZE, dtype=torch.bool)
        self._mask_b = torch.zeros(BLOCK_SIZE, dtype=torch.bool)

    def start(self):
        self.phase = 0.0

    def _compute_polyblep(self, phase, dt, out):
        """out = BLEP(phase, dt): three-region polynomial step residual.

        Region A (0 <= t < dt):  -(u-1)^2          with u = t/dt
        Region B (1-dt < t <= 1): (v+1)^2          with v = (t-1)/dt
        Both closed forms are algebraically identical to the classic
        2u - u^2 - 1 / 2v + v^2 + 1 expressions."""
        torch.div(phase, dt, out=self._blep_a)
        self._blep_a.sub_(1.0).pow_(2).neg_()
        torch.sub(phase, 1.0, out=self._blep_b)
        torch.div(self._blep_b, dt, out=self._blep_b)
        self._blep_b.add_(1.0).pow_(2)
        # Region-B threshold is a TENSOR: dt varies under FM
        torch.sub(1.0, dt, out=self._temp)
        torch.gt(phase, self._temp, out=self._mask_b)
        torch.lt(phase, dt, out=self._mask_a)
        out.zero_()
        out.copy_(self._blep_a).mul_(self._mask_a)
        self._blep_b.mul_(self._mask_b)
        out.add_(self._blep_b)

    def process(self):
        freq_sig = self.in_freq.get_tensor()[0]
        pw_sig = self.in_pw.get_tensor()[0]
        amp = self.params["amp"].value
        wave = int(self.params["waveform"].value)
        out = self.out_sig.buffer[0]

        # 1. Per-sample dt = f / fs, clamped to a stable range
        torch.mul(freq_sig, 1.0 / SAMPLE_RATE, out=self._dt)
        self._dt.clamp_(min=1e-5, max=0.49)

        # 2. Phase accumulation with wrap
        self._phase_buf.copy_(self._dt).cumsum_(dim=0).add_(self.phase).remainder_(1.0)
        self.phase = float(self._phase_buf[-1].item())

        # 3. Waveform generation
        if wave == 0:      # Sine
            torch.mul(self._phase_buf, 2.0 * np.pi, out=self._temp)
            torch.sin(self._temp, out=out)
        elif wave == 1:    # Triangle (naive: no step discontinuity)
            torch.mul(self._phase_buf, 2.0, out=self._temp)
            self._temp.sub_(1.0).abs_().mul_(2.0).sub_(1.0)
            out.copy_(self._temp)
        elif wave == 2:    # Sawtooth + PolyBLEP at the wrap
            torch.mul(self._phase_buf, -2.0, out=self._naive).add_(1.0)
            self._compute_polyblep(self._phase_buf, self._dt, self._blep)
            torch.add(self._naive, self._blep, out=out)
        else:              # Square/Pulse + PolyBLEP at 0 and pw
            torch.lt(self._phase_buf, pw_sig, out=self._mask_a)
            self._naive.copy_(self._mask_a).mul_(2.0).sub_(1.0)
            self._compute_polyblep(self._phase_buf, self._dt, self._blep)
            self._naive.add_(self._blep)
            torch.sub(self._phase_buf, pw_sig, out=self._temp).remainder_(1.0)
            self._compute_polyblep(self._temp, self._dt, self._blep)
            self._naive.sub_(self._blep)
            out.copy_(self._naive)

        out.mul_(amp)


class ColoredNoise(Node):
    category = "Sources"
    label = "Colored Noise Generator"

    TAPS = 127
    HIST = TAPS - 1

    def __init__(self, name=""):
        super().__init__(name)
        self.out_sig = self.add_output("out", channels=CHANNELS)

        self.add_menu_param("type", ["White", "Pink (-3dB/oct)", "Brown (-6dB/oct)", "Blue (+3dB/oct)", "Violet (+6dB/oct)"], 0)
        self.add_float_param("amp", 0.2, 0.0, 1.0)

        self._raw_noise = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._conv_in = torch.zeros((1, CHANNELS, self.HIST + BLOCK_SIZE), dtype=DTYPE)
        self._tail = torch.zeros((CHANNELS, self.HIST), dtype=DTYPE)
        self._kernels = self._init_fir_kernels()

    def _init_fir_kernels(self):
        n = self.TAPS
        freqs = torch.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
        freqs[0] = 1.0

        mag_pink = freqs.pow(-0.5)
        mag_brown = freqs.pow(-1.0)
        mag_blue = freqs.pow(0.5)
        mag_violet = freqs.pow(1.0)

        mag_pink[0] = mag_pink[1]
        mag_brown[0] = mag_brown[1]
        mag_blue[0] = mag_blue[1]
        mag_violet[0] = mag_violet[1]

        def mag_to_ir(mag_spec):
            phase = torch.rand_like(mag_spec) * 2 * torch.pi
            spec = torch.complex(mag_spec * torch.cos(phase), mag_spec * torch.sin(phase))
            ir = torch.fft.irfft(spec, n=n)
            win = torch.hann_window(n, dtype=torch.float32)
            ir = ir * win
            ir = ir / (ir.abs().sum() + 1e-9)
            return ir

        kernels = torch.zeros((4, 1, n), dtype=torch.float32)
        kernels[0, 0] = mag_to_ir(mag_pink)
        kernels[1, 0] = mag_to_ir(mag_brown)
        kernels[2, 0] = mag_to_ir(mag_blue)
        kernels[3, 0] = mag_to_ir(mag_violet)
        return kernels

    def start(self):
        self._tail.zero_()

    def process(self):
        color_idx = int(self.params["type"].value)
        amp = self.params["amp"].value
        out = self.outputs["out"].buffer

        self._raw_noise.uniform_(-1.0, 1.0)

        if color_idx == 0:
            out.copy_(self._raw_noise).mul_(amp)
            return

        self._conv_in[0, :, :self.HIST].copy_(self._tail)
        self._conv_in[0, :, self.HIST:].copy_(self._raw_noise)
        self._tail.copy_(self._raw_noise[:, BLOCK_SIZE - self.HIST:])

        kernel = self._kernels[color_idx - 1].expand(CHANNELS, 1, self.TAPS)
        filtered = torch.nn.functional.conv1d(self._conv_in, kernel, groups=CHANNELS)
        out.copy_(filtered[0]).mul_(amp)
