import torch
import numpy as np
from base import Node, BLOCK_SIZE, DTYPE, SAMPLE_RATE, CHANNELS


class SineOscillator(Node):
    def __init__(self, name="Sine"):
        super().__init__(name)
        self.add_float_param("freq", 440.0, 20.0, 20000.0)
        self.add_float_param("amp", 0.5, 0.0, 1.0)
        self.in_freq = self.add_input("freq_in", "freq")
        self.in_amp = self.add_input("amp_in", "amp")
        self.out_sig = self.add_output("signal", channels=1)
        self.two_pi = 2 * np.pi
        self.sr_recip = 1.0 / SAMPLE_RATE
        self.phase = 0.0
        self._phase_buffer = torch.zeros(BLOCK_SIZE, dtype=DTYPE)

    def process(self):
        freq_sig = self.in_freq.get_tensor()[0]
        amp_sig = self.in_amp.get_tensor()[0]
        torch.mul(freq_sig, self.two_pi * self.sr_recip, out=self._phase_buffer)
        self._phase_buffer.cumsum_(dim=0)
        self._phase_buffer.add_(self.phase)
        torch.sin(self._phase_buffer, out=self.out_sig.buffer[0])
        self.out_sig.buffer[0].mul_(amp_sig)
        self.phase = self._phase_buffer[-1].item() % self.two_pi


class StereoToMono(Node):
    def __init__(self, name="To Mono"):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=1)

    def process(self):
        t = self.inp.get_tensor()

        # FIX: Ensure output buffer is clean.
        # Since we only write to buffer[0], buffer[1] (if it exists) would retain stale data.
        self.out.buffer.zero_()

        if t.shape[0] == 1:
            self.out.buffer[0].copy_(t[0])
        else:
            torch.add(t[0], t[1], out=self.out.buffer[0])
            self.out.buffer[0].mul_(0.5)


class MonoToStereo(Node):
    def __init__(self, name="To Stereo"):
        super().__init__(name)
        self.add_float_param("pan", 0.0, -1.0, 1.0)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=2)

    def process(self):
        t = self.inp.get_tensor()
        pan = self.params["pan"].value
        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2
        torch.mul(t[0], left_gain, out=self.out.buffer[0])
        torch.mul(t[0], right_gain, out=self.out.buffer[1])


class Gain(Node):
    def __init__(self, name="Gain"):
        super().__init__(name)
        self.add_float_param("vol", 1.0, 0.0, 2.0)
        self.inp = self.add_input("in")
        self.gain_mod = self.add_input("mod", "vol")
        self.out = self.add_output("out")

    def process(self):
        t = self.inp.get_tensor()
        mod = self.gain_mod.get_tensor()
        torch.mul(t, mod, out=self.out.buffer)
