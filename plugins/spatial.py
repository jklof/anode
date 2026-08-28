"""
Spatial / imaging utility nodes (Utilities).

StereoPanner: constant-power pan law (gL^2 + gR^2 == 2, unity at center)
combined with a mid/side width matrix. Mono inputs treat R = L; both output
channels are always written (anti-ghosting). pan_mod is param-bound to
"pan" so an unconnected slot falls back to the parameter constant cache.

MidSideEncoder / MidSideDecoder: orthonormal mid/side matrix pair
(Mid=(L+R)/sqrt2, Side=(L-R)/sqrt2). Encode -> decode is an exact
roundtrip. The decoder falls back to Side = 0 for mono input.

Real-time notes: pure in-place vector math, no state, nothing to reset.
"""

import numpy as np
import torch

from base import Node, BLOCK_SIZE, DTYPE

SQRT2 = float(np.sqrt(2.0))


class StereoPanner(Node):
    category = "Utilities"
    label = "Stereo Panner"
    description = (
        "Constant-power stereo panner with width control: gL^2 + gR^2 == 2 (unity "
        "at center, +3 dB hard-panned) combined with a mid/side width matrix. Mono "
        "inputs are treated as R = L and both output channels are always written. "
        "The pan input is parameter-bound for audio-rate panning."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Signal to pan; mono inputs are upmixed to stereo.")
        self.pan_mod = self.add_input("pan_mod", "pan",
                                      help="Audio-rate pan modulation (-1..+1). Unconnected: uses 'pan' parameter.")
        self.out = self.add_output("out", channels=2, help="Panned stereo output.")

        self.add_float_param("pan", 0.0, -1.0, 1.0,
                             help="Pan position: -1 hard left, 0 center, +1 hard right.")
        self.add_float_param("width", 1.0, 0.0, 2.0, unit="x",
                             help="Stereo width scaling: 0 = mono, 1 = normal, 2 = extra wide.")

        self._mid = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._side = torch.zeros(BLOCK_SIZE, dtype=DTYPE)

    def process(self):
        sig = self.inp.get_tensor()
        pan = self.params["pan"].value
        width = self.params["width"].value

        l = sig[0]
        r = sig[1] if sig.shape[0] > 1 else sig[0]

        # 1. Width matrix: L_w = M - S*w, R_w = M + S*w with M=(L+R)/2, S=(R-L)/2
        torch.add(l, r, out=self._mid).mul_(0.5)
        torch.sub(r, l, out=self._side).mul_(0.5 * width)

        # 2. Constant-power gains (unity at center, +3 dB hard panned)
        theta = (pan + 1.0) * (np.pi / 4.0)
        g_l = float(np.cos(theta)) * SQRT2
        g_r = float(np.sin(theta)) * SQRT2

        torch.sub(self._mid, self._side, out=self.out.buffer[0]).mul_(g_l)
        torch.add(self._mid, self._side, out=self.out.buffer[1]).mul_(g_r)


class MidSideEncoder(Node):
    category = "Utilities"
    label = "Mid/Side Encoder"
    description = (
        "Encodes a stereo signal into orthonormal mid/side components: "
        "Mid = (L + R) / sqrt(2), Side = (L - R) / sqrt(2). Pair with "
        "MidSideDecoder for an exact round trip. Zero latency, no state."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Stereo signal to encode; mono inputs set Side to zero.")
        self.out_mid = self.add_output("mid", channels=1, help="Mono mid (center) component.")
        self.out_side = self.add_output("side", channels=1, help="Mono side (stereo difference) component.")

    def process(self):
        t = self.inp.get_tensor()
        l = t[0]
        r = t[1] if t.shape[0] > 1 else t[0]
        torch.add(l, r, out=self.out_mid.buffer[0]).mul_(1.0 / SQRT2)
        torch.sub(l, r, out=self.out_side.buffer[0]).mul_(1.0 / SQRT2)


class MidSideDecoder(Node):
    category = "Utilities"
    label = "Mid/Side Decoder"
    description = (
        "Decodes orthonormal mid/side components back to stereo: "
        "L = (M + S) / sqrt(2), R = (M - S) / sqrt(2). Falls back to Side = 0 "
        "for mono input. Zero latency, no state."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Mid/side signal to decode (channel 0 = Mid, channel 1 = Side).")
        self.out = self.add_output("out", channels=2, help="Reconstructed stereo signal.")
        self._zero_side = torch.zeros(BLOCK_SIZE, dtype=DTYPE)

    def process(self):
        t = self.inp.get_tensor()
        m = t[0]
        s = t[1] if t.shape[0] > 1 else self._zero_side
        torch.add(m, s, out=self.out.buffer[0]).mul_(1.0 / SQRT2)
        torch.sub(m, s, out=self.out.buffer[1]).mul_(1.0 / SQRT2)
