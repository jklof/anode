"""
Bitcrusher — sample-and-hold decimation + bit-depth quantizer (Effects).

Real-time notes (architecture per docs/new_node_specifications.md §7):
- The hold grid is anchored to a GLOBAL sample index n (n % D == 0), so the
  plateau phase stays continuous across block boundaries when D changes or
  not dividing BLOCK_SIZE evenly.
- Hold points can reach up to D-1 < DECIM_MAX samples into the previous
  block; these are resolved by gathering from an extended domain
  [previous tail | current block] instead of any per-sample Python loop.
- Fully vectorized integer-index kernels with pre-allocated int64 index
  tensors and gather(out=); zero .item() calls.
"""

import torch

from base import Node, CHANNELS, BLOCK_SIZE, DTYPE

DECIM_MAX = 64


class Bitcrusher(Node):
    category = "Effects"
    label = "Bitcrusher"

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=CHANNELS)

        self.add_int_param("bits", 8, 1, 16)
        self.add_int_param("downsample", 1, 1, DECIM_MAX)
        self.add_float_param("mix", 1.0, 0.0, 1.0)

        self._g0 = 0  # global sample offset of the next block (reset in start())
        self._ext = torch.zeros((CHANNELS, BLOCK_SIZE + DECIM_MAX), dtype=DTYPE)
        self._tail = torch.zeros((CHANNELS, DECIM_MAX), dtype=DTYPE)
        self._held = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._g = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
        self._kh = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
        self._jidx = torch.zeros(BLOCK_SIZE, dtype=torch.int64)

    def start(self):
        self._g0 = 0
        self._tail.zero_()

    def process(self):
        sig = self.inp.get_tensor()
        d = int(self.params["downsample"].value)
        steps = 2.0 ** (int(self.params["bits"].value) - 1)
        mix = self.params["mix"].value
        out = self.out.buffer

        # Extended gather domain: [previous tail (DECIM_MAX) | current block]
        # Tail MUST come first: hold points reach up to D-1 samples into the
        # previous block, i.e. to negative local offsets shifted by DECIM_MAX.
        self._ext[:, :DECIM_MAX].copy_(self._tail)
        self._ext[:, DECIM_MAX:].copy_(sig)          # broadcasts mono -> stereo

        # Global sample indices of this block
        torch.arange(self._g0, self._g0 + BLOCK_SIZE, out=self._g)
        # Most recent global hold point k = floor(n / D) * D
        torch.div(self._g, d, rounding_mode="floor", out=self._kh).mul_(d)
        # Index into ext domain: local offset shifted past the tail region
        self._jidx.copy_(self._kh).sub_(self._g0).add_(DECIM_MAX)
        for c in range(CHANNELS):
            torch.gather(self._ext[c], 0, self._jidx, out=self._held[c])
        self._tail.copy_(sig[:, BLOCK_SIZE - DECIM_MAX:])
        self._g0 += BLOCK_SIZE

        # Quantize to `bits` (|x| <= 1 keeps round(x*s)/s within [-1, 1])
        self._held.mul_(steps).round_().div_(steps)

        # Dry/wet crossfade without temporaries
        out.copy_(sig)
        out.mul_(1.0 - mix).add_(self._held, alpha=mix)
