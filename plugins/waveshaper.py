"""
WaveShaper — zero-latency nonlinear saturation (Effects).

Real-time notes:
- Fully vectorized in-place math; every branch writes pre-allocated
  scratch buffers only (no fresh tensors in process()).
- drive_mod is param-bound to "drive": an unconnected slot returns the
  parameter constant cache, so drive always has a defined value.
- Mono inputs broadcast into the stereo chain via copy_; both output
  channels are always written (anti-ghosting).
"""

import torch

from base import Node, CHANNELS, BLOCK_SIZE, DTYPE

MODES = ["Tanh (Tape)", "Soft Clip", "Hard Clip", "Wavefolder", "Asymmetric Tube"]


class WaveShaper(Node):
    category = "Effects"
    label = "WaveShaper / Saturation"
    description = (
        "Zero-latency nonlinear saturation with five transfer functions: Tanh "
        "(tape-style), cubic soft clip, hard clip, sine wavefolder, and "
        "asymmetric tube. Drive, bias, mix, and output level shape the response; "
        "drive is audio-rate modulatable via a parameter-bound input."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Signal to saturate; mono inputs broadcast into the stereo chain.")
        self.drive_mod = self.add_input("drive_mod", "drive",
                                        help="Audio-rate drive modulation. Unconnected: uses 'drive' parameter.")
        self.out = self.add_output("out", channels=CHANNELS, help="Saturated stereo output.")

        self.add_menu_param("mode", MODES, 0,
                            help="Nonlinear transfer function to apply.")
        self.add_float_param("drive", 1.0, 0.1, 20.0, unit="x",
                             help="Input gain into the nonlinear stage; used when no modulation is connected.")
        self.add_float_param("bias", 0.0, -1.0, 1.0,
                             help="DC offset applied before shaping (asymmetry control).")
        self.add_float_param("mix", 1.0, 0.0, 1.0,
                             help="Dry/wet crossfade between input and shaped signal.")
        self.add_float_param("output_level", 1.0, 0.0, 2.0, unit="x",
                             help="Output trim applied after the mix.")

        self._driven = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._shaped = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._dry = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._tmp = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._tmp2 = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._sel = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._maskb = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.bool)

    def process(self):
        sig = self.inp.get_tensor()
        drive_row = self.drive_mod.get_tensor()[0]
        mix = self.params["mix"].value
        bias = self.params["bias"].value
        level = self.params["output_level"].value
        mode = int(self.params["mode"].value)

        self._dry.copy_(sig)
        # copy_ broadcasts a mono input into the fixed stereo buffer; an
        # out= reduction here would RESIZE _driven down to the mono shape.
        self._driven.copy_(sig)
        self._driven.mul_(drive_row)
        self._driven.add_(bias)

        if mode == 0:      # Tanh (tape)
            self._shaped.copy_(self._driven).tanh_()
        elif mode == 1:    # Soft clip: x - x^3/3 for |x|<=1, else sign(x)*2/3
            # Branch-free form with no boolean mask and no implicit sync:
            # clamping first makes the cubic identity exact for |x| > 1 too,
            # since clamp(±x, -1, 1) = ±1 -> ±1 - (±1)³/3 = ±2/3. The old
            # form used torch.gt(..., out=bool_mask) + .any() + torch.where,
            # which allocated a mask tensor and forced a host sync per block.
            self._shaped.copy_(self._driven).clamp_(-1.0, 1.0)
            torch.pow(self._shaped, 3, out=self._tmp)
            self._shaped.sub_(self._tmp.mul_(1.0 / 3.0))
        elif mode == 2:    # Hard clip
            self._shaped.copy_(self._driven).clamp_(-1.0, 1.0)
        elif mode == 3:    # Foldback / wavefolder
            self._shaped.copy_(self._driven).sin_()
        else:              # Asymmetric tube
            # Positive branch: x / (1 + x)   (denominator >= 1 for x >= 0)
            # Negative branch: x / (1 - x) - 0.1 x^2  — clamped so the
            # denominator never reaches 0 at x = 1 (division blow-up).
            torch.gt(self._driven, 0.0, out=self._maskb)
            self._tmp.copy_(self._driven)
            self._tmp2.copy_(self._driven).add_(1.0)
            torch.div(self._tmp, self._tmp2, out=self._tmp)
            self._sel.copy_(self._driven)
            self._tmp2.copy_(self._driven).neg_().add_(1.0).clamp_(min=0.25)
            torch.div(self._sel, self._tmp2, out=self._sel)
            torch.pow(self._driven, 2, out=self._tmp2)
            self._sel.sub_(self._tmp2.mul_(0.1))
            torch.where(self._maskb, self._tmp, self._sel, out=self._shaped)

        # Constant-power-style dry/wet crossfade with output trim
        self.out.buffer.copy_(self._dry)
        self.out.buffer.mul_(1.0 - mix).add_(self._shaped, alpha=mix).mul_(level)
