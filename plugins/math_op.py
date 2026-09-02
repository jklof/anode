"""
MathOp — vectorized arithmetic & CV conditioner (Utilities).

Real-time notes:
- Every branch writes the pre-allocated output buffer via out=/in-place ops;
  the only scratch is self._tmp (used by the sign-correct divide).
- Scalar operand paths avoid torch.tensor() construction entirely: Min/Max
  use the clamp identities min(A,s)==A.clamp(max=s), max(A,s)==A.clamp(min=s).
- Divide epsilon is sign-correct: A / (B + sign(B)*1e-6).
- Result channel count follows the widest operand (torch broadcasting into
  the stereo out buffer): mono A x stereo B -> stereo, by design.
- Modes 6-8 are unary (B ignored); mode 9 uses only scalar/offset params.
"""

import torch

from base import Node, CHANNELS, BLOCK_SIZE, DTYPE


class MathOp(Node):
    category = "Utilities"
    label = "Math Operator"
    description = (
        "Vectorized arithmetic and CV conditioner. Applies Add, Subtract, "
        "Multiply, Divide, Min, Max, Invert, Absolute, Clamp [0,1], or Scale & "
        "Offset to input A, using input B (if connected) or the scalar parameter "
        "as the second operand. Output channel count follows the widest operand. "
        "Divide is zero-safe via a sign-correct epsilon."
    )

    OPERATIONS = ["Add", "Subtract", "Multiply", "Divide", "Min", "Max",
                  "Invert", "Absolute", "Clamp", "Scale & Offset"]

    def __init__(self, name=""):
        super().__init__(name)
        self.in_a = self.add_input("in_a", help="Primary operand (also the signal passed through unary modes).")
        self.in_b = self.add_input("in_b", help="Second operand. Unconnected: the 'scalar' parameter is used instead.")
        self.out = self.add_output("out", channels=CHANNELS, help="Result of the selected operation.")

        self.add_menu_param("op", self.OPERATIONS, 0,
                            help="Arithmetic operation to apply.")
        self.add_float_param("scalar", 1.0, -100.0, 100.0,
                             help="Scalar operand used when input B is unconnected.")
        self.add_float_param("offset", 0.0, -100.0, 100.0,
                             help="Additive offset used by the Scale & Offset mode.")

        self._tmp = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._maskf = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._maskb = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.bool)

    def process(self):
        sig_a = self.in_a.get_tensor()
        op = int(self.params["op"].value)
        b_conn = bool(self.in_b.connected_outputs)
        sig_b = self.in_b.get_tensor() if b_conn else None
        scalar = self.params["scalar"].value
        offset = self.params["offset"].value
        out = self.out.buffer

        # copy_-first everywhere: an out= reduction with a mono source would
        # RESIZE the stereo out buffer down to (1, BLOCK).
        if op == 0:      # Add
            out.copy_(sig_a)
            out.add_(sig_b if b_conn else scalar)
        elif op == 1:    # Subtract
            out.copy_(sig_a)
            out.sub_(sig_b if b_conn else scalar)
        elif op == 2:    # Multiply
            out.copy_(sig_a)
            out.mul_(sig_b if b_conn else scalar)
        elif op == 3:    # Divide (sign-correct epsilon, zero-safe)
            if b_conn:
                # denom = B + sign(B)*eps, treating exact zeros as positive
                torch.sign(sig_b, out=self._tmp)
                torch.eq(self._tmp, 0.0, out=self._maskb)
                self._maskf.copy_(self._maskb)
                self._tmp.add_(self._maskf).mul_(1e-6).add_(sig_b)
                out.copy_(sig_a).div_(self._tmp)
            else:
                d = scalar + (1e-6 if scalar >= 0.0 else -1e-6)
                out.copy_(sig_a).div_(d)
        elif op == 4:    # Min
            if b_conn:
                # copy_-first: torch.minimum(mono, mono, out=stereo) RESIZES
                # the out buffer down to (1, BLOCK) (AGENTS.md §2). Broadcast
                # both operands into full-width buffers first.
                out.copy_(sig_a)
                self._tmp.copy_(sig_b)
                torch.minimum(out, self._tmp, out=out)
            else:
                out.copy_(sig_a).clamp_(max=scalar)
        elif op == 5:    # Max
            if b_conn:
                out.copy_(sig_a)
                self._tmp.copy_(sig_b)
                torch.maximum(out, self._tmp, out=out)
            else:
                out.copy_(sig_a).clamp_(min=scalar)
        elif op == 6:    # Invert
            out.copy_(sig_a).neg_()
        elif op == 7:    # Absolute
            out.copy_(sig_a).abs_()
        elif op == 8:    # Clamp to [0, 1]
            out.copy_(sig_a).clamp_(0.0, 1.0)
        elif op == 9:    # Scale & Offset
            out.copy_(sig_a).mul_(scalar).add_(offset)
