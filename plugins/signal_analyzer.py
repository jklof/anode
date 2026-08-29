"""
SignalAnalyzer — real-time audio signal metric extraction (Utilities).

Calculates instantaneous block-rate metrics (RMS, peak, DC offset, crest factor,
and zero-crossing rate) as monophonic CV tensors on every processed block.

Audio-thread notes:
- Pass-through is bit-exact: the input is copied to the output untouched
  (mono inputs broadcast to both output channels, MonoToStereo convention).
- All five metrics are computed fresh on EVERY block (no rate limiting), so
  downstream CV consumers see per-block values with zero added latency.
- ZCR is measured block-locally on channel 0 only, normalized by
  (BLOCK_SIZE - 1). Note that exact-zero samples adjacent to nonzero samples
  count as transitions, because torch.sign(0) == 0 differs from the sign of
  any nonzero neighbor.
- Pre-allocated scratch buffers for the RMS squaring reduction and the ZCR
  sign/diff reduction; process() performs zero net heap allocation.
- Metrics are computed from scalar .item() reductions, so the CV outputs are
  constant-valued rows, never stale (every output is refilled each block).
"""

import math

import torch

from base import Node, BLOCK_SIZE, CHANNELS, DTYPE


class SignalAnalyzer(Node):
    category = "Utilities"
    label = "Signal Analyzer"
    description = (
        "Real-time signal analysis node. Computes block-rate RMS, peak amplitude, "
        "DC offset, crest factor, and zero-crossing rate from the input signal as "
        "monophonic CV tensors (channels=1) on every audio block with zero latency. "
        "Audio passes through unaltered."
    )

    # Peak below this linear amplitude (-120 dBFS) reports crest factor 0.0
    # so a silent/disconnected input does not read as a huge crest ratio.
    SILENCE_FLOOR = 1e-6

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in", help="Signal to analyze; mono inputs are duplicated to stereo.")
        self.out = self.add_output("out", channels=CHANNELS, help="Pass-through copy of the input, unaltered.")

        self.out_rms = self.add_output(
            "rms_out", channels=1, help="Mono CV of block RMS amplitude (linear, all channels pooled)."
        )
        self.out_peak = self.add_output(
            "peak_out", channels=1, help="Mono CV of peak absolute amplitude (all channels pooled)."
        )
        self.out_dc = self.add_output(
            "dc_out", channels=1, help="Mono CV of DC offset (mean amplitude, all channels pooled)."
        )
        self.out_crest = self.add_output(
            "crest_out", channels=1, help="Mono CV of crest factor (peak / RMS; 0.0 on silence)."
        )
        self.out_zcr = self.add_output(
            "zcr_out",
            channels=1,
            help=(
                "Mono CV of normalized zero-crossing rate on channel 0 [0.0, 1.0] "
                "(block-local transitions / (BLOCK_SIZE-1); sign(0)==0 counts as a transition)."
            ),
        )

        # Pre-allocated scratch tensors (zero net allocation in process()).
        self._squared = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._sign = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._diff = torch.zeros(BLOCK_SIZE - 1, dtype=DTYPE)

    def start(self):
        # Transport restart: no stale audio or stale CV values may leak.
        self.out.buffer.zero_()
        self.out_rms.buffer.zero_()
        self.out_peak.buffer.zero_()
        self.out_dc.buffer.zero_()
        self.out_crest.buffer.zero_()
        self.out_zcr.buffer.zero_()

    def process(self):
        sig = self.inp.get_tensor()

        # 1. Bit-exact audio pass-through (copy_ broadcasts mono -> stereo
        #    without resizing the (CHANNELS, BLOCK_SIZE) output buffer).
        self.out.buffer.copy_(sig)

        # 2. Block-rate metric analysis (scalar reductions; no temp tensors).
        min_val = float(torch.min(sig).item())
        max_val = float(torch.max(sig).item())
        mean_val = float(torch.mean(sig).item())
        abs_max = max(abs(min_val), abs(max_val))

        self._squared.copy_(sig)
        self._squared.pow_(2)
        rms_val = math.sqrt(max(0.0, float(torch.mean(self._squared).item())))

        if abs_max >= self.SILENCE_FLOOR:
            crest_val = abs_max / (rms_val + 1e-9)
        else:
            crest_val = 0.0

        # ZCR on Channel 0 (block-local transitions normalized by BLOCK_SIZE - 1).
        torch.sign(sig[0], out=self._sign)
        torch.sub(self._sign[1:], self._sign[:-1], out=self._diff)
        zcr_val = float(torch.count_nonzero(self._diff).item()) / float(BLOCK_SIZE - 1)

        # 3. Fill ALL CV Outputs Every Block (Anti-Ghosting Contract):
        #    a loud block followed by silence must drop every metric to 0.0
        #    immediately, never carry stale values across the boundary.
        self.out_rms.buffer[0].fill_(rms_val)
        self.out_peak.buffer[0].fill_(abs_max)
        self.out_dc.buffer[0].fill_(mean_val)
        self.out_crest.buffer[0].fill_(crest_val)
        self.out_zcr.buffer[0].fill_(zcr_val)
