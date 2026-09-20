"""Shared sliding log-STFT core for Spectrum/Spectrogram visual nodes.

Owns the DSP only (pre-allocated ring/unwrap/window/rfft/mag/dB/log-bin
buffers, no per-block allocation beyond the documented rfft transient).
Telemetry, params, and widgets stay per-node: SpectrumDisplay (256 bins,
capacity 4, smoothing curves) vs SpectrogramDisplay (128 bins, capacity 16,
waterfall LUT).
"""

import numpy as np
import torch

from base import BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE


class SlidingLogSTFT:
    """Sliding-window log-frequency STFT with pre-allocated buffers."""

    # Never auto-registered as a palette node (plain helper, not a Node).
    is_abstract = True

    def __init__(self, num_bins, fft_size=2048,
                 min_freq=20.0, max_freq=20000.0):
        self.fft_size = fft_size
        self.num_bins = num_bins
        self.fft_bins = fft_size // 2 + 1
        self._ring = torch.zeros((CHANNELS, fft_size), dtype=DTYPE)
        self._write_pos = 0
        self._unwrapped = torch.zeros((CHANNELS, fft_size), dtype=DTYPE)
        self._window = torch.hann_window(fft_size, dtype=DTYPE)
        # Canonical norm (2/window_sum): a full-scale sine peaks at ~0 dBFS.
        self._norm = 2.0 / float(self._window.sum())
        self._windowed = torch.zeros((CHANNELS, fft_size), dtype=DTYPE)
        self._mag = torch.zeros((CHANNELS, self.fft_bins), dtype=DTYPE)
        self._db = torch.zeros((CHANNELS, self.fft_bins), dtype=DTYPE)
        self._out = torch.zeros((CHANNELS, num_bins), dtype=DTYPE)
        bin_freqs = torch.linspace(0, SAMPLE_RATE / 2.0, steps=self.fft_bins)
        targets = torch.tensor(
            np.logspace(np.log10(min_freq), np.log10(max_freq),
                        num=num_bins),
            dtype=DTYPE,
        )
        self._log_indices = torch.searchsorted(
            bin_freqs, targets).clamp_(0, self.fft_bins - 1)

    def reset(self):
        """Clear history so a transport restart cannot smear stale audio."""
        self._ring.zero_()
        self._write_pos = 0

    def analyze_block(self, sig, min_db, max_db):
        """Ingest one block and return the [0, 1] log-magnitude view.

        Returns ``self._out`` (caller pushes ``.numpy()`` to telemetry).
        Allocation-free except the documented rfft transient.
        """
        in_ch = sig.shape[0]
        # Ring write. Safe slice: FFT_SIZE % BLOCK_SIZE == 0.
        wp = self._write_pos
        seg = self._ring[:, wp:wp + BLOCK_SIZE]
        if in_ch >= CHANNELS:
            seg.copy_(sig[:CHANNELS])
        else:
            # Mono (or narrow): duplicate the last available channel into
            # every remaining ring row.
            for c in range(CHANNELS):
                seg[c].copy_(sig[min(c, in_ch - 1)])
        self._write_pos = (wp + BLOCK_SIZE) % self.fft_size
        # Unwrap circular -> chronological [oldest ... newest].
        tail = self.fft_size - self._write_pos
        self._unwrapped[:, :tail].copy_(self._ring[:, self._write_pos:])
        self._unwrapped[:, tail:].copy_(self._ring[:, :self._write_pos])
        # Window + FFT (rfft has no out=; documented transient).
        torch.mul(self._unwrapped, self._window, out=self._windowed)
        spectrum = torch.fft.rfft(self._windowed, n=self.fft_size, dim=1)
        # Magnitude -> dBFS.
        torch.abs(spectrum, out=self._mag)
        self._mag.mul_(self._norm).clamp_(min=1e-9)
        torch.log10(self._mag, out=self._db)
        self._db.mul_(20.0)
        # Log-frequency resample + normalize into [0, 1].
        torch.index_select(self._db, 1, self._log_indices, out=self._out)
        db_range = max(1.0, max_db - min_db)
        self._out.sub_(min_db).div_(db_range).clamp_(0.0, 1.0)
        return self._out
