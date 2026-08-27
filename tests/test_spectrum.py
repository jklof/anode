import math
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE


# ==============================================================================
# Helpers (same conventions as test_spectrogram.py)
# ==============================================================================


def make_node(class_name="SpectrumDisplay"):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def drop_pending(node):
    """Emulate the UI consuming frames (ring buffer holds maxsize=4; overflow
    frames are dropped by design)."""
    queue = getattr(node, "monitor_queue", None)
    if queue:
        # Consume all available frames (pop_all returns list, last frame is newest)
        frames = queue.pop_all()
        # Return nothing - the frames are consumed
        return len(frames)
    return 0


def stream(node, tensor, blocks):
    """Feed a tensor through node.process(), consuming the SPSC buffer every
    block exactly like the live widget. Returns the newest frame."""
    slot = node.inputs["in"]
    drop_pending(node)
    last = None
    for _ in range(blocks):
        slot.get_tensor = lambda t=tensor: t
        node.process()
        queue = getattr(node, "monitor_queue", None)
        if queue:
            latest = queue.pop_latest()
            if latest is not None:
                last = latest
    return last


def stereo_block(left_freq, right_freq, amp_l=0.5, amp_r=0.5):
    n = np.arange(BLOCK_SIZE)
    l = amp_l * np.sin(2 * np.pi * left_freq * n / SAMPLE_RATE)
    r = amp_r * np.sin(2 * np.pi * right_freq * n / SAMPLE_RATE)
    return torch.from_numpy(np.vstack([l, r]).astype(np.float32))


# ==============================================================================
# Registration / params
# ==============================================================================


def test_spectrum_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SpectrumDisplay")
    assert cls is not None
    assert cls.category == "Visual"
    assert cls.label == "Spectrum Visualizer"


def test_spectrum_parameter_sync():
    node = make_node()
    assert "min_db" in node.params
    assert "max_db" in node.params
    assert "smoothing" in node.params

    node.params["min_db"].set(-90.0)
    node.params["max_db"].set(12.0)
    node.params["smoothing"].set(0.8)
    node.sync()

    assert node.params["min_db"].value == -90.0
    assert node.params["max_db"].value == 12.0
    assert node.params["smoothing"].value == 0.8


# ==============================================================================
# Audio path
# ==============================================================================


def test_spectrum_stereo_pass_through_and_dual_peaks():
    node = make_node()
    # Left = 1 kHz (bin ~145), Right = 4 kHz (bin ~196) on the log axis
    signal = stereo_block(1000.0, 4000.0)

    points = stream(node, signal, 10)

    # Pass-through integrity
    assert torch.allclose(node.out.buffer, signal)

    # Frame shape / range
    assert isinstance(points, np.ndarray)
    assert points.shape == (CHANNELS, 256)
    assert points.min() >= 0.0 and points.max() <= 1.0

    # Distinct per-channel peaks at their own frequencies
    left_peak = int(np.argmax(points[0]))
    right_peak = int(np.argmax(points[1]))
    assert 130 <= left_peak <= 160, f"left peak bin {left_peak}, expected ~1 kHz"
    assert 180 <= right_peak <= 210, f"right peak bin {right_peak}, expected ~4 kHz"


def test_spectrum_true_mono_channel_count():
    """Regression: genuine (1, BLOCK) input must analyse and duplicate cleanly."""
    node = make_node()
    n = np.arange(BLOCK_SIZE)
    row = (0.5 * np.sin(2 * np.pi * 1000.0 * n / SAMPLE_RATE)).astype(np.float32)
    true_mono = torch.from_numpy(row).unsqueeze(0)  # shape (1, 512)

    col = stream(node, true_mono, 10)

    out = node.out.buffer
    assert out.shape[0] == CHANNELS
    assert torch.allclose(out[0], true_mono[0])
    assert torch.allclose(out[1], true_mono[0])

    # Both channels analyse the duplicated mono: identical peaks near 1 kHz
    left_peak = int(np.argmax(col[0]))
    right_peak = int(np.argmax(col[1]))
    assert abs(left_peak - right_peak) <= 2
    assert 130 <= left_peak <= 160


def test_spectrum_dbfs_level_calibration():
    """Normalization check against the [-70, +6] dB defaults."""
    node = make_node()
    set_min, set_max = -70.0, 6.0

    loud = stream(node, stereo_block(1000.0, 1000.0, amp_l=0.5, amp_r=0.5), 10)[0].max()
    expected_loud = (20 * math.log10(0.5) - set_min) / (set_max - set_min)
    assert abs(loud - expected_loud) < 0.12, f"loud level {loud:.2f} vs {expected_loud:.2f}"

    quiet = stream(node, stereo_block(1000.0, 1000.0, amp_l=0.005, amp_r=0.005), 10)[0].max()
    expected_quiet = (20 * math.log10(0.005) - set_min) / (set_max - set_min)
    assert abs(quiet - expected_quiet) < 0.12, f"quiet level {quiet:.2f} vs {expected_quiet:.2f}"


def test_spectrum_silence_is_dark():
    node = make_node()
    stream(node, stereo_block(1000.0, 4000.0, 0.9, 0.9), 8)

    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    col = stream(node, silence, 10)

    assert col.max() < 0.05, f"silence leaves residual energy {col.max():.3f}"


def test_spectrum_start_clears_history():
    node = make_node()
    noise = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.8
    stream(node, noise, 8)

    node.start()  # transport restart
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    col = stream(node, silence, 8)

    assert col.max() < 0.05, "stale audio leaked across transport restart"


# ==============================================================================
# Real-time memory behaviour
# ==============================================================================


def test_spectrum_no_net_allocation_over_blocks():
    node = make_node()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    stream(node, blk, 5)  # warm up lazy paths / fill queue to steady state

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    stream(node, blk, 50)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"