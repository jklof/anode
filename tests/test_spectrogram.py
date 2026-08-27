import math
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE


# ==============================================================================
# Helpers
# ==============================================================================


def make_node(class_name="SpectrogramDisplay"):
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


def stream(node, tensor, blocks):
    """Feed a tensor through node.process(), consuming the SPSC buffer every
    block exactly like the live widget. Clears any pre-phase backlog first
    (ring buffer holds maxsize=4; overflow frames are dropped by design). Returns
    the newest frame produced by the final block, or None."""
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


def sine_block(freq, amp=0.5, phase=0.0):
    n = np.arange(BLOCK_SIZE)
    tone = amp * np.sin(2 * np.pi * freq * n / SAMPLE_RATE + phase)
    return torch.from_numpy(np.tile(tone.astype(np.float32), (CHANNELS, 1)))


# ==============================================================================
# Registration / params
# ==============================================================================


def test_spectrogram_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SpectrogramDisplay")
    assert cls is not None
    assert cls.category == "Visual"
    assert cls.label == "Spectrogram"


def test_spectrogram_parameters_sync():
    node = make_node()
    assert "colormap" in node.params
    assert "min_db" in node.params
    assert "max_db" in node.params

    node.params["colormap"].set(1)  # Viridis
    node.params["min_db"].set(-60.0)
    node.sync()

    assert node.params["colormap"].value == 1
    assert node.params["min_db"].value == -60.0


# ==============================================================================
# Audio path
# ==============================================================================


def test_spectrogram_pass_through_stereo():
    node = make_node()

    n = np.arange(BLOCK_SIZE)
    l = 0.5 * np.sin(2 * np.pi * 1000.0 * n / SAMPLE_RATE)
    r = 0.25 * np.sin(2 * np.pi * 3000.0 * n / SAMPLE_RATE)
    stereo = torch.from_numpy(np.vstack([l, r]).astype(np.float32))

    stream(node, stereo, 8)

    assert torch.allclose(node.out.buffer, stereo), "pass-through must be bit-exact"


def test_spectrogram_mono_duplicates_pass_through():
    node = make_node()
    mono = sine_block(500.0)

    stream(node, mono, 4)

    out = node.out.buffer
    assert torch.allclose(out[0], mono[0])
    assert torch.allclose(out[1], mono[0]), "mono must duplicate to both channels"


def test_spectrogram_true_mono_channel_count():
    """Regression: a genuine (1, BLOCK) input must not crash the ring write
    (copy_ cannot broadcast (1, B) into the (2, 1) column slice)."""
    node = make_node()
    n = np.arange(BLOCK_SIZE)
    row = (0.5 * np.sin(2 * np.pi * 1000.0 * n / SAMPLE_RATE)).astype(np.float32)
    true_mono = torch.from_numpy(row).unsqueeze(0)  # shape (1, 512)

    col = stream(node, true_mono, 8)

    out = node.out.buffer
    assert out.shape[0] == CHANNELS
    assert torch.allclose(out[0], true_mono[0])
    assert torch.allclose(out[1], true_mono[0])
    assert col is not None and col.shape == (CHANNELS, 128)
    peak = int(np.argmax(col[0]))
    assert 40 <= peak <= 90, f"mono analysis peak at bin {peak}, expected ~1 kHz"


def test_spectrogram_column_shape_and_range():
    node = make_node()
    col = stream(node, sine_block(1000.0), 8)

    assert isinstance(col, np.ndarray)
    assert col.shape == (CHANNELS, 128)
    assert col.min() >= 0.0 and col.max() <= 1.0


def test_spectrogram_peak_at_tone_frequency_log_axis():
    node = make_node()
    # 1 kHz maps to norm_y=(3-1.301)/(4.301-1.301)=0.566 -> bin ~72
    col = stream(node, sine_block(1000.0, amp=0.9), 10)

    peak_bin = int(np.argmax(col[0]))
    assert 40 <= peak_bin <= 90, f"peak at bin {peak_bin}, expected near 1 kHz"
    # R channel analyses the same duplicated signal: peaks should agree closely
    assert abs(int(np.argmax(col[1])) - peak_bin) <= 2


def test_spectrogram_dbfs_level_calibration():
    """Normalization check: amplitude maps to ~20*log10(amp) dBFS."""
    node = make_node()
    set_min, set_max = -80.0, 0.0

    # Full-scale-ish tone (-6 dBFS): expect norm ≈ (80-6)/80 ≈ 0.93 ± scalloping
    loud = stream(node, sine_block(1000.0, amp=0.5), 10)[0].max()
    expected_loud = (20 * math.log10(0.5) - set_min) / (set_max - set_min)
    assert abs(loud - expected_loud) < 0.12, f"loud level {loud:.2f} vs {expected_loud:.2f}"

    # Quiet tone (-46 dBFS): expect norm ≈ 34/80 ≈ 0.43
    quiet = stream(node, sine_block(1000.0, amp=0.005), 10)[0].max()
    expected_quiet = (20 * math.log10(0.005) - set_min) / (set_max - set_min)
    assert abs(quiet - expected_quiet) < 0.12, f"quiet level {quiet:.2f} vs {expected_quiet:.2f}"


def test_spectrogram_silence_is_dark():
    node = make_node()
    stream(node, sine_block(1000.0, amp=0.9), 8)

    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    col = stream(node, silence, 10)  # fully flush the 2048-sample window

    assert col.max() < 0.05, f"silence leaves residual energy {col.max():.3f}"


def test_spectrogram_start_clears_history():
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


def test_spectrogram_no_net_allocation_over_blocks():
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

    # Queue saturates at maxsize=2 and overflow skips the payload copy, so
    # steady-state net growth should be near zero (rfft transients are freed).
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"