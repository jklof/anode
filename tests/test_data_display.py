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


def make_node(class_name="DataDisplayNode"):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def drop_pending(node):
    """Emulate the UI consuming frames (queue holds maxsize=1; overflow
    frames are dropped by design)."""
    while not node.monitor_queue.empty():
        node.monitor_queue.get_nowait()


def stream(node, tensor, blocks):
    """Feed a tensor through node.process(), consuming the monitor queue every
    block exactly like the live widget. Clears any pre-phase backlog first.
    Returns the newest frame produced by the final block, or None."""
    drop_pending(node)
    last = None
    for _ in range(blocks):
        node.inp.get_tensor = lambda t=tensor: t
        node.process()
        while not node.monitor_queue.empty():
            last = node.monitor_queue.get_nowait()
    return last


def sine_block(freq, amp=0.5):
    n = np.arange(BLOCK_SIZE)
    tone = amp * np.sin(2 * np.pi * freq * n / SAMPLE_RATE)
    return torch.from_numpy(np.tile(tone.astype(np.float32), (CHANNELS, 1)))


# ==============================================================================
# Registration
# ==============================================================================


def test_data_display_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("DataDisplayNode")
    assert cls is not None
    assert cls.category == "Visual"
    assert cls.label == "Data Display"


# ==============================================================================
# Audio path: pass-through integrity
# ==============================================================================


def test_data_display_pass_through_stereo():
    node = make_node()

    n = np.arange(BLOCK_SIZE)
    l = 0.5 * np.sin(2 * np.pi * 1000.0 * n / SAMPLE_RATE)
    r = 0.25 * np.sin(2 * np.pi * 3000.0 * n / SAMPLE_RATE)
    stereo = torch.from_numpy(np.vstack([l, r]).astype(np.float32))

    stream(node, stereo, 8)

    assert torch.allclose(node.out.buffer, stereo), "pass-through must be bit-exact"


def test_data_display_mono_pass_through_broadcasts():
    """A genuine (1, BLOCK) mono input must pass through duplicated to both
    channels without crashing the broadcast copy_."""
    node = make_node()
    n = np.arange(BLOCK_SIZE)
    row = (0.5 * np.sin(2 * np.pi * 1000.0 * n / SAMPLE_RATE)).astype(np.float32)
    true_mono = torch.from_numpy(row).unsqueeze(0)  # shape (1, 512)

    frame = stream(node, true_mono, 8)

    out = node.out.buffer
    assert out.shape[0] == CHANNELS
    assert torch.allclose(out[0], true_mono[0])
    assert torch.allclose(out[1], true_mono[0])
    # Regression: the RMS scratch buffer must never be resized by out= ops
    assert node._squared.shape == (CHANNELS, BLOCK_SIZE)
    # Stats must still classify as a Tensor with the mono shape reported
    assert frame is not None and frame["type"] == "Tensor"
    assert "(1, 512)" in frame["shape"]


# ==============================================================================
# Statistics calibration
# ==============================================================================


def test_data_display_sine_stats():
    node = make_node()

    t = torch.linspace(0, BLOCK_SIZE / SAMPLE_RATE, BLOCK_SIZE)
    test_sig = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    test_sig[0] = 0.5 * torch.sin(2 * np.pi * 1000.0 * t)
    test_sig[1] = 0.5 * torch.sin(2 * np.pi * 1000.0 * t)

    stream(node, test_sig, node.UPDATE_INTERVAL_BLOCKS + 1)

    telemetry = node.get_telemetry()
    assert telemetry["type"] == "Tensor"
    assert telemetry["dtype"] == "float32"
    assert not telemetry["is_constant"]

    # Peak should be ~0.5 (-6.02 dBFS)
    assert telemetry["peak"] == pytest.approx(0.5, abs=0.05)
    assert telemetry["peak_db"] == pytest.approx(-6.02, abs=0.5)

    # RMS for 0.5 amplitude sine is 0.5 / sqrt(2) ~= 0.3535 (-9.03 dBFS)
    assert telemetry["rms"] == pytest.approx(0.3535, abs=0.05)
    assert telemetry["rms_db"] == pytest.approx(-9.03, abs=0.5)

    # DC offset should be near 0.0
    assert telemetry["mean"] == pytest.approx(0.0, abs=0.05)

    # Crest factor of a pure sine is sqrt(2)
    assert telemetry["crest_factor"] == pytest.approx(math.sqrt(2), rel=0.15)


def test_data_display_constant_detection():
    node = make_node()

    dc_sig = torch.full((CHANNELS, BLOCK_SIZE), 0.75, dtype=DTYPE)
    stream(node, dc_sig, node.UPDATE_INTERVAL_BLOCKS + 1)

    telemetry = node.get_telemetry()
    assert telemetry["is_constant"] is True
    assert telemetry["type"] == "Constant"
    assert telemetry["constant_val"] == pytest.approx(0.75, abs=1e-5)
    assert telemetry["std"] == pytest.approx(0.0, abs=1e-6)


# ==============================================================================
# Classification: silence must never read as Constant
# ==============================================================================


def test_data_display_silence_classified_not_constant():
    """Regression: an all-zero (silent/disconnected) input trivially satisfies
    the constant test; it must report Silent instead."""
    node = make_node()
    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    stream(node, silence, node.UPDATE_INTERVAL_BLOCKS + 1)

    t = node.get_telemetry()
    assert t["type"] == "Silent"
    assert t["is_constant"] is False


def test_data_display_loud_to_silence_transition():
    node = make_node()
    stream(node, sine_block(1000.0, amp=0.9), 8)

    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    stream(node, silence, node.UPDATE_INTERVAL_BLOCKS + 1)

    t = node.get_telemetry()
    assert t["type"] == "Silent"
    # Zero peak clamps to the 1e-9 dB floor (-180 dB), well below any audible level
    assert t["peak_db"] <= -120.0


def test_data_display_very_quiet_signal_is_tensor_not_constant():
    """A non-silent, non-constant signal above the -120 dBFS floor stays a
    Tensor even at very low amplitude (a tiny constant DC would correctly
    read Constant; this exercises the varying case)."""
    node = make_node()
    n = np.arange(BLOCK_SIZE)
    tone = (5e-4 * np.sin(2 * np.pi * 1000.0 * n / SAMPLE_RATE)).astype(np.float32)
    quiet = torch.from_numpy(np.tile(tone, (CHANNELS, 1)))  # ~ -66 dBFS

    stream(node, quiet, node.UPDATE_INTERVAL_BLOCKS + 1)

    t = node.get_telemetry()
    assert t["type"] == "Tensor"
    assert t["is_constant"] is False
    assert t["peak_db"] == pytest.approx(20 * math.log10(5e-4), abs=0.5)


# ==============================================================================
# Transport restart & rate limiting
# ==============================================================================


def test_data_display_start_resets_state():
    node = make_node()
    stream(node, sine_block(1000.0, amp=0.9), 8)
    assert node.get_telemetry()["type"] == "Tensor"

    node.start()  # transport restart

    t = node.get_telemetry()
    assert t["type"] == "None"
    assert t["peak_db"] == -120.0
    assert t["is_constant"] is False
    # Counter reset means analysis does NOT fire on the first block back
    drop_pending(node)
    node.inp.get_tensor = lambda: sine_block(500.0)
    node.process()
    assert node.monitor_queue.empty(), "analysis must not fire before UPDATE_INTERVAL_BLOCKS"


def test_data_display_rate_limits_analysis():
    node = make_node()
    blk = sine_block(1000.0)

    drop_pending(node)
    for i in range(1, node.UPDATE_INTERVAL_BLOCKS):
        node.inp.get_tensor = lambda b=blk: b
        node.process()
        assert node.monitor_queue.empty(), f"frame dispatched early at block {i}"

    node.process()  # Nth block triggers analysis
    assert not node.monitor_queue.empty()


def test_data_display_queue_dispatch_drain():
    node = make_node()
    blk = sine_block(1000.0)
    frame = stream(node, blk, 12)

    assert isinstance(frame, dict)
    assert frame is node.get_telemetry(), "queue payload must be the live telemetry snapshot"


# ==============================================================================
# Real-time memory behaviour
# ==============================================================================


def test_data_display_no_net_allocation_over_blocks():
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

    # Queue saturates at maxsize=1, telemetry dict mutates in place, and the
    # RMS scratch buffer is pre-allocated, so steady-state growth is near zero.
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"
