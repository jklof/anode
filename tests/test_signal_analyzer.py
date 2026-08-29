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


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SignalAnalyzer")
    assert cls is not None, "SignalAnalyzer not registered"
    return cls()


def process_block(node, tensor):
    """Isolated-node pattern: stub the input slot to return the given tensor
    (no graph connections involved), then process exactly one block."""
    node.inp.get_tensor = lambda t=tensor: t
    node.process()


def sine_block(freq=1000.0, amp=0.5):
    n = np.arange(BLOCK_SIZE)
    tone = (amp * np.sin(2 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    return torch.from_numpy(np.tile(tone, (CHANNELS, 1)))


# ==============================================================================
# Registration & metadata
# ==============================================================================


def test_signal_analyzer_registration_and_metadata():
    node = make_node()
    assert node.category == "Utilities"
    assert node.label == "Signal Analyzer"
    assert len(node.description) >= 15

    doc = plugin_system.get_node_documentation("SignalAnalyzer")
    assert "in" in doc["inputs"] and doc["inputs"]["in"]["help"]
    assert "out" in doc["outputs"] and doc["outputs"]["out"]["channels"] == CHANNELS

    for cv_port in ("rms_out", "peak_out", "dc_out", "crest_out", "zcr_out"):
        assert cv_port in doc["outputs"]
        assert doc["outputs"][cv_port]["channels"] == 1
        assert doc["outputs"][cv_port]["slot_type"] == "audio"
        assert doc["outputs"][cv_port]["help"]


# ==============================================================================
# Metric correctness
# ==============================================================================


def test_signal_analyzer_sine_wave_immediacy():
    node = make_node()
    blk = sine_block(freq=1000.0, amp=0.5)

    # Process exactly ONE block
    process_block(node, blk)

    # Assert immediate per-block CV output calculations
    assert node.out_rms.buffer[0, 0].item() == pytest.approx(0.5 / math.sqrt(2), abs=0.03)
    assert node.out_peak.buffer[0, 0].item() == pytest.approx(0.5, abs=0.01)
    # Tolerance accounts for the phase-dependent mean of 10.67 non-integer
    # cycles across 512 samples at 1 kHz / 48 kHz.
    assert node.out_dc.buffer[0, 0].item() == pytest.approx(0.0, abs=0.03)
    assert node.out_crest.buffer[0, 0].item() == pytest.approx(math.sqrt(2), abs=0.1)
    assert node.out_zcr.buffer[0, 0].item() == pytest.approx((1000.0 * 2.0) / SAMPLE_RATE, abs=0.01)


def test_signal_analyzer_dc_offset():
    node = make_node()
    dc = torch.full((CHANNELS, BLOCK_SIZE), 0.75, dtype=DTYPE)

    process_block(node, dc)

    assert node.out_dc.buffer[0, 0].item() == pytest.approx(0.75, abs=1e-5)
    assert node.out_peak.buffer[0, 0].item() == pytest.approx(0.75, abs=1e-5)
    assert node.out_rms.buffer[0, 0].item() == pytest.approx(0.75, abs=1e-5)
    assert node.out_crest.buffer[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert node.out_zcr.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-5)


def test_signal_analyzer_silence():
    node = make_node()
    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    process_block(node, silence)

    assert node.out_rms.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_peak.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_dc.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_crest.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_zcr.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)


# ==============================================================================
# Anti-ghosting / lifecycle
# ==============================================================================


def test_signal_analyzer_anti_ghosting_loud_to_silence():
    """A loud block immediately followed by silence must drop ALL CV outputs
    to 0.0 on that very block (no stale values across the boundary)."""
    node = make_node()
    loud = sine_block(freq=1000.0, amp=0.9)
    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

    process_block(node, loud)
    assert node.out_peak.buffer[0, 0].item() > 0.5

    # Process silence: all outputs must drop immediately to 0.0
    process_block(node, silence)
    assert node.out_rms.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_peak.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_dc.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_crest.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_zcr.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_signal_analyzer_disconnected_input():
    """An unpatched InputSlot returns a zeroed scratch buffer; every metric
    must read silence rather than crash or report garbage."""
    node = make_node()
    node.process()  # no connections, no param attachment

    assert node.out_rms.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_peak.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_dc.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_crest.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert node.out_zcr.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.all(node.out.buffer == 0.0)


def test_signal_analyzer_start_resets_buffers():
    node = make_node()

    # Pollute all buffers
    node.out.buffer.fill_(0.99)
    node.out_rms.buffer.fill_(0.99)
    node.out_peak.buffer.fill_(0.99)
    node.out_dc.buffer.fill_(0.99)
    node.out_crest.buffer.fill_(0.99)
    node.out_zcr.buffer.fill_(0.99)

    node.start()

    assert torch.all(node.out.buffer == 0.0)
    assert torch.all(node.out_rms.buffer == 0.0)
    assert torch.all(node.out_peak.buffer == 0.0)
    assert torch.all(node.out_dc.buffer == 0.0)
    assert torch.all(node.out_crest.buffer == 0.0)
    assert torch.all(node.out_zcr.buffer == 0.0)


# ==============================================================================
# Channel adaptation / port contract
# ==============================================================================


def test_signal_analyzer_mono_pass_through_and_cv_shapes():
    """Mono input must broadcast to the stereo pass-through output, and the
    PyTorch out= buffer-shrinkage regression guard: all CV buffers must keep
    their strict (1, BLOCK_SIZE) shape after processing a mono input."""
    node = make_node()
    mono = torch.full((1, BLOCK_SIZE), 0.42, dtype=DTYPE)

    process_block(node, mono)

    # Pass-through output must broadcast to stereo (2, BLOCK_SIZE)
    assert node.out.buffer.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.allclose(node.out.buffer[0], mono[0])
    assert torch.allclose(node.out.buffer[1], mono[0])

    # All CV outputs must strictly remain (1, BLOCK_SIZE)
    assert node.out_rms.buffer.shape == (1, BLOCK_SIZE)
    assert node.out_peak.buffer.shape == (1, BLOCK_SIZE)
    assert node.out_dc.buffer.shape == (1, BLOCK_SIZE)
    assert node.out_crest.buffer.shape == (1, BLOCK_SIZE)
    assert node.out_zcr.buffer.shape == (1, BLOCK_SIZE)


# ==============================================================================
# Real-time memory behaviour
# ==============================================================================


def test_signal_analyzer_zero_net_allocation():
    node = make_node()
    blk = sine_block(freq=1000.0, amp=0.5)

    # 5 warm-up blocks to trigger any lazy PyTorch kernel initializations
    for _ in range(5):
        process_block(node, blk)

    import gc

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, blk)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Scratch buffers are pre-allocated and metric values are scalar fills,
    # so steady-state growth is near zero.
    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"
