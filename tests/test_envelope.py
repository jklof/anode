import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("EnvelopeFollower")
    assert cls is not None, "EnvelopeFollower not registered (library build missing?)"
    return cls()


def sine_block(amp):
    n = np.arange(BLOCK_SIZE)
    tone = amp * np.sin(2 * np.pi * 1000.0 * n / 48000.0)
    return torch.from_numpy(np.tile(tone.astype(np.float32), (CHANNELS, 1)))


def process_block(node, blk):
    node.inputs["in"].get_tensor = lambda b=blk: b
    node.process()
    return node.outputs["cv_out"].buffer[0].clone(), node.outputs["gate_out"].buffer[0].clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()
    node._sync_params_to_cpp()


def test_envelope_registration_and_library_load():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("EnvelopeFollower")
    assert cls is not None
    assert cls.category == "Utilities"
    node = make_node()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"


def test_peak_mode_tracks_amplitude():
    node = make_node()
    set_params(node, mode=0, attack_ms=1.0, release_ms=50.0)
    for _ in range(10):   # converge well past the 1 ms attack
        cv, _ = process_block(node, sine_block(0.8))
    assert float(cv.max()) == pytest.approx(0.8, abs=0.05)


def test_rms_mode_averages_uncorrelated_channels():
    """Per-sample RMS differs from peak only across UNCORRELATED channels:
    constant L=0.6, R=0 -> rms=sqrt(0.36/2)=0.4243, peak=0.6."""
    node = make_node()
    blk = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    blk[0] = 0.6

    set_params(node, mode=1, attack_ms=1.0, release_ms=50.0)
    for _ in range(10):
        cv, _ = process_block(node, blk)
    assert float(cv.mean()) == pytest.approx(np.sqrt(0.36 / 2.0), abs=0.01)

    set_params(node, mode=0)
    for _ in range(10):
        cv, _ = process_block(node, blk)
    assert float(cv.mean()) == pytest.approx(0.6, abs=0.01)


def test_gate_hysteresis_holds_open_between_thresholds():
    node = make_node()
    set_params(node, mode=0, attack_ms=1.0, release_ms=20.0, gain=1.0,
               gate_thresh=0.5)

    for _ in range(10):                       # loud: opens
        _, gate = process_block(node, sine_block(0.8))
    assert torch.all(gate == 1.0)

    for _ in range(15):                       # mid: env ~0.4 in [0.25, 0.5)
        _, gate = process_block(node, sine_block(0.4))
    assert torch.all(gate == 1.0), "gate must hold open until env < thresh/2"

    for _ in range(15):                       # quiet: closes
        _, gate = process_block(node, sine_block(0.1))
    assert torch.all(gate == 0.0)


def test_cv_can_exceed_one_with_gain():
    node = make_node()
    set_params(node, mode=0, gain=5.0)
    for _ in range(10):
        cv, _ = process_block(node, sine_block(0.6))
    assert float(cv.max()) > 1.5   # documented: CV range is [0, gain]


def test_mono_input_works():
    node = make_node()
    set_params(node, mode=0, attack_ms=1.0, release_ms=20.0)
    mono = sine_block(0.6)[0].unsqueeze(0)      # genuine (1, BLOCK)
    for _ in range(10):
        node.inputs["in"].get_tensor = lambda m=mono: m
        node.process()
        cv = node.outputs["cv_out"].buffer[0].clone()
    assert float(cv.max()) == pytest.approx(0.6, abs=0.06)


def test_start_resets_envelope_state():
    node = make_node()
    set_params(node, mode=0, attack_ms=1.0, release_ms=500.0)
    for _ in range(10):
        process_block(node, sine_block(0.9))
    node.start()

    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    cv, gate = process_block(node, silence)
    assert float(cv.abs().max()) < 1e-6, "reset must zero the envelope"
    assert torch.all(gate == 0.0)


def test_envelope_no_net_allocation():
    node = make_node()
    blk = sine_block(0.5)
    process_block(node, blk)

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, blk)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"
