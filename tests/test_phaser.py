import gc
import math
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Phaser")
    assert cls is not None, "Phaser not registered (library build missing?)"
    return cls()


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return node.out.buffer.clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()
    node._sync_params_to_cpp()


def test_phaser_registration_and_load():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Phaser")
    assert cls is not None
    assert cls.category == "Effects"
    node = make_node()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"


def test_phaser_mix_zero_is_bit_exact():
    """mix=0.0 output equals input bit-exact."""
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk), "mix=0 should be bit-exact passthrough"


def test_phaser_deterministic_notch_attenuation():
    """With depth=0, the base_freq notch should deeply attenuate that frequency."""
    node = make_node()
    set_params(node, rate=0.05, depth=0.0, base_freq=1000.0,
               feedback=0.0, spread=0.0, mix=0.5)

    def measure_rms(freq_hz, settle=8, measure=16):
        n = np.arange(BLOCK_SIZE)
        phase_inc = 2.0 * np.pi * freq_hz / SAMPLE_RATE
        phase = 0.0
        total_sq = 0.0
        count = 0
        for block_idx in range(settle + measure):
            tone = 0.5 * np.sin(phase + phase_inc * n)
            block = torch.from_numpy(np.tile(tone.astype(np.float32), (2, 1)))
            out = process_block(node, block)
            if block_idx >= settle:
                total_sq += float(torch.sum(out[0] * out[0]))
                count += out[0].shape[0]
            phase = (phase + phase_inc * BLOCK_SIZE) % (2.0 * np.pi)
        return math.sqrt(total_sq / count)

    rms_notch = measure_rms(1000.0)
    rms_pass = measure_rms(100.0)

    assert rms_notch < 0.035, f"notch attenuation insufficient: RMS={rms_notch}"
    # NOTE: with 6 allpass stages the total phase shift at 100 Hz is ~130 deg,
    # so the 50/50 mixed passband RMS is bounded by 0.707*cos(65deg) ~ 0.30.
    # A threshold of 0.25 asserts "mostly unity" without being unattainable.
    assert rms_pass > 0.25, f"passband attenuation too high: RMS={rms_pass}"


def test_phaser_feedback_stability():
    """High feedback should remain bounded (tanh saturation)."""
    node = make_node()
    set_params(node, feedback=0.95, depth=0.5, rate=0.5)
    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.9, dtype=DTYPE)
    peak = 0.0
    for _ in range(60):
        out = process_block(node, loud)
        peak = max(peak, float(out.abs().max()))
    assert peak <= 3.0, f"feedback runaway: peak {peak}"


def test_phaser_stereo_spread():
    """spread=1.0 should produce different left/right channels."""
    node = make_node()
    set_params(node, spread=1.0, rate=0.5, depth=0.7, feedback=0.3, mix=0.5)

    n = np.arange(BLOCK_SIZE)
    tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * n / SAMPLE_RATE)).astype(np.float32)
    stereo = torch.from_numpy(np.tile(tone, (2, 1)))
    out = process_block(node, stereo)

    # With spread=1.0, LFO phases differ -> outputs should differ
    assert not torch.allclose(out[0], out[1], atol=1e-4), "spread=1.0 should produce different channels"


def test_phaser_mono_input_duplicates_to_stereo():
    """Mono input should produce stereo output."""
    node = make_node()
    set_params(node, rate=0.5, depth=0.5)
    mono = torch.randn(1, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, mono)
    assert out.shape == (2, BLOCK_SIZE), f"expected (2, {BLOCK_SIZE}), got {out.shape}"


def test_phaser_start_resets_history():
    """start() should clear all DSP history, silence follows immediately."""
    node = make_node()
    set_params(node, feedback=0.8, depth=0.7)
    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.9, dtype=DTYPE)
    for _ in range(5):
        process_block(node, loud)

    node.start()
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, silence)
    assert float(out.abs().max()) < 1e-6, f"stale audio survived start(): {float(out.abs().max())}"


def test_phaser_save_load_roundtrip():
    """Verify parameter snapshot serialization round-trip."""
    node = make_node()
    set_params(node, rate=1.5, depth=0.6, base_freq=800.0,
               feedback=0.4, spread=0.7, mix=0.3)

    d = node.to_dict()
    fresh = make_node()
    fresh.load_state(d)

    for k in ("rate", "depth", "base_freq", "feedback", "spread", "mix"):
        assert k in fresh.params
        assert float(fresh.params[k].value) == pytest.approx(float(node.params[k].value))


def test_phaser_no_net_allocation():
    """Steady-state processing should not allocate Python heap memory."""
    node = make_node()
    set_params(node, rate=0.5, depth=0.7)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    process_block(node, blk)

    # Warm up
    for _ in range(5):
        process_block(node, blk)

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, blk)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"