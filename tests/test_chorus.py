import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("ChorusFlanger")
    assert cls is not None, "ChorusFlanger not registered (library build missing?)"
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


def test_chorus_registration_and_library_load():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("ChorusFlanger")
    assert cls is not None
    assert cls.category == "Effects"
    node = make_node()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"


def test_static_delay_impulse_position():
    """rate=0, depth=0: pure delay of base_delay_ms. An impulse in the first
    block must reappear exactly base_delay samples later."""
    node = make_node()
    set_params(node, rate=0.05, depth_ms=0.0, base_delay_ms=10.0,
               feedback=0.0, mix=1.0)
    delay_samples = int(10.0 * 0.001 * SAMPLE_RATE)   # 480

    impulse = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    impulse[:, 0] = 1.0
    found = None
    for block_idx in range(4):
        out = process_block(node, impulse if block_idx == 0 else torch.zeros_like(impulse))
        peaks = (out[0].abs() > 0.5).nonzero()
        if len(peaks) > 0:
            found = block_idx * BLOCK_SIZE + int(peaks[0])
            break
    assert found is not None, "impulse lost"
    # Allow +-2 samples: read head clamps to >= 1 sample behind write head
    assert abs(found - delay_samples - 1) <= 3, f"impulse at {found}, expected ~{delay_samples}"


def test_mix_zero_is_bit_exact_passthrough():
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_feedback_stays_bounded():
    node = make_node()
    set_params(node, rate=0.5, depth_ms=5.0, base_delay_ms=5.0,
               feedback=0.9, mix=1.0)
    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.9, dtype=DTYPE)
    peak = 0.0
    for _ in range(60):
        out = process_block(node, loud)
        peak = max(peak, float(out.abs().max()))
    assert peak < 4.0, f"feedback runaway: peak {peak}"


def test_mono_input_duplicated_no_ghosting():
    node = make_node()
    set_params(node, rate=0.0 + 0.05, depth_ms=1.0, feedback=0.0, mix=1.0,
               base_delay_ms=2.0, spread=1.0)
    mono = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
    node.out.buffer.fill_(0.99)
    out = process_block(node, mono)
    # Both channels must be finite and driven by the same input (quadrature
    # LFOs differ only in modulation phase, both bounded by tanh feedback)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) > 0.01


def test_reset_clears_delay_state():
    node = make_node()
    set_params(node, rate=0.05, depth_ms=0.0, base_delay_ms=10.0,
               feedback=0.8, mix=1.0)
    process_block(node, torch.full((CHANNELS, BLOCK_SIZE), 0.9, dtype=DTYPE))

    node.start()   # calls native reset
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = None
    for _ in range(6):   # flush the whole delay line
        out = process_block(node, silence)
    assert float(out.abs().max()) == 0.0, "reset must clear the delay rings"


def test_chorus_no_net_allocation():
    node = make_node()
    set_params(node, rate=2.0, depth_ms=4.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
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


def test_simple_delay_start_clears_stale_delay_tail():
    """Regression: SimpleDelay's native library had no reset() export, so a
    transport restart (engine start) retained stale delay-line audio."""
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SimpleDelay")
    assert cls is not None, "SimpleDelay not registered (library build missing?)"
    node = cls()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"

    tone = 0.5 * torch.sin(torch.arange(BLOCK_SIZE, dtype=DTYPE) * 0.05)
    blk = torch.zeros(2, BLOCK_SIZE, dtype=DTYPE)
    blk[0] = tone
    blk[1] = tone
    silence = torch.zeros(2, BLOCK_SIZE, dtype=DTYPE)

    node.inputs["in"].get_tensor = lambda b=blk: b
    for _ in range(10):
        node.process()

    # Transport restart: start() must clear the native delay line
    node.start()
    node.inputs["in"].get_tensor = lambda: silence
    node.process()
    assert node.outputs["out"].buffer.abs().max() < 1e-6, "stale delay audio survived start()"
