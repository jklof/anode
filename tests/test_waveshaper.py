import math

import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node(class_name="WaveShaper"):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def sine_block(amp=0.5):
    n = np.arange(BLOCK_SIZE)
    tone = amp * np.sin(2 * np.pi * 1000.0 * n / 48000.0)
    return torch.from_numpy(np.tile(tone.astype(np.float32), (CHANNELS, 1)))


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return node.out.buffer.clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()


def test_waveshaper_registration():
    cls = plugin_system.NODE_REGISTRY.get("WaveShaper")
    assert cls is not None
    assert cls.category == "Effects"


def test_mix_zero_is_bit_exact_passthrough():
    node = make_node()
    set_params(node, mode=1, mix=0.0)   # soft clip, fully dry
    blk = sine_block(0.9)
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_tanh_calibration():
    node = make_node()
    set_params(node, mode=0, drive=2.0, bias=0.0, mix=1.0, output_level=1.0)
    blk = sine_block(0.5)               # driven peak = 1.0 -> tanh(1)
    out = process_block(node, blk)
    expected = math.tanh(1.0)
    assert out.abs().max() == pytest.approx(expected, abs=0.01)


def test_soft_clip_continuous_at_knee():
    node = make_node()
    set_params(node, mode=1, drive=2.0, mix=1.0)
    at_knee = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=DTYPE)   # driven = 1.0
    over = torch.full((CHANNELS, BLOCK_SIZE), 0.6, dtype=DTYPE)      # driven = 1.2
    out_at = process_block(node, at_knee)[0, 0]
    out_over = process_block(node, over)[0, 0]
    # Both branches meet at sign(x)*2/3 as x^3/3 -> 2/3 at x=1
    assert out_at == pytest.approx(2.0 / 3.0, abs=0.02)
    assert out_over == pytest.approx(2.0 / 3.0, abs=0.02)


def test_hard_clip_bounds():
    node = make_node()
    set_params(node, mode=2, drive=1.0, mix=1.0)
    blk = torch.full((CHANNELS, BLOCK_SIZE), 1.5, dtype=DTYPE)
    out = process_block(node, blk)
    assert torch.all(out == 1.0)


def test_wavefolder_folds():
    node = make_node()
    set_params(node, mode=3, drive=1.0, mix=1.0)
    blk = torch.full((CHANNELS, BLOCK_SIZE), 1.5707964, dtype=DTYPE)  # pi/2
    out = process_block(node, blk)
    assert out[0, 0] == pytest.approx(1.0, abs=0.01)


def test_asymmetric_tube_shifts_dc():
    node = make_node()
    set_params(node, mode=4, drive=2.0, mix=1.0)
    out = process_block(node, sine_block(0.5))
    mean_dc = float(out.mean())
    assert abs(mean_dc) > 0.01, "asymmetric tube must produce DC by design"


def test_mono_input_writes_both_channels_exactly():
    """Genuine (1, BLOCK) input: no stale pollution may survive on either
    output channel (exact-value anti-ghosting assert)."""
    node = make_node()
    set_params(node, mode=2, drive=1.0, mix=1.0)   # hard clip
    mono = torch.full((1, BLOCK_SIZE), 0.25, dtype=DTYPE)
    node.out.buffer.fill_(0.99)
    out = process_block(node, mono)
    assert torch.allclose(out[0], torch.full_like(out[0], 0.25))
    assert torch.allclose(out[1], torch.full_like(out[1], 0.25))


def test_drive_mod_param_binding():
    """drive_mod unconnected must fall back to the param value (no silence)."""
    node = make_node()
    set_params(node, mode=0, drive=2.0, mix=1.0)
    blk = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=DTYPE)
    out = process_block(node, blk)
    assert out[0, 0] == pytest.approx(math.tanh(1.0), abs=0.01)


def test_waveshaper_no_net_allocation():
    node = make_node()
    set_params(node, mode=1)
    blk = sine_block(0.5)
    process_block(node, blk)

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"
