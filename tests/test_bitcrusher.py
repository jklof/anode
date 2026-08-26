import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE

DECIM_MAX = 64


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Bitcrusher")
    assert cls is not None
    return cls()


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return node.out.buffer.clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()


def test_bitcrusher_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("Bitcrusher")
    assert cls is not None
    assert cls.category == "Effects"


def test_max_bits_min_decim_is_near_passthrough():
    node = make_node()
    set_params(node, bits=16, downsample=1, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.5
    out = process_block(node, blk)
    assert (out - blk).abs().max() < 2 ** -14


def test_one_bit_quantizes_to_ternary():
    node = make_node()
    set_params(node, bits=1, downsample=1, mix=1.0)
    blk = torch.linspace(-1, 1, BLOCK_SIZE, dtype=DTYPE).repeat(CHANNELS, 1)
    out = process_block(node, blk)
    uniq = torch.unique(out)
    for v in uniq:
        assert v.item() in (-1.0, 0.0, 1.0), f"unexpected quantized value {v}"


def test_downsample_creates_plateaus():
    node = make_node()
    set_params(node, bits=16, downsample=8, mix=1.0)
    n = torch.arange(BLOCK_SIZE, dtype=DTYPE)
    ramp = (n / BLOCK_SIZE).unsqueeze(0).repeat(CHANNELS, 1)   # slow ramp
    out = process_block(node, ramp)

    diffs = int(torch.count_nonzero(torch.diff(out[0]).abs() > 1e-6))
    assert diffs <= BLOCK_SIZE // 8 + 1, f"too many steps: {diffs}"


def test_hold_grid_continuous_across_blocks():
    """Second block's hold values must follow the GLOBAL grid: sample at
    global index g holds the value of floor(g/D)*D — no re-triggered
    plateau edge at the block seam."""
    node = make_node()
    set_params(node, bits=16, downsample=8, mix=1.0)
    d = 8
    ramp = torch.arange(2 * BLOCK_SIZE, dtype=DTYPE).div_(2000.0)
    two_blocks = ramp.unsqueeze(0).repeat(CHANNELS, 1)

    first = process_block(node, two_blocks[:, :BLOCK_SIZE].contiguous())
    second = process_block(node, two_blocks[:, BLOCK_SIZE:].contiguous())

    # Verify a few samples in the second block against the global grid.
    # Tolerance covers bits=16 quantization (quantum ~3e-5).
    g0 = BLOCK_SIZE
    for local in (0, 3, d, d + 5):
        g = g0 + local
        k = (g // d) * d
        expected = two_blocks[0, k]
        assert second[0, local] == pytest.approx(expected.item(), abs=4e-5)


def test_mono_input_writes_both_channels_exactly():
    node = make_node()
    set_params(node, bits=16, downsample=4, mix=1.0)
    mono = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
    node.out.buffer.fill_(0.99)
    out = process_block(node, mono)
    assert torch.allclose(out[0], torch.full_like(out[0], 0.5))
    assert torch.allclose(out[1], torch.full_like(out[1], 0.5))


def test_mix_zero_is_bit_exact_passthrough():
    node = make_node()
    set_params(node, bits=2, downsample=32, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_start_resets_hold_grid_and_tail():
    node = make_node()
    set_params(node, bits=16, downsample=8, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    process_block(node, blk)
    assert node._g0 == BLOCK_SIZE

    node.start()
    assert node._g0 == 0
    assert torch.all(node._tail == 0.0)


def test_bitcrusher_no_net_allocation():
    node = make_node()
    set_params(node, bits=8, downsample=8)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
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
