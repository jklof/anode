import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("MathOp")
    assert cls is not None
    return cls()


def process(node, a, b=None):
    node.in_a.get_tensor = lambda t=a: t
    if b is not None:
        # Simulate a connected B input (get_tensor semantics mocked per convention)
        node.in_b.get_tensor = lambda t=b: t
        if not node.in_b.connected_outputs:
            node.in_b.connected_outputs.append(object())
    else:
        node.in_b.connected_outputs = []
        node.in_b.get_tensor = lambda: torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    node.process()
    return node.out.buffer.clone()


def set_op(node, idx):
    node.params["op"].set(idx)
    node.sync()


def test_math_op_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("MathOp")
    assert cls is not None
    assert cls.category == "Utilities"
    assert len(cls.OPERATIONS) == 10


def test_binary_operations_table():
    node = make_node()
    a = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=DTYPE)
    b = torch.full((CHANNELS, BLOCK_SIZE), 0.2, dtype=DTYPE)

    cases = {
        0: a + b,          # Add
        1: a - b,          # Subtract
        2: a * b,          # Multiply
    }
    for op_idx, expected in cases.items():
        set_op(node, op_idx)
        out = process(node, a, b)
        assert torch.allclose(out, expected), f"op {op_idx} failed"


def test_divide_sign_correct_epsilon():
    """A / (B + sign(B)*eps): tiny negative B must yield negative results."""
    node = make_node()
    set_op(node, 3)
    a = torch.ones(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    b_neg = torch.full((CHANNELS, BLOCK_SIZE), -1e-9, dtype=DTYPE)
    out = process(node, a, b_neg)
    assert (out < 0).all(), "negative denominator must give negative quotient"

    b_zero = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = process(node, a, b_zero)
    assert torch.isfinite(out).all(), "epsilon must prevent division by zero"
    assert (out > 0).all()


def test_scalar_min_max_clamp_identities():
    node = make_node()
    a = torch.linspace(-1, 1, BLOCK_SIZE, dtype=DTYPE).repeat(CHANNELS, 1)

    set_op(node, 4)   # Min with disconnected B uses scalar via clamp(max=s)
    node.params["scalar"].set(0.3)
    node.sync()
    out = process(node, a)
    assert torch.allclose(out, a.clamp(max=0.3))

    set_op(node, 5)   # Max -> clamp(min=s)
    out = process(node, a)
    assert torch.allclose(out, a.clamp(min=0.3))


def test_unary_operations():
    node = make_node()
    a = torch.full((CHANNELS, BLOCK_SIZE), -0.7, dtype=DTYPE)

    set_op(node, 6)
    assert torch.all(process(node, a) == 0.7)

    set_op(node, 7)
    assert torch.all(process(node, a) == 0.7)

    set_op(node, 8)
    assert torch.all(process(node, a) == 0.0)   # clamped into [0, 1]


def test_scale_and_offset_ignores_b():
    node = make_node()
    set_op(node, 9)
    node.params["scalar"].set(2.0)
    node.params["offset"].set(0.25)
    node.sync()
    a = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=DTYPE)
    out = process(node, a)
    assert torch.allclose(out, torch.full_like(a, 1.25))


def test_mono_a_stereo_b_broadcast_polarity():
    """Result channel count follows the widest operand (documented)."""
    node = make_node()
    set_op(node, 2)   # Multiply
    mono_a = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
    stereo_b = torch.full((CHANNELS, BLOCK_SIZE), 4.0, dtype=DTYPE)
    node.out.buffer.fill_(0.99)
    out = process(node, mono_a, stereo_b)
    assert out.shape[0] == CHANNELS
    assert torch.all(out == 2.0)


def test_math_op_no_net_allocation():
    node = make_node()
    set_op(node, 3)
    a = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.5
    b = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.5 + 0.1
    process(node, a, b)

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"
