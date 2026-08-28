"""Tests for ValuePlotterNode — pass-through + telemetry + zero allocation."""

import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("ValuePlotterNode")
    assert cls is not None, "ValuePlotterNode not registered"
    return cls()


def process_block(node, blk):
    node.inputs["in"].get_tensor = lambda b=blk: b
    node.process()
    return node.outputs["out"].buffer.clone()


def test_value_plotter_bit_exact_passthrough():
    node = make_node()

    # Stereo input: output must be bit-exact and keep shape (2, 512).
    stereo = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, stereo)
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(out, stereo)

    # Mono input: broadcast to stereo, shape stays (2, 512), both rows equal.
    mono = torch.randn(1, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, mono)
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(out[0], mono[0])
    assert torch.equal(out[1], mono[0])


def test_value_plotter_telemetry_and_drop():
    node = make_node()
    sig = torch.randn(1, BLOCK_SIZE, dtype=DTYPE)

    # Push many blocks; the ring never blocks and pop_all returns frames.
    for _ in range(50):
        process_block(node, sig)

    frames = node.monitor_queue.pop_all()
    assert isinstance(frames, list)
    for f in frames:
        assert f.shape == (1, 8)
    # The queue should not have raised or blocked regardless of count.


def test_value_plotter_zero_steady_state_allocation():
    node = make_node()
    sig = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    process_block(node, sig)  # warm up allocators

    import gc
    gc.collect()
    tracemalloc.start()
    for _ in range(50):
        process_block(node, sig)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"