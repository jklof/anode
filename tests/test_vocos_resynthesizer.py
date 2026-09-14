"""
tests/test_vocos_resynthesizer.py
Verifies registration, primitive geometry, two-part ring wrap with known ramp,
active wet-path execution under 64 KB heap allocation limit, and save/load state.
"""
import gc
import tracemalloc
import pytest
import torch
import numpy as np

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE
from plugins.vocos_resynthesizer import build_halfband_kernel, build_vocos_mel_filterbank


def test_dsp_primitives_geometry():
    kernel = build_halfband_kernel()
    assert kernel.shape == (1, 1, 63)
    assert kernel.sum().item() == pytest.approx(1.0, abs=1e-3)

    mel_fb = build_vocos_mel_filterbank()
    assert mel_fb.shape == (100, 513)
    assert mel_fb.dtype == torch.float32


def test_vocos_registration_and_docs():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    assert cls is not None
    assert cls.category == "Effects"
    doc = plugin_system.get_node_documentation("VocosResynthesizer")
    assert "in" in doc["inputs"]
    assert "out" in doc["outputs"]
    assert "model_path" in doc["params"]
    assert "mix" in doc["params"]


def test_delay_ring_two_part_wrap_with_known_ramp():
    """
    Simulates wrap boundary crossing and asserts exact sample equality across the seam.
    """
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    node = cls()

    # Pre-populate delay ring with a continuous ramp (0.0 to 16383.0)
    ramp = torch.arange(node.in_ring_size, dtype=DTYPE).repeat(CHANNELS, 1)
    node.in_delay_ring.copy_(ramp)

    # Position wp so rp lands at 16322 (crosses boundary: 62 samples at tail, 450 at head)
    node.write_pos = 1536
    rp = (1536 - node.LATENCY_SAMPLES) & node.in_ring_mask
    assert rp == 16322

    blk = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    node.inp.get_tensor = lambda: blk

    # Process block (triggers read across circular wrap)
    node.process()

    # Build expected output from known ramp
    k = node.in_ring_size - rp
    expected = torch.cat([ramp[:, rp:], ramp[:, :BLOCK_SIZE - k]], dim=1)

    assert torch.allclose(node.buf_dry, expected), "Sample corruption detected across ring wrap boundary!"
    assert node.out.buffer.shape == (CHANNELS, BLOCK_SIZE)


def test_vocos_allocation_under_64kb_with_active_worker():
    """
    Exercises the wet path and pool transfers under the 64 KB memory limit.
    """
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    node = cls()

    # Simulate active session and pre-filled worker output pool
    node.session = object()
    node.params["mix"].set(1.0)
    for i in range(len(node.pool_out)):
        node.pool_out[i].fill(0.35)

    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.2
    node.inp.get_tensor = lambda: blk

    # Prime queue
    for i in range(30):
        node.queue_out.try_push(i % len(node.pool_out))

    # Warmup
    for _ in range(10):
        node.queue_out.try_push(1)
        node.process()

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for i in range(50):
        node.queue_out.try_push(i % len(node.pool_out))
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert growth - before < 64 * 1024, f"Allocated {growth - before} bytes (limit: 64 KB)"


def test_vocos_save_load_roundtrip():
    """Verify load_state restores parameters cleanly without missing-file errors."""
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    node = cls()

    # Empty model path in test avoids missing file warnings
    node.params["model_path"].set("")
    node.params["mix"].set(0.65)
    state = node.to_dict()

    restored = cls()
    restored.load_state(state)
    assert restored.params["mix"].value == pytest.approx(0.65)
    assert restored.params["model_path"].value == ""
