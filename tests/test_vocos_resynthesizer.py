"""
tests/test_vocos_resynthesizer.py
Verifies registration, primitive geometry, two-part ring wrap with known ramp,
active wet-path execution under 64 KB heap allocation limit, and save/load state.
"""
import gc
import time
import tracemalloc
import pytest
import torch
import numpy as np

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE, SPSCRingBuffer
from plugins.vocos_resynthesizer import (
    build_halfband_kernel, build_vocos_mel_filterbank, _pop_newest,
)
from plugins.vocos_resynthesizer import VocosResynthesizer


def test_stride_latency_bookkeeping():
    """Taken hops must land exactly latency_blocks before playout so wet
    aligns with the dry delay line (LATENCY_SAMPLES), in every quality mode.
    Hop j <-> context frame j; a burst emitted during block N plays N+1..,
    so latency_blocks == window - oldest_take uniformly."""
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    node = cls()
    assert (node.CONTEXT_FRAMES, node.INFER_STRIDE, node.TAKE_HOPS,
            node.LATENCY_SAMPLES) == (8, 3, (4, 5, 6), 4 * BLOCK_SIZE + 62)
    node.params["quality"].set(1)
    node.params["quality"].sync()
    node.on_ui_param_change("quality")
    assert (node.CONTEXT_FRAMES, node.INFER_STRIDE, node.TAKE_HOPS,
            node.LATENCY_SAMPLES) == (24, 4, (9, 10, 11, 12), 15 * BLOCK_SIZE + 62)
    for mode in cls.QUALITY_MODES:
        w, k, take, lb = mode["window"], mode["stride"], mode["take"], mode["latency_blocks"]
        assert len(take) == k  # one emitted hop per stride step
        assert max(take) <= w - 2  # decodes of w frames yield w-1 hops
        assert lb == w - min(take)  # oldest take sets the uniform latency


def test_pop_newest_skips_stale_backlog():
    """_pop_newest returns the newest pending index (empty -> (None, False)),
    so a lagging worker fast-forwards instead of building a permanent
    stale backlog."""
    q = SPSCRingBuffer(capacity=8)
    assert _pop_newest(q) == (None, False)
    for i in range(5):
        assert q.try_push(i) is True
    assert _pop_newest(q) == (4, True)
    assert _pop_newest(q) == (None, False)


def test_thread_pools_capped_single():
    """base.py must cap BLAS/OpenMP fan-out: 512-sample block DSP never
    amortizes it, and spinning pools were measured burning ~6 CPU cores
    while starving the single-threaded ONNX worker next to them."""
    import torch as _t
    assert _t.get_num_threads() == 1


def test_vocos_nrt_load_installs_despite_epoch_bumps():
    """Regression: the NRT executor's supersede epoch is shared across tags on
    a node. During engine startup, load_state() and start() used to submit the
    same load twice; the stale result's destroy-submit then bumped the epoch so
    the VALID load result was routed to on_nrt_discarded() and destroyed — the
    session never installed and the node stayed on "Loading Model..." forever.
    The node must install the result whose plugin epoch still matches."""
    import os
    import types
    from core import NRTExecutor
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    node = cls()

    nrt = NRTExecutor()
    fake_engine = types.SimpleNamespace(nrt=nrt)
    fake_graph = types.SimpleNamespace(engine=fake_engine)
    node.graph = fake_graph

    path = "models/vocos_mel_24k.onnx"
    if not os.path.exists(path):
        pytest.skip("vocos ONNX model not exported")

    # load_state() then start(): the same path submitted twice (before the
    # in-flight dedupe, or via any concurrent path churn).
    node._load_onnx_model(path)
    node._load_onnx_model(path)  # second call must dedupe, but drain must be safe either way
    assert node._loading_path == path

    # Wait for the NRT pool job to actually finish before draining.
    deadline = time.time() + 30
    while node in nrt._in_flight and time.time() < deadline:
        time.sleep(0.1)
    nrt.drain(node)  # deliver load result(s); destroy submits bump the epoch
    nrt.drain(node)
    assert node.session is not None, "session failed to install after load"
    assert node.current_model_path == path
    assert node._loading_path == ""

    # Scenario 2: an unrelated submit (e.g. a discard_sess destroy) bumps the
    # executor epoch while a fresh load is in flight. The load result arrives
    # "superseded" but its plugin epoch still matches -> must be installed.
    node.session = None
    node.current_model_path = ""
    node._load_onnx_model(path)
    nrt.submit(node, lambda: ("dummy",), (), tag="discard_sess")  # epoch bump
    deadline = time.time() + 30
    while node in nrt._in_flight and time.time() < deadline:
        time.sleep(0.1)
    nrt.drain(node)
    assert node.session is not None, (
        "valid load result was destroyed because an unrelated submit "
        "bumped the shared supersede epoch")
    node.stop()


def test_vocos_nrt_discarded_stale_result_destroyed():
    """A genuinely stale load result (plugin epoch behind) must NOT install."""
    import types
    from core import NRTExecutor
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocosResynthesizer")
    node = cls()
    nrt = NRTExecutor()
    node.graph = types.SimpleNamespace(
        engine=types.SimpleNamespace(nrt=nrt))

    # Simulate an already-loaded session for path A, then a load for B that
    # completes late and is superseded: it must be destroyed, not installed.
    node.session = object()  # sentinel "old session"
    node.current_model_path = "a.onnx"
    node._load_epoch = 5
    node.on_nrt_discarded("load_vocos", True, (object(), 2, "b.onnx"))
    assert node.current_model_path == "a.onnx"  # unchanged
    assert node.session is not None



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
    assert cls.category == "Voice & Pitch"
    doc = plugin_system.get_node_documentation("VocosResynthesizer")
    assert "in" in doc["inputs"]
    assert "out" in doc["outputs"]
    assert "model_path" in doc["params"]
    assert "studio_model_path" in doc["params"]
    assert "mix" in doc["params"]
    assert "quality" in doc["params"]


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

    # Position wp so rp lands 62 samples before the ring end (crosses
    # boundary: 62 samples at tail, 450 at head). Derived from
    # LATENCY_SAMPLES rather than hardcoded so latency retunes keep working.
    rp_want = node.in_ring_size - 62
    node.write_pos = (rp_want + node.LATENCY_SAMPLES) & node.in_ring_mask
    rp = (node.write_pos - node.LATENCY_SAMPLES) & node.in_ring_mask
    assert rp == rp_want

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
    node.params["quality"].set(1)
    node.params["quality"].sync()
    node.on_ui_param_change("quality")
    assert node.LATENCY_SAMPLES == 15 * BLOCK_SIZE + 62
    assert node._active_model_path() == node.params["studio_model_path"].value
    state = node.to_dict()

    restored = cls()
    restored.load_state(state)
    assert restored.params["mix"].value == pytest.approx(0.65)
    assert restored.params["model_path"].value == ""
    assert restored.params["quality"].value == 1
    assert restored.LATENCY_SAMPLES == 15 * BLOCK_SIZE + 62
