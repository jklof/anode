import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("NoiseGate")
    assert cls is not None
    return cls()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()


def rms_db(x):
    return 10.0 * float(torch.log10(torch.mean(x.pow(2))) + 1e-9)


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return node.out.buffer.clone()


GATED = {"thresh": -40.0, "ratio": 10.0, "attack": 1.0, "hold": 10.0,
         "release": 30.0, "range": 60.0}


def test_noise_gate_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("NoiseGate")
    assert cls is not None
    assert cls.category == "Effects"
    assert cls.label == "Noise Gate"


def test_loud_signal_passes_near_unity():
    node = make_node()
    set_params(node, **GATED)
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(BLOCK_SIZE) / 48000.0)).astype(np.float32)
    blk = torch.from_numpy(np.tile(tone, (CHANNELS, 1)))     # ~ -10 dBFS

    out = None
    for _ in range(10):
        out = process_block(node, blk)
    reduction_db = rms_db(out) - rms_db(blk)
    assert reduction_db > -1.5, f"open-gate loss {reduction_db:.2f} dB"


def test_quiet_signal_attenuated_toward_range():
    node = make_node()
    set_params(node, **GATED)
    noise = (torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.0005)   # ~ -66 dBFS

    out = None
    for _ in range(30):   # let release reach the floor
        out = process_block(node, noise)
    reduction_db = rms_db(out) - rms_db(noise)
    assert reduction_db <= -(60.0 - 12.0), f"only {reduction_db:.1f} dB attenuation"


def test_hold_freezes_release_during_short_gaps():
    node = make_node()
    set_params(node, thresh=-40.0, ratio=10.0, attack=1.0, hold=50.0,
               release=10.0, range=60.0)

    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=DTYPE)
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)

    for _ in range(10):
        process_block(node, loud)                 # gate fully open
    process_block(node, silence)
    # Hold of 50 ms (2400 samples) spans ~4.7 blocks; gate must still be open
    assert node.hold_left > 0 or node.gr_db == pytest.approx(0.0, abs=0.1)


def test_mono_sidechain_detection_without_crash():
    """Mono sidechain (1, BLOCK) into stereo gate must stay in-bounds."""
    node = make_node()
    set_params(node, **GATED)
    mono_sc = torch.full((1, BLOCK_SIZE), 0.001, dtype=DTYPE)   # silent-level key
    node.sc.connected_outputs.append(object())                  # mark connected
    node.sc.get_tensor = lambda m=mono_sc: m

    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=DTYPE)
    out = None
    for _ in range(20):
        out = process_block(node, loud)
    rms = float(torch.sqrt(torch.mean(out.pow(2))))
    assert rms < 0.01, "silent sidechain must gate the loud main signal"


def test_lookahead_ring_flushed_on_start():
    node = make_node()
    set_params(node, **GATED)
    process_block(node, torch.ones(CHANNELS, BLOCK_SIZE, dtype=DTYPE))
    assert not torch.all(node._ring == 0.0)

    node.start()
    assert torch.all(node._ring == 0.0)
    assert node.gr_db == 0.0
    assert node.hold_left == 0


def test_noise_gate_no_net_allocation():
    node = make_node()
    set_params(node, **GATED)
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
