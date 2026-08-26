import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node(class_name):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return {name: slot.buffer.clone() for name, slot in node.outputs.items()}


def test_spatial_registrations():
    for name in ["StereoPanner", "MidSideEncoder", "MidSideDecoder"]:
        cls = plugin_system.NODE_REGISTRY.get(name)
        assert cls is not None, name
        assert cls.category == "Utilities"


def test_panner_center_identity():
    node = make_node("StereoPanner")
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)["out"]
    assert torch.allclose(out[0], blk[0], atol=1e-6)
    assert torch.allclose(out[1], blk[1], atol=1e-6)


def test_panner_hard_left_constant_power():
    node = make_node("StereoPanner")
    node.params["pan"].set(-1.0)
    node.sync()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)["out"]
    # w=1: M-S == L exactly, so hard left is sqrt(2)*L and R is exactly 0
    assert torch.allclose(out[0], blk[0] * np.sqrt(2.0), atol=1e-5)
    assert torch.all(out[1] == 0.0)


def test_panner_width_zero_collapse_to_mono():
    node = make_node("StereoPanner")
    node.params["width"].set(0.0)
    node.sync()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)["out"]
    assert torch.allclose(out[0], out[1], atol=1e-6)
    mid = (blk[0] + blk[1]) * 0.5
    assert torch.allclose(out[0], mid, atol=1e-6)


def test_panner_mono_input_no_ghosting():
    node = make_node("StereoPanner")
    mono = torch.full((1, BLOCK_SIZE), 0.25, dtype=DTYPE)
    node.out.buffer.fill_(0.99)
    out = process_block(node, mono)["out"]
    assert torch.allclose(out[0], mono[0], atol=1e-6)
    assert torch.allclose(out[1], mono[0], atol=1e-6)


def test_mid_side_roundtrip_is_identity():
    enc = make_node("MidSideEncoder")
    dec = make_node("MidSideDecoder")
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.4

    encoded = process_block(enc, blk)
    # Encoder exposes separate mid/side outputs; stack them into the stereo
    # buffer the decoder expects (ch0 = mid, ch1 = side).
    stacked = torch.stack([encoded["mid"][0], encoded["side"][0]])
    dec.inp.get_tensor = lambda e=stacked: e
    dec.process()
    decoded = dec.out.buffer.clone()

    assert torch.allclose(decoded[0], blk[0], atol=1e-5)
    assert torch.allclose(decoded[1], blk[1], atol=1e-5)


def test_encoder_mono_side_is_zero():
    enc = make_node("MidSideEncoder")
    mono = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
    outs = process_block(enc, mono)
    # Mono duplicates R = L, so Mid = (L + R)/sqrt2 = 2*0.5/sqrt2
    assert torch.allclose(outs["mid"][0], mono[0] * np.sqrt(2.0), atol=1e-6)
    assert torch.all(outs["side"][0].abs() < 1e-6)


def test_decoder_mono_input_treats_side_as_zero():
    dec = make_node("MidSideDecoder")
    mono = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
    node_out = dec.out.buffer
    node_out.fill_(0.99)
    dec.inp.get_tensor = lambda m=mono: m
    dec.process()
    expected = 0.5 / np.sqrt(2.0)
    assert torch.allclose(dec.out.buffer[0], torch.full_like(node_out[0], expected), atol=1e-6)
    assert torch.allclose(dec.out.buffer[1], torch.full_like(node_out[1], expected), atol=1e-6)


def test_panner_no_net_allocation():
    node = make_node("StereoPanner")
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
