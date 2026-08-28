"""Tests for modulation & CV generator nodes (ADSRNode, LFONode, GateButtonNode)."""

import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, SAMPLE_RATE, DTYPE


def make_node(class_name):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def test_modulation_registration():
    plugin_system.load_plugins("plugins")
    for name in ("ADSRNode", "LFONode", "GateButtonNode"):
        cls = plugin_system.NODE_REGISTRY.get(name)
        assert cls is not None, f"{name} not registered"
        doc = plugin_system.get_node_documentation(name)
        assert doc["label"], f"{name} missing label"
        assert doc["description"], f"{name} missing description"
        for p in doc["inputs"].values():
            assert p["help"], f"{name} input missing help"
        for p in doc["outputs"].values():
            assert p["help"], f"{name} output missing help"
        for p in doc["params"].values():
            assert p["help"], f"{name} param missing help"

    assert plugin_system.NODE_REGISTRY["LFONode"].category == "Sources"
    assert plugin_system.NODE_REGISTRY["GateButtonNode"].category == "Sources"
    assert plugin_system.NODE_REGISTRY["ADSRNode"].category == "Utilities"


# --------------------------------------------------------------------------
# ADSRNode
# --------------------------------------------------------------------------
def _feed_adsr(node, gate):
    node.inputs["gate"].get_tensor = lambda g=gate: g
    node.process()
    return node.outputs["out"].buffer[0].clone()


def test_adsr_progression():
    node = make_node("ADSRNode")
    set_params(node, attack=0.01, decay=0.1, sustain=0.7, release=0.3)
    node.start()

    gate_on = torch.ones((1, BLOCK_SIZE), dtype=DTYPE)
    gate_off = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)

    # 1. Idle block: envelope stays at zero.
    out0 = _feed_adsr(node, gate_off)
    assert float(out0.abs().max()) < 1e-6

    # 2. Rising gate engages Attack; within one block it ramps to 1.0.
    out1 = _feed_adsr(node, gate_on)
    assert float(out1.max()) == pytest.approx(1.0, abs=0.02)
    # Linear ramp up: samples before the peak should be increasing from 0.
    assert float(out1[0]) > 0.0

    # 3. Continue holding: envelope decays to sustain and pins there.
    last = None
    for _ in range(15):
        last = _feed_adsr(node, gate_on)
    assert float(last[-1]) == pytest.approx(0.7, abs=0.01)
    # Pinned: holding further produces the exact same sustain value.
    for _ in range(5):
        held = _feed_adsr(node, gate_on)
    assert float(held[-1]) == pytest.approx(0.7, abs=0.01)

    # 4. Drop the gate: envelope decays exponentially towards 0.0.
    prev = float(last[-1])
    released = _feed_adsr(node, gate_off)
    assert float(released[0]) < prev      # immediate decay
    final = None
    for _ in range(130):
        final = _feed_adsr(node, gate_off)
    assert float(final[-1]) < 0.01


def test_adsr_release_from_mid_attack():
    node = make_node("ADSRNode")
    set_params(node, attack=0.07, decay=0.1, sustain=0.7, release=0.3)
    node.start()

    gate_on = torch.ones((1, BLOCK_SIZE), dtype=DTYPE)
    gate_off = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)

    level = 0.0
    for _ in range(2):
        out = _feed_adsr(node, gate_on)
        level = float(out[-1])

    # Two blocks puts us mid-attack (approx 512*2 / (0.07*48000) ~ 0.30),
    # well below the sustain level.
    assert 0.2 <= level <= 0.5

    # Dropping the gate must release from that mid-attack level without ever
    # snapping up to the sustain value.
    released = _feed_adsr(node, gate_off)
    assert float(released.max()) < level + 1e-9
    assert float(released.max()) > 0.0
    assert not torch.allclose(released, torch.full_like(released, 0.7), atol=1e-3)

    # And it should continue exponentially decaying toward zero.
    final = released
    for _ in range(130):
        final = _feed_adsr(node, gate_off)
    assert float(final[-1]) < 0.01


# --------------------------------------------------------------------------
# LFONode
# --------------------------------------------------------------------------
def _set_freq_in(node, freq):
    tensor = torch.full((1, BLOCK_SIZE), freq, dtype=DTYPE)
    node.inputs["freq_in"].get_tensor = lambda t=tensor: t
    node.params["freq"].set(min(freq, 50.0))
    node.sync()


def test_lfo_waveform_shapes():
    node = make_node("LFONode")
    node.start()
    # 93.75 Hz -> exactly one cycle in a 512-sample block (start phase 0).
    _set_freq_in(node, SAMPLE_RATE / BLOCK_SIZE)
    node.process()

    sine = node.outputs["sine"].buffer[0].clone()
    tri = node.outputs["triangle"].buffer[0].clone()
    saw = node.outputs["saw"].buffer[0].clone()
    sq = node.outputs["square"].buffer[0].clone()

    # All bipolar outputs span [-1, +1].
    for w in (sine, tri, saw, sq):
        assert float(w.max()) == pytest.approx(1.0, abs=0.03)
        assert float(w.min()) == pytest.approx(-1.0, abs=0.03)

    # Sine starts near 0 (phase 0) and traverses the full cycle (peak-to-peak,
    # dip below zero).
    assert abs(float(sine[0])) < 0.02

    # Triangle / square start at +1 (phase 0).
    assert float(tri[0]) == pytest.approx(1.0, abs=0.02)
    assert float(sq[0]) == pytest.approx(1.0, abs=1e-6)

    # Sawtooth: first sample is +1.
    assert float(saw[0]) == pytest.approx(1.0, abs=0.02)

    # Square is bipolar binary.
    assert torch.all((sq == 1.0) | (sq == -1.0))


def test_lfo_hard_sync_resets_phase():
    node = make_node("LFONode")
    node.start()
    F = SAMPLE_RATE / BLOCK_SIZE  # 93.75 Hz
    _set_freq_in(node, F)

    # Prime the sync slot: a fully-low block so _prev_sync ends at 0.
    low = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)
    node.inputs["sync"].connected_outputs = [object()]
    node.inputs["sync"].get_tensor = lambda l=low: l
    node.process()

    # Mid-block rising edge at k = 256.
    k = 256
    sync = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)
    sync[0, k:] = 1.0
    node.inputs["sync"].get_tensor = lambda s=sync: s
    node.process()

    saw = node.outputs["saw"].buffer[0].clone()
    # At the edge index the phase resets to 0.0 -> saw approaches +1.0
    # (instead of ~0.0 without a reset).
    assert float(saw[k]) > 0.9
    # The internal phase really restarted from 0 at the edge sample.
    assert float(node._phase_buf[k].item()) < 0.01


def test_lfo_unipolar_mode():
    node = make_node("LFONode")
    set_params(node, bipolar=False)
    node.start()
    _set_freq_in(node, 8.0)

    for _ in range(4):
        node.process()

    for name in ("sine", "triangle", "saw", "square"):
        buf = node.outputs[name].buffer[0].clone()
        assert torch.all(buf >= 0.0), f"{name} went below 0 in unipolar mode"
        assert torch.all(buf <= 1.0), f"{name} went above 1 in unipolar mode"


# --------------------------------------------------------------------------
# GateButtonNode
# --------------------------------------------------------------------------
def test_gate_button_output():
    node = make_node("GateButtonNode")
    set_params(node, state=False)
    node.process()
    assert torch.all(node.outputs["out"].buffer[0] == 0.0)

    set_params(node, state=True)
    node.process()
    assert torch.all(node.outputs["out"].buffer[0] == 1.0)


# --------------------------------------------------------------------------
# Zero steady-state allocation
# --------------------------------------------------------------------------
def test_modulation_zero_steady_state_allocation():
    adsr = make_node("ADSRNode")
    set_params(adsr, attack=0.01, decay=0.1, sustain=0.7, release=0.3)
    adsr.start()
    gate_on = torch.ones((1, BLOCK_SIZE), dtype=DTYPE)
    gate_off = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)

    lfo = make_node("LFONode")
    lfo.start()

    gate_btn = make_node("GateButtonNode")

    def run_all(blocks):
        for i in range(blocks):
            if i % 4 == 0:
                adsr.inputs["gate"].get_tensor = lambda g=gate_on: g
            else:
                adsr.inputs["gate"].get_tensor = lambda g=gate_off: g
            adsr.process()
            lfo.process()
            gate_btn.process()

    run_all(5)  # warm up
    import gc
    gc.collect()
    tracemalloc.start()
    run_all(50)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"