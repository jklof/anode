"""Tests for the RubberBandPitchShifter plugin (requires pylibrb)."""

import tracemalloc

import numpy as np
import pytest
import torch

pytest.importorskip("pylibrb")

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("RubberBandPitchShifter")
    assert cls is not None, "RubberBandPitchShifter not registered"
    return cls()


@pytest.fixture
def rb_node():
    """Node whose native stretcher is released deterministically on teardown,
    so pylibrb handles never outlive the test (nanobind leak guard)."""
    node = make_node()
    yield node
    node.remove()


def sine_block(freq=440.0, start_sample=0, amp=0.4, channels=CHANNELS):
    n = np.arange(start_sample, start_sample + BLOCK_SIZE)
    tone = (amp * np.sin(2 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    return torch.from_numpy(np.tile(tone, (channels, 1)))


def feed(node, block):
    node.inp.get_tensor = lambda b=block: b
    node.process()
    return node.out.buffer


def test_rubberband_registration_and_documentation(rb_node):
    node = rb_node
    assert node.category == "Effects"
    assert node.label == "RubberBand Pitch Shifter"
    assert len(node.description) >= 15

    doc = plugin_system.get_node_documentation("RubberBandPitchShifter")
    assert doc["category"] == "Effects"
    assert doc["description"]
    for port in ("in", "pitch_mod", "formant_mod"):
        assert port in doc["inputs"] and doc["inputs"][port]["help"]
    assert doc["inputs"]["pitch_mod"]["param_name"] == "pitch_shift"
    assert doc["inputs"]["formant_mod"]["param_name"] == "formant_shift"
    assert doc["outputs"]["out"]["channels"] == CHANNELS

    params = doc["params"]
    assert params["pitch_shift"]["meta"]["min"] == -24.0
    assert params["pitch_shift"]["meta"]["max"] == 24.0
    assert params["formant_shift"]["meta"]["min"] == -24.0
    assert params["mix"]["value"] == 1.0
    assert params["formant_mode"]["type"] == "menu"

    tel = node.get_telemetry()
    assert isinstance(tel["latency_samples"], int) and tel["latency_samples"] >= 0
    assert tel["latency_ms"] >= 0.0


def test_pitch_shift_frequency_accuracy(rb_node):
    node = rb_node
    node.params["pitch_shift"].set(12.0)
    node.sync()

    blocks = []
    for i in range(64):
        blocks.append(
            feed(node, sine_block(440.0, start_sample=i * BLOCK_SIZE)).clone())

    # Trim the latency-fill period and a settling margin before analysis.
    y = torch.cat(blocks, dim=1)[0].numpy()[8192:]
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, 1.0 / SAMPLE_RATE)
    peak = freqs[np.argmax(spec)]
    assert abs(peak - 880.0) < 5.0


def test_formant_modes_run_clean(rb_node):
    node = rb_node
    node.params["pitch_shift"].set(7.0)
    node.params["formant_shift"].set(-3.0)
    for mode in (0, 1, 2):
        node.start()
        node.params["formant_mode"].set(mode)
        node.sync()
        for i in range(30):
            out = feed(node, sine_block(220.0, start_sample=i * BLOCK_SIZE))
            assert out.shape == (CHANNELS, BLOCK_SIZE)
            assert torch.isfinite(out).all()


def test_mono_input_channel_adaptation(rb_node):
    node = rb_node
    node.params["pitch_shift"].set(12.0)
    node.sync()
    for i in range(24):
        out = feed(node, sine_block(440.0, start_sample=i * BLOCK_SIZE, channels=1))
        assert out.shape == (CHANNELS, BLOCK_SIZE)
        # Mono input is duplicated: both output channels stay identical and
        # the buffer never shrinks (out= shrinkage regression guard).
        assert torch.allclose(out[0], out[1])
    assert out.abs().max() > 0.0  # wet audio flows after the latency fill


def test_mix_zero_is_bit_exact_passthrough(rb_node):
    node = rb_node
    node.params["mix"].set(0.0)
    node.sync()
    for i in range(20):
        block = sine_block(440.0, start_sample=i * BLOCK_SIZE)
        out = feed(node, block)
        assert torch.equal(out, block)


def test_transport_restart_clears_buffers(rb_node):
    node = rb_node
    for i in range(20):
        feed(node, sine_block(440.0, start_sample=i * BLOCK_SIZE))
    tel = node.get_telemetry()
    assert tel["buffered_samples"] > 0
    assert tel["primed"]

    node.start()
    tel = node.get_telemetry()
    assert tel["buffered_samples"] == 0
    assert not tel["primed"]

    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    for _ in range(20):
        out = feed(node, silence)
        assert out.abs().max() == 0.0


def test_zero_steady_state_heap_allocations(rb_node):
    node = rb_node
    for i in range(20):
        feed(node, sine_block(440.0, start_sample=i * BLOCK_SIZE))

    block = sine_block(440.0, start_sample=999 * BLOCK_SIZE)
    node.inp.get_tensor = lambda: block
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        node.process()
    current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert (current - before) < 64 * 1024


def test_missing_library_fallback_bypass(rb_node):
    node = rb_node
    node._stretcher = None
    mono = sine_block(440.0, channels=1)
    out = feed(node, mono)
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(out[0], mono[0])
    assert torch.equal(out[1], mono[0])


def test_remove_releases_native_stretcher_deterministically(rb_node):
    """remove() must drop the node's reference to the native handle. The
    attribute is the ONLY reference (the stretcher is not part of any cycle),
    so refcounting frees the native object immediately instead of leaving it
    to interpreter teardown (nanobind leak guard). This is the hook core.py's
    delete handler and the app's shutdown path rely on."""
    node = rb_node
    assert node._stretcher is not None
    node.remove()
    assert node._stretcher is None
    # Telemetry must degrade safely after release.
    tel = node.get_telemetry()
    assert tel["latency_samples"] == 0
    assert not tel["primed"]
