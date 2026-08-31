import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SwiftF0Node")
    assert cls is not None, "SwiftF0Node not registered"
    return cls()


def process_block(node, tensor):
    node.inp.get_tensor = lambda t=tensor: t
    node.process()


def sine_block(freq=440.0, amp=0.5):
    n = np.arange(BLOCK_SIZE)
    tone = (amp * np.sin(2 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    return torch.from_numpy(np.tile(tone, (CHANNELS, 1)))


def test_swift_f0_registration_and_metadata():
    node = make_node()
    assert node.category == "Utilities"
    assert node.label == "SwiftF0 Pitch & MIDI Tracker"
    assert len(node.description) >= 15

    doc = plugin_system.get_node_documentation("SwiftF0Node")
    assert "in" in doc["inputs"] and doc["inputs"]["in"]["help"]
    assert "out" in doc["outputs"] and doc["outputs"]["out"]["channels"] == CHANNELS

    for cv_port in ("pitch_out", "gate_out", "confidence_out"):
        assert cv_port in doc["outputs"]
        assert doc["outputs"][cv_port]["channels"] == 1
        assert doc["outputs"][cv_port]["slot_type"] == "audio"
        assert doc["outputs"][cv_port]["help"]

    assert "midi_out" in doc["outputs"]
    assert doc["outputs"]["midi_out"]["slot_type"] == "midi"
    assert doc["outputs"]["midi_out"]["help"]


def test_swift_f0_fmin_boundary_clamping():
    """Verify fmin below MODEL_MIN_F0 (46.875) is safely clamped and does not raise ValueError."""
    node = make_node()
    detector, epoch = node._build_detector_nrt(30.0, 1200.0, 0.4, epoch=1)
    if detector is not None:
        assert detector.fmin >= 46.875


def test_swift_f0_pass_through_and_cv_shape_guards():
    node = make_node()
    mono = torch.full((1, BLOCK_SIZE), 0.35, dtype=DTYPE)

    process_block(node, mono)

    # Audio pass-through broadcasts mono -> stereo
    assert node.out.buffer.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.allclose(node.out.buffer[0], mono[0])
    assert torch.allclose(node.out.buffer[1], mono[0])

    # All CV outputs maintain shape (1, BLOCK_SIZE)
    assert node.pitch_out.buffer.shape == (1, BLOCK_SIZE)
    assert node.gate_out.buffer.shape == (1, BLOCK_SIZE)
    assert node.conf_out.buffer.shape == (1, BLOCK_SIZE)
    assert node.midi_out.packet.messages == []


def test_swift_f0_mock_tracking_and_midi_emission():
    node = make_node()
    node.params["glide_ms"].set(0.0)
    node.params["glide_ms"].sync()

    class FakeMsg:
        def __init__(self, type, note, velocity):
            self.type = type
            self.note = note
            self.velocity = velocity

    fake_payload = {
        "gen": node._worker_generation,
        "f0": 220.0,
        "conf": 0.95,
        "voiced": True,
        "midi_msgs": [FakeMsg("note_on", 57, 80)],
    }
    node._results_queue.try_push(fake_payload)

    blk = sine_block(220.0)
    process_block(node, blk)

    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(220.0, abs=1e-3)
    assert node.gate_out.buffer[0, 0].item() == pytest.approx(1.0, abs=1e-3)
    assert node.conf_out.buffer[0, 0].item() == pytest.approx(0.95, abs=1e-3)
    assert len(node.midi_out.packet.messages) == 1
    assert node.midi_out.packet.messages[0][1].note == 57

    # Anti-ghosting: packet is cleared on the subsequent block
    process_block(node, blk)
    assert len(node.midi_out.packet.messages) == 0
    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(220.0, abs=1e-3)


def test_swift_f0_unvoiced_holds_pitch():
    node = make_node()
    node._target_f0 = 330.0
    node._current_f0 = 330.0

    node._results_queue.try_push({
        "gen": node._worker_generation,
        "f0": 0.0,
        "conf": 0.1,
        "voiced": False,
        "midi_msgs": [],
    })

    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    process_block(node, silence)

    # Gate drops to 0.0, pitch holds previous valid frequency (330 Hz)
    assert node.gate_out.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-3)
    assert node.conf_out.buffer[0, 0].item() == pytest.approx(0.1, abs=1e-3)
    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(330.0, abs=1e-3)


def test_swift_f0_glide_trajectory():
    node = make_node()
    node.params["glide_ms"].set(50.0)
    node.params["glide_ms"].sync()

    node._current_f0 = 220.0
    node._target_f0 = 440.0
    node._current_gate = 1.0

    blk = sine_block(440.0)
    process_block(node, blk)

    start_p = node.pitch_out.buffer[0, 0].item()
    end_p = node.pitch_out.buffer[0, -1].item()

    assert start_p == pytest.approx(220.0, abs=5.0)
    assert end_p > start_p
    assert end_p < 440.0


def test_swift_f0_start_stop_lifecycle():
    node = make_node()
    try:
        node.start()
        assert node.out.buffer.abs().max().item() == pytest.approx(0.0, abs=1e-6)
        assert node.pitch_out.buffer.abs().max().item() == pytest.approx(0.0, abs=1e-6)
    finally:
        node.stop()
    assert node._worker_thread is None or not node._worker_thread.is_alive()


def test_swift_f0_zero_net_allocation():
    node = make_node()
    blk = sine_block(440.0)

    for _ in range(5):
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


def test_swift_f0_decimator_lengths_and_dtype():
    """3:1 FIR decimation must reproduce resampy's int(L*16000/48000) length."""
    from plugins.swift_f0_node import ANALYSIS_SR as A_SR, _build_decim_kernel

    kernel = _build_decim_kernel()
    pad = (kernel.shape[-1] - 3) // 2

    for L in (7200, 7680, 10800, 14400):
        y = torch.nn.functional.conv1d(
            torch.zeros(1, 1, L, dtype=DTYPE), kernel, stride=3, padding=pad
        )
        assert y.shape[-1] == int(L * A_SR / SAMPLE_RATE)
        assert y.dtype == DTYPE


def test_swift_f0_decimator_passband_and_alias_rejection():
    """Kernel passes 440 Hz at unity amplitude and rejects 12 kHz.

    Without the anti-aliasing lowpass, 12 kHz would alias into the 0-8 kHz
    band after 3:1 decimation and corrupt the pitch estimate.
    """
    from plugins.swift_f0_node import _build_decim_kernel

    kernel = _build_decim_kernel()
    pad = (kernel.shape[-1] - 3) // 2
    L = 7200
    t = np.arange(L) / SAMPLE_RATE

    def rms(x):
        return float(torch.sqrt((x ** 2).mean()))

    # Unity DC gain: constant input produces constant output.
    dc_out = torch.nn.functional.conv1d(
        torch.ones(1, 1, L, dtype=DTYPE), kernel, stride=3, padding=pad
    )
    assert dc_out.mean().item() == pytest.approx(1.0, abs=1e-2)

    # Passband: 440 Hz tone keeps its amplitude (expected rms 0.5/sqrt(2)).
    tone = torch.from_numpy((0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)).view(1, 1, L)
    tone_out = torch.nn.functional.conv1d(tone, kernel, stride=3, padding=pad)
    assert rms(tone_out) == pytest.approx(0.5 / np.sqrt(2), rel=1e-2)

    # Alias rejection: 12 kHz is above the post-decimation Nyquist (8 kHz) and
    # must be attenuated far below in-band level (resampy kaiser_best parity).
    hi = torch.from_numpy((0.5 * np.sin(2 * np.pi * 12000.0 * t)).astype(np.float32)).view(1, 1, L)
    hi_out = torch.nn.functional.conv1d(hi, kernel, stride=3, padding=pad)
    assert rms(hi_out) < 0.01, "12 kHz must not alias into the 0-8 kHz analysis band"


def test_swift_f0_pool_handoff_sends_index_not_copy():
    """With a live worker, process() must hand the worker an integer pool index
    referencing a pre-allocated buffer, never a freshly allocated numpy copy
    per block (RT allocation regression guard)."""
    node = make_node()
    blk = sine_block(440.0)
    process_block(node, blk)  # warm-up (worker off: push path skipped)

    class _FakeWorker:
        def is_alive(self):
            return True

    node._worker_thread = _FakeWorker()
    try:
        for _ in range(10):
            process_block(node, blk)

        indices = []
        while True:
            payload, ok = node._audio_queue.try_pop()
            if not ok:
                break
            indices.append(payload)

        assert node._pool_seq >= 10
        assert len(indices) == 10, "all 10 blocks must be queued (capacity 64)"
        # Every payload is an integer pool index — the old code pushed a fresh
        # numpy array here (self._mono_scratch.copy()).
        assert all(isinstance(i, int) for i in indices)
        assert len(set(indices)) == 10, "pool indices must be unique during a cycle"
        assert all(0 <= i < len(node._mono_pool) for i in indices)

        # The referenced pool slot holds the exact downmixed audio for that
        # block (stereo inputs are averaged: (L + R) * 0.5).
        expected = blk[0].numpy().copy()
        for i in indices:
            assert node._mono_pool[i].shape == (BLOCK_SIZE,)
            assert np.allclose(node._mono_pool[i], expected, atol=1e-7)
    finally:
        node._worker_thread = None
