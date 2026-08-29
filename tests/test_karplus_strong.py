import gc
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("KarplusStrong")
    assert cls is not None, "KarplusStrong not registered (library build missing?)"
    return cls()


def process_block(node, trig):
    node.inputs["trigger"].get_tensor = lambda b=trig: b
    node.process()
    return node.outputs["out"].buffer.clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()
    node._sync_params_to_cpp()


def test_karplus_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("KarplusStrong")
    assert cls is not None
    assert cls.category == "Sources"


def test_karplus_pluck_edge_vs_held_gate():
    """A constant gate should pluck only on the rising edge, not re-trigger."""
    node = make_node()
    set_params(node, freq=220.0, damping=0.3, brightness=0.8, decay=0.99)

    gate = torch.ones(CHANNELS, BLOCK_SIZE, dtype=DTYPE)

    # Block 0: rising edge (from 0 to 1) triggers pluck
    out0 = process_block(node, gate)
    peak0 = float(out0.abs().max())

    # Block 1: sustained high gate should NOT re-trigger
    out1 = process_block(node, gate)
    peak1 = float(out1.abs().max())

    # Peak in block 1 should be less than peak in block 0 (natural decay)
    assert peak1 < peak0, f"held gate re-triggered: peak1={peak1} >= peak0={peak0}"

    # Feed silence -> output decays. At 220 Hz there are ~218 recirculations
    # per 512-sample block; 200 blocks (~0.9 s) is ample for decay=0.99.
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    for _ in range(200):
        out_sil = process_block(node, silence)
    assert float(out_sil.abs().max()) < 0.01, "string did not decay to silence"


def test_karplus_pitch_accuracy():
    """Verify the fundamental frequency matches the requested pitch."""
    node = make_node()
    set_params(node, freq=440.0, damping=0.0, decay=1.0, brightness=1.0)

    # Trigger with impulse at sample 0
    trig = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    trig[:, 0] = 1.0

    # Collect 8 blocks (4096 samples) after the pluck
    signal = []
    out = process_block(node, trig)
    signal.append(out[0].clone())

    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    for _ in range(7):
        out = process_block(node, silence)
        signal.append(out[0].clone())

    audio = torch.cat(signal)

    # Window and zero-pad FFT
    n_fft = 32768
    spec = torch.fft.rfft(audio * torch.hann_window(len(audio)), n=n_fft).abs()
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)

    # Find peak in expected range
    f_lo, f_hi = 200.0, 800.0
    idx_lo = int(f_lo / (SAMPLE_RATE / n_fft))
    idx_hi = int(f_hi / (SAMPLE_RATE / n_fft))
    peak_idx = spec[idx_lo:idx_hi].argmax().item() + idx_lo
    peak_freq = freqs[peak_idx].item()

    assert peak_freq == pytest.approx(440.0, abs=6.0), f"peak at {peak_freq} Hz, expected ~440 Hz"


def test_karplus_damping_effect():
    """Higher damping should decay high-frequency energy faster."""
    def run_damping(damping_val):
        node = make_node()
        set_params(node, freq=440.0, damping=damping_val, brightness=1.0, decay=0.99)

        trig = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
        trig[:, 0] = 1.0
        silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)

        # Pluck
        process_block(node, trig)
        # Collect block 2 and block 15
        block2 = None
        block15 = None
        for i in range(16):
            out = process_block(node, silence)
            if i == 1:
                block2 = out[0].clone()
            if i == 14:
                block15 = out[0].clone()

        # High-frequency energy proxy: use diff (spectral high-frequency content)
        hf2 = float(torch.sum(block2.diff().abs()))
        hf15 = float(torch.sum(block15.diff().abs()))
        return hf2, hf15

    hf2_low, hf15_low = run_damping(0.05)
    hf2_high, hf15_high = run_damping(0.85)

    decay_ratio_low = hf15_low / (hf2_low + 1e-9)
    decay_ratio_high = hf15_high / (hf2_high + 1e-9)

    # High damping should decay HF at least 5x faster
    assert decay_ratio_high < decay_ratio_low / 5.0, (
        f"damping=0.85 decay ratio {decay_ratio_high:.4f} not 5x faster than "
        f"damping=0.05 ratio {decay_ratio_low:.4f}"
    )


def test_karplus_freq_in_disconnect_restores_staged():
    """Disconnecting freq_in should restore the staged freq parameter."""
    node = make_node()
    set_params(node, freq=220.0)

    # Record set_param calls to verify what the native DSP receives
    calls = []
    orig_set_param = node.lib.set_param
    node.lib.set_param = lambda h, i, v: (calls.append((i, v)), orig_set_param(h, i, v))[1]

    # Simulate freq_in connection with 880 Hz (fake both connectivity and data)
    freq_block = torch.full((1, BLOCK_SIZE), 880.0, dtype=DTYPE)
    node.inputs["freq_in"].get_tensor = lambda: freq_block
    node.inputs["freq_in"].connected_outputs = [object()]
    node._was_freq_mod_connected = True

    # Process with modulation active: native should receive 880 Hz
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    process_block(node, silence)
    assert calls and calls[-1][0] == node.PARAM_MAP["freq"] and calls[-1][1] == pytest.approx(880.0)

    # Disconnect freq_in (remove the fake connection): process() must detect
    # the disconnect and re-push the staged 220 Hz value to the native side.
    del node.inputs["freq_in"].get_tensor
    node.inputs["freq_in"].connected_outputs = []
    calls.clear()
    process_block(node, silence)
    freq_calls = [v for (i, v) in calls if i == node.PARAM_MAP["freq"]]
    assert freq_calls and freq_calls[-1] == pytest.approx(220.0), (
        f"disconnect did not restore staged freq: {freq_calls}"
    )
    assert not node._was_freq_mod_connected


def test_karplus_mono_trigger_stereo_shape():
    """Mono trigger input should produce stereo output with identical channels."""
    node = make_node()
    set_params(node, freq=220.0)

    mono_trig = torch.zeros(1, BLOCK_SIZE, dtype=DTYPE)
    mono_trig[0, 0] = 1.0

    out = process_block(node, mono_trig)
    assert out.shape == (2, BLOCK_SIZE), f"expected (2, {BLOCK_SIZE}), got {out.shape}"
    assert torch.allclose(out[0], out[1]), "mono trigger should produce identical stereo channels"


def test_karplus_start_resets_state():
    """start() should clear all DSP state, silence follows immediately."""
    node = make_node()
    set_params(node, freq=220.0, decay=0.999)

    # Excite the string
    trig = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    trig[:, 0] = 1.0
    out = process_block(node, trig)
    assert float(out.abs().max()) > 0.01, "string did not ring"

    # Reset via start()
    node.start()

    # Feed silence -> output should be immediately near zero
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, silence)
    assert float(out.abs().max()) < 1e-6, f"stale audio survived start(): {float(out.abs().max())}"


def test_karplus_save_load_roundtrip():
    """Verify dictionary round-trip preserves parameter values."""
    node = make_node()
    set_params(node, freq=330.0, damping=0.4, brightness=0.6, decay=0.95)

    d = node.to_dict()
    fresh = make_node()
    fresh.load_state(d)

    for k in ("freq", "damping", "brightness", "decay"):
        assert k in fresh.params
        assert float(fresh.params[k].value) == pytest.approx(float(node.params[k].value))


def test_karplus_no_net_allocation():
    """Steady-state processing should not allocate Python heap memory."""
    node = make_node()
    set_params(node, freq=220.0)

    trig = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    trig[:, 0] = 1.0
    process_block(node, trig)

    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    # Warm up
    for _ in range(5):
        process_block(node, silence)

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, silence)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"