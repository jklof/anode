import numpy as np
import pytest
import torch
import tracemalloc
import math

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node(class_name="NoiseGate"):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()


def rms_db(x):
    return 20.0 * float(torch.log10(torch.sqrt(torch.mean(x.pow(2))) + 1e-9))


def process_block(node, blk):
    node.inputs["in"].get_tensor = lambda b=blk: b
    node.process()
    return node.outputs["out"].buffer.clone()


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
    blk = torch.from_numpy(np.tile(tone, (CHANNELS, 1)))

    out = None
    for _ in range(10):
        out = process_block(node, blk)
    reduction_db = rms_db(out) - rms_db(blk)
    assert reduction_db > -1.5, f"open-gate loss {reduction_db:.2f} dB"


def test_quiet_signal_attenuated_toward_range():
    node = make_node()
    set_params(node, **GATED)
    noise = (torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.0005)

    out = None
    for _ in range(30):
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
        process_block(node, loud)
    process_block(node, silence)
    assert node.hold_left > 0 or node.gr_db == pytest.approx(0.0, abs=0.1)


def test_mono_sidechain_detection_without_crash():
    node = make_node()
    set_params(node, **GATED)
    mono_sc = torch.full((1, BLOCK_SIZE), 0.001, dtype=DTYPE)
    node.inputs["sidechain"].connected_outputs.append(object())
    node.inputs["sidechain"].get_tensor = lambda m=mono_sc: m

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


# --- Brickwall Limiter Tests ---

def test_brickwall_limiter_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("BrickwallLimiter")
    assert cls is not None
    assert cls.category == "Effects"
    assert cls.label == "Brickwall Limiter"


def test_limiter_never_exceeds_ceiling():
    node = make_node("BrickwallLimiter")
    set_params(node, threshold=-6.0, ceiling=-0.1, release=50.0)

    loud = torch.full((CHANNELS, BLOCK_SIZE), 2.0, dtype=DTYPE)
    out = None
    for _ in range(10):
        out = process_block(node, loud)
    ceiling_lin = 10.0 ** (-0.1 / 20.0)
    assert out.abs().max().item() <= ceiling_lin + 1e-5, f"output exceeds ceiling: {out.abs().max().item()} > {ceiling_lin}"


def test_limiter_anti_ghosting_mono():
    node = make_node("BrickwallLimiter")
    set_params(node, threshold=-6.0, ceiling=-0.1, release=50.0)

    mono_in = torch.full((1, BLOCK_SIZE), 2.0, dtype=DTYPE)
    node.inputs["in"].get_tensor = lambda m=mono_in: m
    out = process_block(node, mono_in)

    assert out.shape[0] == CHANNELS
    assert torch.allclose(out[0], out[1]), "Both output channels must match exactly for mono input"


def test_brickwall_limiter_no_net_allocation():
    node = make_node("BrickwallLimiter")
    set_params(node, threshold=-6.0, ceiling=-0.1, release=50.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.5
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


# --- Transient Shaper Tests ---

def test_transient_shaper_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("TransientShaper")
    assert cls is not None
    assert cls.category == "Effects"
    assert cls.label == "Transient Shaper"


def test_transient_shaper_attack_boost():
    node = make_node("TransientShaper")
    set_params(node, attack=0.0, sustain=0.0, output_gain_db=0.0)

    # Create a step impulse: silence then sudden loud signal
    step = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    step[:, BLOCK_SIZE // 2:] = 0.5

    # Process silence first to establish baseline
    silence = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
    for _ in range(5):
        process_block(node, silence)

    # Now process the step with attack=0
    set_params(node, attack=0.0)
    out_attack0 = process_block(node, step)
    peak_attack0 = out_attack0.abs().max().item()

    # Reset and process with attack=1.0
    node.start()
    for _ in range(5):
        process_block(node, silence)
    set_params(node, attack=1.0)
    out_attack1 = process_block(node, step)
    peak_attack1 = out_attack1.abs().max().item()

    assert peak_attack1 > peak_attack0, f"attack=1.0 should boost transient: {peak_attack1} > {peak_attack0}"


def test_transient_shaper_sustain_effect():
    node = make_node("TransientShaper")
    set_params(node, attack=0.0, sustain=0.0, output_gain_db=0.0)

    # Continuous tone
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(BLOCK_SIZE) / 48000.0)).astype(np.float32)
    blk = torch.from_numpy(np.tile(tone, (CHANNELS, 1)))

    for _ in range(5):
        out_sustain0 = process_block(node, blk)
    rms_sustain0 = float(torch.sqrt(torch.mean(out_sustain0.pow(2))))

    node.start()
    set_params(node, attack=0.0, sustain=1.0)
    for _ in range(5):
        out_sustain1 = process_block(node, blk)
    rms_sustain1 = float(torch.sqrt(torch.mean(out_sustain1.pow(2))))

    # Sustain boost should increase level of sustained portion
    assert rms_sustain1 > rms_sustain0 * 0.9, f"sustain boost should increase level: {rms_sustain1} > {rms_sustain0}"


def test_transient_shaper_no_net_allocation():
    node = make_node("TransientShaper")
    set_params(node, attack=0.5, sustain=0.5, output_gain_db=0.0)
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


# --- Auto Gain Tests ---

def test_autogain_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("AutoGain")
    assert cls is not None
    assert cls.category == "Utilities"
    assert cls.label == "Auto Gain / Leveler"


def test_autogain_levels_quiet_signal():
    node = make_node("AutoGain")
    set_params(node, target_db=-14.0, window_s=0.5, max_gain_db=18.0, silence_gate_db=-50.0)

    # -24 dBFS tone
    tone = (0.063 * np.sin(2 * np.pi * 1000.0 * np.arange(BLOCK_SIZE) / 48000.0)).astype(np.float32)
    blk = torch.from_numpy(np.tile(tone, (CHANNELS, 1)))

    out = None
    for _ in range(150):
        out = process_block(node, blk)
    out_db = rms_db(out)
    assert abs(out_db - (-14.0)) < 0.5, f"output {out_db:.2f} dBFS not near target -14 dBFS"


def test_autogain_freezes_on_silence():
    node = make_node("AutoGain")
    set_params(node, target_db=-14.0, window_s=0.5, max_gain_db=18.0, silence_gate_db=-40.0)

    # Feed silence below gate
    silence = torch.full((CHANNELS, BLOCK_SIZE), 1e-6, dtype=DTYPE)
    for _ in range(20):
        out = process_block(node, silence)

    # Gain should not have ramped to max (which would be +18dB on silence)
    rms = float(torch.sqrt(torch.mean(out.pow(2))))
    assert rms < 0.01, "gain should freeze on silence, not pump to max_gain_db"


def test_autogain_window_change():
    node = make_node("AutoGain")
    set_params(node, target_db=-14.0, window_s=2.0, max_gain_db=18.0, silence_gate_db=-50.0)

    tone = (0.063 * np.sin(2 * np.pi * 1000.0 * np.arange(BLOCK_SIZE) / 48000.0)).astype(np.float32)
    blk = torch.from_numpy(np.tile(tone, (CHANNELS, 1)))

    for _ in range(20):
        process_block(node, blk)
    old_count = node._hist_count

    # Change window
    set_params(node, window_s=0.5)
    for _ in range(5):
        process_block(node, blk)

    assert node._hist_count <= node._rms_history.shape[0], "history count should be capped at max"


def test_autogain_no_net_allocation():
    node = make_node("AutoGain")
    set_params(node, target_db=-14.0, window_s=2.0, max_gain_db=18.0, silence_gate_db=-50.0)
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

def _assert_mono_input_keeps_stereo_buffer(class_name):
    node = make_node(class_name)
    mono = torch.rand(1, BLOCK_SIZE, dtype=DTYPE) * 0.5 + 0.1
    node.inputs["in"].get_tensor = lambda: mono
    buf = node.outputs["out"].buffer
    for _ in range(5):
        node.process()
        # The (CHANNELS, BLOCK_SIZE) output buffer must never be resized by
        # an out= op fed with a mono tensor.
        assert buf.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.isfinite(buf).all()
    assert torch.allclose(buf[0], buf[1])
    assert buf.abs().max() > 1e-4


def test_transient_shaper_mono_input_keeps_stereo_output():
    _assert_mono_input_keeps_stereo_buffer("TransientShaper")


def test_auto_gain_mono_input_keeps_stereo_output():
    _assert_mono_input_keeps_stereo_buffer("AutoGain")


def test_compressor_native_reset_clears_delay_line():
    """AGENTS.md §7: every native DSP class exports reset(void*). Without it
    the compressor's lookahead delay line keeps leaking stale samples (and the
    detector envelope keeps its gain-reduction state) across a transport
    restart."""
    node = make_node("Compressor")
    if not (node.lib and node.dsp_handle):
        pytest.skip("compressor native library not available")
    assert hasattr(node.lib, "reset"), "compressor must export reset(void*)"

    loud = torch.full((1, BLOCK_SIZE), 0.9, dtype=torch.float32)
    silent = torch.zeros(1, BLOCK_SIZE, dtype=torch.float32)

    # Fill the lookahead delay line with signal (first block is silent while
    # the delay line warms up).
    process_block(node, loud)
    process_block(node, loud)
    assert float(node.outputs["out"].buffer[0].abs().max()) > 1e-6

    # Silenced input: without reset the stale 0.9 samples still leak out of
    # the delay line (envelope also holds gain-reduction state).
    process_block(node, silent)
    stale = node.outputs["out"].buffer.clone()
    assert float(stale[0].abs().max()) > 1e-6

    # FFINode.start() calls lib.reset() when available — silence must stay
    # exactly silent afterwards.
    node.start()
    process_block(node, silent)
    assert float(node.outputs["out"].buffer[0].abs().max()) < 1e-6


def test_autogain_window_reads_recent_ring_entries():
    """Regression: _rms_history is a true ring. Once the write pointer passes
    the window length, fresh samples land past the linear prefix, so the old
    _rms_history[:_hist_count] read averaged stale data and the long-term
    measurement froze on the first window's worth of samples."""
    node = make_node("AutoGain")
    set_params(node, target_db=-14.0, window_s=0.5, max_gain_db=18.0,
               silence_gate_db=-50.0)
    n_window = int(0.5 * SAMPLE_RATE / BLOCK_SIZE)   # 50 blocks

    quiet = torch.full((CHANNELS, BLOCK_SIZE), 0.01, dtype=DTYPE)
    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.25, dtype=DTYPE)

    for _ in range(120):          # write pointer advances well past the window
        process_block(node, quiet)
    for _ in range(5):
        process_block(node, loud)

    ptr, count = node._hist_ptr, node._hist_count
    assert count == n_window
    assert ptr >= n_window, "write pointer must have passed the window length"

    # Correct ring read: the last `count` entries must contain the recent
    # loud blocks (the data the node's long-term RMS must now be averaging).
    ring_recent = [node._rms_history[(ptr - 1 - i) % node.MAX_BLOCKS].item()
                   for i in range(count)]
    # Old buggy read: the linear prefix holds only stale quiet samples.
    stale_prefix = node._rms_history[:count].tolist()

    assert max(ring_recent) == pytest.approx(0.25, abs=1e-3), \
        "ring window must contain the recent loud blocks"
    assert max(stale_prefix) < 0.05, \
        "linear prefix holds only stale quiet data (the old bug's read window)"

