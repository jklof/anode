import math
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, SAMPLE_RATE, DTYPE


# ==============================================================================
# Helpers
# ==============================================================================


def make_node(class_name):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def feed_tone(node, freq_hz, amplitude=0.5, channels=1, settle=6, measure=16):
    """Stream a phase-continuous sine through node.process(); returns output RMS.

    Note: an unconnected InputSlot.get_tensor() zeroes its scratch buffer on
    every call, so we mock get_tensor with the tone block (existing test
    convention, see test_gain)."""
    n = np.arange(BLOCK_SIZE)
    phase_inc = 2.0 * np.pi * freq_hz / SAMPLE_RATE
    phase = 0.0
    total_sq = 0.0
    count = 0
    slot = node.inputs["in"]
    for block_idx in range(settle + measure):
        tone = amplitude * np.sin(phase + phase_inc * n)
        block = torch.from_numpy(np.tile(tone.astype(np.float32), (channels, 1)))
        slot.get_tensor = lambda b=block: b
        node.process()
        if block_idx >= settle:
            out = node.out.buffer[0]
            total_sq += float(torch.sum(out * out))
            count += out.shape[0]
        phase = (phase + phase_inc * BLOCK_SIZE) % (2.0 * np.pi)
    rms_out = math.sqrt(total_sq / count)
    return rms_out


def stream_blocks(node, tensor, blocks):
    """Feed the same stereo tensor repeatedly via mocked get_tensor."""
    slot = node.inputs["in"]
    for _ in range(blocks):
        slot.get_tensor = lambda t=tensor: t
        node.process()


def gain_db(rms_out, amplitude):
    rms_in = amplitude / math.sqrt(2.0)
    return 20.0 * math.log10(max(rms_out, 1e-12) / rms_in)


def set_params(node, **kwargs):
    for name, value in kwargs.items():
        node.params[name].set(value)
        node.params[name].sync()


# ==============================================================================
# BiquadFilter — frequency response
# ==============================================================================


def test_biquad_low_pass_response():
    bq = make_node("BiquadFilter")
    set_params(bq, type=0, cutoff=1000.0, q=0.707)

    passed = gain_db(feed_tone(bq, 100.0), 0.5)
    assert passed > -1.0, f"100 Hz passband attenuation {passed:.2f} dB"
    assert passed < 1.0, f"passband must stay near unity gain (+{passed:.2f} dB)"

    attenuated = gain_db(feed_tone(bq, 15000.0), 0.5)
    assert attenuated < -20.0, f"15 kHz stopband only {attenuated:.2f} dB"


def test_biquad_high_pass_response():
    bq = make_node("BiquadFilter")
    set_params(bq, type=1, cutoff=1000.0, q=0.707)

    attenuated = gain_db(feed_tone(bq, 100.0), 0.5)
    assert attenuated < -20.0, f"100 Hz stopband only {attenuated:.2f} dB"

    passed = gain_db(feed_tone(bq, 15000.0), 0.5)
    assert passed > -1.0, f"15 kHz passband attenuation {passed:.2f} dB"


def test_biquad_notch_attenuates_center():
    bq = make_node("BiquadFilter")
    set_params(bq, type=3, cutoff=1000.0, q=4.0)

    center = gain_db(feed_tone(bq, 1000.0), 0.5)
    assert center < -20.0
    away = gain_db(feed_tone(bq, 8000.0), 0.5)
    assert away > -1.0


def test_biquad_peaking_boost_accuracy():
    bq = make_node("BiquadFilter")
    set_params(bq, type=4, cutoff=1000.0, q=0.707, gain_db=6.0)

    boosted = gain_db(feed_tone(bq, 1000.0), 0.5)
    assert abs(boosted - 6.0) <= 0.5, f"peaking boost {boosted:.2f} dB != 6 +/- 0.5"

    set_params(bq, gain_db=-6.0)
    cut = gain_db(feed_tone(bq, 1000.0), 0.5)
    assert abs(cut + 6.0) <= 0.5, f"peaking cut {cut:.2f} dB != -6 +/- 0.5"


def test_biquad_shelf_gain():
    bq = make_node("BiquadFilter")
    set_params(bq, type=5, cutoff=500.0, gain_db=6.0)  # low shelf: below corner boosted

    low = gain_db(feed_tone(bq, 80.0), 0.5)
    assert abs(low - 6.0) <= 0.75, f"low shelf asymptote {low:.2f} dB != 6"
    high = gain_db(feed_tone(bq, 8000.0), 0.5)
    assert abs(high) <= 1.0, f"low shelf must leave highs untouched ({high:.2f} dB)"


# ==============================================================================
# Channel handling / anti-ghosting / state reset
# ==============================================================================


def test_biquad_mono_input_duplicated_to_second_channel():
    bq = make_node("BiquadFilter")
    set_params(bq, type=0, cutoff=1000.0)

    rms = feed_tone(bq, 200.0, channels=1)
    assert rms > 1e-3, "channel 0 should carry filtered audio"

    # Channel adaptation: mono input is duplicated to both output channels,
    # not muted on the right channel (see ffi_base.FFINode.process()).
    ch1 = bq.out.buffer[1]
    assert torch.allclose(bq.out.buffer[0], ch1), "channel 1 not duplicated on mono input"
    assert ch1.abs().max() > 1e-3
    assert torch.isfinite(bq.out.buffer).all()


def test_biquad_stereo_channels_independent():
    bq = make_node("BiquadFilter")
    set_params(bq, type=0, cutoff=500.0, q=2.0)

    n = np.arange(BLOCK_SIZE)
    l = 0.5 * np.sin(2 * np.pi * 100.0 * n / SAMPLE_RATE)
    r = 0.5 * np.sin(2 * np.pi * 12000.0 * n / SAMPLE_RATE)
    stereo = torch.from_numpy(np.vstack([l, r]).astype(np.float32))

    stream_blocks(bq, stereo, 10)

    assert float(bq.out.buffer[0].abs().mean()) > 0.05, "100 Hz must pass LP@500"
    assert float(bq.out.buffer[1].abs().mean()) < 0.02, "12 kHz must be attenuated"


def test_biquad_state_reset_on_start():
    bq = make_node("BiquadFilter")
    set_params(bq, type=0, cutoff=1000.0, q=6.0)  # resonant -> long ring

    noise = torch.randn(2, BLOCK_SIZE, dtype=DTYPE)
    stream_blocks(bq, noise, 5)
    loud_tail = float(bq.out.buffer.abs().max())
    assert loud_tail > 0.01

    bq.start()  # transport restart clears filter state

    silence = torch.zeros(2, BLOCK_SIZE, dtype=DTYPE)
    stream_blocks(bq, silence, 2)
    residual = float(bq.out.buffer.abs().max())
    assert residual < 1e-4, f"state not cleared on start(): residual {residual}"


# ==============================================================================
# LinearPhaseEQ — frequency response
# ==============================================================================


def test_fir_low_pass_response():
    eq = make_node("LinearPhaseEQ")
    set_params(eq, type=0, cutoff=1000.0)

    passed = gain_db(feed_tone(eq, 100.0, settle=8), 0.5)
    assert passed > -1.5, f"100 Hz passband {passed:.2f} dB"
    assert passed < 1.5, f"passband must stay near unity gain (+{passed:.2f} dB)"

    attenuated = gain_db(feed_tone(eq, 15000.0, settle=8), 0.5)
    assert attenuated < -20.0, f"15 kHz stopband only {attenuated:.2f} dB"


def test_fir_high_pass_response():
    eq = make_node("LinearPhaseEQ")
    set_params(eq, type=1, cutoff=1000.0)

    attenuated = gain_db(feed_tone(eq, 60.0, settle=8), 0.5)
    assert attenuated < -20.0

    passed = gain_db(feed_tone(eq, 12000.0, settle=8), 0.5)
    assert passed > -1.5


def test_fir_band_pass_response():
    eq = make_node("LinearPhaseEQ")
    set_params(eq, type=2, cutoff=2000.0, q=1.0)  # band ~1000..3000 Hz

    center = gain_db(feed_tone(eq, 2000.0, settle=8), 0.5)
    assert center > -1.5, f"band center {center:.2f} dB"

    low = gain_db(feed_tone(eq, 300.0, settle=8), 0.5)
    high = gain_db(feed_tone(eq, 6000.0, settle=8), 0.5)
    assert low < -20.0 and high < -20.0


def test_fir_notch_response():
    eq = make_node("LinearPhaseEQ")
    set_params(eq, type=3, cutoff=2000.0, q=1.0)

    center = gain_db(feed_tone(eq, 2000.0, settle=8), 0.5)
    assert center < -30.0, f"notch depth only {center:.2f} dB"

    away = gain_db(feed_tone(eq, 6000.0, settle=8), 0.5)
    assert away > -1.0


def test_fir_mono_input_and_latency_telemetry():
    eq = make_node("LinearPhaseEQ")

    rms = feed_tone(eq, 200.0, channels=1, settle=8)
    assert rms > 1e-3
    # Channel adaptation: a mono input is DUPLICATED to both output channels,
    # not muted on the right channel (see ffi_base.process()).
    assert torch.isfinite(eq.out.buffer).all()
    assert torch.allclose(eq.out.buffer[0], eq.out.buffer[1])
    assert eq.out.buffer[1].abs().max() > 1e-3

    tel = eq.get_telemetry()
    assert tel["latency_samples"] == (eq.NUM_TAPS - 1) // 2


def test_fir_channel_change_clears_history():
    eq = make_node("LinearPhaseEQ")
    set_params(eq, type=0, cutoff=500.0)

    # Stereo run leaves history full of signal
    noise = torch.randn(2, BLOCK_SIZE, dtype=DTYPE) * 0.5
    stream_blocks(eq, noise, 5)

    # Switch to mono: stale history must be flushed, silence stays silent
    silence_mono = torch.zeros(1, BLOCK_SIZE, dtype=DTYPE)
    stream_blocks(eq, silence_mono, 2)  # channel change detected on first block
    assert float(eq.out.buffer[0].abs().max()) < 1e-3


# ==============================================================================
# Real-time memory behaviour
# ==============================================================================


def _net_growth(node, tensor, blocks=50):
    stream_blocks(node, tensor, 1)  # warm up lazy paths / caches
    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    stream_blocks(node, tensor, blocks)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return growth


def test_biquad_no_net_allocation_over_blocks():
    bq = make_node("BiquadFilter")
    set_params(bq, type=0, cutoff=1000.0)
    scratch = torch.randn(2, BLOCK_SIZE, dtype=DTYPE) * 0.3

    growth = _net_growth(bq, scratch)
    assert growth < 64 * 1024, f"biquad net allocation {growth} bytes over 50 blocks"


def test_fir_no_net_allocation_over_blocks():
    eq = make_node("LinearPhaseEQ")
    set_params(eq, type=0, cutoff=1000.0)
    scratch = torch.randn(2, BLOCK_SIZE, dtype=DTYPE) * 0.3

    growth = _net_growth(eq, scratch)
    # conv1d transients are freed each block; allow small allocator slack.
    assert growth < 128 * 1024, f"FIR net allocation {growth} bytes over 50 blocks"


# ==============================================================================
# Save/load compatibility with the generic param system
# ==============================================================================


def test_filters_serialize_params():
    for cls_name in ("BiquadFilter", "LinearPhaseEQ"):
        node = make_node(cls_name)
        d = node.to_dict()
        fresh = make_node(cls_name)
        fresh.load_state(d)
        for k in ("type", "cutoff", "q"):
            assert k in fresh.params


def test_biquad_mono_input_duplicated_to_stereo():
    """Regression: FFINode used to process min(in, out) channels and zero the
    rest, muting the right channel for mono inputs. Mono inputs are now
    duplicated to both output channels."""
    node = make_node("BiquadFilter")
    node.params["cutoff"].set(1000.0)
    node.params["q"].set(0.7)
    node.sync()
    node._sync_params_to_cpp()

    n = np.arange(BLOCK_SIZE)
    tone = (0.5 * np.sin(2.0 * np.pi * 200.0 * n / SAMPLE_RATE)).astype(np.float32)
    mono = torch.from_numpy(np.tile(tone, (1, 1)))
    stream_blocks(node, mono, 8)

    assert node.out.buffer.shape[0] == 2
    assert torch.isfinite(node.out.buffer).all()
    assert torch.allclose(node.out.buffer[0], node.out.buffer[1])
    assert node.out.buffer[1].abs().max() > 1e-3


def test_biquad_shelf_boost_and_cut_accuracy():
    """Stronger shelf settings than test_biquad_shelf_gain: both +12 dB and
    -12 dB must realize the RBJ asymptote below the corner and unity above
    it, for either polarity. A sign error in the shelf denominator
    coefficients (a2/a0) would skew the poles and the curve."""
    bq = make_node("BiquadFilter")
    for gain in (12.0, -12.0):
        set_params(bq, type=5, cutoff=500.0, q=0.707, gain_db=gain)

        low = gain_db(feed_tone(bq, 100.0), 0.5)
        assert abs(low - gain) <= 0.75, \
            f"{gain} dB shelf at 100 Hz measured {low:.2f} dB"

        high = gain_db(feed_tone(bq, 8000.0), 0.5)
        assert abs(high) <= 1.0, \
            f"{gain} dB shelf at 8 kHz measured {high:.2f} dB (expected ~0 dB)"

