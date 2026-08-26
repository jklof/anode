import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE


def make_node(class_name="WaveformOscillator"):
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get(class_name)
    assert cls is not None, f"{class_name} not registered"
    return cls()


def run(node, blocks=8):
    for _ in range(blocks):
        node.process()
    return node.out_sig.buffer[0].clone()


def zero_crossings(x):
    return int(torch.count_nonzero((x[:-1] <= 0) & (x[1:] > 0)))


def set_wave(node, idx):
    node.params["waveform"].set(idx)
    node.sync()


def _spectral_slope(node, blocks=128, warmup=20):
    """Collect `blocks` blocks and compute spectral slope in dB/octave."""
    node.start()
    for _ in range(warmup):
        node.process()
    chunks = []
    for _ in range(blocks):
        node.process()
        chunks.append(node.outputs["out"].buffer[0].clone())
    x = torch.cat(chunks)
    spec = torch.fft.rfft(x).abs()
    freqs = torch.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)
    log_f = torch.log2(freqs[1:])
    log_s = torch.log2(spec[1:] + 1e-12)
    slope = torch.sum((log_f - log_f.mean()) * (log_s - log_s.mean())) / torch.sum((log_f - log_f.mean()) ** 2)
    return float(slope * 6.0206)  # Convert to dB/octave


def test_oscillator_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("WaveformOscillator")
    assert cls is not None
    assert cls.category == "Sources"
    assert cls.label == "Waveform Oscillator"


def test_sine_peak_and_frequency():
    node = make_node()
    node.params["freq"].set(1000.0)
    node.params["amp"].set(0.8)
    node.sync()
    out = run(node)

    assert out.abs().max() == pytest.approx(0.8, abs=0.02)
    expected = 1000.0 * BLOCK_SIZE / SAMPLE_RATE
    assert zero_crossings(out) == pytest.approx(expected, abs=2)


def test_triangle_peak():
    node = make_node()
    set_wave(node, 1)
    node.params["freq"].set(500.0)
    node.params["amp"].set(0.7)
    node.sync()
    out = run(node)
    assert out.abs().max() == pytest.approx(0.7, abs=0.02)


def test_sawtooth_polyblep_smooths_wrap_jump():
    node = make_node()
    set_wave(node, 2)
    node.params["freq"].set(500.0)
    node.params["amp"].set(1.0)
    node.sync()
    out = run(node)

    max_delta = float(torch.diff(out).abs().max())
    assert max_delta < 1.3, f"wrap jump not smoothed: {max_delta}"
    assert out.abs().max() == pytest.approx(1.0, abs=0.05)


def _harmonic_specter(node, freq=500.0, periods=64):
    samples_per_period = SAMPLE_RATE / freq
    blocks = int(round(periods * samples_per_period / BLOCK_SIZE))
    chunks = []
    for _ in range(blocks):
        node.process()
        chunks.append(node.out_sig.buffer[0].clone())
    x = torch.cat(chunks)

    spec = torch.fft.rfft(x).abs()
    fund_bin = int(round(periods))
    harmonics = set()
    k = 1
    while k * fund_bin < len(spec) // 2:
        for h in (k * fund_bin - 1, k * fund_bin, k * fund_bin + 1):
            harmonics.add(h)
        k += 1
    fundamental = float(spec[fund_bin])
    non_harm = [float(spec[i]) for i in range(2, len(spec) // 2) if i not in harmonics]
    return 20.0 * np.log10(max(non_harm) / fundamental + 1e-12)


def test_sawtooth_polyblep_suppresses_aliasing():
    node = make_node()
    set_wave(node, 2)
    node.params["freq"].set(500.0)
    node.params["amp"].set(1.0)
    node.sync()

    inharmonic_db = _harmonic_specter(node)
    assert inharmonic_db < -35.0, f"inharmonic content {inharmonic_db:.1f} dB too high"


def test_square_pulse_width_duty_cycle():
    node = make_node()
    set_wave(node, 3)
    node.params["freq"].set(200.0)
    node.params["amp"].set(1.0)
    node.params["pulse_width"].set(0.25)
    node.sync()
    out = run(node)

    assert float(out.mean()) == pytest.approx(2.0 * 0.25 - 1.0, abs=0.05)
    assert out.abs().max() == pytest.approx(1.0, abs=0.05)


def test_freq_modulation_via_bound_input():
    node = make_node()
    node.params["freq"].set(440.0)
    node.sync()

    hi = torch.full((CHANNELS, BLOCK_SIZE), 4000.0, dtype=DTYPE)
    node.in_freq.get_tensor = lambda: hi
    out = run(node)
    expected = 4000.0 * BLOCK_SIZE / SAMPLE_RATE
    assert zero_crossings(out) == pytest.approx(expected, abs=3)


def test_start_resets_phase():
    node = make_node()
    node.params["freq"].set(1000.0)
    node.sync()
    first = run(node, 3)
    again = run(node, 1)
    assert not torch.allclose(first[-1], again)

    node.start()
    after_reset = run(node, 1)
    ref = make_node()
    ref.params["freq"].set(1000.0)
    ref.sync()
    reference = run(ref, 1)
    assert torch.allclose(after_reset, reference)


def test_oscillator_no_net_allocation():
    node = make_node()
    set_wave(node, 3)
    run(node, 5)

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    run(node, 50)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"


def test_colored_noise_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("ColoredNoise")
    assert cls is not None
    assert cls.category == "Sources"
    assert cls.label == "Colored Noise Generator"


def test_colored_noise_white_peak():
    node = make_node("ColoredNoise")
    node.params["type"].set(0)
    node.params["amp"].set(1.0)
    node.sync()
    out = run(node)
    assert out.abs().max() <= 1.0


def test_colored_noise_spectral_slopes():
    node = make_node("ColoredNoise")
    node.params["amp"].set(1.0)
    node.sync()

    node.params["type"].set(1)
    node.sync()
    slope_pink = _spectral_slope(node)
    assert slope_pink < -2.0, f"Pink slope {slope_pink:.2f} dB/oct not negative enough"

    node.params["type"].set(3)
    node.sync()
    slope_blue = _spectral_slope(node)
    assert slope_blue > 2.0, f"Blue slope {slope_blue:.2f} dB/oct not positive enough"


def test_colored_noise_no_net_allocation():
    node = make_node("ColoredNoise")
    node.params["type"].set(1)
    node.params["amp"].set(0.5)
    node.sync()
    run(node, 5)

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    run(node, 50)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"