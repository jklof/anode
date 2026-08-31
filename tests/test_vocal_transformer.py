import gc
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE

LATENCY = 4608                                 # native kLatency (fixed emission delay)
SETTLE_BLOCKS = (LATENCY // BLOCK_SIZE) + 2    # 11 blocks: fully flush the pipeline
NFFT = 16384                                   # zero-padded FFT for peak interpolation


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocalTransformer")
    assert cls is not None, "VocalTransformer not registered (library build missing?)"
    node = cls()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"
    return node


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return node.out.buffer.clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()
    node._sync_params_to_cpp()


def dominant_peak_hz(spec, nfft):
    """Quadratic peak interpolation over a zero-padded magnitude spectrum."""
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        d = 0.0
    return (k + d) * SAMPLE_RATE / nfft


def tone_blocks(freq, n_blocks, amp=0.4):
    """Continuous-phase sine split into successive (CHANNELS, BLOCK_SIZE) blocks."""
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = (amp * np.sin(2.0 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    t2 = np.tile(t, (CHANNELS, 1))
    return [torch.from_numpy(t2[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(n_blocks)]


def saw_blocks(freq, n_blocks, n_harm=30, amp=0.3):
    """Continuous-phase bandlimited sawtooth (voice-like harmonic stack) split
    into successive (CHANNELS, BLOCK_SIZE) blocks. Pitch-shift tests use this
    because a pure sine's spectral hull does not follow the shifted harmonic
    under formant-preserving envelope replacement."""
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = np.zeros_like(n)
    for h in range(1, n_harm + 1):
        if freq * h >= SAMPLE_RATE / 2:
            break
        t += (1.0 / h) * np.sin(2.0 * np.pi * freq * h * n / SAMPLE_RATE)
    t = (t / np.max(np.abs(t)) * amp).astype(np.float32)
    t2 = np.tile(t, (CHANNELS, 1))
    return [torch.from_numpy(t2[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(n_blocks)]


def block_peak_hz(block, nfft=NFFT):
    x = block[0].numpy().astype(np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    return dominant_peak_hz(spec, nfft)


def autocorr_pitch(x, fmin=60.0, fmax=900.0):
    """Envelope-robust pitch estimate via normalized autocorrelation.

    Formant-preserving pitch shifts relocate the dominant SPECTRAL peak to
    surviving formant regions (e.g. a 440 Hz saw shifted -12 st keeps its
    strongest partial at 440 Hz because the original hull peaks there), so
    dominant-peak tests are invalid for shifted outputs. Autocorrelation
    measures the harmonic structure instead."""
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(SAMPLE_RATE / fmax)
    hi = min(int(SAMPLE_RATE / fmin), len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SAMPLE_RATE / lag


def test_vocal_transformer_registration():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocalTransformer")
    assert cls is not None
    assert cls.category == "Effects"
    assert cls.label
    assert cls.description
    node = make_node()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"
    for port in ("in", "pitch_mod", "formant_mod"):
        assert node.inputs[port].help, f"{port} missing help"
    assert node.inputs["pitch_mod"].param_name == "pitch_shift"
    assert node.inputs["formant_mod"].param_name == "formant_shift"
    assert node.params["pitch_shift"].meta.get("min") == -24.0
    assert node.params["pitch_shift"].meta.get("max") == 24.0
    assert node.params["gender_morph"].meta.get("min") == -1.0
    assert node.get_telemetry()["latency_samples"] == 4608


def test_vocal_transformer_mix_zero_bit_exact():
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_vocal_transformer_zero_shift_spectrum():
    """mix=1, all shifts neutral: COLA-normalized identity mapping. Compare
    magnitude spectra (a phase vocoder cannot be waveform-accurate)."""
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=0.0, gender_morph=0.0,
               mix=1.0, sibilant_bypass=0.0, breathiness=0.0)
    blocks = tone_blocks(440.0, SETTLE_BLOCKS + 8)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)

    settled = blocks[SETTLE_BLOCKS + 4]
    out = process_block(node, settled)

    x = settled[0].numpy().astype(np.float64)
    y = out[0].numpy().astype(np.float64)
    win = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * win, NFFT))
    Y = np.abs(np.fft.rfft(y * win, NFFT))
    peak = dominant_peak_hz(Y, NFFT)
    assert abs(peak - 440.0) < 10.0, f"zero-shift peak at {peak:.2f} Hz"
    e_in = float(np.sum(X ** 2))
    e_out = float(np.sum(Y ** 2))
    ratio_db = 10.0 * np.log10(e_out / e_in + 1e-12)
    assert -3.0 < ratio_db < 3.0, f"zero-shift energy off by {ratio_db:.2f} dB"


def test_vocal_transformer_pitch_accuracy():
    node = make_node()
    set_params(node, pitch_shift=12.0, formant_shift=0.0, gender_morph=0.0,
               mix=1.0, sibilant_bypass=0.0, breathiness=0.0)
    blocks = saw_blocks(440.0, SETTLE_BLOCKS + 12)
    for b in blocks[:SETTLE_BLOCKS + 4]:
        process_block(node, b)
    out = process_block(node, blocks[-1])
    peak = block_peak_hz(out)
    assert abs(peak - 880.0) < 4.0, f"+12 st peak at {peak:.2f} Hz (want 880)"


def test_vocal_transformer_modulation_inputs():
    """pitch_mod drives pitch at block rate; disconnecting restores the staged
    parameter (disconnect re-sync contract). Feeds input CONTIGUOUSLY and burns
    the fixed emission latency (4608 samples) after each change before
    measuring — otherwise pre-change OLA content still in the ring (440 Hz)
    dominates the measured window. Pitch is measured by autocorrelation: a
    formant-preserving -12 st shift of a 440 Hz saw keeps its dominant partial
    at 440 Hz (the original hull peak weights the output's 2nd harmonic), so
    spectral-peak tests would be wrong."""
    node = make_node()
    set_params(node, pitch_shift=0.0, mix=1.0, sibilant_bypass=0.0, breathiness=0.0)
    blocks = saw_blocks(440.0, SETTLE_BLOCKS * 3 + 24)
    bi = 0

    # 1. Fill the pipeline at neutral pitch (440 Hz).
    for b in blocks[bi:bi + SETTLE_BLOCKS]:
        process_block(node, b)
    bi += SETTLE_BLOCKS

    # 2. Connect -12 st CV; burn the latency; measure a long window.
    cv = torch.full((CHANNELS, BLOCK_SIZE), -12.0, dtype=DTYPE)
    node.pitch_mod.connected_outputs = [object()]      # truthy gate
    node.pitch_mod.get_tensor = lambda: cv
    for b in blocks[bi:bi + SETTLE_BLOCKS + 2]:
        process_block(node, b)
    bi += SETTLE_BLOCKS + 2
    outs = [process_block(node, b) for b in blocks[bi:bi + 16]]
    bi += 16
    x = np.concatenate([o[0].numpy().astype(np.float64) for o in outs])
    p = autocorr_pitch(x)
    assert abs(p - 220.0) < 10.0, f"pitch_mod -12 st gave {p:.2f} Hz (want 220)"

    # 3. Disconnect: re-push restores the staged parameter (0 st -> 440 Hz).
    node.pitch_mod.connected_outputs = []
    for b in blocks[bi:bi + SETTLE_BLOCKS + 2]:
        process_block(node, b)
    bi += SETTLE_BLOCKS + 2
    outs2 = [process_block(node, b) for b in blocks[bi:bi + 16]]
    x2 = np.concatenate([o[0].numpy().astype(np.float64) for o in outs2])
    p2 = autocorr_pitch(x2)
    assert abs(p2 - 440.0) < 10.0, f"after disconnect pitch {p2:.2f} Hz (want 440)"


def test_vocal_transformer_extremes_stability():
    node = make_node()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.2
    for pitch in (-24.0, 0.0, 24.0):
        for formant in (-12.0, 0.0, 12.0):
            for gender in (-1.0, 1.0):
                set_params(node, pitch_shift=pitch, formant_shift=formant,
                           gender_morph=gender, breathiness=1.0,
                           sibilant_bypass=1.0, mix=1.0)
                out = process_block(node, blk)
                assert torch.isfinite(out).all(), \
                    f"non-finite at pitch={pitch} formant={formant} gender={gender}"
                assert float(out.abs().max()) < 8.0
    for br in (0.0, 1.0):
        for sb in (0.0, 1.0):
            set_params(node, pitch_shift=7.0, formant_shift=-5.0, gender_morph=0.5,
                       breathiness=br, sibilant_bypass=sb, mix=1.0)
            out = process_block(node, blk)
            assert torch.isfinite(out).all()


def test_vocal_transformer_mono_input_channel_adaptation():
    """Mono (1, 512) in -> strictly (2, 512) out with identical channels
    (anti-shrinkage guard through the node's process() override)."""
    node = make_node()
    set_params(node, mix=1.0, sibilant_bypass=0.0, breathiness=0.0)
    blocks = tone_blocks(440.0, SETTLE_BLOCKS + 2)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    mono = blocks[SETTLE_BLOCKS][0:1].clone()
    node.inp.get_tensor = lambda b=mono: b
    node.process()
    out = node.out.buffer
    assert out.shape == (CHANNELS, BLOCK_SIZE), f"output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) > 0.01
    assert torch.allclose(out[0], out[1], atol=1e-6)


def test_vocal_transformer_reset_on_start():
    node = make_node()
    set_params(node, mix=1.0, sibilant_bypass=0.0, breathiness=0.0)
    blocks = tone_blocks(440.0, SETTLE_BLOCKS + 2)
    for b in blocks:
        process_block(node, b)
    assert float(node.out.buffer.abs().max()) > 0.01

    node.start()   # -> _call_reset() -> native reset()
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = None
    for _ in range(6):   # flush the whole pipeline
        out = process_block(node, silence)
    assert float(out.abs().max()) < 1e-6, "reset must clear ring/OLA/phase memory"


def test_vocal_transformer_zero_steady_state_allocation():
    node = make_node()
    set_params(node, mix=1.0, sibilant_bypass=0.0, breathiness=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    process_block(node, blk)   # warm-up (first-call lazy init)

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, blk)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"
