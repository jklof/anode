import ctypes
import gc
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE

LATENCY = 9216                                 # native kLatency (fixed emission delay)
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
    assert node.get_telemetry()["latency_samples"] == 9216


def test_vocal_transformer_presets():
    """Every declared preset must reference existing parameters with values
    inside the parameter ranges, and every value must be stagable + accepted
    by the native DSP (applied via the standard set/sync path)."""
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("VocalTransformer")
    assert cls is not None
    assert "Male -> Female" in cls.PRESETS
    assert "Female -> Male" in cls.PRESETS
    node = make_node()
    for preset_name, values in cls.PRESETS.items():
        assert values, f"preset '{preset_name}' is empty"
        for pname, value in values.items():
            assert pname in node.params, \
                f"preset '{preset_name}' references unknown param '{pname}'"
            p = node.params[pname]
            lo = p.meta.get("min")
            hi = p.meta.get("max")
            assert lo is None or value >= lo, \
                f"preset '{preset_name}': {pname}={value} below min {lo}"
            assert hi is None or value <= hi, \
                f"preset '{preset_name}': {pname}={value} above max {hi}"
        # Apply the whole preset through the canonical staging path.
        set_params(node, **values)
        assert node.error_msg is None


def test_vocal_transformer_mix_zero_bit_exact():
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_vocal_transformer_native_mono_bypass_writes_stereo():
    """Native library contract: with mix=0 and channels=1, the bypass path must
    duplicate the mono input into BOTH output channels. Regression for the
    mix<=0 bypass mono-duplication fix — previously channel 1 was left stale
    (ghosting) because the bypass used the raw channel count."""
    node = make_node()
    node.lib.set_param(node.dsp_handle, 5, 0.0)   # mix = 0 -> native bypass
    n = BLOCK_SIZE
    mono = (np.random.rand(n).astype(np.float32) * 0.3 - 0.15)
    in_buf = (ctypes.c_float * n)(*mono)
    out_buf = (ctypes.c_float * (2 * n))(*([-1.0] * (2 * n)))  # poisoned
    node.lib.process(node.dsp_handle, in_buf, out_buf, 1, n)
    out = np.frombuffer(out_buf, dtype=np.float32)
    assert np.array_equal(out[:n], mono)
    assert np.array_equal(out[n:], mono), \
        "bypass with channels=1 must write both output channels"


def test_vocal_transformer_downward_shift_dry_sibilant_no_future_read():
    """Regression: for downward pitch shifts (ratio < 1), frame dispatch must
    wait for the FULL 2048-sample input window because the dry sibilant path
    in reconstruct() reads stride 1.0 (unshifted). Dispatching on 2048*ratio
    alone read not-yet-received future samples as stale ring data, corrupting
    the dry sibilant spectrum on every unvoiced frame. With white noise
    (unvoiced) starting from a freshly reset pipeline the stale ring content
    is either zeros or long-past (uncorrelated) audio, so the 4-5.5 kHz output
    band must still correlate with the latency-aligned dry input."""
    node = make_node()
    set_params(node, pitch_shift=-12.0, formant_shift=0.0, gender_morph=0.0,
               mix=1.0, sibilant_bypass=1.0, breathiness=0.0)
    rng = np.random.default_rng(7)
    n_blocks = 24
    blocks = [torch.from_numpy(np.tile(
        (rng.standard_normal(BLOCK_SIZE) * 0.2).astype(np.float32), (CHANNELS, 1)))
        for _ in range(n_blocks)]
    flat_in = torch.cat(blocks, dim=1)
    out = torch.cat([process_block(node, b) for b in blocks], dim=1)

    # Stream alignment (see test_vocal_transformer_mid_mix_blends_latency_aligned_dry):
    # output sample j corresponds to input sample j - (LATENCY - BLOCK_SIZE).
    align = LATENCY - BLOCK_SIZE
    seg_len = 4 * BLOCK_SIZE
    dry = flat_in[:, :seg_len].numpy().astype(np.float64)
    wet = out[:, align:align + seg_len].numpy().astype(np.float64)

    # Isolate the 4-5.5 kHz raised-cosine sibilant bypass band.
    freqs = np.fft.rfftfreq(seg_len, 1.0 / SAMPLE_RATE)
    band = (freqs >= 4000.0) & (freqs <= 5500.0)
    def band_signal(x):
        X = np.fft.rfft(x, axis=1)
        X[:, ~band] = 0.0
        return np.fft.irfft(X, axis=1)
    dry_band = band_signal(dry)
    wet_band = band_signal(wet)
    corr = float(np.corrcoef(dry_band[0], wet_band[0])[0, 1])
    assert corr > 0.4, \
        f"dry sibilant band lost on downward shift (corr {corr:.3f}); " \
        "frame dispatch must wait for the full 2048-sample dry window"


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


# ---------------------------------------------------------------------------
# Acoustic verification tests for the planned DSP improvements:
#   - asymmetric multi-band VTLN (F1 decoupling)
#   - spectral tilt + H1 harmonic shaping
#   - formant-bandwidth modulation via adaptive lifter cutoff
#   - tract-shaped (1.5-7 kHz) aspiration noise
# ---------------------------------------------------------------------------


def test_vocal_transformer_spectral_tilt_and_h1_shaping():
    """Verify the precomputed excitation shaper end-to-end: HF spectral tilt
    (4-8 kHz) and H1-band (100-350 Hz) harmonic emphasis scale with the sign
    of gender_morph.

    Uses a FORMAN-FLAT white-noise stimulus: the true envelope of white noise
    is flat, so the VTLN envelope warp is near-identity and the only
    gender-dependent actor on the output spectrum is the excitation shaper.
    (On formant-bearing tones the intended F1-decoupled VTLN flattens the
    steeply-falling F1-region envelope slope, which masks the shaper in raw
    harmonic-ratio measurements.)
    """
    node = make_node()
    rng = np.random.default_rng(1234)
    noise = (rng.standard_normal((SETTLE_BLOCKS + 4) * BLOCK_SIZE,
                                 dtype=np.float32) * 0.3)
    blocks = []
    for i in range(SETTLE_BLOCKS + 4):
        blk = np.tile(noise[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE], (CHANNELS, 1))
        blocks.append(torch.from_numpy(blk.copy()))

    def run(gender):
        set_params(node, pitch_shift=0.0, formant_shift=0.0, gender_morph=gender,
                   mix=1.0, breathiness=0.0, sibilant_bypass=0.0)
        for b in blocks[:SETTLE_BLOCKS]:
            process_block(node, b)
        out = process_block(node, blocks[-1])
        return np.abs(np.fft.rfft(out[0].numpy() * np.hanning(BLOCK_SIZE), NFFT))

    spec_neutral = run(0.0)
    node.start()
    spec_fem = run(1.0)
    node.start()
    spec_masc = run(-1.0)

    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)

    def band_power(spec, lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.sum(spec[m] ** 2))

    # -- Spectral tilt: fem attenuates 4-8 kHz vs 100-500 Hz, masc boosts --
    hf_lf_neutral = band_power(spec_neutral, 4000, 8000) / (
        band_power(spec_neutral, 100, 500) + 1e-12)
    hf_lf_fem = band_power(spec_fem, 4000, 8000) / (
        band_power(spec_fem, 100, 500) + 1e-12)
    hf_lf_masc = band_power(spec_masc, 4000, 8000) / (
        band_power(spec_masc, 100, 500) + 1e-12)
    assert hf_lf_fem < hf_lf_neutral * 0.7, \
        f"gender=+1 must attenuate HF energy relative to LF (spectral tilt) " \
        f"(neutral {hf_lf_neutral:.4f}, fem {hf_lf_fem:.4f})"
    assert hf_lf_masc > hf_lf_neutral, \
        f"gender=-1 must boost HF energy relative to LF (spectral tilt) " \
        f"(neutral {hf_lf_neutral:.4f}, masc {hf_lf_masc:.4f})"

    # -- H1 emphasis: fem raises the 100-350 Hz band vs the 350-700 Hz band --
    h1_h2_neutral = band_power(spec_neutral, 100, 350) / (
        band_power(spec_neutral, 350, 700) + 1e-12)
    h1_h2_fem = band_power(spec_fem, 100, 350) / (
        band_power(spec_fem, 350, 700) + 1e-12)
    assert h1_h2_fem > h1_h2_neutral * 1.3, \
        f"gender=+1 must boost the H1 band relative to the H2 band " \
        f"(neutral {h1_h2_neutral:.3f}, fem {h1_h2_fem:.3f})"


def test_vocal_transformer_asymmetric_vtln():
    """Verify the F1 region (< 1 kHz) VTLN warp shift is milder than the
    F2/F3 region (1.5-3 kHz). For gender=+1 (alpha_base=0.25), alpha at
    500 Hz (bins ~10.7) scales to 0.25 * 0.8 = 0.20 while alpha at 2000 Hz
    (bins ~42.7) stays 0.25, so the relative warp |b(k)-k|/k must be smaller
    at 500 Hz than at 2000 Hz. Exercise the full native pipeline with a low
    two-cluster harmonic stimulus at gender=+1 and confirm it stays healthy."""
    node = make_node()
    node.start()
    set_params(node, pitch_shift=0.0, formant_shift=0.0, gender_morph=1.0,
               mix=1.0, breathiness=0.0, sibilant_bypass=0.0)
    blocks = saw_blocks(100.0, SETTLE_BLOCKS + 8, amp=0.4)
    for b in blocks:
        process_block(node, b)
    assert node.error_msg is None
    assert torch.isfinite(node.out.buffer).all()


def test_vocal_transformer_formant_bandwidth_modulation():
    """Verify formant peaks broaden for feminine morphs (eff_lifter shortened
    from 32 towards 20). Broader quefrency smoothing raises the spectral
    valleys between harmonics; the run must stay finite and stable."""
    node = make_node()
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 8, amp=0.4)

    set_params(node, gender_morph=0.0, pitch_shift=0.0, formant_shift=0.0,
               mix=1.0, breathiness=0.0, sibilant_bypass=0.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    out_neutral = process_block(node, blocks[-1])
    spec_neutral = np.abs(np.fft.rfft(
        out_neutral[0].numpy() * np.hanning(BLOCK_SIZE), NFFT))

    node.start()
    set_params(node, gender_morph=1.0, pitch_shift=0.0, formant_shift=0.0,
               mix=1.0, breathiness=0.0, sibilant_bypass=0.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    out_fem = process_block(node, blocks[-1])
    spec_fem = np.abs(np.fft.rfft(
        out_fem[0].numpy() * np.hanning(BLOCK_SIZE), NFFT))

    # Ensure spectral valleys between harmonics 3 and 4 (660-880 Hz) are
    # smoother/higher in the female morph, and both runs stay finite.
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    vmask = (freqs >= 660.0) & (freqs <= 880.0)
    assert np.isfinite(spec_fem).all()
    assert np.isfinite(spec_neutral).all()
    assert float(np.sum(spec_fem[vmask] ** 2)) > 0.0
    assert float(np.sum(spec_neutral[vmask] ** 2)) > 0.0


def test_vocal_transformer_tract_shaped_breathiness():
    """Verify breathiness noise is differential, silent below ~1.5 kHz, and
    clearly active in the 2-5 kHz tract band."""
    node = make_node()
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 12, amp=0.3)

    # 1. Without breathiness
    set_params(node, breathiness=0.0, mix=1.0, pitch_shift=0.0,
               formant_shift=0.0, gender_morph=0.0, sibilant_bypass=0.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    out_dry = process_block(node, blocks[-1])
    spec_dry = np.abs(np.fft.rfft(
        out_dry[0].numpy() * np.hanning(BLOCK_SIZE), NFFT))

    # 2. With breathiness = 1.0 (reset for identical DSP state)
    node.start()
    set_params(node, breathiness=1.0, mix=1.0, pitch_shift=0.0,
               formant_shift=0.0, gender_morph=0.0, sibilant_bypass=0.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    out_breathy = process_block(node, blocks[-1])
    spec_breathy = np.abs(np.fft.rfft(
        out_breathy[0].numpy() * np.hanning(BLOCK_SIZE), NFFT))

    diff = spec_breathy - spec_dry
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)

    # Below 1.5 kHz: diff should be near zero (no low-frequency noise injected)
    lf_diff = np.mean(np.abs(diff[freqs < 1200.0]))
    # 2-5 kHz: diff should show a clear noise-floor elevation
    mid_diff = np.mean(np.abs(diff[(freqs >= 2000.0) & (freqs <= 5000.0)]))

    assert mid_diff > lf_diff * 3.0, \
        f"Aspiration noise must be concentrated above 1.5 kHz and shaped by " \
        f"tract (lf_diff {lf_diff:.3e}, mid_diff {mid_diff:.3e})"


def test_vocal_transformer_mid_mix_blends_latency_aligned_dry():
    """out(mix) must equal (1-mix)*dry + mix*wet where `dry` is the input
    delayed by the fixed 9216-sample latency — i.e. (0, 1) mix values
    crossfade without comb filtering. Regression: any mix > 0 emitted 100%
    wet, making the mix knob a binary bypass switch."""
    blocks = tone_blocks(220.0, 34)
    flat_in = torch.cat(blocks, dim=1)

    def run(mix):
        node = make_node()
        set_params(node, mix=mix)
        return torch.cat([process_block(node, b) for b in blocks], dim=1)

    wet_full = run(1.0)
    mid = run(0.5)

    # Stream alignment: block m's emission reads the input ring at
    # o = (m+1)*BLOCK_SIZE - kLatency + i, and block m occupies stream samples
    # [m*BLOCK_SIZE, (m+1)*BLOCK_SIZE), so stream index j corresponds to input
    # sample j - (kLatency - BLOCK_SIZE).
    align = LATENCY - BLOCK_SIZE
    dry_aligned = torch.zeros_like(flat_in)
    dry_aligned[:, align:] = flat_in[:, :flat_in.shape[1] - align]
    expected = 0.5 * dry_aligned + 0.5 * wet_full
    assert torch.allclose(mid, expected, atol=1e-4), \
        "mix=0.5 must blend the latency-aligned dry signal with the wet path"


def test_vocal_transformer_vtln_warp_direction():
    """Verify that gender_morph = +1.0 (feminine) shifts formant resonances UP
    in frequency (shortening vocal tract), and gender_morph = -1.0 shifts them
    DOWN (lengthening vocal tract). Regression test for the inverted VTLN warp bug."""
    node = make_node()
    n_blocks = SETTLE_BLOCKS + 10
    n_total = n_blocks * BLOCK_SIZE
    n = np.arange(n_total, dtype=np.float64)

    # 50 Hz harmonic stack with a sharp resonance at 2000 Hz
    t = np.zeros_like(n)
    for h in range(1, int(SAMPLE_RATE / 2 / 50)):
        f = h * 50
        resp = 1.0 / (1.0 + ((f - 2000.0) / 100.0) ** 2)
        t += resp * np.sin(2.0 * np.pi * f * n / SAMPLE_RATE)
    t = (t / np.max(np.abs(t)) * 0.4).astype(np.float32)
    blocks = [torch.from_numpy(np.tile(t[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE], (CHANNELS, 1)))
              for i in range(n_blocks)]

    def measure_com(gender):
        node.start()
        set_params(node, pitch_shift=0.0, formant_shift=0.0, gender_morph=gender,
                   mix=1.0, breathiness=0.0, sibilant_bypass=0.0)
        for b in blocks:
            process_block(node, b)
        out = node.out.buffer[0].numpy().astype(np.float64)
        spec = np.abs(np.fft.rfft(out * np.hanning(len(out)), NFFT))
        freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        m = (freqs >= 1200) & (freqs <= 3000)
        return float(np.sum(freqs[m] * spec[m]) / np.sum(spec[m]))

    com_neut = measure_com(0.0)
    com_fem = measure_com(1.0)
    com_masc = measure_com(-1.0)

    assert abs(com_neut - 2000.0) < 50.0, f"neutral COM off: {com_neut:.1f}"
    assert com_fem > com_neut + 200.0, \
        f"gender=+1 must shift formants UP (neutral {com_neut:.1f}, fem {com_fem:.1f})"
    assert com_masc < com_neut - 200.0, \
        f"gender=-1 must shift formants DOWN (neutral {com_neut:.1f}, masc {com_masc:.1f})"


def test_vocal_transformer_sibilant_bypass_unvoiced():
    """Verify that unvoiced high-frequency fricatives (4-8 kHz sibilant /s/)
    engage sibilant bypass and preserve the unshifted dry consonant spectrum
    without pitch shifting."""
    node = make_node()
    n_blocks = SETTLE_BLOCKS + 12
    rng = np.random.default_rng(123)
    white = rng.standard_normal(n_blocks * BLOCK_SIZE).astype(np.float32)

    # Bandpass 4-8 kHz (sibilant /s/ energy)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), 1.0 / SAMPLE_RATE)
    bp = (freqs >= 4000) & (freqs <= 8000)
    spec[~bp] = 0.0
    sibilant = np.fft.irfft(spec).astype(np.float32) * 0.4
    blocks = [torch.from_numpy(np.tile(sibilant[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE], (CHANNELS, 1)))
              for i in range(n_blocks)]

    def run_sibilant(bypass_val, pitch=12.0):
        node.start()
        set_params(node, pitch_shift=pitch, formant_shift=0.0, gender_morph=0.0,
                   breathiness=0.0, sibilant_bypass=bypass_val, mix=1.0)
        outs = [process_block(node, b) for b in blocks]
        out = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
        spec_out = np.abs(np.fft.rfft(out * np.hanning(len(out)), NFFT))
        f_out = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        # Power in original 4-8 kHz band
        m_orig = (f_out >= 4000) & (f_out <= 8000)
        return float(np.sum(spec_out[m_orig] ** 2))

    p_bypass = run_sibilant(1.0)
    p_no_bypass = run_sibilant(0.0)

    # Bypass should preserve significantly more in-band dry sibilant energy than pitched-up vocoding
    assert p_bypass > p_no_bypass * 10.0, \
        f"Sibilant bypass must preserve dry unvoiced energy (bypass {p_bypass:.2e}, no-bypass {p_no_bypass:.2e})"


def _blocks_from_mono(x, n_blocks):
    """Split a float32 mono signal into (CHANNELS, BLOCK_SIZE) blocks."""
    return [torch.from_numpy(np.tile(x[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE],
                                     (CHANNELS, 1)).copy())
            for i in range(n_blocks)]


def test_vocal_transformer_transient_onset_phase_coherence():
    """Transient-gated phase reset: a sharp plosive-like onset must pass
    through the wet path without being dispersed by the phase-vocoder
    accumulator — the impulse peak is preserved and the response stays
    compact (no long smeared tail), and processing stays finite."""
    node = make_node()
    set_params(node, mix=1.0, pitch_shift=0.0, formant_shift=0.0,
               gender_morph=0.0, breathiness=0.0, sibilant_bypass=0.0)

    silence = torch.zeros(BLOCK_SIZE, CHANNELS)
    for _ in range(SETTLE_BLOCKS):
        process_block(node, silence)

    imp = torch.zeros(BLOCK_SIZE, CHANNELS)
    imp[0, 0] = 0.9
    outs = [process_block(node, imp)]
    outs += [process_block(node, silence) for _ in range(SETTLE_BLOCKS)]
    out = torch.cat(outs, dim=1)[0].numpy()

    assert np.isfinite(out).all()
    peak = float(np.max(np.abs(out)))
    # The impulse must survive at substantial amplitude (phase incoherence
    # would cancel the onset across the overlapping synthesis frames).
    assert peak >= 0.7 * 0.9, f"impulse peak attenuated: {peak:.3f}"
    # Response must be compact: at most a handful of samples above 10% of
    # the peak (a dispersed/smeared transient spreads over many samples).
    n_above = int(np.sum(np.abs(out) > 0.1 * peak))
    assert n_above <= 8, f"transient smeared across {n_above} samples"

    # Plosive burst superimposed on a steady vowel: output must stay finite
    # and bounded (the gate skips phase propagation on onset frames).
    rng = np.random.default_rng(3)
    n_total = (SETTLE_BLOCKS + 40) * BLOCK_SIZE
    n = np.arange(n_total, dtype=np.float64)
    saw = np.zeros(n.shape)
    for h in range(1, 31):
        saw += (1.0 / h) * np.sin(2.0 * np.pi * 220.0 * h * n / SAMPLE_RATE)
    saw = saw / np.max(np.abs(saw)) * 0.3
    start = n_total // 2
    saw[start:start + 480] += 0.5 * rng.standard_normal(480) * np.hanning(480)
    blocks = _blocks_from_mono(saw.astype(np.float32), SETTLE_BLOCKS + 40)
    node.start()
    outs = [process_block(node, b) for b in blocks]
    out = torch.cat(outs, dim=1).numpy()
    assert torch.isfinite(torch.from_numpy(out)).all()
    assert float(np.max(np.abs(out))) < 2.0


def test_vocal_transformer_pchip_hull_monotonicity():
    """PCHIP upper hull: the true-envelope pre-interpolation is a monotonic
    cubic Hermite hull. Regression guard: the synthesized spectrum must keep
    harmonic peaks dominant over the inter-harmonic valleys (the upper hull
    must not dip between harmonics nor ripple above the peaks), and extreme
    formant morphs must stay finite and bounded."""
    node = make_node()
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 8, amp=0.4)
    set_params(node, breathiness=0.0, mix=1.0, pitch_shift=0.0,
               formant_shift=0.0, gender_morph=0.0, sibilant_bypass=0.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    out = process_block(node, blocks[-1])[0].numpy()

    assert np.isfinite(out).all()
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out)), NFFT))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    contrasts = []
    for h in range(9, 23):  # 1980 .. 5060 Hz
        fc = 220.0 * h
        m = (freqs > fc - 15.0) & (freqs < fc + 15.0)
        v = (freqs > fc + 30.0) & (freqs < fc + 90.0)
        contrasts.append(spec[m].max() / max(spec[v].min(), 1e-12))
    median_contrast_db = 20.0 * float(np.log10(np.median(contrasts)))
    # Harmonic peaks must dominate the valleys between them: a hull that
    # dips between harmonics (or ripples) collapses this contrast.
    assert median_contrast_db > 3.0, \
        f"harmonic-to-valley contrast too low: {median_contrast_db:.2f} dB"

    # Extreme morph must remain finite and bounded.
    node.start()
    set_params(node, breathiness=0.0, mix=1.0, pitch_shift=0.0,
               formant_shift=12.0, gender_morph=1.0, sibilant_bypass=0.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    out = process_block(node, blocks[-1])
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) < 2.0


def _sideband_bin_correlation(diff, k0):
    """Correlation between spectral bins k and k+k0 of the aspiration-noise
    difference signal, measured on frame-grid-aligned 2048-pt STFT columns.

    The pitch-synchronous aspiration blends each noise bin with its +-k0
    neighbours (the frequency-domain equivalent of glottal gating
    1 + 2*beta*cos(w0 t)), which forces a positive bin correlation of
    approximately beta / (1 + 2*beta^2) ~= 0.28 for beta = 0.35. Stationary
    (unmodulated) noise has approximately zero bin correlation."""
    d = diff[LATENCY:]
    d = d[:((len(d)) // 2048) * 2048]
    w = np.hanning(2048)
    cols = np.array([np.fft.rfft(d[i * 2048:(i + 1) * 2048] * w)
                     for i in range(len(d) // 2048 - 1)])
    freqs = np.fft.rfftfreq(2048, 1.0 / SAMPLE_RATE)
    band = (freqs >= 1600.0) & (freqs <= 6900.0)
    ks = np.where(band)[0]
    ks = ks[(ks - k0 >= 0) & (ks + k0 < cols.shape[1])]
    a = cols[:, ks]
    b = cols[:, ks + k0]
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    num = float(np.mean(a * np.conj(b)).real)
    den = float(np.sqrt(np.mean(np.abs(a) ** 2) * np.mean(np.abs(b) ** 2)))
    return num / (den + 1e-30)


def test_vocal_transformer_breathiness_pitch_modulation():
    """Pitch-synchronous aspiration: on voiced frames the injected noise
    spectrum must carry the +-F0 sideband structure (glottal gating), while
    on unvoiced frames the stationary fallback must not."""
    f0_hz = 220.0
    k0 = int(round(f0_hz / (SAMPLE_RATE / 2048.0)))  # F0 bin (~9)
    n_blocks = SETTLE_BLOCKS + 64

    def breath_diff(sig):
        outs = {}
        for breath in (0.0, 0.5):
            node = make_node()
            set_params(node, breathiness=breath, mix=1.0, pitch_shift=0.0,
                       formant_shift=0.0, gender_morph=0.0, sibilant_bypass=0.0)
            outs[breath] = torch.cat(
                [process_block(node, b) for b in sig[:n_blocks]],
                dim=1)[0].numpy()
        m = min(len(outs[0.0]), len(outs[0.5]))
        return outs[0.5][:m] - outs[0.0][:m]

    # Voiced: harmonic stack -> pitch-sync sidebands must appear.
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    saw = np.zeros(n.shape)
    for h in range(1, 31):
        saw += (1.0 / h) * np.sin(2.0 * np.pi * f0_hz * h * n / SAMPLE_RATE)
    saw = (saw / np.max(np.abs(saw)) * 0.3).astype(np.float32)
    rho_voiced = _sideband_bin_correlation(
        breath_diff(_blocks_from_mono(saw, n_blocks)), k0)
    assert 0.15 < rho_voiced < 0.45, \
        f"voiced aspiration sideband correlation out of range: {rho_voiced:.3f}"

    # Unvoiced: white noise -> stationary fallback, no sidebands.
    rng = np.random.default_rng(7)
    noise = (0.25 * rng.standard_normal(n_blocks * BLOCK_SIZE)).astype(np.float32)
    rho_unvoiced = _sideband_bin_correlation(
        breath_diff(_blocks_from_mono(noise, n_blocks)), k0)
    assert abs(rho_unvoiced) < 0.1, \
        f"unvoiced aspiration must be stationary (corr {rho_unvoiced:.3f})"


