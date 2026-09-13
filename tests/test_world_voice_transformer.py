import gc
import time
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE

LATENCY = 1024  # native kLatency (fixed emission delay)
SETTLE_BLOCKS = (LATENCY // BLOCK_SIZE) + 8  # flush priming + synth lookahead
NFFT = 16384  # zero-padded FFT for peak interpolation


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("WorldVoiceTransformer")
    assert cls is not None, "WorldVoiceTransformer not registered (library build missing?)"
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


def saw_blocks(freq, n_blocks, n_harm=30, amp=0.3):
    """Continuous-phase bandlimited sawtooth (voice-like harmonic stack)."""
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


def tone_blocks(freq, n_blocks, amp=0.4):
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = (amp * np.sin(2.0 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    t2 = np.tile(t, (CHANNELS, 1))
    return [torch.from_numpy(t2[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(n_blocks)]


def autocorr_pitch(x, fmin=60.0, fmax=900.0):
    """Periodicity-based F0 estimate (robust to formant-preserving shifts,
    where the dominant spectral peak may stay at a formant region)."""
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(SAMPLE_RATE / fmax)
    hi = min(int(SAMPLE_RATE / fmin), len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SAMPLE_RATE / lag


def test_world_registration_docs_and_telemetry():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("WorldVoiceTransformer")
    assert cls is not None
    assert cls.category == "Effects"
    assert cls.label
    assert cls.description
    node = make_node()
    for port in ("in", "pitch_mod", "formant_mod", "f0_in", "gate_in"):
        assert node.inputs[port].help, f"{port} missing help"
    assert node.inputs["pitch_mod"].param_name == "pitch_shift"
    assert node.inputs["formant_mod"].param_name == "formant_shift"
    assert node.params["pitch_shift"].meta.get("min") == -12.0
    assert node.params["pitch_shift"].meta.get("max") == 12.0
    assert node.params["formant_shift"].meta.get("min") == -12.0
    assert node.params["formant_shift"].meta.get("max") == 12.0
    assert node.params["gender_morph"].meta.get("min") == -1.0
    assert node.params["gender_morph"].meta.get("max") == 1.0
    assert node.params["breathiness"].meta.get("min") == 0.0
    assert node.params["breathiness"].meta.get("max") == 1.0
    assert node.params["output_gain"].meta.get("min") == -12.0
    telem = node.get_telemetry()
    assert telem["latency_samples"] == LATENCY
    assert telem["latency_ms"] == pytest.approx(LATENCY / SAMPLE_RATE * 1000.0, abs=0.01)


def test_world_presets_defined_and_valid():
    """PRESETS conform to the NodeItem schema: known params, in-range values,
    stagable through the canonical set/sync path."""
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("WorldVoiceTransformer")
    assert hasattr(cls, "PRESETS")
    assert "Male -> Female" in cls.PRESETS
    assert "Female -> Male" in cls.PRESETS
    node = make_node()
    for name, pdict in cls.PRESETS.items():
        assert pdict, f"preset '{name}' is empty"
        for k, v in pdict.items():
            assert k in node.params, f"unknown param '{k}' in preset '{name}'"
            p = node.params[k]
            lo, hi = p.meta.get("min"), p.meta.get("max")
            assert lo is None or v >= lo
            assert hi is None or v <= hi
            p.set(v)
        node.sync()
        node._sync_params_to_cpp()


def test_world_mix_zero_bit_exact():
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_world_mono_to_stereo_identical():
    """Mono (1, 512) in -> (2, 512) out with bit-identical channels (the
    native side publishes channel 0 twice for duplicated inputs so
    independent synthesis noise cannot diverge)."""
    node = make_node()
    set_params(node, mix=1.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 2)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    mono = blocks[SETTLE_BLOCKS][0:1].clone()
    node.inp.get_tensor = lambda b=mono: b
    node.process()
    out = node.out.buffer
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], out[1])


def test_world_identity_pitch_and_energy():
    """Neutral settings preserve F0 (autocorrelation) and overall energy
    once the fixed-latency pipeline has primed."""
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0, output_gain=0.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in blocks]
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert float(np.abs(tail).max()) > 0.01
    assert abs(autocorr_pitch(tail) - 220.0) < 5.0
    dry = torch.cat(blocks[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    ratio_db = 10.0 * np.log10(np.sum(tail ** 2) / (np.sum(dry ** 2) + 1e-12))
    assert -1.5 < ratio_db < 1.5, f"identity energy off by {ratio_db:.2f} dB"


def test_world_pitch_accuracy():
    node = make_node()
    set_params(node, pitch_shift=12.0, formant_shift=0.0, mix=1.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[SETTLE_BLOCKS:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 440.0) < 10.0

    node.start()
    set_params(node, pitch_shift=-12.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[SETTLE_BLOCKS:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 110.0) < 10.0


def test_world_pitch_modulation_and_disconnect():
    """pitch_mod drives pitch at block rate; disconnecting restores the
    staged parameter (disconnect re-sync contract)."""
    node = make_node()
    set_params(node, pitch_shift=0.0, mix=1.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS * 3 + 32)
    bi = 0
    for b in blocks[bi:bi + SETTLE_BLOCKS]:
        process_block(node, b)
    bi += SETTLE_BLOCKS

    cv = torch.full((CHANNELS, BLOCK_SIZE), -12.0, dtype=DTYPE)
    node.pitch_mod.connected_outputs = [object()]
    node.pitch_mod.get_tensor = lambda: cv
    for b in blocks[bi:bi + SETTLE_BLOCKS + 2]:
        process_block(node, b)
    bi += SETTLE_BLOCKS + 2
    outs = [process_block(node, b) for b in blocks[bi:bi + 16]]
    bi += 16
    x = np.concatenate([o[0].numpy().astype(np.float64) for o in outs])
    assert abs(autocorr_pitch(x) - 110.0) < 10.0

    node.pitch_mod.connected_outputs = []
    for b in blocks[bi:bi + SETTLE_BLOCKS + 2]:
        process_block(node, b)
    bi += SETTLE_BLOCKS + 2
    outs2 = [process_block(node, b) for b in blocks[bi:bi + 16]]
    x2 = np.concatenate([o[0].numpy().astype(np.float64) for o in outs2])
    assert abs(autocorr_pitch(x2) - 220.0) < 10.0


def test_world_formant_direction_and_pitch_independence():
    """Formant shifts move resonance energy in the right direction while the
    fundamental stays put (pitch-formant independence)."""
    n_blocks = SETTLE_BLOCKS + 10
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = np.zeros_like(n)
    for h in range(1, int(SAMPLE_RATE / 2 / 100)):
        f = h * 100.0
        resp = 1.0 / (1.0 + ((f - 2000.0) / 100.0) ** 2)
        t += resp * np.sin(2.0 * np.pi * f * n / SAMPLE_RATE)
    t = (t / np.max(np.abs(t)) * 0.4).astype(np.float32)
    blocks = [torch.from_numpy(np.tile(t[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE], (CHANNELS, 1)))
              for i in range(n_blocks)]

    def measure_com(formant):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=formant, mix=1.0)
        for b in blocks:
            process_block(node, b)
        y = node.out.buffer[0].numpy().astype(np.float64)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), NFFT))
        freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        m = (freqs >= 1200) & (freqs <= 3200)
        return float(np.sum(freqs[m] * spec[m]) / np.sum(spec[m]))

    com_neut = measure_com(0.0)
    assert abs(com_neut - 2000.0) < 150.0, f"neutral COM off: {com_neut:.1f}"
    assert measure_com(6.0) > com_neut + 150.0
    assert measure_com(-6.0) < com_neut - 150.0

    # Pitch independence: F0 of a 220 Hz saw is untouched by formant warp.
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=5.0, mix=1.0)
    vblocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in vblocks]
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 220.0) < 5.0


def resonant_stack_blocks(n_blocks, base_hz=100.0, res_hz=2000.0, amp=0.4):
    """Harmonic stack with a sharp Lorentzian resonance (formant assay).
    Base stays above the 71 Hz F0 floor so tracking stays voiced."""
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = np.zeros_like(n)
    for h in range(1, int(SAMPLE_RATE / 2 / base_hz)):
        f = h * base_hz
        resp = 1.0 / (1.0 + ((f - res_hz) / 100.0) ** 2)
        t += resp * np.sin(2.0 * np.pi * f * n / SAMPLE_RATE)
    t = (t / np.max(np.abs(t)) * amp).astype(np.float32)
    return [torch.from_numpy(np.tile(t[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE],
                                      (CHANNELS, 1)).copy())
            for i in range(n_blocks)]


def resonant_com(out_buffer, lo=1200.0, hi=3200.0):
    """Spectral center of mass of the settled tail output."""
    y = out_buffer[0].numpy().astype(np.float64)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), NFFT))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(freqs[m] * spec[m]) / np.sum(spec[m]))


def test_world_warp_gate_gender_alone():
    """Gender morph warps formants even when formant_shift is 0 (unified
    warp-gate regression guard)."""
    n_blocks = SETTLE_BLOCKS + 10
    blocks = resonant_stack_blocks(n_blocks)

    def measure_com(gender):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=0.0,
                   gender_morph=gender, mix=1.0, breathiness=0.0)
        for b in blocks:
            process_block(node, b)
        return resonant_com(node.out.buffer)

    com_neut = measure_com(0.0)
    assert abs(com_neut - 2000.0) < 150.0, f"neutral COM off: {com_neut:.1f}"
    com_fem = measure_com(1.0)
    assert com_fem > com_neut + 100.0, \
        f"gender=+1 with formant=0 must shift up (neut {com_neut:.1f}, fem {com_fem:.1f})"


def test_world_unified_semantics_gender_vs_formant():
    """gender=+1.0 (equiv +3 st warp) moves resonance the same direction as
    formant_shift=+3.0. Margin is loose: gender additionally tilts the
    spectrum, which pulls COM down relative to pure warp."""
    n_blocks = SETTLE_BLOCKS + 10
    blocks = resonant_stack_blocks(n_blocks)

    def measure_com(formant, gender):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=formant,
                   gender_morph=gender, mix=1.0, breathiness=0.0)
        for b in blocks:
            process_block(node, b)
        return resonant_com(node.out.buffer)

    com_neut = measure_com(0.0, 0.0)
    com_gender = measure_com(0.0, 1.0)
    com_formant = measure_com(3.0, 0.0)
    assert com_gender > com_neut + 50.0
    assert com_formant > com_neut + 50.0
    assert abs(com_gender - com_formant) < 150.0, \
        f"gender=+1 ({com_gender:.1f}) and formant=+3 ({com_formant:.1f}) must agree in magnitude"


def test_world_gender_spectral_tilt():
    """Feminine morph attenuates HF relative to LF; masculine enhances it."""
    blocks = saw_blocks(150.0, SETTLE_BLOCKS + 10)

    def hf_lf_ratio(gender):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=0.0,
                   gender_morph=gender, mix=1.0, breathiness=0.0)
        for b in blocks:
            process_block(node, b)
        y = node.out.buffer[0].numpy().astype(np.float64)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), NFFT))
        freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        lf = float(np.sum(spec[(freqs >= 100) & (freqs <= 500)] ** 2))
        hf = float(np.sum(spec[(freqs >= 3000) & (freqs <= 8000)] ** 2))
        return hf / (lf + 1e-12)

    ratio_neut = hf_lf_ratio(0.0)
    assert hf_lf_ratio(1.0) < ratio_neut * 0.85, "feminine must tilt HF down"
    assert hf_lf_ratio(-1.0) > ratio_neut * 1.15, "masculine must tilt HF up"


def test_world_breathiness_monotonic_response():
    """Breathiness raises inter-harmonic noise floor monotonically in the
    1.5-7 kHz band (spectral flatness metric: WORLD's power-complementary
    split keeps total band power roughly constant, so power is the wrong
    metric). LF periodic power must stay preserved."""
    blocks = saw_blocks(200.0, SETTLE_BLOCKS + 12)

    def measure(breath_val):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=0.0, gender_morph=0.0,
                   breathiness=breath_val, mix=1.0)
        outs = [process_block(node, b) for b in blocks]
        y = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), NFFT)) + 1e-12
        freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        band = spec[(freqs >= 2500) & (freqs <= 5500)]
        flatness = float(np.exp(np.mean(np.log(band))) / np.mean(band))
        lf = float(np.sum(spec[(freqs >= 100) & (freqs <= 800)] ** 2))
        return flatness, lf

    f0, lf0 = measure(0.0)
    f5, lf5 = measure(0.5)
    f10, lf10 = measure(1.0)
    assert f10 > f5 > f0, \
        f"aspiration flatness must rise monotonically: 0={f0:.3f} 0.5={f5:.3f} 1={f10:.3f}"
    assert abs(lf10 - lf0) / (lf0 + 1e-12) < 0.20, "LF periodic power must survive breathiness"


def test_world_mid_mix_blends_latency_aligned_dry():
    """out(mix) == (1-mix)*dry_aligned + mix*wet where dry is delayed by the
    fixed latency — intermediate mix values crossfade without comb filtering."""
    blocks = tone_blocks(220.0, 34)
    flat_in = torch.cat(blocks, dim=1)

    def run(mix):
        node = make_node()
        set_params(node, mix=mix)
        return torch.cat([process_block(node, b) for b in blocks], dim=1)

    wet_full = run(1.0)
    mid = run(0.5)
    align = LATENCY - BLOCK_SIZE
    dry_aligned = torch.zeros_like(flat_in)
    dry_aligned[:, align:] = flat_in[:, :flat_in.shape[1] - align]
    expected = 0.5 * dry_aligned + 0.5 * wet_full
    assert torch.allclose(mid, expected, atol=1e-4)


def test_world_extremes_stability_and_no_dropouts():
    """Extreme settings stay finite/bounded, and steady voiced output never
    starves (latency-margin guard for the fixed read pointer)."""
    node = make_node()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.2
    for pitch in (-12.0, 0.0, 12.0):
        for formant in (-12.0, 0.0, 12.0):
            set_params(node, pitch_shift=pitch, formant_shift=formant, mix=1.0)
            out = process_block(node, blk)
            assert torch.isfinite(out).all()
            assert float(out.abs().max()) < 8.0

    for freq in (110.0, 220.0, 440.0):
        node.start()
        set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0)
        blocks = saw_blocks(freq, SETTLE_BLOCKS + 16)
        outs = [process_block(node, b) for b in blocks]
        tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)
        assert torch.isfinite(tail).all()
        # No synthesis starvation: every block carries significant energy.
        for i in range(tail.shape[1] // BLOCK_SIZE):
            seg = tail[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            assert float(seg.abs().max()) > 0.005, f"dropout at {freq}Hz block {i}"


def test_world_silence_in_silence_out():
    node = make_node()
    set_params(node, mix=1.0)
    for b in saw_blocks(220.0, SETTLE_BLOCKS):
        process_block(node, b)
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    outs = [process_block(node, silence) for _ in range(SETTLE_BLOCKS)]
    tail = torch.cat(outs[-4:], dim=1)
    assert float(tail.abs().max()) < 0.02


def test_world_reset_on_start():
    node = make_node()
    set_params(node, mix=1.0)
    for b in saw_blocks(220.0, SETTLE_BLOCKS + 2):
        process_block(node, b)
    assert float(node.out.buffer.abs().max()) > 0.01
    node.start()
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = None
    for _ in range(SETTLE_BLOCKS):
        out = process_block(node, silence)
    assert float(out.abs().max()) < 1e-6, "reset must clear rings/FIFO/synth memory"


def test_world_save_load_roundtrip():
    node = make_node()
    set_params(node, pitch_shift=5.0, formant_shift=-3.0, gender_morph=0.75,
               breathiness=0.3, mix=0.7, output_gain=2.0)
    snapshot = node.to_dict()
    node2 = make_node()
    node2.load_state(snapshot)
    for k in ("pitch_shift", "formant_shift", "gender_morph", "breathiness",
              "mix", "output_gain"):
        assert node2.params[k].value == pytest.approx(node.params[k].value)


def test_world_zero_python_allocation():
    node = make_node()
    set_params(node, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    process_block(node, blk)  # warm-up

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, blk)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth - before < 64 * 1024, f"net Python allocation {growth - before} bytes over 50 blocks"


def test_world_native_processing_budget():
    """Native C++ steady-state cost stays well under the 10.67 ms block
    budget (RTF/timing per AGENTS.md §4 — tracemalloc does not apply).
    Best of 3 rounds filters desktop scheduling noise; the bar still
    catches >60% regressions (typical idle cost is ~4.5 ms)."""
    node = make_node()
    set_params(node, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    process_block(node, blk)
    for _ in range(10):
        process_block(node, blk)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        n = 60
        for _ in range(n):
            process_block(node, blk)
        best = min(best, (time.perf_counter() - t0) / n * 1000.0)
    assert best < 6.5, f"best process() {best:.2f} ms exceeds budget"


# ---------------------------------------------------------------------------
# External F0 guidance (e.g. SwiftF0 pitch_out/gate_out -> f0_in/gate_in):
# NSDF bypass with gated / fallback voicing and disconnect revert.
# ---------------------------------------------------------------------------


def _connect_ext_f0(node, f0_hz, gate=None):
    """Wire block-constant external F0 (+ optional gate) CV into the node,
    mimicking a connected upstream CV output slot."""
    from base import DTYPE as _DTYPE

    f0_tensor = torch.full((1, BLOCK_SIZE), f0_hz, dtype=_DTYPE)
    node.f0_in.connected_outputs = [object()]
    node.f0_in.get_tensor = lambda: f0_tensor
    if gate is not None:
        gate_tensor = torch.full((1, BLOCK_SIZE), gate, dtype=_DTYPE)
        node.gate_in.connected_outputs = [object()]
        node.gate_in.get_tensor = lambda: gate_tensor


def test_world_ext_f0_drives_pitch():
    """External F0 at 440 Hz with a 220 Hz saw input resynthesizes at the
    external fundamental (pitch_shift = 0)."""
    node = make_node()
    assert "f0_in" in node.inputs and "gate_in" in node.inputs
    set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0, output_gain=0.0)
    _connect_ext_f0(node, 440.0, gate=1.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in blocks]
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 440.0) < 10.0
    assert node._was_f0_connected is True


def test_world_ext_gate_unvoiced_devoices():
    """Gate 0 vetoes the held CV: with f0_in = 440 Hz but gate 0, output
    must not lock to 440 Hz (unvoiced excitation through the envelope)."""
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0)
    _connect_ext_f0(node, 440.0, gate=0.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in blocks]
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert torch.isfinite(torch.cat(outs, dim=1)).all()
    assert abs(autocorr_pitch(tail) - 440.0) > 50.0, \
        "unvoiced gate must not synthesize the held 440 Hz CV"


def test_world_ext_f0_disconnect_reverts():
    """Disconnecting f0_in publishes mode 0 once and reverts to internal
    tracking (220 Hz saw passes at ~220 Hz with pitch_shift = 0)."""
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0)
    _connect_ext_f0(node, 440.0, gate=1.0)
    process_block(node, saw_blocks(220.0, 1)[0])
    assert node._was_f0_connected is True
    assert node._last_ext_mode == pytest.approx(1.0)

    node.f0_in.connected_outputs = []
    node.gate_in.connected_outputs = []
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in blocks]
    assert node._was_f0_connected is False
    assert node._last_ext_mode == pytest.approx(0.0)
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 220.0) < 10.0


def test_world_ext_path_zero_allocation():
    """External-F0 steady state (gated 220 Hz CV): 100 blocks allocate
    < 64 KB net, matching the internal tracking-path budget."""
    node = make_node()
    set_params(node, mix=1.0)
    _connect_ext_f0(node, 220.0, gate=1.0)

    audio_in = torch.randn((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    node.inp.get_tensor = lambda: audio_in

    for _ in range(20):  # warm-up (mode push + pipeline settle)
        node.process()

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(100):
        node.process()
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth - before < 64 * 1024, \
        f"net allocation {growth - before} bytes over 100 ext-F0 blocks"


def _low_pitch(x, fmin=40.0, fmax=120.0):
    """Bass-range periodicity: returns (freq_hz, peak_strength).

    The shared autocorr_pitch() searches up to 900 Hz, where a strong
    formant ringing on each glottal pulse outscores the true 55 Hz pulse
    rate. Restricting to <= 120 Hz measures the pulse rate itself. Returns
    the peak normalized autocorrelation too, so callers can distinguish a
    voiced pulse train (~0.6) from unvoiced excitation (~0.07)."""
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(SAMPLE_RATE / fmax)
    hi = min(int(SAMPLE_RATE / fmin), len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SAMPLE_RATE / lag, float(ac[lag])


def test_world_low_bass_voicing():
    """Fundamentals in [50, 71) Hz (male bass) must stay voiced.

    The voicing-acceptance floor is 50 Hz while CheapTrick/D4C analyze with
    a 71 Hz floor to preserve N_FFT = 2048; synthesis uses the true F0.
    A 55 Hz (A1) saw must resynthesize as a 55 Hz pulse train, not collapse
    to unvoiced excitation (pre-fix: f0 = 0 below the 71 Hz floor).

    Note: bass output is legitimately pulsatile at the block scale (glottal
    period ~873 samples > BLOCK_SIZE with breathiness = 0), so there is no
    per-block energy assertion here — only overall energy, pulse-rate F0,
    and voicing strength. Covers both the internal NSDF path and the
    external-F0 path."""
    for ext_f0 in (None, 55.0):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0,
                   output_gain=0.0)
        if ext_f0 is not None:
            _connect_ext_f0(node, ext_f0, gate=1.0)
        blocks = saw_blocks(55.0, SETTLE_BLOCKS + 32, n_harm=30, amp=0.3)
        outs = [process_block(node, b) for b in blocks]
        assert torch.isfinite(torch.cat(outs, dim=1)).all()
        tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
        assert float(np.abs(tail).max()) > 0.01, \
            f"bass output went silent (ext_f0={ext_f0})"
        f, strength = _low_pitch(tail)
        assert strength > 0.3, \
            f"bass output not a voiced pulse train (ext_f0={ext_f0}): " \
            f"periodicity {strength:.3f} (unvoiced excitation ~0.07)"
        assert abs(f - 55.0) < 5.0, \
            f"bass F0 mistracked (ext_f0={ext_f0}): {f:.2f} Hz (want 55)"
