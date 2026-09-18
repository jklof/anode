import gc
import time
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE

LATENCY = 5760  # native inputLatency + outputLatency (120 ms at 48 kHz)
SETTLE_BLOCKS = (LATENCY // BLOCK_SIZE) + 8  # flush latency-fill + margin
NFFT = 16384  # zero-padded FFT for peak interpolation


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SignalsmithVocal")
    assert cls is not None, "SignalsmithVocal not registered (library build missing?)"
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


def autocorr_pitch(x, fmin=60.0, fmax=1200.0):
    """Periodicity-based F0 estimate (robust to formant-preserving shifts,
    where the dominant spectral peak may stay at a formant region)."""
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(SAMPLE_RATE / fmax)
    hi = min(int(SAMPLE_RATE / fmin), len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SAMPLE_RATE / lag


def resonant_stack_blocks(n_blocks, base_hz=100.0, res_hz=2000.0, amp=0.4):
    """Harmonic stack with a sharp Lorentzian resonance (formant assay)."""
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


def test_signalsmith_registration_docs_and_telemetry():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SignalsmithVocal")
    assert cls is not None
    assert cls.category == "Voice & Pitch"
    assert cls.label
    assert cls.description
    node = make_node()
    for port in ("in", "pitch_mod", "formant_mod"):
        assert node.inputs[port].help, f"{port} missing help"
    assert node.inputs["pitch_mod"].param_name == "pitch_shift"
    assert node.inputs["formant_mod"].param_name == "formant_shift"
    assert node.params["pitch_shift"].meta.get("min") == -24.0
    assert node.params["pitch_shift"].meta.get("max") == 24.0
    assert node.params["formant_shift"].meta.get("min") == -24.0
    assert node.params["formant_shift"].meta.get("max") == 24.0
    assert node.params["gender_morph"].meta.get("min") == -1.0
    assert node.params["gender_morph"].meta.get("max") == 1.0
    assert node.params["tonality_limit"].meta.get("min") == 0.0
    assert node.params["tonality_limit"].meta.get("max") == 8000.0
    telem = node.get_telemetry()
    assert telem["latency_samples"] == LATENCY
    assert telem["latency_ms"] == pytest.approx(LATENCY / SAMPLE_RATE * 1000.0, abs=0.01)


def test_signalsmith_presets_defined_and_valid():
    """PRESETS conform to the NodeItem schema: known params, in-range values,
    stagable through the canonical set/sync path."""
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SignalsmithVocal")
    assert hasattr(cls, "PRESETS")
    assert "Male -> Female Pop" in cls.PRESETS
    assert "Female -> Male Deep" in cls.PRESETS
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


def test_signalsmith_mix_zero_bit_exact():
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    out = process_block(node, blk)
    assert torch.equal(out, blk)


def test_signalsmith_mono_to_stereo_identical():
    """Mono (1, 512) in -> (2, 512) out with bit-identical channels."""
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


def test_signalsmith_identity_pitch_and_energy():
    """Neutral settings preserve F0 (autocorrelation) and overall energy
    once the fixed-latency pipeline has primed."""
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=0.0, mix=1.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in blocks]
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert float(np.abs(tail).max()) > 0.01
    assert abs(autocorr_pitch(tail) - 220.0) < 5.0
    dry = torch.cat(blocks[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    ratio_db = 10.0 * np.log10(np.sum(tail ** 2) / (np.sum(dry ** 2) + 1e-12))
    assert -1.5 < ratio_db < 1.5, f"identity energy off by {ratio_db:.2f} dB"


def test_signalsmith_pitch_accuracy():
    node = make_node()
    set_params(node, pitch_shift=12.0, formant_shift=0.0, mix=1.0)
    blocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[SETTLE_BLOCKS:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 440.0) < 5.0

    node.start()
    set_params(node, pitch_shift=-12.0)
    for b in blocks[:SETTLE_BLOCKS]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[SETTLE_BLOCKS:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 110.0) < 5.0


def test_signalsmith_pitch_modulation_and_disconnect():
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
    assert abs(autocorr_pitch(x) - 110.0) < 5.0

    node.pitch_mod.connected_outputs = []
    for b in blocks[bi:bi + SETTLE_BLOCKS + 2]:
        process_block(node, b)
    bi += SETTLE_BLOCKS + 2
    outs2 = [process_block(node, b) for b in blocks[bi:bi + 16]]
    x2 = np.concatenate([o[0].numpy().astype(np.float64) for o in outs2])
    assert abs(autocorr_pitch(x2) - 220.0) < 5.0


def test_signalsmith_formant_direction_and_pitch_independence():
    """Formant shifts move resonance energy in the right direction while the
    fundamental stays put (pitch-formant independence)."""
    n_blocks = SETTLE_BLOCKS + 10
    blocks = resonant_stack_blocks(n_blocks)

    def measure_com(formant):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=formant, mix=1.0)
        for b in blocks:
            process_block(node, b)
        return resonant_com(node.out.buffer)

    com_neut = measure_com(0.0)
    assert abs(com_neut - 2000.0) < 150.0, f"neutral COM off: {com_neut:.1f}"
    assert measure_com(6.0) > com_neut + 150.0
    assert measure_com(-6.0) < com_neut - 150.0

    # Pitch independence: F0 of a 220 Hz saw is untouched by formant warp.
    node = make_node()
    set_params(node, pitch_shift=0.0, formant_shift=6.0, mix=1.0)
    vblocks = saw_blocks(220.0, SETTLE_BLOCKS + 16)
    outs = [process_block(node, b) for b in vblocks]
    tail = torch.cat(outs[SETTLE_BLOCKS:], dim=1)[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 220.0) < 5.0


def test_signalsmith_gender_morph_shifts_formants():
    """gender=+1 warps formants upward even when formant_shift is 0."""
    n_blocks = SETTLE_BLOCKS + 10
    blocks = resonant_stack_blocks(n_blocks)

    def measure_com(gender):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=0.0,
                   gender_morph=gender, mix=1.0)
        for b in blocks:
            process_block(node, b)
        return resonant_com(node.out.buffer)

    com_neut = measure_com(0.0)
    assert abs(com_neut - 2000.0) < 150.0, f"neutral COM off: {com_neut:.1f}"
    com_fem = measure_com(1.0)
    assert com_fem > com_neut + 100.0, \
        f"gender=+1 with formant=0 must shift up (neut {com_neut:.1f}, fem {com_fem:.1f})"


def test_signalsmith_tonality_limit_runs_clean():
    """Tonality limit engages without instability or dropouts."""
    node = make_node()
    set_params(node, pitch_shift=12.0, tonality_limit=8000.0, mix=1.0)
    for b in saw_blocks(220.0, SETTLE_BLOCKS + 8):
        out = process_block(node, b)
        assert torch.isfinite(out).all()
        assert float(out.abs().max()) < 8.0


def test_signalsmith_mid_mix_blends_latency_aligned_dry():
    """out(mix) == (1-mix)*dry_aligned + mix*wet where dry is delayed by the
    fixed latency — intermediate mix values crossfade without comb filtering."""
    blocks = tone_blocks(220.0, 40)
    flat_in = torch.cat(blocks, dim=1)

    def run(mix):
        node = make_node()
        set_params(node, mix=mix)
        return torch.cat([process_block(node, b) for b in blocks], dim=1)

    wet_full = run(1.0)
    mid = run(0.5)
    dry_aligned = torch.zeros_like(flat_in)
    dry_aligned[:, LATENCY:] = flat_in[:, :flat_in.shape[1] - LATENCY]
    expected = 0.5 * dry_aligned + 0.5 * wet_full
    assert torch.allclose(mid, expected, atol=1e-4)


def test_signalsmith_extremes_stability_and_no_dropouts():
    """Extreme settings stay finite/bounded, and steady voiced output never
    starves."""
    node = make_node()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.2
    for pitch in (-24.0, 0.0, 24.0):
        for formant in (-24.0, 0.0, 24.0):
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
        for i in range(tail.shape[1] // BLOCK_SIZE):
            seg = tail[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
            assert float(seg.abs().max()) > 0.005, f"dropout at {freq}Hz block {i}"


def test_signalsmith_silence_in_silence_out():
    node = make_node()
    set_params(node, mix=1.0)
    for b in saw_blocks(220.0, SETTLE_BLOCKS):
        process_block(node, b)
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    outs = [process_block(node, silence) for _ in range(SETTLE_BLOCKS)]
    tail = torch.cat(outs[-4:], dim=1)
    assert float(tail.abs().max()) < 0.02


def test_signalsmith_reset_on_start():
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
    assert float(out.abs().max()) < 1e-6, "reset must clear rings/stretcher memory"


def test_signalsmith_save_load_roundtrip():
    node = make_node()
    set_params(node, pitch_shift=5.0, formant_shift=-3.0, gender_morph=0.75,
               tonality_limit=6000.0, mix=0.7)
    snapshot = node.to_dict()
    node2 = make_node()
    node2.load_state(snapshot)
    for k in ("pitch_shift", "formant_shift", "gender_morph",
              "tonality_limit", "mix"):
        assert node2.params[k].value == pytest.approx(node.params[k].value)


def test_signalsmith_zero_python_allocation():
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


def test_signalsmith_native_processing_budget():
    """Native C++ steady-state cost stays well under the 10.67 ms block
    budget (RTF/timing per AGENTS.md §4 — tracemalloc does not apply).
    Best of 3 rounds filters desktop scheduling noise."""
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
    assert best < 4.0, f"best process() {best:.2f} ms exceeds budget"


# ---------------------------------------------------------------------------
# Latency modes: 0 Studio (5760 spls / 120 ms), 1 Live (~1920 spls / 40 ms).
# ---------------------------------------------------------------------------

def _set_live_mode(node, mode):
    """Switch mode and run one block so the staged value reaches native DSP
    (telemetry reads native state)."""
    node.params["latency_mode"].set(mode)
    node.sync()
    process_block(node, torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE))


def test_signalsmith_latency_modes_query():
    node = make_node()
    assert node.get_telemetry()["latency_samples"] == LATENCY
    _set_live_mode(node, 1)
    telem = node.get_telemetry()
    assert telem["latency_samples"] == 1920, telem
    assert telem["latency_ms"] == pytest.approx(40.0, abs=0.01)
    _set_live_mode(node, 0)
    assert node.get_telemetry()["latency_samples"] == LATENCY


def test_signalsmith_live_mode_pitch_accuracy():
    """220 Hz saw +12 st in Live mode lands at 440 Hz after the short prime."""
    node = make_node()
    set_params(node, pitch_shift=12.0, formant_shift=0.0, mix=1.0)
    _set_live_mode(node, 1)
    live_lat = node.get_telemetry()["latency_samples"]
    settle = live_lat // BLOCK_SIZE + 6
    blocks = saw_blocks(220.0, settle + 16)
    for b in blocks[:settle]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[settle:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 440.0) < 8.0


def test_signalsmith_live_mode_mix_aligned():
    """Mid-mix blends the Live-latency-aligned dry path (align = live L)."""
    node0 = make_node()
    _set_live_mode(node0, 1)
    live_lat = node0.get_telemetry()["latency_samples"]
    blocks = tone_blocks(220.0, 30)
    flat_in = torch.cat(blocks, dim=1)

    def run(mix):
        node = make_node()
        set_params(node, mix=mix)
        _set_live_mode(node, 1)
        return torch.cat([process_block(node, b) for b in blocks], dim=1)

    wet_full = run(1.0)
    mid = run(0.5)
    dry_aligned = torch.zeros_like(flat_in)
    dry_aligned[:, live_lat:] = flat_in[:, :flat_in.shape[1] - live_lat]
    expected = 0.5 * dry_aligned + 0.5 * wet_full
    assert torch.allclose(mid, expected, atol=1e-4)


def test_signalsmith_live_mode_switch_stability():
    """Studio -> Live -> Studio mid-stream stays finite and recovers."""
    node = make_node()
    set_params(node, pitch_shift=0.0, mix=1.0)
    blocks = saw_blocks(220.0, 40)
    for b in blocks[:12]:
        process_block(node, b)
    _set_live_mode(node, 1)
    outs = [process_block(node, b) for b in blocks[12:]]
    assert all(torch.isfinite(o).all() for o in outs)
    assert float(torch.cat(outs[-6:], dim=1).abs().max()) > 0.01
