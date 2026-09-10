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
    for port in ("in", "pitch_mod", "formant_mod"):
        assert node.inputs[port].help, f"{port} missing help"
    assert node.inputs["pitch_mod"].param_name == "pitch_shift"
    assert node.inputs["formant_mod"].param_name == "formant_shift"
    assert node.params["pitch_shift"].meta.get("min") == -12.0
    assert node.params["pitch_shift"].meta.get("max") == 12.0
    assert node.params["formant_shift"].meta.get("min") == -12.0
    assert node.params["formant_shift"].meta.get("max") == 12.0
    assert node.params["output_gain"].meta.get("min") == -12.0
    telem = node.get_telemetry()
    assert telem["latency_samples"] == LATENCY
    assert telem["latency_ms"] == pytest.approx(LATENCY / SAMPLE_RATE * 1000.0, abs=0.01)


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
    set_params(node, pitch_shift=5.0, formant_shift=-3.0, mix=0.7, output_gain=2.0)
    snapshot = node.to_dict()
    node2 = make_node()
    node2.load_state(snapshot)
    for k in ("pitch_shift", "formant_shift", "mix", "output_gain"):
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
    budget (RTF/timing per AGENTS.md §4 — tracemalloc does not apply)."""
    node = make_node()
    set_params(node, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    process_block(node, blk)
    for _ in range(10):
        process_block(node, blk)
    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        process_block(node, blk)
    dt = (time.perf_counter() - t0) / n * 1000.0
    assert dt < 5.0, f"mean process() {dt:.2f} ms exceeds 50% of block budget"
