import gc
import time
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("DeEsser")
    assert cls is not None, "DeEsser not registered (library build missing?)"
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


def ess_blocks(n_blocks=12, seed=5, amp=0.3):
    """Sustained /s/-like energy: white noise bandpassed to 5-8.5 kHz."""
    rng = np.random.default_rng(seed)
    n = n_blocks * BLOCK_SIZE
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    fr = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    X[(fr < 5000) | (fr > 8500)] = 0.0
    ess = np.fft.irfft(X).astype(np.float32)
    ess = (ess / np.max(np.abs(ess)) * amp).astype(np.float32)
    return [torch.from_numpy(np.tile(ess[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE],
                                      (CHANNELS, 1)).copy())
            for i in range(n_blocks)]


def saw_blocks(freq, n_blocks, amp=0.3, n_harm=30):
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


def band_energy(x, lo, hi):
    """Total power of a mono signal in [lo, hi] Hz."""
    X = np.abs(np.fft.rfft(x.astype(np.float64)))
    fr = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
    m = (fr >= lo) & (fr <= hi)
    return float(np.sum(X[m] ** 2))


def test_deesser_registration_docs_and_telemetry():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("DeEsser")
    assert cls is not None
    assert cls.category == "Effects"
    assert cls.label
    assert cls.description
    node = make_node()
    assert node.inputs["in"].help
    assert node.outputs["out"].buffer.shape[0] == CHANNELS
    assert node.params["frequency"].meta.get("min") == 3000.0
    assert node.params["attack_ms"].meta.get("max") == 10.0
    assert node.params["release_ms"].meta.get("max") == 300.0
    telem = node.get_telemetry()
    assert telem["latency_samples"] == 0
    assert telem["gr_db"] == pytest.approx(0.0)


def test_deesser_mix_zero_bit_exact():
    node = make_node()
    set_params(node, mix=0.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    assert torch.equal(process_block(node, blk), blk)


def test_deesser_idle_transparent():
    """With depth=0 the processor is magnitude-transparent: the LR4
    crossover sums flat (its phase shift is inaudible steady-state), so
    overall and per-band energies are preserved."""
    node = make_node()
    set_params(node, depth=0.0, mix=1.0)
    blocks = saw_blocks(220.0, 8) + ess_blocks(8)
    outs = torch.cat([process_block(node, b) for b in blocks], dim=1)
    dry = torch.cat(blocks, dim=1)
    w = outs[0].numpy().astype(np.float64)
    d = dry[0].numpy().astype(np.float64)
    overall = 10.0 * np.log10(np.sum(w ** 2) / (np.sum(d ** 2) + 1e-12))
    assert abs(overall) < 0.05, f"idle must preserve level (got {overall:+.3f} dB)"
    for lo, hi in ((0, 3000), (6000, 9000)):
        b = 10.0 * np.log10(band_energy(w, lo, hi) / (band_energy(d, lo, hi) + 1e-12))
        assert abs(b) < 0.1, f"idle band {lo}-{hi} moved {b:+.3f} dB"


def test_deesser_ess_cut_and_vowel_transparency():
    """Hot ess in the HF band is cut toward `depth`; vowel body is untouched
    (split-band: LF energy preserved, only HF ducks)."""
    node = make_node()
    set_params(node, frequency=6500.0, threshold=-30.0, depth=6.0,
               attack_ms=1.0, release_ms=60.0, listen=False, mix=1.0)

    eblocks = ess_blocks(16)
    eouts = torch.cat([process_block(node, b) for b in eblocks], dim=1)
    wet = eouts[0, 4 * BLOCK_SIZE:].numpy()
    dry = torch.cat(eblocks, dim=1)[0, 4 * BLOCK_SIZE:].numpy()
    hf_cut = 10.0 * np.log10(band_energy(wet, 6500, 9000) /
                              (band_energy(dry, 6500, 9000) + 1e-12))
    assert hf_cut < -3.0, f"HF ess band must duck (got {hf_cut:+.2f} dB)"
    assert node.get_telemetry()["gr_db"] < -2.0

    node.start()
    vblocks = saw_blocks(220.0, 16)
    vouts = torch.cat([process_block(node, b) for b in vblocks], dim=1)
    wv = vouts[0, 4 * BLOCK_SIZE:].numpy()
    dv = torch.cat(vblocks, dim=1)[0, 4 * BLOCK_SIZE:].numpy()
    overall = 10.0 * np.log10(np.sum(wv ** 2) / (np.sum(dv ** 2) + 1e-12))
    assert abs(overall) < 0.5, f"vowel body must pass (got {overall:+.2f} dB)"
    lf = 10.0 * np.log10(band_energy(wv, 0, 3000) / (band_energy(dv, 0, 3000) + 1e-12))
    assert abs(lf) < 0.3, f"LF band must be untouched (got {lf:+.2f} dB)"


def test_deesser_mid_mix_blends_comb_free():
    """mix=0.5 lands between dry-reconstructed and full-wet HF energy (no
    crossover notch: blending stays inside the phase-consistent domain)."""
    eblocks = ess_blocks(16)

    def run(mix):
        node = make_node()
        set_params(node, frequency=6500.0, threshold=-30.0, depth=6.0, mix=mix)
        return torch.cat([process_block(node, b) for b in eblocks], dim=1)

    dry_mix = run(0.0)
    mid = run(0.5)
    wet = run(1.0)
    seg = slice(4 * BLOCK_SIZE, None)
    e_dry = band_energy(dry_mix[0, seg].numpy(), 6500, 9000)
    e_mid = band_energy(mid[0, seg].numpy(), 6500, 9000)
    e_wet = band_energy(wet[0, seg].numpy(), 6500, 9000)
    assert e_wet < e_mid < e_dry, "mix must interpolate HF energy monotonically"
    assert torch.isfinite(mid).all()


def test_deesser_no_false_trigger_on_lf_thump():
    """Loud low-frequency content must not engage HF reduction, and overall
    level is preserved (LR4 phase shift aside, energy is untouched)."""
    node = make_node()
    set_params(node, threshold=-30.0, depth=6.0, mix=1.0)
    n = np.arange(12 * BLOCK_SIZE, dtype=np.float64)
    thump = (0.5 * np.sin(2.0 * np.pi * 80.0 * n / SAMPLE_RATE)).astype(np.float32)
    blocks = [torch.from_numpy(np.tile(thump[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE],
                                        (CHANNELS, 1)).copy())
              for i in range(12)]
    outs = torch.cat([process_block(node, b) for b in blocks], dim=1)
    w = outs[0].numpy().astype(np.float64)
    overall = 10.0 * np.log10(np.sum(w ** 2) / (np.sum(thump.astype(np.float64) ** 2) + 1e-12))
    assert abs(overall) < 0.05
    assert node.get_telemetry()["gr_db"] == pytest.approx(0.0, abs=0.5)


def test_deesser_listen_solos_detector_band():
    node = make_node()
    set_params(node, frequency=6500.0, listen=True, mix=1.0)
    vblocks = saw_blocks(220.0, 8)
    outs = torch.cat([process_block(node, b) for b in vblocks], dim=1)
    wet = outs[0].numpy()
    dry = torch.cat(vblocks, dim=1)[0].numpy()
    # LF vowel body is gone from the solo; HF sizzle remains audible.
    lf_ratio = band_energy(wet, 0, 2000) / (band_energy(dry, 0, 2000) + 1e-12)
    assert 10.0 * np.log10(lf_ratio + 1e-12) < -20.0
    assert band_energy(wet, 6000, 9000) > 0.0


def test_deesser_attack_release_ballistics():
    """Fast attack catches short bursts; release recovers to zero GR."""
    node = make_node()
    set_params(node, threshold=-30.0, depth=6.0, attack_ms=1.0,
               release_ms=60.0, mix=1.0)
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    for _ in range(4):
        process_block(node, silence)
    for b in ess_blocks(4):
        process_block(node, b)
    assert node.get_telemetry()["gr_db"] < -2.0, "attack must engage within a few blocks"
    for _ in range(60):
        process_block(node, silence)
    assert node.get_telemetry()["gr_db"] == pytest.approx(0.0, abs=0.2)


def test_deesser_gr_reports_block_minimum():
    """Metering captures a mid-block spike: ess in the first quarter of a
    block with fast release still reads near-full reduction, even though
    the end-of-block value has already released."""
    node = make_node()
    set_params(node, frequency=6500.0, threshold=-30.0, depth=6.0,
               attack_ms=1.0, release_ms=10.0, mix=1.0)
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    for _ in range(4):
        process_block(node, silence)
    eblocks = ess_blocks(2)
    burst = eblocks[0].clone()
    burst[:, BLOCK_SIZE // 4:] = 0.0  # ess only in the first quarter
    process_block(node, burst)
    assert node.get_telemetry()["gr_db"] < -4.0


def test_deesser_mono_to_stereo_identical():
    node = make_node()
    set_params(node, mix=1.0)
    blocks = ess_blocks(8)
    for b in blocks[:4]:
        process_block(node, b)
    mono = blocks[4][0:1].clone()
    node.inp.get_tensor = lambda b=mono: b
    node.process()
    out = node.out.buffer
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], out[1])


def test_deesser_reset_clears_state():
    node = make_node()
    set_params(node, threshold=-30.0, depth=6.0, mix=1.0)
    for b in ess_blocks(8):
        process_block(node, b)
    assert node.get_telemetry()["gr_db"] < -1.0
    node.start()
    silence = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, silence)
    assert float(out.abs().max()) == 0.0
    assert node.get_telemetry()["gr_db"] == pytest.approx(0.0)


def test_deesser_extremes_stability():
    node = make_node()
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    for freq in (3000.0, 9000.0):
        for depth in (0.0, 12.0):
            for listen in (False, True):
                set_params(node, frequency=freq, threshold=-40.0, depth=depth,
                           attack_ms=0.1, release_ms=300.0, listen=listen, mix=1.0)
                out = process_block(node, blk)
                assert torch.isfinite(out).all()
                assert float(out.abs().max()) < 8.0


def test_deesser_save_load_roundtrip():
    node = make_node()
    set_params(node, frequency=7000.0, threshold=-24.0, depth=8.0,
               attack_ms=0.5, release_ms=120.0, listen=True, mix=0.8)
    snapshot = node.to_dict()
    node2 = make_node()
    node2.load_state(snapshot)
    for k in ("frequency", "threshold", "depth", "attack_ms", "release_ms",
              "listen", "mix"):
        assert node2.params[k].value == pytest.approx(node.params[k].value)


def test_deesser_zero_python_allocation():
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
    assert growth - before < 64 * 1024, f"net Python allocation {growth - before} bytes"


def test_deesser_native_processing_budget():
    node = make_node()
    set_params(node, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    for _ in range(10):
        process_block(node, blk)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        process_block(node, blk)
    dt = (time.perf_counter() - t0) / n * 1000.0
    assert dt < 0.5, f"mean process() {dt:.3f} ms exceeds budget"
