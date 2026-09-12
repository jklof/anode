"""Latency modes for VocalTransformer (Studio/Live/Ultra-Low).

Mode table (48 kHz):
  0 Studio:    N=2048, H=256, L=9216, +-24 st
  1 Live:      N=1024, H=128, L=2560, +-12 st
  2 Ultra-Low: N=1024, H=128, L=2048, +-7 st
"""

import gc
import json
import os
import time
import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE
from core import Graph

MODES = {0: 9216, 1: 2560, 2: 2048}


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


def set_mode(node, mode):
    """Switch latency mode and run one block so the staged value reaches
    native DSP (telemetry reads native state)."""
    node.params["latency_mode"].set(mode)
    node.sync()
    process_block(node, torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE))


def saw_blocks(freq, n_blocks, n_harm=30, amp=0.3):
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


def autocorr_pitch(x, fmin=60.0, fmax=1200.0):
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(SAMPLE_RATE / fmax)
    hi = min(int(SAMPLE_RATE / fmin), len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SAMPLE_RATE / lag


def resonant_stack_blocks(n_blocks, base_hz=100.0, res_hz=2000.0, amp=0.4):
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


def resonant_com(out_buffer, lo=1200.0, hi=3200.0, nfft=16384):
    y = out_buffer[0].numpy().astype(np.float64)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / SAMPLE_RATE)
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(freqs[m] * spec[m]) / np.sum(spec[m]))


def test_latency_query_matches_mode():
    node = make_node()
    assert node.get_telemetry()["latency_samples"] == 9216
    for mode, expect in MODES.items():
        set_mode(node, mode)
        telem = node.get_telemetry()
        assert telem["latency_samples"] == expect, f"mode {mode}"
        assert telem["latency_ms"] == pytest.approx(expect / SAMPLE_RATE * 1000.0, abs=0.01)


def test_live_mode_pitch_accuracy():
    """220 Hz saw +12 st in Live mode lands at 440 Hz after the short prime."""
    node = make_node()
    set_params(node, pitch_shift=12.0, formant_shift=0.0, mix=1.0)
    set_mode(node, 1)
    settle = MODES[1] // BLOCK_SIZE + 6
    blocks = saw_blocks(220.0, settle + 16)
    for b in blocks[:settle]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[settle:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 440.0) < 5.0


def test_ultra_mode_pitch_and_range_clamp():
    """Ultra-Low: +7 st lands near 329.6 Hz; +24 st is clamped into range
    (finite, bounded, no stale reads past the 2048-sample budget)."""
    node = make_node()
    set_params(node, pitch_shift=7.0, mix=1.0)
    set_mode(node, 2)
    settle = MODES[2] // BLOCK_SIZE + 6
    blocks = saw_blocks(220.0, settle + 16)
    for b in blocks[:settle]:
        process_block(node, b)
    tail = torch.cat(
        [process_block(node, b) for b in blocks[settle:]], dim=1
    )[0].numpy().astype(np.float64)
    assert abs(autocorr_pitch(tail) - 329.63) < 8.0

    set_params(node, pitch_shift=24.0)
    for b in saw_blocks(220.0, settle + 8):
        out = process_block(node, b)
        assert torch.isfinite(out).all()
        assert float(out.abs().max()) < 8.0


def test_live_mode_formant_direction():
    """gender=+1 shifts spectral COM up in Live mode (N=1024, lifter 24)."""
    n_blocks = MODES[1] // BLOCK_SIZE + 12
    blocks = resonant_stack_blocks(n_blocks)

    def measure_com(gender):
        node = make_node()
        set_params(node, pitch_shift=0.0, formant_shift=0.0,
                   gender_morph=gender, mix=1.0)
        set_mode(node, 1)
        for b in blocks:
            process_block(node, b)
        return resonant_com(node.out.buffer)

    com_neut = measure_com(0.0)
    assert abs(com_neut - 2000.0) < 250.0, f"live neutral COM off: {com_neut:.1f}"
    assert measure_com(1.0) > com_neut + 100.0


def test_f0_floor_grace_in_live_mode():
    """80 Hz material below the Live F0 floor degrades gracefully: finite,
    bounded output, no NaN/Inf."""
    node = make_node()
    set_params(node, pitch_shift=0.0, mix=1.0)
    set_mode(node, 1)
    for b in saw_blocks(80.0, MODES[1] // BLOCK_SIZE + 12):
        out = process_block(node, b)
        assert torch.isfinite(out).all()
        assert float(out.abs().max()) < 8.0


def test_mid_stream_mode_switch_stability():
    """Switching modes mid-stream stays finite and recovers energy after
    one latency cycle (configure_mode clears transient state)."""
    node = make_node()
    set_params(node, pitch_shift=0.0, mix=1.0)
    blocks = saw_blocks(220.0, 60)
    for b in blocks[:20]:
        process_block(node, b)
    set_mode(node, 1)
    outs = [process_block(node, b) for b in blocks[20:]]
    assert all(torch.isfinite(o).all() for o in outs)
    tail = torch.cat(outs[-8:], dim=1)
    assert float(tail.abs().max()) > 0.01, "no recovery after mode switch"
    set_mode(node, 0)
    outs0 = [process_block(node, b) for b in blocks[20:]]
    assert all(torch.isfinite(o).all() for o in outs0)
    assert float(torch.cat(outs0[-8:], dim=1).abs().max()) > 0.01


def test_zero_python_heap_allocation_across_modes():
    node = make_node()
    set_params(node, mix=1.0)
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    for mode in (0, 1):
        set_mode(node, mode)
        process_block(node, blk)  # warm-up
        gc.collect()
        tracemalloc.start()
        before, _ = tracemalloc.get_traced_memory()
        for _ in range(50):
            process_block(node, blk)
        growth, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert growth - before < 64 * 1024, \
            f"net Python allocation {growth - before} bytes in mode {mode}"


def test_save_load_latency_mode_roundtrip():
    node = make_node()
    set_params(node, latency_mode=1, pitch_shift=5.0, mix=0.7)
    snapshot = node.to_dict()
    node2 = make_node()
    node2.load_state(snapshot)
    assert node2.params["latency_mode"].value == 1
    set_mode(node2, 1)  # push staged value native-side
    assert node2.get_telemetry()["latency_samples"] == 2560


def test_shootout_patch_regression():
    """vocal_shootout.json loads with all engines defaulting to Studio."""
    path = os.path.join(os.path.dirname(__file__), "..", "patches",
                        "vocal_shootout.json")
    plugin_system.load_plugins("plugins")
    with open(path) as f:
        data = json.load(f)
    graph = Graph()
    for n_data in data.get("nodes", []):
        cls = plugin_system.NODE_REGISTRY.get(n_data.get("type"))
        assert cls is not None, f"unknown node type {n_data.get('type')}"
        node = cls(n_data.get("name", ""))
        node.id = n_data["id"]
        graph.add_node(node)
        node.load_state(n_data)
    for c in data.get("connections", []):
        assert graph.connect(c["src_id"], c["src_port"],
                             c["dst_id"], c["dst_port"]), f"connect failed: {c}"
    for vid in ("voxClassic", "voxTuned"):
        assert graph.node_map[vid].params["latency_mode"].value == 0
    assert graph.node_map["voxSmith"].params["latency_mode"].value == 0


def _bench_node(mode, blk, n=40):
    """Per-block ms for one mode (fresh node per call). Note the Live win is
    small by construction — FFT work per second scales as (sr/H)*N*logN with
    H=N/8, i.e. ~9% less, plus fixed per-block overhead — so timing guards
    against gross regressions (e.g. Live accidentally running 2048-pt FFTs
    would cost ~2x), not a 2x speedup. Live mode's real win is latency."""
    node = make_node()
    set_params(node, mix=1.0)
    set_mode(node, mode)
    for _ in range(10):
        process_block(node, blk)
    t0 = time.perf_counter()
    for _ in range(n):
        process_block(node, blk)
    return (time.perf_counter() - t0) / n * 1000.0


def test_cpu_ratio_live_vs_studio_interleaved():
    """Interleaved Studio/Live timing pairs share load conditions, so the
    ratio is robust against machine drift that breaks back-to-back bests.
    Median pair ratio must clear a loose 5% bar."""
    blk = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE) * 0.3
    ratios = []
    for _ in range(5):
        studio = _bench_node(0, blk)
        live = _bench_node(1, blk)
        ratios.append(live / studio)
    med = float(np.median(ratios))
    assert med < 0.95, f"median live/studio ratio {med:.3f} ({ratios})"
