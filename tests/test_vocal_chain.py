import json
import os

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE
from core import Graph

PATCH = os.path.join(os.path.dirname(__file__), "..", "patches",
                     "vocal_broadcast_chain.json")


def load_patch_graph(path):
    """Rebuild a Graph from a patch JSON file, mirroring Engine load logic
    (instantiate -> attach -> load_state -> connect)."""
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
    return graph


def run_graph(graph, blocks):
    order = graph.execution_order
    first = order[0]
    last = order[-1]
    outs = []
    for blk in blocks:
        first.inp.get_tensor = lambda b=blk: b
        for node in order:
            node.sync()
        for node in order:
            node.process()
        outs.append(last.outputs["out"].buffer.clone())
    return torch.cat(outs, dim=1)


def saw_blocks(freq, n_blocks, amp=0.3):
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = np.zeros_like(n)
    for h in range(1, 31):
        if freq * h >= SAMPLE_RATE / 2:
            break
        t += (1.0 / h) * np.sin(2.0 * np.pi * freq * h * n / SAMPLE_RATE)
    t = (t / np.max(np.abs(t)) * amp).astype(np.float32)
    return [torch.from_numpy(np.tile(t[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE],
                                      (CHANNELS, 1)).copy())
            for i in range(n_blocks)]


def test_vocal_chain_patch_loads():
    graph = load_patch_graph(PATCH)
    assert len(graph.nodes) == 5
    assert graph.node_map["vox"].params["mix"].value == pytest.approx(1.0)
    assert graph.node_map["deess"].params["frequency"].value == pytest.approx(6500.0)
    # Chain order follows the wiring.
    names = [n.id for n in graph.execution_order]
    assert names == ["hpf", "gate", "vox", "deess", "limit"]


def test_vocal_chain_neutral_end_to_end():
    """All stages neutral: a 220 Hz saw passes through finite, voiced, and
    roughly level-preserved (WORLD identity + transparent dynamics)."""
    graph = load_patch_graph(PATCH)
    n_blocks = 40
    blocks = saw_blocks(220.0, n_blocks)
    out = run_graph(graph, blocks)
    assert torch.isfinite(out).all()
    tail = out[:, 20 * BLOCK_SIZE:].numpy().astype(np.float64)
    assert float(np.abs(tail).max()) > 0.01, "chain went silent"
    dry = torch.cat(blocks, dim=1)[:, 20 * BLOCK_SIZE:].numpy().astype(np.float64)
    ratio_db = 10.0 * np.log10(np.sum(tail ** 2) / (np.sum(dry ** 2) + 1e-12))
    assert -6.0 < ratio_db < 6.0, f"chain level off by {ratio_db:.2f} dB"


def test_vocal_chain_stopgap_compressor_sidechain():
    """No-code de-esser stopgap: HF-filtered sidechain keying a Compressor
    reduces ess energy (broadband pumping is the documented compromise vs
    the split-band DeEsser node)."""
    plugin_system.load_plugins("plugins")
    R = plugin_system.NODE_REGISTRY
    hpf, comp = R["BiquadFilter"](), R["Compressor"]()
    hpf.params["type"].set(1)
    hpf.params["cutoff"].set(6000.0)
    comp.params["thresh"].set(-40.0)
    comp.params["ratio"].set(10.0)
    for n in (hpf, comp):
        n.sync()
        if hasattr(n, "_sync_params_to_cpp"):
            n._sync_params_to_cpp()

    rng = np.random.default_rng(5)
    n = np.arange(20 * BLOCK_SIZE, dtype=np.float64)
    white = rng.standard_normal(len(n))
    X = np.fft.rfft(white)
    fr = np.fft.rfftfreq(len(n), 1.0 / SAMPLE_RATE)
    X[(fr < 5000) | (fr > 8500)] = 0.0
    ess = (np.fft.irfft(X).astype(np.float32))
    ess = (ess / np.max(np.abs(ess)) * 0.3).astype(np.float32)
    blks = [torch.from_numpy(np.tile(ess[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE],
                                      (CHANNELS, 1)).copy())
            for i in range(20)]
    comp.inputs["sidechain"].connected_outputs = [hpf.outputs["out"]]
    outs = []
    for b in blks:
        hpf.inputs["in"].get_tensor = lambda b=b: b
        hpf.process()
        comp.inputs["in"].get_tensor = lambda b=b: b
        comp.process()
        outs.append(comp.outputs["out"].buffer.clone())
    out = torch.cat(outs, dim=1)
    assert torch.isfinite(out).all()

    def band(x, lo, hi):
        Xx = np.abs(np.fft.rfft(x.astype(np.float64)))
        f = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE)
        m = (f >= lo) & (f <= hi)
        return float(np.sum(Xx[m] ** 2))

    w = out[0, 4 * BLOCK_SIZE:].numpy()
    d = np.tile(ess, (2, 1))[0, 4 * BLOCK_SIZE:]
    cut = 10.0 * np.log10(band(w, 6000, 9000) / (band(d, 6000, 9000) + 1e-12))
    assert cut < -3.0, f"stopgap must duck ess HF (got {cut:+.2f} dB)"
