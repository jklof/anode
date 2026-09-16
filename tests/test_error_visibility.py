"""Error visibility (AGENTS.md: defer reporting to control/UI) + Vocos audio path.

Covers two gaps found while debugging a silent VocosResynthesizer:
* the engine cleared every node's error_msg after each successful block, so
  async (NRT) failures never survived to a snapshot, and no snapshot was
  emitted when errors changed -> the UI showed nothing;
* the node's mel filterbank did not match its checkpoint (Slaney vs
  unnormalized HTK), attenuating mel ~34x so the vocoder read silence.
"""
import queue
import time

import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, Node
from core import Engine


def _drain_snapshots(engine):
    snaps = []
    while True:
        try:
            msg = engine.output_queue.get_nowait()
        except queue.Empty:
            return snaps
        if isinstance(msg, dict) and msg.get("type") == "graph_update":
            snaps.append(msg)


def _wait_for(pred, timeout=5.0, desc="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {desc}")


def test_async_error_survives_successful_blocks():
    """An error set outside process() (e.g. NRT load failure) must not be
    wiped by subsequent successful audio blocks."""
    plugin_system.load_plugins("plugins")
    engine = Engine()
    node = plugin_system.NODE_REGISTRY["Gain"]()
    engine.start()
    try:
        engine.push_command(("add", node, "g1", (0, 0), None))
        assert _wait_for(lambda: engine._active_plan.nodes, desc="plan compile")
        node.error_msg = "async boom"
        time.sleep(0.3)  # ~28 blocks of successful process()
        assert node.error_msg == "async boom"
        assert "g1" not in engine._process_error_ids
    finally:
        engine.stop()


def test_process_error_clears_on_recovery():
    """Errors raised by process() itself are still set and then cleared once
    the node recovers (existing loop behavior preserved)."""
    plugin_system.load_plugins("plugins")
    engine = Engine()

    class Flaky(Node):
        def __init__(self):
            super().__init__()
            self.out = self.add_output("out")
            self.calls = 0

        def process(self):
            self.calls += 1
            self.out.buffer.zero_()
            if self.calls < 10:
                raise RuntimeError("transient")

    node = Flaky()
    engine.start()
    try:
        engine.push_command(("add", node, "f1", (0, 0), None))
        assert _wait_for(lambda: node.calls >= 1, desc="first failing block")
        assert _wait_for(lambda: node.error_msg == "transient", desc="error set")
        assert "f1" in engine._process_error_ids
        assert _wait_for(lambda: node.error_msg is None, desc="error recovery")
        assert "f1" not in engine._process_error_ids
    finally:
        engine.stop()


def test_snapshot_emitted_on_error_change():
    """The UI must be told about error transitions without waiting for a
    structural change."""
    plugin_system.load_plugins("plugins")
    engine = Engine()
    node = plugin_system.NODE_REGISTRY["Gain"]()
    engine.start()
    try:
        engine.push_command(("add", node, "g2", (0, 0), None))
        assert _wait_for(lambda: engine._active_plan.nodes, desc="plan compile")
        _drain_snapshots(engine)
        node.error_msg = "nrt load failed"
        assert _wait_for(
            lambda: any(n["id"] == "g2" and n.get("error") == "nrt load failed"
                        for s in _drain_snapshots(engine) for n in s["nodes"]),
            desc="error snapshot")
        node.error_msg = None
        assert _wait_for(
            lambda: any(n["id"] == "g2" and n.get("error") is None
                        for s in _drain_snapshots(engine) for n in s["nodes"]),
            desc="recovery snapshot")
    finally:
        engine.stop()


def test_vocos_filterbank_matches_checkpoint():
    """The runtime mel filterbank must equal charactr/vocos-mel-24khz's own
    MelScale (norm=None, htk). Slaney normalization attenuated mel ~34x and
    the vocoder read every input as silence."""
    vocos_pkg = pytest.importorskip("vocos")
    from plugins.vocos_resynthesizer import build_vocos_mel_filterbank
    vocos = vocos_pkg.Vocos.from_pretrained("charactr/vocos-mel-24khz")
    ms = vocos.feature_extractor.mel_spec.mel_scale
    assert ms.n_mels == 100
    assert ms.norm is None
    assert ms.mel_scale == "htk"
    mine = build_vocos_mel_filterbank()
    assert mine.shape == (100, 513)
    # Float32 construction-order noise only; the Slaney bug was ~34x off.
    max_diff = float((mine - ms.fb.T.contiguous()).abs().max())
    assert max_diff < 1e-4, max_diff


def test_vocos_wet_path_produces_audio():
    """End-to-end wet path with voice-like input must produce audible (not
    digital-silence) output. Needs the exported model; slow (~3 s)."""
    import os
    path = "models/vocos_mel_24k.onnx"
    if not os.path.exists(path):
        pytest.skip("vocos ONNX model not exported")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    plugin_system.load_plugins("plugins")
    from base import DTYPE
    cls = plugin_system.NODE_REGISTRY["VocosResynthesizer"]
    node = cls()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    node._install_session(sess, path)

    node.start()  # no engine: falls back to a raw worker thread
    try:
        sr = 48000
        for hop_i in range(40):
            t = torch.arange(BLOCK_SIZE, dtype=DTYPE) + hop_i * BLOCK_SIZE
            secs = t / sr
            f0 = 110 * (1 + 0.01 * torch.sin(2 * torch.pi * 5 * secs))
            ph = 2 * torch.pi * torch.cumsum(f0 / sr, dim=0)
            buzz = sum(torch.sin(k * ph) / k for k in range(1, 41)) * 0.25
            blk = torch.stack([buzz, buzz]).float()
            node.inp.get_tensor = lambda b=blk: b
            node.process()
            time.sleep(0.005)
        assert _wait_for(
            lambda: max(float(np.abs(p).max()) for p in node.pool_out) > 1e-3,
            timeout=10.0, desc="wet audio in worker pool")
        assert float(node.buf_wet.abs().max()) > 0
    finally:
        node.stop()


def _voice_block(hop_i):
    from base import DTYPE
    sr = 48000
    t = torch.arange(BLOCK_SIZE, dtype=DTYPE) + hop_i * BLOCK_SIZE
    secs = t / sr
    f0 = 110 * (1 + 0.01 * torch.sin(2 * torch.pi * 5 * secs))
    ph = 2 * torch.pi * torch.cumsum(f0 / sr, dim=0)
    buzz = sum(torch.sin(k * ph) / k for k in range(1, 41)) * 0.25
    return torch.stack([buzz, buzz]).float()


def test_vocos_studio_path_produces_audio():
    """Studio quality (24-frame model) end-to-end: routes to the w24 model
    file and produces audible wet audio. Needs both exports; slow (~8 s)."""
    import os
    if not os.path.exists("models/vocos_mel_24k_w24.onnx"):
        pytest.skip("vocos studio ONNX model not exported")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY["VocosResynthesizer"]
    node = cls()
    node.params["quality"].set(1)
    node.params["quality"].sync()
    node.on_ui_param_change("quality")
    assert node.LATENCY_SAMPLES == 15 * BLOCK_SIZE + 62
    assert node._active_model_path() == "models/vocos_mel_24k_w24.onnx"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(node._active_model_path(), opts,
                                providers=["CPUExecutionProvider"])
    node._install_session(sess, node._active_model_path())

    node.start()  # no engine: falls back to a raw worker thread
    try:
        for hop_i in range(60):
            blk = _voice_block(hop_i)
            node.inp.get_tensor = lambda b=blk: b
            node.process()
            time.sleep(0.005)
        assert _wait_for(
            lambda: max(float(np.abs(p).max()) for p in node.pool_out) > 1e-3,
            timeout=15.0, desc="studio wet audio in worker pool")
        assert float(node.buf_wet.abs().max()) > 0
    finally:
        node.stop()
