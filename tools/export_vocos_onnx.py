"""
tools/export_vocos_onnx.py
Exports charactr/vocos-mel-24khz to ONNX, checks numerical parity,
measures exact iSTFT hop length, and enforces the 10.67 ms CPU deadline.

Export-only dependencies (NOT runtime deps, NOT in environment.yml):
    conda run -n anode-dev pip install vocos onnx onnxscript

Notes:
- The legacy TorchScript exporter cannot be used here: Vocos's iSTFT head
  uses complex tensors and fails with
  ``RuntimeError: Unknown number type: complex``. This script therefore
  uses the dynamo-based exporter (``dynamo=True``, requires ``onnxscript``).
- The dynamo graph passes ``onnx.checker`` but older ONNX Runtime builds
  reject it (``ScatterND`` with int32 indices, INVALID_GRAPH). If ORT load
  fails, retry on the faster machine with a current onnxruntime before
  debugging the model itself.
- Native-torch reference (i3-6006U, 1 thread): median ~45 ms / p95 ~57 ms
  per 8-frame inference vs the 10.67 ms hop-1 deadline, and 8 input frames
  decode to 1792 samples (7 hops), not 2048. Hop-1 realtime needs faster
  hardware; otherwise move to infer-stride > 1 (see node LATENCY_SAMPLES).
"""
import os
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from vocos import Vocos


def export_and_benchmark():
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/vocos_mel_24k.onnx"

    print("1. Loading pretrained Vocos model (charactr/vocos-mel-24khz)...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()

    class VocosWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, mel):
            # mel shape: (batch=1, n_mels=100, frames)
            return self.model.decode(mel)

    wrapper = VocosWrapper(vocos)
    context_frames = 8
    dummy_mel = torch.randn(1, 100, context_frames, dtype=torch.float32)

    print("2. Exporting to ONNX (dynamo exporter; legacy fails on complex istft)...")
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        onnx_path,
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch", 2: "frames"},
            "audio": {0: "batch", 1: "samples"},
        },
        dynamo=True,
    )

    print("3. Validating ONNX graph...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    print("4. Verifying numerical parity (PyTorch vs ONNX Runtime)...")
    with torch.no_grad():
        py_out = wrapper(dummy_mel).numpy()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"mel": dummy_mel.numpy()})[0]

    max_err = float(np.max(np.abs(py_out - ort_out)))
    print(f"Max absolute error: {max_err:.2e}")
    assert max_err < 1e-4, f"Parity mismatch: {max_err}"

    out_samples = ort_out.shape[-1]
    print(f"Model output length for {context_frames} frames: {out_samples} samples")

    print("5. Benchmarking CPU inference latency (50 iterations)...")
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, {"mel": dummy_mel.numpy()})
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(times))
    p95_ms = float(np.percentile(times, 95))
    deadline_ms = (512 / 48000.0) * 1000.0  # 10.67 ms
    print(f"Latency: Median = {median_ms:.2f} ms, p95 = {p95_ms:.2f} ms (Deadline: {deadline_ms:.2f} ms)")

    if p95_ms > deadline_ms:
        raise RuntimeError(
            f"Benchmark gate failed: p95 latency ({p95_ms:.2f} ms) exceeds 10.67 ms deadline! "
            "Increase infer-stride or reduce context frames before proceeding."
        )
    print("Benchmark gate PASSED.")


if __name__ == "__main__":
    export_and_benchmark()
