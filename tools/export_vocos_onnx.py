"""
tools/export_vocos_onnx.py
Exports charactr/vocos-mel-24khz to ONNX, checks numerical parity,
measures exact iSTFT hop length, and enforces the 10.67 ms CPU deadline.

Export-only dependencies (NOT runtime deps, NOT in environment.yml):
    conda run -n anode-dev pip install vocos onnx onnxscript

Notes:
- Vocos's iSTFT head builds a complex64 spectrogram (mag * (cos p + i sin p))
  and runs torch.istft/torch.fft.irfft on it. Neither the legacy TorchScript
  exporter (``RuntimeError: Unknown number type: complex``) nor the dynamo
  exporter (``No ONNX function found for prims.convert_element_type`` with a
  complex-valued input) can translate complex ops, so the head is swapped for
  ExportISTFTHead before export: identical math (real/imag split, real-IDFT
  matrix multiply, Hann window, fold overlap-add, divide by OLA(window^2),
  center trim) using float32 real ops only. Parity vs torch.istft is ~5e-6.
- The wrapper calls backbone+head directly instead of Vocos.decode():
  decode() runs under @torch.inference_mode and torch.export's
  functionalization fails on the resulting inference tensors
  (``Cannot set version_counter for inference tensor``).
- Shapes are exported static (1, 100, 8): torch.export's shape-env rejects
  dynamic batch/frames for this model (backbone constraints), and the runtime
  node always feeds exactly (1, 100, CONTEXT_FRAMES=8) anyway.
- The dynamo graph passes ``onnx.checker`` but ONNX Runtime rejects it:
  onnxscript emits ``ScatterND`` with int32 index tensors while the ONNX
  spec (and ORT) require int64. This is fixed by a graph post-processing
  pass (``fix_scatternd_indices``) that inserts ``Cast`` nodes to int64
  in front of every non-int64 ``ScatterND`` indices input.
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


class ExportISTFTHead(torch.nn.Module):
    """Real-valued drop-in for vocos.heads.ISTFTHead (export only).

    The stock head builds a complex64 spectrum (mag * (cos p + i sin p)) and
    runs torch.istft on it; no ONNX exporter translates complex dtypes, so
    this replays the identical math with float32 real ops: split magnitude /
    phase into real/imag planes, inverse-DFT each frame with a precomputed
    real basis (matmul), Hann-window, fold overlap-add, divide by
    OLA(window^2), center-trim n_fft//2 per side. Matches torch.istft
    (center=True) to ~5e-6. The trained ``out`` Linear is reused unchanged.
    """

    def __init__(self, orig_head):
        super().__init__()
        self.out = orig_head.out
        istft = orig_head.istft
        self.n_fft = int(istft.n_fft)
        self.hop_length = int(istft.hop_length)
        self.win_length = int(istft.win_length)
        window = istft.window.detach().clone().float()
        self.register_buffer("window", window)
        k = torch.arange(self.n_fft, dtype=torch.float32)
        ang = 2.0 * torch.pi * torch.outer(k, k) / float(self.n_fft)
        self.register_buffer("dft_cos", torch.cos(ang))
        self.register_buffer("dft_sin", torch.sin(ang))

    def forward(self, x):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        s_re = mag * torch.cos(p)
        s_im = mag * torch.sin(p)
        return self.real_istft(s_re, s_im)

    def real_istft(self, s_re, s_im):
        # Full conjugate-symmetric spectrum as separate real/imag planes.
        fr = torch.cat([s_re, s_re[:, 1:-1, :].flip(1)], dim=1)
        fi = torch.cat([s_im, (-s_im[:, 1:-1, :]).flip(1)], dim=1)
        fr_t = fr.transpose(1, 2)
        fi_t = fi.transpose(1, 2)
        frame_t = torch.matmul(fr_t, self.dft_cos) - torch.matmul(fi_t, self.dft_sin)
        ifft = frame_t.transpose(1, 2) / float(self.n_fft)
        ifft = ifft * self.window[None, :, None]
        t_frames = s_re.shape[2]
        output_size = (t_frames - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )
        y = y[:, 0, 0, :]
        w2 = self.window.square().expand(t_frames, -1).transpose(0, 1).unsqueeze(0)
        env = torch.nn.functional.fold(
            w2, output_size=(1, output_size),
            kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )
        env = env[:, 0, 0, :]
        y = y / env
        pad = self.n_fft // 2
        return y[:, pad:output_size - pad]


def _elem_type(model, name):
    """Return the element type of a graph/initializer or None if unknown."""
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        if init.name == name:
            return init.data_type
    return None


def fix_scatternd_indices(model):
    """Insert Cast to int64 in front of ScatterND indices inputs.

    The dynamo exporter (onnxscript) emits ScatterND with int32 indices in
    the iSTFT overlap-add decomposition; the ONNX spec requires int64 and
    ONNX Runtime rejects the graph otherwise.
    """
    INT64 = onnx.TensorProto.INT64
    produced = set()
    for node in model.graph.node:
        for out in node.output:
            produced.add(out)

    inserted = 0
    # Copy the list: new Cast nodes are appended while iterating.
    for node in list(model.graph.node):
        if node.op_type != "ScatterND":
            continue
        idx_name = node.input[1]
        et = _elem_type(model, idx_name)
        if et == INT64 or et is None:
            continue
        cast_name = f"{idx_name}_as_int64"
        if cast_name in produced:
            # Already cast (shared index tensor); just rewire if needed.
            pass
        else:
            cast = onnx.helper.make_node(
                "Cast", [idx_name], [cast_name], to=INT64,
                name=f"cast_indices_int64_{inserted}",
            )
            # Insert immediately before the consumer to keep the node list
            # topologically sorted.
            pos = list(model.graph.node).index(node)
            model.graph.node.insert(pos, cast)
            produced.add(cast_name)
            inserted += 1
        node.input[1] = cast_name
    return inserted


def _slice_node(name, data, starts, ends, steps, axes, output):
    return onnx.helper.make_node(
        "Slice", [data, starts, ends, axes, steps], [output], name=name)


def fix_dft_irfft(model):
    """Rewrite DFT(inverse=1, onesided=1) (IRFFT) into an ORT-supported form.

    ONNX Runtime does not implement the IRFFT case of the DFT operator
    (``onesided=1, inverse=1`` -> ShapeInferenceError). Rebuild it from
    supported ops: reconstruct the full conjugate-symmetric spectrum
    (Slice + conj + Concat), run DFT(inverse=1, onesided=0), and Slice
    the real part. Output values are identical to IRFFT.
    """
    g = model.graph
    init_map = {i.name: i for i in g.initializer}
    replaced = 0
    for node in list(g.node):
        if node.op_type != "DFT":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("inverse") != 1 or attrs.get("onesided") != 1:
            continue
        x, dft_len_name = node.input[0], node.input[1]
        axis_name = node.input[2] if len(node.input) > 2 else None
        out = node.output[0]

        def add_init(arr, name):
            if name not in init_map:
                t = onnx.numpy_helper.from_array(arr, name)
                g.initializer.append(t)
                init_map[name] = t
            return name

        n_fft = int(onnx.numpy_helper.to_array(init_map[dft_len_name]))
        if axis_name is not None:
            axis_val = int(onnx.numpy_helper.to_array(init_map[axis_name]))
        else:
            axis_val = 1  # spec default
        f_half = n_fft // 2 + 1

        def iv(name, arr):
            return add_init(np.array(arr, dtype=np.int64), name)

        p = f"vocos_dft{replaced}"
        head = f"{p}_head"
        tail_rev = f"{p}_tail"
        tail_conj = f"{p}_tail_conj"
        full_spec = f"{p}_full_spec"
        ifft_out = f"{p}_ifft"

        # head = x[0 : f_half] along the DFT axis (all onesided bins)
        head_node = _slice_node(
            head + "_n", x, iv(f"{head}_s", [0]), iv(f"{head}_e", [f_half]),
            iv(f"{head}_st", [1]), iv(f"{head}_ax", [axis_val]), head)
        # tail = conj(reverse(x[1 : f_half - 1])) -> bins f_half .. n_fft-1
        tail_node = _slice_node(
            tail_rev + "_n", x, iv(f"{tail_rev}_s", [f_half - 2]),
            iv(f"{tail_rev}_e", [0]), iv(f"{tail_rev}_st", [-1]),
            iv(f"{tail_rev}_ax", [axis_val]), tail_rev)
        conj_mul = add_init(
            np.array([1.0, -1.0], dtype=np.float32), f"{p}_conj_mul")
        conj_node = onnx.helper.make_node(
            "Mul", [tail_rev, conj_mul], [tail_conj], name=f"{p}_conj")
        # full spectrum along the DFT axis -> (..., n_fft, 2)
        concat_node = onnx.helper.make_node(
            "Concat", [head, tail_conj], [full_spec], axis=axis_val,
            name=f"{p}_concat")
        # inverse DFT over the full spectrum (ORT-supported)
        new_in = [full_spec, dft_len_name] + ([axis_name] if axis_name else [])
        dft_node = onnx.helper.make_node(
            "DFT", new_in, [ifft_out], name=node.name + "_full", inverse=1)
        # take the real part (last dim) and feed the original consumers
        real = f"{p}_real"
        real_node = _slice_node(
            real + "_n", ifft_out, iv(f"{real}_s", [0]), iv(f"{real}_e", [1]),
            iv(f"{real}_st", [1]), iv(f"{real}_ax", [-1]), out)

        # Insert the replacement subgraph at the original node's position so
        # the node list stays topologically sorted.
        new_nodes = [head_node, tail_node, conj_node, concat_node, dft_node, real_node]
        pos = list(g.node).index(node)
        for i, nn in enumerate(new_nodes):
            g.node.insert(pos + i, nn)
        g.node.remove(node)  # now at pos + len(new_nodes)
        replaced += 1
    return replaced


def export_and_benchmark(context_frames=8, stride=1):
    """Export one model variant. Live uses 8 static frames (stride 1); the
    studio quality mode needs a 24-frame window (stride 4), which must be a
    separate static-shape export (torch.export rejects dynamic frames)."""
    os.makedirs("models", exist_ok=True)
    suffix = "" if context_frames == 8 else f"_w{context_frames}"
    onnx_path = f"models/vocos_mel_24k{suffix}.onnx"

    print(f"1. Loading pretrained Vocos model (charactr/vocos-mel-24khz)...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()
    # Swap the complex iSTFT head for a real-valued equivalent (see class
    # note below); the trained Linear weights are reused unchanged.
    vocos.head = ExportISTFTHead(vocos.head)

    class VocosWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, mel):
            # mel shape: (batch=1, n_mels=100, frames=context_frames)
            # Bypass Vocos.decode() (@torch.inference_mode): inference
            # tensors break torch.export functionalization.
            return self.model.head(self.model.backbone(mel))

    wrapper = VocosWrapper(vocos)
    dummy_mel = torch.randn(1, 100, context_frames, dtype=torch.float32)

    print("2. Exporting to ONNX (dynamo exporter, real-valued iSTFT head)...")
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        onnx_path,
        input_names=["mel"],
        output_names=["audio"],
        dynamo=True,
    )

    print("3. Validating ONNX graph...")
    model = onnx.load(onnx_path)
    n_casts = fix_scatternd_indices(model)
    n_dfts = fix_dft_irfft(model)
    if n_casts or n_dfts:
        print(f"   Patched {n_casts} ScatterND indices input(s) to int64 "
              f"and rewrote {n_dfts} IRFFT DFT node(s).")
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)

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
    # Strided inference runs once per `stride` blocks, so its budget is
    # stride block periods (studio: 4 x 10.67 ms).
    deadline_ms = (512 / 48000.0) * 1000.0 * stride
    print(f"Latency: Median = {median_ms:.2f} ms, p95 = {p95_ms:.2f} ms (Deadline: {deadline_ms:.2f} ms)")

    if p95_ms > deadline_ms:
        raise RuntimeError(
            f"Benchmark gate failed: p95 latency ({p95_ms:.2f} ms) exceeds "
            f"{deadline_ms:.2f} ms deadline! "
            "Increase infer-stride or reduce context frames before proceeding."
        )
    print("Benchmark gate PASSED.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    export_and_benchmark(args.context, args.stride)
