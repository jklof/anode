"""
plugins/vocos_resynthesizer.py
Real-time neural vocal resynthesizer / de-vocoder powered by Vocos ONNX.
Cleans up phase-vocoder metallic artifacts and restores natural vocal acoustics
with latency-compensated dry/wet crossfade.
"""
import os
import threading
import logging
import torch
import numpy as np

from base import Node, BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE, SPSCRingBuffer

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

VOCOS_SR = 24000
VOCOS_N_FFT = 1024
VOCOS_HOP = 256
VOCOS_N_MELS = 100
VOCOS_FMIN = 0.0
VOCOS_FMAX = 12000.0


def build_halfband_kernel():
    taps = 63
    cutoff = 11500.0 / 48000.0
    n = np.arange(taps) - (taps - 1) // 2
    h = np.sinc(2 * cutoff * n) * np.blackman(taps)
    h /= np.sum(h)
    return torch.from_numpy(h.astype(np.float32)).view(1, 1, taps)


def build_vocos_mel_filterbank():
    from torchaudio.functional import melscale_fbanks
    # Must match charactr/vocos-mel-24khz's own MelScale exactly
    # (norm=None, mel_scale="htk"): Slaney area-normalization attenuates the
    # mel energies ~34x, the backbone reads near-silence, and with mix=1.0
    # the node outputs digital silence despite healthy input.
    fb = melscale_fbanks(
        n_freqs=VOCOS_N_FFT // 2 + 1,
        f_min=VOCOS_FMIN,
        f_max=VOCOS_FMAX,
        n_mels=VOCOS_N_MELS,
        sample_rate=VOCOS_SR,
        norm=None,
        mel_scale="htk"
    )
    return fb.T.contiguous()


def _pop_newest(queue_in):
    """Pop all pending input indices, return the newest one.

    Freshness over completeness: if inference fell behind (ORT warmup, CPU
    jitter), skipping stale hops and processing the newest keeps wet aligned
    with dry. Without this, any transient lag becomes a permanent backlog:
    wet plays ever-more-stale while dry stays at design latency
    (partial-mix comb/slapback), and the queue eventually fills and drops
    (wet/dry chattering pops).
    """
    idx_in, ok = queue_in.try_pop()
    while ok:
        nxt, ok2 = queue_in.try_pop()
        if not ok2:
            break
        idx_in = nxt
    return idx_in, ok


class VocosResynthesizer(Node):
    category = "Effects"
    label = "Vocos Neural Resynthesizer"
    description = (
        "Real-time neural vocal resynthesizer powered by Vocos ONNX. Takes pitch/formant "
        "shifted vocal audio, extracts 100-band log-mel representations at 24 kHz, "
        "and reconstructs natural vocal waveforms via an iSTFT neural head. Eliminates "
        "metallic phase vocoder artifacts while preserving vocal articulation. "
        "Features latency-aligned dry/wet crossfade. Live mode (44 ms) keeps an "
        "audible frame-rate buzz on harmonic input; Studio mode (161 ms) is clean."
    )

    CONTEXT_FRAMES = 8
    LOOKAHEAD_FRAMES = 3
    # Inference runs every INFER_STRIDE input hops and emits that many taken
    # hops per run (4, 5, 6 -> frames newest-3, newest-2, newest-1). The worker
    # cannot sustain one full inference per 10.67 ms block on typical CPUs
    # (~7-14 ms measured all-in); striding brings the average pace to
    # ~2.5-5 ms/block with headroom to spare. LATENCY_SAMPLES accounts for
    # the oldest taken frame (see below); hops from one decode share its
    # ISTFT, so intra-burst seams are continuous.
    INFER_STRIDE = 3
    # Output-hop indices taken per inference (hop j <-> context frame j).
    TAKE_HOPS = (4, 5, 6)
    # Latency: (LOOKAHEAD_FRAMES + 1 queue buffer) * 512 + 62 FIR samples = 2110 samples (44.0 ms)
    LATENCY_SAMPLES = (LOOKAHEAD_FRAMES + 1) * BLOCK_SIZE + 62

    # Inference window presets (class defaults above == "live"). The Vocos
    # checkpoint phase-locks to short inference windows: 8-frame windows
    # imprint a ~93.75 Hz buzz on harmonic content (measured ratio ~0.3-0.5
    # vs signal). Clean output needs ~12+ frames of future context, which
    # costs latency: "studio" trades 44 ms -> 161 ms for ~-35 dB buzz.
    # latency_blocks covers oldest-taken-frame age + 1 queue buffer;
    # LATENCY_SAMPLES = latency_blocks * BLOCK_SIZE + 62 FIR.
    QUALITY_MODES = (
        {"label": "Live (44 ms)", "window": 8, "stride": 3,
         "take": (4, 5, 6), "latency_blocks": 4},
        {"label": "Studio (161 ms)", "window": 24, "stride": 4,
         "take": (9, 10, 11, 12), "latency_blocks": 15},
    )

    def __init__(self, name=""):
        super().__init__(name)

        # 1. Ports
        self.inp = self.add_input("in", help="Vocal audio to resynthesize (mono or stereo).")
        self.out = self.add_output("out", channels=CHANNELS, help="Resynthesized natural vocal audio.")

        # 2. Parameters
        self.add_file_param(
            "model_path", "models/vocos_mel_24k.onnx",
            filter="ONNX Models (*.onnx)",
            help="Pretrained Vocos 24kHz ONNX model file (Live quality)."
        )
        self.add_file_param(
            "studio_model_path", "models/vocos_mel_24k_w24.onnx",
            filter="ONNX Models (*.onnx)",
            help="24-frame Vocos ONNX model file (Studio quality). "
                 "Export with tools/export_vocos_onnx.py --context 24 --stride 4."
        )
        self.add_float_param(
            "mix", 1.0, 0.0, 1.0,
            help="Dry/wet crossfade (0 = latency-aligned input, 1 = neural resynthesis)."
        )
        self.add_menu_param(
            "quality", [m["label"] for m in self.QUALITY_MODES], 0,
            help="Live: 44 ms latency, audible frame-rate buzz on harmonic input. "
                 "Studio: 161 ms latency, clean (wide inference window). "
                 "Switching causes a brief dropout."
        )
        self._apply_quality()

        # 3. Audio Thread Scratchpads (Zero Allocations in process())
        self.in_ring_size = 16384  # Must be a power of 2 and multiple of 512
        self.in_ring_mask = self.in_ring_size - 1
        self.in_delay_ring = torch.zeros((CHANNELS, self.in_ring_size), dtype=DTYPE)
        self.write_pos = 0

        self.buf_mono = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self.buf_dry = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self.buf_wet = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

        # 4. SPSC Communication Queues & Pools (512-sample 48kHz blocks)
        self.queue_in = SPSCRingBuffer(capacity=32)
        self.queue_out = SPSCRingBuffer(capacity=32)
        self.pool_in = [np.zeros(BLOCK_SIZE, dtype=np.float32) for _ in range(32)]
        self.pool_out = [np.zeros(BLOCK_SIZE, dtype=np.float32) for _ in range(32)]
        self.pool_seq_in = 0
        self.pool_seq_out = 0

        # Telemetry metrics
        self.drops_in = 0
        self.drops_out = 0

        # 5. Lifecycles, Generation & Thread Safety
        self._load_epoch = 0
        self._loading_path = ""  # path currently loading (dedupes start vs load_state)
        self._worker_generation = 0
        self.session = None
        self.current_model_path = ""
        self.worker_thread = None
        self.stop_event = threading.Event()
        # Wakes the worker when the audio thread enqueues a hop (see
        # process()/start()/stop()). A fixed sleep here overshoots 2-15 ms
        # on coarse platform timers and makes the worker chronically late.
        self._wake_event = threading.Event()

    def _apply_quality(self):
        """(Re)compute instance inference config from the quality menu index.

        Runs on control/UI contexts (init, param change, load_state); the
        worker picks the values up live each iteration and rebuilds its mel
        context when the window size changes. Switching modes causes a brief
        dropout (dry delay jumps, wet context refills).
        """
        try:
            idx = int(self.params["quality"].value)
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(self.QUALITY_MODES) - 1))
        mode = self.QUALITY_MODES[idx]
        self.CONTEXT_FRAMES = mode["window"]
        self.INFER_STRIDE = mode["stride"]
        self.TAKE_HOPS = mode["take"]
        self.LATENCY_SAMPLES = mode["latency_blocks"] * BLOCK_SIZE + 62

    def _active_model_path(self) -> str:
        """ONNX file for the current quality mode (single session: switching
        quality reloads the other variant, ~1-2 s gap)."""
        try:
            studio = int(self.params["quality"].value) == 1
        except Exception:
            studio = False
        key = "studio_model_path" if studio else "model_path"
        return self.params[key].value if key in self.params else ""

    def _maybe_reload_model(self):
        path = self._active_model_path()
        if path and path != self.current_model_path:
            self._load_onnx_model(path)

    def on_ui_param_change(self, param_name: str):
        if param_name == "quality":
            self._apply_quality()
            self._maybe_reload_model()
            return
        if param_name in ("model_path", "studio_model_path"):
            self._maybe_reload_model()
            return

    def load_state(self, data: dict):
        super().load_state(data)
        self._apply_quality()
        self._maybe_reload_model()

    def _destroy_session_blocking(self, sess):
        """NRT background deallocation of the ONNX session."""
        del sess

    def _load_onnx_model(self, path: str):
        """Asynchronously loads ONNX model via NRT, tracked with _load_epoch."""
        if not ORT_AVAILABLE or not os.path.exists(path):
            self.error_msg = f"Model missing or onnxruntime unavailable: {path}"
            return
        if path == self._loading_path:
            return  # already loading this exact path (e.g. load_state + start)

        self._load_epoch += 1
        epoch = self._load_epoch
        self._loading_path = path

        def _worker_load():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
                return sess, epoch, path
            except Exception as e:
                logger.error(f"Failed to load Vocos ONNX: {e}")
                return None, epoch, path

        self.submit_nrt(_worker_load, tag="load_vocos")

    def _install_session(self, sess, path: str):
        """Install a freshly loaded session and retire the previous one."""
        old_sess = self.session
        self.session = sess
        self.current_model_path = path
        self._loading_path = ""
        self.error_msg = None
        if old_sess is not None:
            self.submit_nrt(self._destroy_session_blocking, old_sess, tag="discard_sess")
        logger.info(f"Installed Vocos model: {path}")

    def on_nrt_complete(self, tag, ok, result):
        if tag == "load_vocos" and ok and result is not None:
            sess, epoch, path = result
            if epoch != self._load_epoch:
                if sess is not None:
                    self.submit_nrt(self._destroy_session_blocking, sess, tag="discard_sess")
                return
            if sess is None:
                self._loading_path = ""
                self.error_msg = f"Failed to load model: {path}"
                return
            self._install_session(sess, path)

    def on_nrt_discarded(self, tag, ok, result):
        if tag == "load_vocos" and ok and result is not None:
            sess, epoch, path = result
            # The executor's supersede epoch is shared across ALL tags on this
            # node, so being "superseded" does not necessarily mean a newer
            # load exists: a discard_sess/stop_stream submit made while this
            # load was in flight also bumps the epoch (e.g. the stale-result
            # destroy in on_nrt_complete). If this result still matches the
            # load the user asked for, install it; otherwise destroy it.
            if (
                sess is not None
                and epoch == self._load_epoch
                and path == self.params["model_path"].value
            ):
                self._install_session(sess, path)
            elif sess is not None:
                self.submit_nrt(self._destroy_session_blocking, sess, tag="discard_sess")
            elif path == self._loading_path:
                self._loading_path = ""

    def _worker_loop(self, generation: int):
        """
        Background Worker:
        Runs 48k -> 24k downsampling, STFT, Mel extraction, ONNX inference,
        and 24k -> 48k upsampling completely off the audio thread.
        """
        mel_fb = build_vocos_mel_filterbank()
        fft_window = torch.hann_window(VOCOS_N_FFT, dtype=torch.float32)
        resampler_hb = build_halfband_kernel()
        hb_taps = resampler_hb.shape[-1]

        down_state = torch.zeros(1, 1, hb_taps - 1)
        up_state = torch.zeros(1, 1, hb_taps - 1)

        analysis_ring = torch.zeros(VOCOS_N_FFT, dtype=torch.float32)
        mel_context = np.full((1, VOCOS_N_MELS, self.CONTEXT_FRAMES), -11.5129, dtype=np.float32)
        hop_counter = 0

        while not self.stop_event.is_set():
            idx_in, ok = _pop_newest(self.queue_in)
            if not ok:
                # Idle: sleep until process() signals a new hop. The
                # clear-then-recheck closes the lost-wakeup race (a push
                # before the recheck is seen; a push after it follows its
                # own set()). The timeout is a backstop only.
                self._wake_event.clear()
                idx_in, ok = _pop_newest(self.queue_in)
                if not ok:
                    self._wake_event.wait(0.050)
                    continue

            if generation != self._worker_generation:
                break  # Stale generation; exit thread immediately

            # Live quality-mode switch: rebuild the mel context when the
            # window size changed (allocates only on switch, NRT thread).
            window = self.CONTEXT_FRAMES
            if mel_context.shape[2] != window:
                mel_context = np.full((1, VOCOS_N_MELS, window), -11.5129, dtype=np.float32)

            chunk_48k = torch.from_numpy(self.pool_in[idx_in])

            # 1. Resample 48 kHz (512) -> 24 kHz (256)
            down_in = torch.cat([down_state[0, 0], chunk_48k]).view(1, 1, -1)
            down_state[0, 0].copy_(chunk_48k[-(hb_taps - 1):])
            chunk_24k = torch.nn.functional.conv1d(down_in, resampler_hb, stride=2)[0, 0, :BLOCK_SIZE // 2]

            # 2. Extract 1 Mel Frame (Hop = 256)
            analysis_ring = torch.roll(analysis_ring, -VOCOS_HOP)
            analysis_ring[-VOCOS_HOP:].copy_(chunk_24k)
            stft_frame = analysis_ring * fft_window
            spec = torch.fft.rfft(stft_frame, n=VOCOS_N_FFT).abs().clamp_min_(1e-5)
            mel_frame = torch.matmul(mel_fb, spec).log_().numpy()

            # 3. Roll context and append newest frame
            mel_context[:, :, :-1] = mel_context[:, :, 1:]
            mel_context[0, :, -1] = mel_frame
            hop_counter += 1

            if self.session is None:
                continue

            # Strided inference: frontend runs per hop (states stay
            # continuous), the model runs every INFER_STRIDE hops and emits
            # that many taken hops. Same average pace at ~half the
            # per-block cost; hops from one decode share its ISTFT, so the
            # intra-pair seam is continuous.
            if hop_counter % self.INFER_STRIDE != 0:
                continue

            # 4. ONNX Inference
            try:
                ort_outs = self.session.run(None, {"mel": mel_context})
                audio_24k = ort_outs[0][0]  # Shape: (samples,)

                for take in self.TAKE_HOPS:
                    take_idx = take * VOCOS_HOP
                    if not 0 <= take_idx <= audio_24k.shape[0] - VOCOS_HOP:
                        logger.warning(
                            f"Vocos output size ({audio_24k.shape[0]}) incompatible with "
                            f"take hop {take}"
                        )
                        break

                    # 5. Resample 24 kHz (256) -> 48 kHz (512)
                    synth_hop_24k = torch.from_numpy(audio_24k[take_idx:take_idx + VOCOS_HOP])
                    stuffed = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
                    stuffed[::2] = synth_hop_24k
                    up_in = torch.cat([up_state[0, 0], stuffed]).view(1, 1, -1)
                    up_state[0, 0].copy_(stuffed[-(hb_taps - 1):])
                    synth_hop_48k = torch.nn.functional.conv1d(up_in, resampler_hb, stride=1)[0, 0, :BLOCK_SIZE] * 2.0

                    idx_out = self.pool_seq_out % len(self.pool_out)
                    self.pool_seq_out += 1
                    np.copyto(self.pool_out[idx_out], synth_hop_48k.numpy())
                    if not self.queue_out.try_push(idx_out):
                        self.drops_out += 1
            except Exception as e:
                logger.debug(f"Vocos inference error: {e}")

    def start(self):
        # Prevent start/stop race conditions
        self._worker_generation += 1
        gen = self._worker_generation

        self.stop()
        self.stop_event.clear()
        self.write_pos = 0
        self.in_delay_ring.zero_()
        self.drops_in = 0
        self.drops_out = 0

        while self.queue_in.try_pop()[1]:
            pass
        while self.queue_out.try_pop()[1]:
            pass

        # Only (re)load when the desired path differs from what is loading or
        # installed. During engine startup, load_state() has already submitted
        # the load; a second submit here would supersede it and — via the
        # stale-result destroy submit — ultimately prevent ANY result from
        # installing (NRTExecutor epoch is shared across tags on this node).
        model_p = self._active_model_path()
        if model_p and self.current_model_path != model_p:
            self._load_onnx_model(model_p)

        if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
            self.worker_thread = self.graph.engine.nrt.spawn_stream(self._worker_loop, gen)
        else:
            self.worker_thread = threading.Thread(target=self._worker_loop, args=(gen,), daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        self._wake_event.set()  # release a worker parked in the idle wait
        if self.worker_thread is not None:
            if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
                self.graph.engine.nrt.stop_stream(self, lambda: self.stop_event.set(), self.worker_thread)
            else:
                self.worker_thread.join(timeout=0.5)
            self.worker_thread = None

    def remove(self):
        self.stop()
        if self.session is not None:
            sess = self.session
            self.session = None
            if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
                self.submit_nrt(self._destroy_session_blocking, sess, tag="discard_sess")

    def process(self):
        in_tensor = self.inp.get_tensor()
        out_tensor = self.out.buffer

        # 1. Vectorized Write into Delay Ring (Always 512-aligned; never wraps)
        wp = self.write_pos
        self.in_delay_ring[0, wp:wp + BLOCK_SIZE].copy_(in_tensor[0])
        if in_tensor.shape[0] > 1:
            self.in_delay_ring[1, wp:wp + BLOCK_SIZE].copy_(in_tensor[1])
        else:
            self.in_delay_ring[1, wp:wp + BLOCK_SIZE].copy_(in_tensor[0])

        # 2. Vectorized Mono Downmix & Push Pool Index
        if in_tensor.shape[0] > 1:
            torch.add(in_tensor[0], in_tensor[1], out=self.buf_mono).mul_(0.5)
        else:
            self.buf_mono.copy_(in_tensor[0])

        idx_in = self.pool_seq_in % len(self.pool_in)
        self.pool_seq_in += 1
        np.copyto(self.pool_in[idx_in], self.buf_mono.numpy())
        if not self.queue_in.try_push(idx_in):
            self.drops_in += 1
        # Uncontended set (~100 ns, no wait): wakes the worker promptly.
        self._wake_event.set()

        # 3. Two-Part Vectorized Ring Read (Safe against circular boundary wrap)
        rp = (wp - self.LATENCY_SAMPLES) & self.in_ring_mask
        k = min(BLOCK_SIZE, self.in_ring_size - rp)
        self.buf_dry[:, :k].copy_(self.in_delay_ring[:, rp:rp + k])
        if k < BLOCK_SIZE:
            self.buf_dry[:, k:].copy_(self.in_delay_ring[:, :BLOCK_SIZE - k])

        self.write_pos = (wp + BLOCK_SIZE) & self.in_ring_mask

        # 4. Pull Synthesized Audio from Worker
        idx_out, ok = self.queue_out.try_pop()
        mix = float(self.params["mix"].value)

        if ok and self.session is not None:
            wet_mono = torch.from_numpy(self.pool_out[idx_out])
            self.buf_wet[0].copy_(wet_mono)
            self.buf_wet[1].copy_(wet_mono)

            # Vectorized Dry/Wet Crossfade
            out_tensor.copy_(self.buf_dry)
            out_tensor.mul_(1.0 - mix).add_(self.buf_wet, alpha=mix)
        else:
            # Latency-filling / underrun fallback: emit latency-aligned dry audio
            out_tensor.copy_(self.buf_dry)

    def get_telemetry(self) -> dict:
        if self.session is not None:
            status = "Ready"
        elif self.worker_thread is not None and bool(self.params["model_path"].value):
            status = "Loading Model..."
        else:
            status = "Idle (No Model)"

        return {
            "latency_samples": self.LATENCY_SAMPLES,
            "latency_ms": round(self.LATENCY_SAMPLES / SAMPLE_RATE * 1000.0, 1),
            "status": status,
            "drops_in": self.drops_in,
            "drops_out": self.drops_out,
        }
