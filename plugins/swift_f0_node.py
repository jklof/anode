"""
SwiftF0Node — real-time neural pitch tracking & audio-to-MIDI transcription (Utilities).

Extracts continuous fundamental frequency (Hz CV), voicing gate, tracking confidence,
and segmented MIDI Note-On/Note-Off messages from an incoming audio signal.
Model loading runs asynchronously via NRT workers; inference executes on a background
streaming worker to guarantee zero audio dropouts on the real-time processing thread.

NOTE: This module is intentionally named ``swift_f0_node.py`` (not ``swift_f0.py``).
A plugin file named ``swift_f0.py`` would shadow the installed ``swift_f0`` pip
package inside the plugin loader: ``import swift_f0`` at module load time would
resolve to the plugin module itself and ``swift_f0.SwiftF0`` / ``swift_f0.segment_notes``
would raise AttributeError at runtime. The class/port/param names are unaffected.
"""

import logging
import math
import threading
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from base import Node, BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE, SPSCRingBuffer

logger = logging.getLogger(__name__)

try:
    import swift_f0
    SWIFT_F0_AVAILABLE = True
except ImportError:
    swift_f0 = None
    SWIFT_F0_AVAILABLE = False

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    mido = None
    MIDO_AVAILABLE = False

ANALYSIS_SR = 16000
CHUNK_DURATION_S = 0.15  # ~150 ms analysis window
CHUNK_SAMPLES_48K = int(CHUNK_DURATION_S * SAMPLE_RATE)
MODEL_MIN_F0 = 46.875
MODEL_MAX_F0 = 2093.75

# 48 kHz -> 16 kHz is an exact 3:1 integer decimation, so it can be done as a
# strided FIR lowpass (F.conv1d, stride=3) instead of resampy's general
# fractional resampler. The kernel is a DC-normalized Blackman-windowed sinc
# with cutoff at ANALYSIS_SR / 2 (8 kHz), which makes the decimation alias-free.
# Output length floor(L/3) matches resampy.resample(..., axis=-1) exactly for
# the chunk sizes this worker produces, and no external C library is involved.
DECIM_TAPS = 129


@lru_cache(maxsize=1)
def _build_decim_kernel():
    """Anti-aliasing lowpass kernel for the 3:1 (48k -> 16k) decimator.

    Blackman-windowed sinc lowpass with cutoff at ANALYSIS_SR/2 (8 kHz) and
    unity DC gain. Immutable; safe to share read-only across worker threads.
    """
    n = np.arange(DECIM_TAPS) - (DECIM_TAPS - 1) // 2
    kernel = np.sinc(2 * (ANALYSIS_SR / (2.0 * SAMPLE_RATE)) * n) * np.blackman(DECIM_TAPS)
    kernel /= kernel.sum()
    return torch.from_numpy(kernel.astype(np.float32)).view(1, 1, DECIM_TAPS)


class SwiftF0Node(Node):
    category = "Utilities"
    label = "SwiftF0 Pitch & MIDI Tracker"
    description = (
        "Real-time neural pitch tracker and audio-to-MIDI transcriber powered by "
        "SwiftF0. Produces monophonic pitch CV in Hz, binary gate CV, confidence CV, "
        "and discrete MIDI Note-On/Note-Off packets on every block. Neural inference "
        "and model loading run asynchronously off the audio thread with zero latency "
        "impact on audio processing."
    )

    def __init__(self, name=""):
        super().__init__(name)

        # 1. Sockets
        self.inp = self.add_input("in", help="Signal to track; mono inputs are duplicated to stereo.")
        self.out = self.add_output("out", channels=CHANNELS, help="Pass-through copy of the input, unaltered.")
        self.pitch_out = self.add_output(
            "pitch_out", channels=1,
            help="Mono CV of tracked fundamental frequency in Hz (holds last pitch when unvoiced)."
        )
        self.gate_out = self.add_output(
            "gate_out", channels=1,
            help="Mono binary gate CV (1.0 when voiced/confident, 0.0 when unvoiced)."
        )
        self.conf_out = self.add_output(
            "confidence_out", channels=1,
            help="Mono CV of pitch tracking confidence [0.0, 1.0]."
        )
        self.midi_out = self.add_midi_output(
            "midi_out",
            help="Transcribed MIDI stream with segmented Note-On/Note-Off messages."
        )

        # 2. Parameters (fmin clamped >= 50.0 to satisfy underlying model constraints)
        self.add_float_param("fmin", 65.0, 50.0, 500.0, unit="Hz", help="Minimum detectable fundamental frequency.")
        self.add_float_param("fmax", 1200.0, 200.0, 2000.0, unit="Hz", help="Maximum detectable fundamental frequency.")
        self.add_float_param("confidence_thresh", 0.4, 0.1, 0.9, help="Confidence threshold for voicing/gate activation.")
        self.add_float_param("glide_ms", 10.0, 0.0, 100.0, unit="ms", help="Portamento glide time for pitch CV transitions.")
        self.add_int_param("velocity", 80, 1, 127, help="Default velocity for synthesized MIDI Note-On events.")

        # 3. Real-Time Thread State
        self._current_f0 = 0.0
        self._target_f0 = 0.0
        self._current_gate = 0.0
        self._current_conf = 0.0

        self._pitch_buf = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)
        self._mono_scratch = np.zeros(BLOCK_SIZE, dtype=np.float32)

        # 4. Ring Buffers (Audio Thread <-> Worker Thread)
        self._audio_queue = SPSCRingBuffer(capacity=64)
        self._results_queue = SPSCRingBuffer(capacity=64)

        # 5. Worker Thread and Detector State
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._detector = None
        self._detector_epoch = 0
        self._worker_generation = 0
        self._active_midi_note: Optional[int] = None

        if not SWIFT_F0_AVAILABLE:
            self.error_msg = "Missing dependency: swift-f0"

    def _request_detector_rebuild(self):
        """Asynchronously schedules detector build on the NRT worker pool."""
        if not SWIFT_F0_AVAILABLE:
            return
        self._detector_epoch += 1
        fmin = float(self.params["fmin"].value)
        fmax = float(self.params["fmax"].value)
        thresh = float(self.params["confidence_thresh"].value)
        self.submit_nrt(self._build_detector_nrt, fmin, fmax, thresh, self._detector_epoch, tag="init_detector")

    def _build_detector_nrt(self, fmin: float, fmax: float, thresh: float, epoch: int):
        """Runs on NRT thread pool. Performs ONNX session load off the audio thread."""
        try:
            safe_fmin = max(fmin, MODEL_MIN_F0)
            safe_fmax = min(fmax, MODEL_MAX_F0)
            instance = swift_f0.SwiftF0(fmin=safe_fmin, fmax=safe_fmax, confidence_threshold=thresh)
            return instance, epoch
        except Exception as e:
            logger.error(f"[{self.name}] SwiftF0 NRT build error: {e}")
            return None, epoch

    def on_nrt_complete(self, tag, ok, result):
        """Installed between blocks on the engine/control thread."""
        if tag == "init_detector" and ok:
            instance, epoch = result
            if epoch == self._detector_epoch and instance is not None:
                self._detector = instance
                self.error_msg = None

    def on_ui_param_change(self, param_name: str):
        if param_name in ("fmin", "fmax", "confidence_thresh"):
            self._request_detector_rebuild()

    def _worker_loop(self, generation: int):
        """Streaming background worker for 16 kHz resampling, neural inference, and note segmentation."""
        accum_audio = []
        accum_samples = 0

        while not self._stop_event.is_set():
            # 1. Drain incoming audio from audio thread
            while True:
                block, ok = self._audio_queue.try_pop()
                if not ok:
                    break
                accum_audio.append(block)
                accum_samples += len(block)

            # 2. Unconditional memory cap to prevent slow-motion accumulation.
            # Keep the most recent full analysis window (CHUNK_SAMPLES_48K)
            # rather than a tiny tail so a burst of queued audio cannot stall
            # chunk execution: with only a partial window left, this branch
            # would wait forever for audio that never arrives. Dropping only
            # the oldest samples still bounds memory to ~0.6 s of audio.
            if accum_samples > CHUNK_SAMPLES_48K * 2:
                kept = []
                kept_samples = 0
                for b in reversed(accum_audio):
                    if kept_samples >= CHUNK_SAMPLES_48K:
                        break
                    kept.append(b)
                    kept_samples += len(b)
                accum_audio = kept[::-1]
                accum_samples = kept_samples

            # 3. Guard against execution when detector is compiling or missing
            if self._detector is None:
                self._stop_event.wait(0.02)
                continue

            # 4. When full chunk is ready, run inference
            if accum_samples >= CHUNK_SAMPLES_48K:
                audio_48k = np.concatenate(accum_audio)
                keep_samples = CHUNK_SAMPLES_48K // 2
                accum_audio = [audio_48k[-keep_samples:]]
                accum_samples = keep_samples

                # Resample 48 kHz -> 16 kHz via 3:1 FIR decimation.
                # The DC-normalized Blackman-windowed sinc kernel (cutoff 8 kHz)
                # makes decimation alias-free; output length int(L/3) matches
                # resampy.resample() exactly. audio_48k is contiguous float32,
                # so view() is zero-copy.
                try:
                    x = torch.from_numpy(audio_48k).view(1, 1, -1)
                    y = torch.nn.functional.conv1d(
                        x, _build_decim_kernel(), stride=3, padding=(DECIM_TAPS - 3) // 2
                    )
                    audio_16k = y[0, 0].numpy()
                except Exception:
                    continue

                # Run SwiftF0 inference
                try:
                    res = self._detector.detect_from_array(audio_16k, ANALYSIS_SR)
                except Exception as e:
                    logger.debug(f"[{self.name}] Inference error: {e}")
                    continue

                # Compute voicing & F0
                voiced = np.where(res.voicing)[0]
                conf_thresh = float(self.params["confidence_thresh"].value)
                if len(voiced) > 0:
                    last_idx = voiced[-1]
                    f0 = float(res.pitch_hz[last_idx])
                    conf = float(res.confidence[last_idx])
                    is_voiced = (conf >= conf_thresh) and (f0 > 0.0)
                else:
                    f0 = 0.0
                    conf = 0.0
                    is_voiced = False

                # Segment discrete MIDI notes
                midi_msgs = []
                try:
                    notes = swift_f0.segment_notes(res)
                    detected_note = notes[-1].pitch_midi if notes else None
                except Exception:
                    detected_note = None

                velocity = int(self.params["velocity"].value)

                # MIDI Note State Machine
                if detected_note is not None and is_voiced:
                    if self._active_midi_note != detected_note:
                        if self._active_midi_note is not None and MIDO_AVAILABLE:
                            midi_msgs.append(mido.Message("note_off", note=self._active_midi_note, velocity=0))
                        if MIDO_AVAILABLE:
                            midi_msgs.append(mido.Message("note_on", note=detected_note, velocity=velocity))
                        self._active_midi_note = detected_note
                elif not is_voiced and self._active_midi_note is not None:
                    if MIDO_AVAILABLE:
                        midi_msgs.append(mido.Message("note_off", note=self._active_midi_note, velocity=0))
                    self._active_midi_note = None

                payload = {
                    "gen": generation,
                    "f0": f0,
                    "conf": conf,
                    "voiced": is_voiced,
                    "midi_msgs": midi_msgs,
                }
                self._results_queue.try_push(payload)

            else:
                self._stop_event.wait(0.01)

        # Emit final Note-Off on exit
        if self._active_midi_note is not None and MIDO_AVAILABLE:
            off_msg = mido.Message("note_off", note=self._active_midi_note, velocity=0)
            self._results_queue.try_push({"gen": generation, "f0": 0.0, "conf": 0.0, "voiced": False, "midi_msgs": [off_msg]})
            self._active_midi_note = None

    def start(self):
        self._current_f0 = 0.0
        self._target_f0 = 0.0
        self._current_gate = 0.0
        self._current_conf = 0.0
        self._active_midi_note = None
        self._worker_generation += 1

        self.out.buffer.zero_()
        self.pitch_out.buffer.zero_()
        self.gate_out.buffer.zero_()
        self.conf_out.buffer.zero_()
        self.midi_out.clear_packet()

        self._request_detector_rebuild()
        self._stop_event.clear()

        # Clear communication queues
        while self._audio_queue.try_pop()[1]:
            pass
        while self._results_queue.try_pop()[1]:
            pass

        # Spawn background streaming worker with generation tag
        if SWIFT_F0_AVAILABLE:
            gen = self._worker_generation
            if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
                self._worker_thread = self.graph.engine.nrt.spawn_stream(self._worker_loop, gen)
            else:
                self._worker_thread = threading.Thread(target=self._worker_loop, args=(gen,), daemon=True)
                self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._worker_thread is not None:
            if getattr(self, "graph", None) and getattr(self.graph, "engine", None):
                self.graph.engine.nrt.stop_stream(self, lambda: self._stop_event.set(), self._worker_thread)
            else:
                self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def remove(self):
        self.stop()

    def process(self):
        sig = self.inp.get_tensor()

        # 1. Bit-exact audio pass-through
        self.out.buffer.copy_(sig)

        # 2. Reset MIDI Output Packet at top of block (Anti-Ghosting Contract)
        self.midi_out.clear_packet()

        # 3. Drain background inference updates (with generation validation)
        while True:
            res, ok = self._results_queue.try_pop()
            if not ok:
                break
            if res.get("gen") != self._worker_generation:
                continue  # Stale result from previous session

            if res["voiced"] and res["f0"] > 0.0:
                self._target_f0 = res["f0"]
                if self._current_f0 <= 0.0:
                    self._current_f0 = self._target_f0
            self._current_gate = 1.0 if res["voiced"] else 0.0
            self._current_conf = res["conf"]

            for msg in res["midi_msgs"]:
                self.midi_out.packet.messages.append((0, msg))

        # 4. Monophonic Pitch CV Generation with Portamento Glide
        glide_ms = float(self.params["glide_ms"].value)
        if glide_ms <= 1.0 or self._current_f0 <= 0.0 or abs(self._current_f0 - self._target_f0) < 1e-2:
            self._pitch_buf[0].fill_(self._target_f0)
            self._current_f0 = self._target_f0
        else:
            block_dur = BLOCK_SIZE / SAMPLE_RATE
            alpha = 1.0 - math.exp(-block_dur / (glide_ms / 1000.0))
            end_pitch = self._current_f0 + alpha * (self._target_f0 - self._current_f0)
            torch.linspace(self._current_f0, end_pitch, BLOCK_SIZE, out=self._pitch_buf[0])
            self._current_f0 = end_pitch

        self.pitch_out.buffer.copy_(self._pitch_buf)
        self.gate_out.buffer[0].fill_(self._current_gate)
        self.conf_out.buffer[0].fill_(self._current_conf)

        # 5. Push downmixed mono audio slice to worker queue (no per-block buffer growth)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            if sig.shape[0] > 1:
                np.add(sig[0].cpu().numpy(), sig[1].cpu().numpy(), out=self._mono_scratch)
                self._mono_scratch *= 0.5
            else:
                np.copyto(self._mono_scratch, sig[0].cpu().numpy())
            self._audio_queue.try_push(self._mono_scratch.copy())
