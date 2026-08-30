"""
RubberBandPitchShifter — real-time pitch and formant shifter powered by pylibrb
(the Rubber Band Library R3 engine).

Real-time notes:

- The stretcher runs in Rubber Band's real-time mode (ENGINE_FINER, which is
  required for explicit formant scaling). Its output arrives in bursts with a
  start latency of roughly 30-60 ms at 48 kHz. The output is silent during the
  initial latency-fill period, and the FIFO is primed to
  get_start_delay() + 2 blocks before any wet audio is emitted, which prevents
  steady-state underruns (the stretcher's average output rate exactly equals
  its input rate, so without priming the FIFO hovers empty and underruns).
- pylibrb has no retrieve_into(): retrieve() returns a freshly allocated numpy
  array on every call. This is bounded and freed every block, so the steady
  state does not grow, but it is NOT a zero-allocation path.
- mix blends the UNDELAYED dry signal with the DELAYED wet signal. Below 1.0
  this is a comb-filtering special effect, not a latency-compensated crossfade.
- The pitch_mod/formant_mod inputs are parameter-bound block-rate CV: the
  first sample of each block is used, matching the engine's per-block
  parameter synchronization.
- All FIFO/stretcher state is owned by the audio thread. The stretcher is
  constructed in __init__ (node construction never happens on the audio
  thread); start() only resets() it. Rubber Band requires setPitchScale /
  setFormantScale to be called from the same thread as process(), which this
  node guarantees by only touching them inside process().
"""

import logging

import numpy as np
import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)

try:
    from pylibrb import RubberBandStretcher, Option, AUTO_FORMANT_SCALE
    PYLIBRB_AVAILABLE = True
except ImportError:
    RubberBandStretcher = None
    Option = None
    AUTO_FORMANT_SCALE = 0.0
    PYLIBRB_AVAILABLE = False


class RubberBandPitchShifter(Node):
    category = "Effects"
    label = "RubberBand Pitch Shifter"
    description = (
        "High-quality real-time pitch and formant shifter powered by the "
        "Rubber Band R3 engine. Pitch and formant shift up to +/-24 semitones "
        "with three formant modes: Preserve Formants (automatic tracking), "
        "Shifted Formants (independent formant amount), and Off (Classic: "
        "formants move with pitch). The wet signal is delayed by the "
        "stretcher's start latency (~30-60 ms); mix blends this delayed wet "
        "signal with the undelayed dry signal and is intended as a special "
        "effect, not a latency-compensated crossfade. Mod inputs are "
        "block-rate CV sampled at the start of each block. Output is silent "
        "during the initial latency-fill period; steady-state heap use is "
        "bounded (small transient allocations from the binding each block)."
    )

    FIFO_CAPACITY = 8192
    # Fallback priming target if get_start_delay() is unavailable.
    DEFAULT_PRIME_TARGET = 4 * BLOCK_SIZE

    def __init__(self, name=""):
        super().__init__(name)

        self.inp = self.add_input(
            "in", help="Signal to pitch-shift; mono inputs are duplicated to stereo.")
        self.pitch_mod = self.add_input(
            "pitch_mod", "pitch_shift",
            help="Block-rate pitch CV in semitones (bound to 'pitch_shift'); "
                 "unconnected slots use the parameter value.")
        self.formant_mod = self.add_input(
            "formant_mod", "formant_shift",
            help="Block-rate formant CV in semitones (bound to 'formant_shift'); "
                 "only used in 'Shifted Formants' mode.")
        self.out = self.add_output(
            "out", channels=CHANNELS, help="Pitch-shifted stereo output.")

        self.add_float_param(
            "pitch_shift", 0.0, -24.0, 24.0, unit="st",
            help="Pitch shift amount in semitones.")
        self.add_float_param(
            "formant_shift", 0.0, -24.0, 24.0, unit="st",
            help="Formant shift amount in semitones (Shifted Formants mode).")
        self.add_menu_param(
            "formant_mode",
            ["Preserve Formants", "Shifted Formants", "Off (Classic)"], 0,
            help="Formant processing mode.")
        self.add_float_param(
            "mix", 1.0, 0.0, 1.0,
            help="Dry/wet balance. The wet path carries the stretcher latency, "
                 "so values below 1.0 comb-filter; 1.0 is fully wet.")

        # Pre-allocated audio-thread state.
        self._in_np = np.zeros((CHANNELS, BLOCK_SIZE), dtype=np.float32)
        self._fifo = np.zeros((CHANNELS, self.FIFO_CAPACITY), dtype=np.float32)
        self._fifo_head = 0
        self._fifo_tail = 0
        self._fifo_count = 0
        self._primed = False
        self._prime_target = self.DEFAULT_PRIME_TARGET
        self._underruns = 0
        self._last_pitch_scale = None
        self._last_formant_scale = None

        self._stretcher = None
        if PYLIBRB_AVAILABLE:
            options = (
                Option.PROCESS_REALTIME
                | Option.ENGINE_FINER       # R3 engine: required for formant_scale
                | Option.WINDOW_SHORT       # lower start delay in the R3 engine
                | Option.SMOOTHING_ON
                | Option.FORMANT_PRESERVED
            )
            try:
                self._stretcher = RubberBandStretcher(
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    options=options,
                    initial_time_ratio=1.0,
                    initial_pitch_scale=1.0,
                )
                self._stretcher.set_max_process_size(BLOCK_SIZE)
                self._compute_prime_target()
            except Exception as e:
                self._stretcher = None
                self.error_msg = f"Stretcher Init Failed: {e}"
                logger.error(f"[{self.name}] {self.error_msg}")
        else:
            self.error_msg = "Missing dependency: pylibrb"

    # ------------------------------------------------------------------
    # Lifecycle (control thread)
    # ------------------------------------------------------------------

    def _compute_prime_target(self):
        """Priming level that keeps the FIFO above one block in steady state.

        The stretcher's average output rate equals its input rate, so once the
        FIFO reaches this level it stays there (burst jitter aside). Without
        priming, the FIFO hovers near empty and blocks get dropped to silence.
        """
        try:
            delay = int(self._stretcher.get_start_delay())
        except Exception:
            delay = 0
        self._prime_target = min(
            max(delay, 0) + 2 * BLOCK_SIZE,
            self.FIFO_CAPACITY - 2 * BLOCK_SIZE,
        )

    def start(self):
        self._fifo_head = 0
        self._fifo_tail = 0
        self._fifo_count = 0
        self._fifo.fill(0.0)
        self._primed = False
        self._underruns = 0
        self._last_pitch_scale = None
        self._last_formant_scale = None
        if self._stretcher is not None:
            try:
                self._stretcher.reset()
            except Exception as e:
                self.error_msg = f"Stretcher Reset Failed: {e}"
                return
            self._compute_prime_target()

    def remove(self):
        """Release the native stretcher when the node is deleted.

        NOTE: must not be called while the audio engine is processing this
        node. Callers today are the app's shutdown path (after Engine.stop()
        has joined the audio thread) and test teardown, both on control
        threads. Undo-restore always builds a fresh node instance from the
        serialized memento (see DeleteNodeCommand.undo), so releasing here is
        final by contract.
        """
        self._stretcher = None
        self._primed = False
        self._last_pitch_scale = None
        self._last_formant_scale = None
        self._fifo_count = 0

    # ------------------------------------------------------------------
    # Audio thread
    # ------------------------------------------------------------------

    def _fifo_write(self, block_np):
        """Append a (channels, n) array to the ring buffer (vectorized).

        On overflow the OLDEST samples are dropped so recent audio and the
        wet/dry relationship are preserved. This guard should never trigger
        because the priming target leaves ample headroom.
        """
        n = block_np.shape[1]
        free = self.FIFO_CAPACITY - self._fifo_count
        if n > free:
            drop = n - free
            self._fifo_tail = (self._fifo_tail + drop) % self.FIFO_CAPACITY
            self._fifo_count -= drop
        first = min(n, self.FIFO_CAPACITY - self._fifo_head)
        if first > 0:
            np.copyto(self._fifo[:, self._fifo_head:self._fifo_head + first],
                      block_np[:, :first])
        if n > first:
            np.copyto(self._fifo[:, :n - first], block_np[:, first:])
        self._fifo_head = (self._fifo_head + n) % self.FIFO_CAPACITY
        self._fifo_count += n

    def _fifo_read(self, out):
        """Consume BLOCK_SIZE samples from the ring buffer into `out`."""
        first = min(BLOCK_SIZE, self.FIFO_CAPACITY - self._fifo_tail)
        out[:, :first].copy_(
            torch.from_numpy(self._fifo[:, self._fifo_tail:self._fifo_tail + first]))
        second = BLOCK_SIZE - first
        if second > 0:
            out[:, first:].copy_(torch.from_numpy(self._fifo[:, :second]))
        self._fifo_tail = (self._fifo_tail + BLOCK_SIZE) % self.FIFO_CAPACITY
        self._fifo_count -= BLOCK_SIZE

    def _fallback_output(self, out, sig, mix):
        """Latency-fill / underrun behavior: dry passthrough at mix=0, silence
        otherwise. Keeps the mix=0 path bit-exact from the very first block."""
        if mix <= 0.0:
            out.copy_(sig)
        else:
            out.zero_()

    def process(self):
        out = self.out.buffer
        sig = self.inp.get_tensor()

        if not PYLIBRB_AVAILABLE or self._stretcher is None:
            # Fallback bypass (Tensor.copy_ broadcasts mono to stereo safely).
            out.copy_(sig)
            return

        # 1. Block-rate parameters / modulation (first sample of the block).
        pitch_st = float(self.pitch_mod.get_tensor()[0, 0].item())
        formant_st = float(self.formant_mod.get_tensor()[0, 0].item())
        mode = int(self.params["formant_mode"].value)
        mix = float(self.params["mix"].value)

        # 2. Push pitch/formant ratios to the stretcher (only on change).
        #    Rubber Band requires these setters to run on the process() thread.
        pitch_scale = 2.0 ** (pitch_st / 12.0)
        if pitch_scale != self._last_pitch_scale:
            self._stretcher.pitch_scale = pitch_scale
            self._last_pitch_scale = pitch_scale

        # Rubber Band's formant_scale is RELATIVE to the pitch scale
        # (absolute formant ratio = pitch_scale * formant_scale; auto = 0.0
        # is treated as 1/pitch_scale under OptionFormantPreserved).
        if mode == 0:      # Preserve Formants: automatic tracking
            formant_scale = AUTO_FORMANT_SCALE
        elif mode == 1:    # Shifted Formants: desired absolute ratio / pitch
            formant_scale = (2.0 ** (formant_st / 12.0)) / max(1e-6, pitch_scale)
        else:              # Off (Classic): formants shift with the pitch
            formant_scale = 1.0
        if formant_scale != self._last_formant_scale:
            self._stretcher.formant_scale = formant_scale
            self._last_formant_scale = formant_scale

        # 3. Mono -> stereo into the pre-allocated input array.
        sig_np = sig.cpu().numpy()
        if sig_np.shape[0] == 1:
            np.copyto(self._in_np[0], sig_np[0])
            np.copyto(self._in_np[1], sig_np[0])
        else:
            np.copyto(self._in_np, sig_np[:CHANNELS])

        # 4. Feed the stretcher and drain whatever is available.
        self._stretcher.process(self._in_np, final=False)
        avail = self._stretcher.available()
        if avail > 0:
            self._fifo_write(self._stretcher.retrieve(avail))

        # 5. Prime the FIFO past the stretcher's start latency before any wet
        #    audio is emitted (prevents steady-state underruns, see above).
        if not self._primed:
            if self._fifo_count >= self._prime_target:
                self._primed = True
            else:
                self._fallback_output(out, sig, mix)
                return

        if self._fifo_count >= BLOCK_SIZE:
            self._fifo_read(out)
        else:
            self._underruns += 1
            self._fallback_output(out, sig, mix)
            return

        # 6. Dry/wet. The wet path carries the stretcher latency, so this is a
        #    special effect rather than a latency-compensated crossfade.
        #    (in-place add_ broadcasts a mono sig safely; no out= shrinkage.)
        if mix <= 0.0:
            out.copy_(sig)
        elif mix < 1.0:
            out.mul_(mix).add_(sig, alpha=1.0 - mix)

    # ------------------------------------------------------------------
    # Telemetry (control/UI thread)
    # ------------------------------------------------------------------

    def get_telemetry(self) -> dict:
        delay = 0
        if self._stretcher is not None:
            try:
                delay = int(self._stretcher.get_start_delay())
            except Exception:
                delay = 0
        return {
            "latency_samples": delay,
            "latency_ms": round(delay / SAMPLE_RATE * 1000.0, 2),
            "buffered_samples": self._fifo_count,
            "primed": self._primed,
            "underruns": self._underruns,
        }
