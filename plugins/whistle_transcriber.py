"""
WhistleStreamer — realtime speech-to-text tap (Voice & Pitch).

Live transcription of whatever reaches ``in`` using Whistle, Cactus
Compute's 16.9 MB Apache-2.0 CPU speech-to-text model. Whistle runs on
``libneedle3`` (the native engine from the ``cactus-needle`` pip package,
NOT the audio.cpp sidecar) reading a single ``whistle.cact`` container.

Threading (AGENTS.md sections 3/4/6): ``process()`` only copies the input
to the output and hands the worker a *pool index*. Every heavy step —
loading the engine, the 48 kHz -> 16 kHz resample, the neural decode, the
transcript/word-timestamp file writes — happens on one background worker
thread owned by this node. The audio thread never allocates per block, never
touches the filesystem, never logs, and never waits for the model.

Streaming is pull-based: the worker feeds Whistle a generator of 16 kHz mono
chunks and the engine pulls from it inside ``Whistle.stream()``. The
generator terminates on ``_stop_event``, which lets the stream's own
``finally`` run ``needle_stream_transcribe_stop`` so the tail is committed
before the node flushes its artifacts.

Artifacts are published on the ``transcript_uri`` / ``words_uri`` outputs
when the transport stops, from ``on_nrt_complete`` — a control-thread
callback between blocks. There is deliberately NO ``done`` pulse output:
the flush happens when audio stops running, so a one-block pulse could never
be observed by any downstream node.
"""

import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from base import (
    BLOCK_SIZE,
    CHANNELS,
    Node,
    SAMPLE_RATE,
    SPSCRingBuffer,
    TelemetryDictRingBuffer,
)

try:  # GUI is optional: the node must load headless (tests, --headless).
    from PySide6.QtCore import Qt, QSignalBlocker, QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QLabel,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    GUI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without PySide6
    GUI_AVAILABLE = False

from audiocpp_backend import REPO_ROOT, missing_specs
from download_util import (
    DownloadCancelled,
    DownloadError,
    DownloadSpec,
    fetch_all,
    format_bytes,
)
from whistle_backend import (
    DEFAULT_MODEL_DIR,
    WHISTLE_SAMPLE_RATE,
    assets_ready,
    check_assets,
    engine_library_path,
    extract_engine_wheel,
    language_code,
    parse_keywords,
    weights_path,
    whistle_fetch_specs,
    whistle_language_menu,
)

logger = logging.getLogger(__name__)

# Streaming chunk length, in seconds of 48 kHz engine audio.
#
# The stream decoder commits the words two passes agree on, so every pass
# re-decodes a trailing window and the per-pass cost is dominated by that
# overlap rather than by the new audio. Bigger chunks therefore amortise the
# same work over more audio. Measured on the dev box (40 s of speech, soxr
# HQ, this model), realtime factor and resulting transcript:
#
#     chunk   xRT   mean pass_ms   transcript chars
#       1 s  0.71x           1382                 542
#       2 s  1.22x           1560                 546
#       4 s  1.89x           1926                 549
#       8 s  2.42x           2750                 549
#
# The transcript is effectively identical across all four, so the only real
# cost of a longer chunk is latency. The default is the shortest chunk with
# real headroom over a decode pass: a dropped second is a wrong transcript,
# where extra lag is only slower captions. Machines that still fall behind
# raise chunk_seconds, and the node says so rather than hiding it.
DEFAULT_CHUNK_SECONDS = 4
CHUNK_SECONDS_MIN = 1
CHUNK_SECONDS_MAX = 8

# Headroom, in seconds of audio, reserved on the ring for a decode pass that
# is still running. The audio thread keeps handing over blocks while the
# worker is inside the engine; without this the ring fills mid-pass and the
# surplus is dropped.
PASS_HEADROOM_SECONDS = 5

# Zero-allocation audio handoff (SwiftF0's pool-index ring, same rationale):
# the audio thread pushes a round-robin pool *index* into _audio_queue rather
# than a fresh numpy copy per block, and the worker reads _mono_pool[idx].
#
# Unlike SwiftF0 this size CANNOT be copied from the template. SwiftF0 analyses
# 0.15 s windows in ~10 ms, so 64 slots is ~7 windows of slack. A Whistle pass
# costs 1-4 s, so the slot count is driven by two hard requirements:
#
# 1. Correctness: _chunks() holds references to every block of the current
#    chunk while it accumulates. If the audio thread's round-robin wraps onto
#    a slot the worker still references, the decoded audio is silently
#    corrupted. So the pool must exceed one whole chunk in blocks.
# 2. Throughput: one full decode pass has to fit in the ring, or the excess
#    arriving during it is dropped. That is a pass of seconds, not
#    milliseconds, so 64 slots is three orders of magnitude too small.
#
# Sized as the worst-case chunk plus pass headroom (~13 s = 1219 blocks) and
# rounded up. 2.6 MB per node instance: trivial next to the 17 MB model, and
# the alternative is either corrupted audio or a transcript full of holes.
POOL_SIZE = -(-(CHUNK_SECONDS_MAX + PASS_HEADROOM_SECONDS)
              * SAMPLE_RATE // BLOCK_SIZE)

# The live preview handed to the UI is capped: a multi-hour session would
# otherwise rebuild an ever-growing string on every committed step.
PREVIEW_CHARS = 4000

# Worker error reporting throttle: the first failure always surfaces, then
# every Nth consecutive one, so a persistently broken engine is visible
# without flooding _results_queue once per second.
ERROR_REPORT_EVERY = 10

# Fetch-progress throttle (same policy as AudioCppJob): the download callback
# fires per 1 MB chunk and would otherwise flood the NRT inbox.
_FETCH_PROGRESS_PCT = 5.0
_FETCH_PROGRESS_BYTES = 8 * 1024 * 1024

_ENV_LOCK = threading.Lock()
_ENV_READY = False
_MODEL_LOCK = threading.Lock()
_MODEL = None


def _configure_engine_env(model_dir):
    """Point cactus-needle at the fetched engine + weights, once per process.

    These are process-global variables read by ``cactus-needle`` when it
    resolves the engine and the weights; set only if they are not already
    ours, because every later read is a one-time engine load and churning
    the environment between sessions would only invite a mismatch with the
    model that is already mapped in the native library. ``HF_HUB_OFFLINE=1``
    is the belt-and-braces guarantee that a cache miss inside the package
    raises instead of silently reaching the network from this background
    thread (AGENTS.md section 4).

    ``NEEDLE_TELEMETRY``/``DO_NOT_TRACK`` are the upstream opt-outs: they are
    NOT on by default, ``needle.stream()`` calls ``_track()`` on every call,
    and ``HF_HUB_OFFLINE`` does not cover that path — so without them this
    node would ship anonymous usage counters off the user's machine while its
    own description promises the network is never touched. ANode opts out by
    default rather than making it a user setting.
    """
    global _ENV_READY
    with _ENV_LOCK:
        if _ENV_READY:
            return
        os.environ["NEEDLE3_LIB_PATH"] = str(engine_library_path(model_dir))
        os.environ["NEEDLE_WHISTLE_WEIGHTS"] = str(weights_path(model_dir))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["NEEDLE_TELEMETRY"] = "0"
        os.environ["DO_NOT_TRACK"] = "1"
        _ENV_READY = True


def _load_model(weights):
    """One loaded Whistle per process, shared by every session.

    ``Whistle`` maps the .cact into the native engine and is explicitly not
    thread-safe, and the engine cannot be unloaded once mapped: constructing
    a second instance per session would leak a whole model into the library.
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            import needle

            _MODEL = needle.Whistle(weights=str(weights))
        return _MODEL


class WhistleStreamer(Node):
    # "Voice & Pitch" is the palette bucket for the realtime neural speech /
    # vocal nodes (SwiftF0, vocos, RubberBand shift); test_system_integrity
    # asserts a closed category set and it has no "Utilities" member.
    category = "Voice & Pitch"
    label = "Whistle Live Transcriber"
    description = (
        "Live speech-to-text tap built on Whistle, Cactus Compute's 16.9 MB "
        "Apache-2.0 CPU speech-to-text model (~17 MB download, fetched once "
        "with the Download button or tools/whistle/fetch.py and never "
        "auto-fetched). The incoming audio is downmixed to 16 kHz mono, fed "
        "to the model on a background worker and transcribed live: text is "
        "committed a few seconds behind the audio, with word timestamps "
        "(start, end and probability per word) and keyword biasing for names "
        "and product words. English, German, French, Spanish, Italian, Dutch "
        "and Polish are supported; the language can be forced or detected per "
        "session. The streaming decoder re-reads a trailing window on every "
        "pass, so longer chunks_seconds transcribe more audio per second at "
        "the cost of latency; if the machine cannot keep up with real time the "
        "node drops the oldest audio and says so in its status rather than "
        "silently shipping a transcript with a hole in it. The audio passes "
        "through bit-exact so it can be monitored while it is being "
        "transcribed, and the transcript plus its word timestamps are written "
        "to disk and published on the URI outputs when the transport stops. "
        "Listen can be toggled at any time, including mid-session. Inference "
        "is entirely local and never touches the network; note that the "
        "native engine cannot be unloaded once it has been used for the "
        "first time in a session."
    )

    def __init__(self, name=""):
        super().__init__(name)

        # 1. Sockets
        self.inp = self.add_input(
            "in", help="Audio to transcribe; mono inputs are duplicated to stereo.")
        self.out = self.add_output(
            "out", channels=CHANNELS,
            help="Bit-exact pass-through of the input, unaltered, so the audio "
                 "can be monitored while transcribing.")
        self.add_uri_output(
            "transcript_uri",
            help="Path of the transcript text file written when the transport stops.")
        self.add_uri_output(
            "words_uri",
            help="Path of the word-timestamp JSON (start, end, probability per "
                 "word) written when the transport stops.")

        # 2. Parameters
        self.add_bool_param(
            "listen", False,
            help="Arm live transcription. Can be toggled mid-session: turning "
                 "it on starts the background worker, turning it off stops it "
                 "and flushes the transcript.")
        self.add_menu_param("language", whistle_language_menu(), initial_idx=0,
                            help="Recognition language, or auto-detect. A forced "
                                 "language is more reliable than detection on "
                                 "known-language audio.")
        self.add_string_param(
            "keywords", "",
            help="Comma- or newline-separated biasing terms (names, places, "
                 "product words) the decoder should favour; empty means no biasing.")
        self.add_int_param(
            "chunk_seconds", DEFAULT_CHUNK_SECONDS,
            CHUNK_SECONDS_MIN, CHUNK_SECONDS_MAX,
            help="Seconds of audio per streaming decode pass. The decoder "
                 "re-reads a trailing window on every pass to decide which "
                 "words two passes agree on, so longer chunks transcribe more "
                 "audio per second at the cost of latency; shorter chunks are "
                 "quicker to first text but fall behind real time and make the "
                 "node drop audio. Raise this if the status reports dropped audio.")
        self.add_string_param("model_dir", DEFAULT_MODEL_DIR,
                              help="Model directory holding whistle.cact and bin/ (repo-relative).")
        self.add_bool_param(
            "download", False,
            help="Transient trigger: downloads the missing model + engine in "
                 "the background (resumable), then resets itself.")
        # Last kept artifacts, for save/load re-linking. Written on flush.
        self.add_string_param("last_txt", "",
                              help="Path of the last transcript; re-linked on patch load.")
        self.add_string_param("last_words", "",
                              help="Path of the last word timestamps; re-linked on patch load.")

        # 3. Real-time thread state
        self._mono_scratch = np.zeros(BLOCK_SIZE, dtype=np.float32)

        # 4. Ring buffers (audio thread <-> worker thread)
        self._audio_queue = SPSCRingBuffer(capacity=POOL_SIZE)
        self._results_queue = SPSCRingBuffer(capacity=POOL_SIZE)
        self.monitor_queue = TelemetryDictRingBuffer(capacity=4)

        self._mono_pool = [np.zeros(BLOCK_SIZE, dtype=np.float32)
                           for _ in range(POOL_SIZE)]
        self._pool_seq = 0
        # Blocks the worker never got to because the ring was full: the audio
        # thread increments this (an int add, still allocation-free), the
        # worker reads it when reporting.
        self._dropped_blocks = 0

        # 5. Worker thread and transcript state
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._worker_generation = 0
        self._status = "Idle"
        self._status_detail = "Not listening"
        self._transcript = ""
        self._pending = ""
        self._language = ""
        # Readiness probe cache for the Download button (see _assets_ready).
        self._ready_path = None
        self._ready_value = None

    # ------------------------------------------------------------------
    # helpers (engine/control thread)
    # ------------------------------------------------------------------
    def _engine(self):
        graph = getattr(self, "graph", None)
        return getattr(graph, "engine", None) if graph is not None else None

    def _restage(self, name, value):
        """Deliberate re-stage of a transient trigger (AGENTS.md section 5)."""
        self.params[name].set(value)
        self.params[name].sync()

    def _resolve_model_dir(self):
        path = Path(self.params["model_dir"].value or DEFAULT_MODEL_DIR)
        return path if path.is_absolute() else REPO_ROOT / path

    def _fail(self, message):
        self._status = "Error"
        self._status_detail = message
        self.error_msg = message
        logger.error(f"[{self.name}] {message}")

    # ------------------------------------------------------------------
    # audio thread
    # ------------------------------------------------------------------
    def process(self):
        sig = self.inp.get_tensor()

        # 1. Bit-exact pass-through. copy_ broadcasts a mono (1, B) input
        #    into both output channels without resizing the pre-allocated
        #    destination; a functional out= would shrink it and downgrade the
        #    port (AGENTS.md section 2).
        self.out.buffer.copy_(sig)

        # 2. Drain worker results, dropping anything stamped with a previous
        #    session's generation (stale-result rejection).
        while True:
            res, ok = self._results_queue.try_pop()
            if not ok:
                break
            if res.get("gen") != self._worker_generation:
                continue

            error = res.get("error")
            if error:
                # The worker reports through the ring rather than writing
                # error_msg from another thread: this block *is* the owner of
                # the write, which is what AGENTS.md section 3 requires.
                self._status = "Error"
                self._status_detail = error
                self.error_msg = error
                continue
            self._transcript = res.get("text") or ""
            self._pending = res.get("pending") or ""
            self._language = res.get("language") or ""
            self._status = "Listening"
            self._status_detail = self._describe(res)
            if res.get("recovered"):
                self.error_msg = None

        # 3. Hand the worker the next mono block: pool index, never a copy.
        if self._worker_thread is not None and self._worker_thread.is_alive():
            if sig.shape[0] > 1:
                np.add(sig[0].cpu().numpy(), sig[1].cpu().numpy(),
                       out=self._mono_scratch)
                self._mono_scratch *= 0.5
            else:
                np.copyto(self._mono_scratch, sig[0].cpu().numpy())
            idx = self._pool_seq % len(self._mono_pool)
            self._pool_seq += 1
            np.copyto(self._mono_pool[idx], self._mono_scratch)
            # A full ring means the worker is behind; drop this block rather
            # than block the audio thread (AGENTS.md section 11). This is the
            # path that actually loses audio in practice: a decode pass costs
            # far more than a 10.7 ms block, so the ring fills long before the
            # worker's own overrun cap can engage. Counted so the worker can
            # tell the user the transcript has a gap in it.
            if not self._audio_queue.try_push(idx):
                self._dropped_blocks += 1

    @staticmethod
    def _describe(res):
        received = res.get("received") or 0.0
        pass_ms = res.get("pass_ms") or 0.0
        detail = f"{received:.1f} s transcribed"
        if pass_ms:
            detail += f", {pass_ms:.0f} ms/pass"
        # Falling behind is not a transcription error, but it is a data-loss
        # event the user has to know about: the transcript has a hole in it.
        dropped = res.get("dropped_s") or 0.0
        if dropped >= 0.5:
            detail += f", DROPPED {dropped:.0f} s — raise chunk_seconds"
        return detail

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self._worker_generation += 1
        self._reset_session()
        self._stop_event.clear()
        if self.params["listen"].value:
            self._spawn_worker()

    def stop(self):
        self._stop_event.set()
        thread = self._worker_thread
        if thread is not None:
            engine = self._engine()
            if engine is not None:
                # Non-blocking: stop + join happen on the NRT pool.
                engine.nrt.stop_stream(self, lambda: self._stop_event.set(),
                                       thread)
            else:
                thread.join(timeout=1.0)
            self._worker_thread = None

    def remove(self):
        self.stop()

    def _reset_session(self):
        self.out.buffer.zero_()
        self._status = "Idle"
        self._status_detail = "Not listening"
        self._transcript = ""
        self._pending = ""
        self._language = ""
        self._dropped_blocks = 0
        while self._audio_queue.try_pop()[1]:
            pass
        while self._results_queue.try_pop()[1]:
            pass
        while self.monitor_queue.pop_latest() is not None:
            pass

    def _spawn_worker(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        gen = self._worker_generation
        engine = self._engine()
        if engine is not None:
            self._worker_thread = engine.nrt.spawn_stream(self._worker_loop, gen)
        else:
            self._worker_thread = threading.Thread(
                target=self._worker_loop, args=(gen,), daemon=True)
            self._worker_thread.start()

    def on_ui_param_change(self, param_name: str):
        if param_name == "listen":
            if self.params["listen"].value:
                # Mid-session arm: same reset as a transport start, so a
                # previous session's partial text can never leak into this one.
                self._worker_generation += 1
                self._reset_session()
                self._stop_event.clear()
                self._spawn_worker()
            else:
                self.stop()
        elif param_name == "download":
            if not self.params["download"].value:
                return
            self._restage("download", False)
            self._request_download()

    # ------------------------------------------------------------------
    # worker thread (NRT stream)
    # ------------------------------------------------------------------
    def _push_nrt(self, tag, ok, payload, epoch):
        """Deliver a terminal control-side message to this node's NRT inbox.

        ``submit_nrt`` is never used for the worker, so the inbox has to be
        created here. ``NRTExecutor.drain`` only decrements ``_in_flight`` for
        nodes present in that dict (``if ... node in self._in_flight``), so a
        direct push with an ordinary tag is delivered safely and needs no
        ``_PROGRESS_TAGS`` change.
        """
        inbox = self._nrt_inbox
        if inbox is None:
            inbox = queue.SimpleQueue()
            self._nrt_inbox = inbox
        inbox.put((epoch, tag, ok, payload))

    def _worker_loop(self, generation: int):
        """Background transcription worker: resample, decode, publish."""
        epoch = self._nrt_epoch
        # `listen` normally decided whether this thread exists at all, but the
        # toggle can land between spawn_stream() and the first scheduling:
        # re-read it once so a disarmed node never loads a 17 MB model.
        if not self.params["listen"].value or self._stop_event.is_set():
            return
        # The worker owns its transcript lists outright: nothing outside this
        # thread reads them, so a start()/stop() cycle can never mutate a list
        # a retiring worker is still appending to.
        chunks: list = []
        words: list = []

        model_dir = self._resolve_model_dir()

        try:
            import needle
        except ImportError:
            self._push_nrt("worker_status", False,
                           "Missing dependency: cactus-needle — install it in "
                           "the anode-dev env (see environment.yml)", epoch)
            return
        try:
            import soxr
        except ImportError:
            self._push_nrt("worker_status", False,
                           "Missing dependency: soxr — install "
                           "cactus-needle[mic] in the anode-dev env "
                           "(see environment.yml)", epoch)
            return

        ok, missing = check_assets(model_dir)
        if not ok:
            self._push_nrt(
                "worker_status", False,
                f"Whistle assets missing in {model_dir} ({', '.join(missing)}) "
                "— run: python tools/whistle/fetch.py (or press Download)",
                epoch)
            return

        language = language_code(self.params["language"].value)
        keywords = parse_keywords(self.params["keywords"].value)
        chunk_samples = self._chunk_samples_48k()

        try:
            _configure_engine_env(model_dir)
            model = _load_model(weights_path(model_dir))
        except Exception as exc:
            self._push_nrt("worker_status", False,
                           f"Whistle engine failed to load: {exc}", epoch)
            return

        # soxr carries its anti-alias filter state across calls, so a
        # streaming 48 kHz -> 16 kHz conversion stays phase-continuous where
        # SwiftF0's stateless overlap-save decimator would not have needed to
        # (there the ratio is an exact 3:1; here it is also exactly 3:1 but
        # the stream boundary is where the filter state matters).
        resampler = soxr.ResampleStream(SAMPLE_RATE, WHISTLE_SAMPLE_RATE, 1,
                                        dtype=np.float32, quality="HQ")

        # The engine pulls from self._chunks() inside this loop, so the
        # generator, the resampler and the native stream all live on this one
        # thread: no lock, no cross-thread stream handle.
        failures = 0
        try:
            for step in model.stream(
                    self._chunks(resampler, chunk_samples),
                    language=language, keywords=keywords):
                if not isinstance(step, dict):
                    # Keep streaming across an unusable chunk — the next one
                    # may well be fine — but never stay silent about a
                    # persistent failure (throttled, or a 1 Hz failure would
                    # flood _results_queue and the UI with the same message).
                    failures += 1
                    if failures == 1 or failures % ERROR_REPORT_EVERY == 0:
                        self._results_queue.try_push({
                            "gen": generation,
                            "error": f"Whistle returned an unusable step "
                                     f"({type(step).__name__})",
                        })
                    continue
                failures = 0
                text = step.get("text") or ""
                if text:
                    chunks.append(text)
                step_words = step.get("words") or []
                if step_words:
                    words.extend(step_words)
                transcript = _join(chunks)
                self._results_queue.try_push({
                    "gen": generation,
                    "text": transcript[-PREVIEW_CHARS:],
                    "pending": step.get("pending") or "",
                    "language": step.get("language") or "",
                    "received": step.get("received") or 0.0,
                    "pass_ms": step.get("pass_ms") or 0.0,
                    "dropped_s": _dropped_seconds(self),
                    "recovered": True,
                })
        except Exception as exc:
            self._results_queue.try_push({
                "gen": generation, "error": f"Whistle stream failed: {exc}"})

        self._flush(model_dir, chunks, words, language, epoch,
                    _dropped_seconds(self))

    def _chunk_samples_48k(self):
        """Configured chunk length in 48 kHz samples, clamped to the menu range."""
        try:
            seconds = int(self.params["chunk_seconds"].value)
        except (TypeError, ValueError):
            seconds = DEFAULT_CHUNK_SECONDS
        seconds = min(max(seconds, CHUNK_SECONDS_MIN), CHUNK_SECONDS_MAX)
        return seconds * SAMPLE_RATE

    def _chunks(self, resampler, chunk_samples):
        """16 kHz mono chunk generator pulled by ``Whistle.stream``.

        Runs on the worker thread (the engine pulls from it inside the
        ``for`` loop), terminates on ``_stop_event`` so the stream's own
        ``finally`` gets to call ``needle_stream_transcribe_stop``, and keeps
        only references into ``_mono_pool`` — no per-block copies.

        There is deliberately no local backlog cap here. This generator clears
        its buffer *before* each yield, so its working set never exceeds one
        chunk and a trim threshold could never fire; the only place audio can
        actually be lost is ``process()`` finding the ring full, which is
        counted in ``_dropped_blocks``.
        """
        accum = []
        held = 0
        while not self._stop_event.is_set():
            idx, ok = self._audio_queue.try_pop()
            if not ok:
                self._stop_event.wait(0.005)
                continue
            accum.append(self._mono_pool[idx])
            held += BLOCK_SIZE
            if held < chunk_samples:
                continue
            audio_48k = np.concatenate(accum)
            accum, held = [], 0
            yield resampler.resample_chunk(audio_48k)

        # Drain the resampler's tail so the last sub-block of the session is
        # not silently truncated before the engine's final pass.
        tail = resampler.resample_chunk(np.zeros(0, np.float32), last=True)
        if len(tail):
            yield tail

    def _flush(self, model_dir, chunks, words, language, epoch, dropped_s=0.0):
        """Write the artifacts and hand the paths to the control thread."""
        transcript = _join(chunks)
        if not transcript.strip():
            # Nothing was committed (silence, or a session stopped early).
            # Writing an empty transcript would publish an empty URI and bury
            # the previous one, so leave the last artifacts alone.
            self._push_nrt("worker_status", True, "No speech transcribed", epoch)
            return
        keep = Path(model_dir) / "outputs" / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_whistle")
        try:
            keep.mkdir(parents=True, exist_ok=True)
            txt_path = keep / "transcript.txt"
            words_path = keep / "words.json"
            txt_path.write_text(transcript + "\n", encoding="utf-8")
            words_path.write_text(json.dumps({
                "language": language or "",
                # Audio the overrun cap discarded: the transcript reads as one
                # continuous utterance but a gap here means it is not.
                "dropped_seconds": round(dropped_s, 2),
                "words": words,
            }, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self._push_nrt("worker_status", False,
                           f"Could not write the transcript: {exc}", epoch)
            return
        self._push_nrt("flush", True, {
            "txt": str(txt_path),
            "words": str(words_path),
            "text": transcript,
            "n_words": len(words),
        }, epoch)

    # ------------------------------------------------------------------
    # NRT inbox
    # ------------------------------------------------------------------
    def on_nrt_complete(self, tag, ok, result):
        if tag == "flush":
            if not ok:
                self._fail(f"Writing the transcript failed: {result}")
                return
            txt = result.get("txt") or ""
            self._transcript = result.get("text") or ""
            self._pending = ""
            self.outputs["transcript_uri"].uri = txt
            self.outputs["words_uri"].uri = result.get("words") or ""
            self._restage("last_txt", txt)
            self._restage("last_words", result.get("words") or "")
            self.error_msg = None
            self._status = "Ready"
            self._status_detail = (
                f"{Path(txt).name}: {len(self._transcript)} chars, "
                f"{result.get('n_words', 0)} words")
        elif tag == "worker_status":
            if ok:
                self._status = "Idle"
                self._status_detail = result or "Idle"
            else:
                self._fail(str(result))
        elif tag == "fetch_progress":
            if ok:
                self._status = "Downloading"
                self._status_detail = _format_progress(result)
        elif tag == "fetch":
            self._on_fetch_complete(ok, result)
        # Unknown tags are ignored: stream teardown (stop_stream) delivers
        # completions with tag=None and must never touch node state.

    def on_nrt_discarded(self, tag, ok, result):
        """Release what a superseded/deleted result carried.

        The flush payload is paths plus transcript text — no open handles, no
        native resources — so there is nothing to free; dropping the
        reference keeps a deleted node from retaining the text until GC.
        """
        if tag == "flush" and isinstance(result, dict):
            result.clear()

    # ------------------------------------------------------------------
    # download
    # ------------------------------------------------------------------
    def _request_download(self):
        if self._engine() is None:
            self._fail("Node is not attached to an engine.")
            return
        try:
            payload = self._build_fetch_payload()
        except ValueError as exc:
            self._fail(str(exc))
            return
        if not payload["specs"]:
            self.error_msg = None
            self._status = "Idle"
            self._status_detail = "Model and engine already downloaded"
            return
        self.error_msg = None
        self._status = "Downloading"
        self._status_detail = "Starting download…"
        self.submit_nrt(self._fetch_nrt, payload, tag="fetch")

    def _build_fetch_payload(self):
        model_dir = self._resolve_model_dir()
        specs = whistle_fetch_specs(model_dir)
        if engine_library_path(model_dir).is_file():
            # The library is unpacked from the wheel rather than downloaded
            # directly, so an already-extracted engine means the wheel itself
            # is no longer needed.
            specs = [s for s in specs if s.dest.suffix != ".whl"]
        pending = missing_specs(specs)
        return {
            "specs": [{"url": s.url, "dest": str(s.dest), "size": s.size,
                       "sha256": s.sha256, "label": s.label} for s in pending],
            "model_dir": str(model_dir),
        }

    def _fetch_nrt(self, payload):
        """Download the missing assets with inbox progress. NRT only."""
        epoch = self._nrt_epoch
        inbox = self._nrt_inbox
        last_report = {}

        def _progress(label, done, total, state):
            if inbox is None:
                return
            if state == "downloading":
                prev_pct, prev_done = last_report.get(
                    label, (-1.0, -_FETCH_PROGRESS_BYTES))
                pct = (100.0 * done / total) if total else prev_pct
                complete = bool(total) and done >= total
                if (not complete and (pct - prev_pct) < _FETCH_PROGRESS_PCT
                        and (done - prev_done) < _FETCH_PROGRESS_BYTES):
                    return
                last_report[label] = (pct, done)
            try:
                inbox.put((epoch, "fetch_progress", True,
                           {"label": label, "done": done, "total": total,
                            "state": state}))
            except Exception:
                pass

        model_dir = Path(payload["model_dir"])
        specs = [DownloadSpec(s["url"], Path(s["dest"]), s["size"],
                              s["sha256"], s["label"])
                 for s in payload["specs"]]
        try:
            fetch_all(specs, progress_cb=_progress)
        except DownloadCancelled:
            return {"cancelled": True}
        library = extract_engine_wheel(model_dir)
        return {"library": str(library),
                "fetched": [s.label for s in specs]}

    def _on_fetch_complete(self, ok, result):
        if not ok:
            if isinstance(result, DownloadCancelled):
                self._status = "Idle"
                self._status_detail = "Download cancelled — re-run to resume"
                self.error_msg = None
            else:
                self._fail(f"Download failed: {result}")
            return
        if isinstance(result, dict) and result.get("cancelled"):
            self._status = "Idle"
            self._status_detail = "Download cancelled — re-run to resume"
            self.error_msg = None
            return
        # The files on disk just changed: drop the cached readiness probe so
        # the widget re-enables Listen immediately.
        self._ready_value = None
        self.error_msg = None
        self._status = "Idle"
        self._status_detail = "Download complete — press Listen"

    # ------------------------------------------------------------------
    # telemetry / save-load
    # ------------------------------------------------------------------
    def get_telemetry(self) -> dict:
        data = {
            "status": self._status,
            "audio": self._status_detail,
            "detail": self._preview(),
            # Capped like every other transcript value: this dict is pushed at
            # ~10 Hz and a long session's full text would otherwise be rebuilt
            # and copied into the telemetry ring on every tick.
            "text": self._transcript[-PREVIEW_CHARS:],
            "pending": self._pending,
            "language": self._language,
            "ready": self._assets_ready(),
            "busy": self._busy(),
        }
        self.monitor_queue.push(data)
        return data

    def _assets_ready(self):
        """Cached existence probe for the widget's Download button.

        ``get_telemetry`` runs at ~10 Hz on the engine thread; two ``stat()``
        calls per tick would be filesystem work in the engine loop for a value
        that only changes when model_dir changes or a download finishes.
        """
        path = str(self.params["model_dir"].value)
        if path != self._ready_path:
            self._ready_path = path
            self._ready_value = None
        if self._ready_value is None:
            try:
                self._ready_value = assets_ready(self._resolve_model_dir())
            except OSError:
                self._ready_value = False
        return self._ready_value

    def _preview(self):
        return self._status_detail + (
            "\n" + self._transcript[-PREVIEW_CHARS:] if self._transcript else "")

    def _busy(self):
        thread = self._worker_thread
        return thread is not None and thread.is_alive()

    def load_state(self, data: dict):
        super().load_state(data)
        # A restored patch must never start a model on load.
        self._restage("listen", False)
        self._restage("download", False)
        # Menu index order is part of the patch format; clamp a value written
        # by a build with a different language list.
        items = whistle_language_menu()
        self._restage("language", min(max(int(self.params["language"].value),
                                         0), len(items) - 1))
        self._restage("chunk_seconds", self._chunk_samples_48k() // SAMPLE_RATE)
        self._reset_session()

        # Transcripts are small text: re-link synchronously on the control
        # thread (same call Qwen3ASRTranscriber.load_state makes). No
        # transcript is read back here, only its existence.
        txt = self.params["last_txt"].value
        words = self.params["last_words"].value
        self.outputs["words_uri"].uri = words if words and Path(words).exists() else ""
        if txt and Path(txt).exists():
            self.outputs["transcript_uri"].uri = txt
            self._status = "Ready"
            self._status_detail = f"Re-linked {Path(txt).name}"
        else:
            self.outputs["transcript_uri"].uri = ""
            if txt:
                self._status = "Idle"
                self._status_detail = "Previous transcript is missing"


def _join(chunks):
    """Full committed transcript so far (empty chunks contribute nothing)."""
    return " ".join(chunk for chunk in chunks if chunk).strip()


def _dropped_seconds(node):
    """Seconds of 48 kHz audio this session failed to transcribe.

    One loss path only: a decode pass costs far more than the 10.7 ms of
    audio arriving during it, so the ring fills and ``process()`` never hands
    those blocks over at all. The audio thread counts them in
    ``_dropped_blocks`` (an int add, still allocation-free) and the worker
    reads the total when reporting. A transcript that reads as one continuous
    utterance but is missing speech is worse than one that admits the gap, so
    this reaches the node status and the saved words.json.
    """
    return node._dropped_blocks * BLOCK_SIZE / float(SAMPLE_RATE)


def _format_progress(progress):
    label = progress.get("label", "download")
    done, total = progress.get("done", 0), progress.get("total")
    state = progress.get("state", "")
    if state == "present":
        return f"{label}: already present"
    if state == "verifying":
        return f"{label}: verifying…"
    if total:
        return f"{label}: {100.0 * done / total:.1f}% ({format_bytes(done)} / {format_bytes(total)})"
    return f"{label}: {format_bytes(done)}"


if GUI_AVAILABLE:

    class WhistleStreamerWidget(QWidget):
        """Listen toggle + language/keywords + live transcript browser."""

        IS_NODE_UI = True
        NODE_CLASS_NAME = "WhistleStreamer"

        def __init__(self, node_proxy):
            super().__init__()
            self.proxy = node_proxy
            self.setMinimumWidth(280)
            self._rendered = None

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(6)

            self.btn_listen = QPushButton("Listen")
            self.btn_listen.setCheckable(True)
            self.btn_listen.setToolTip(
                "Arm/disarm live transcription. Switching it off stops the "
                "worker and writes the transcript.")
            self.btn_listen.toggled.connect(self._on_listen_toggled)
            layout.addWidget(self.btn_listen)

            self.param_widgets = {}
            for key in ("language", "keywords"):
                widget = self.proxy.create_param_widget(key)
                self.param_widgets[key] = widget
                layout.addWidget(widget)

            self.btn_download = QPushButton("Download model (~17 MB)")
            self.btn_download.clicked.connect(
                lambda: self.proxy.set_parameter("download", True))
            layout.addWidget(self.btn_download)

            self.lbl_status = QLabel("Idle")
            self.lbl_status.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.lbl_status)

            self.browser = QTextBrowser()
            self.browser.setFont(QFont("Monospace", 8))
            self.browser.setReadOnly(True)
            self.browser.setMinimumHeight(96)
            layout.addWidget(self.browser)

            self._ready = False
            self.timer = QTimer(self)
            self.timer.setInterval(100)  # engine telemetry cadence
            self.timer.timeout.connect(self._poll_monitor)
            self.timer.start()

        # -- parameter plumbing -----------------------------------------
        def _on_listen_toggled(self, checked):
            # Commit the keywords editor first: pressing Listen must not start
            # a session with the previous, unconfirmed biasing terms.
            widget = self.param_widgets.get("keywords")
            commit = getattr(widget, "_commit_text", None)
            if callable(commit):
                commit()
            self.proxy.set_parameter("listen", bool(checked))

        def update_from_params(self, params):
            for key, widget in self.param_widgets.items():
                if key in params:
                    widget.update_from_backend(params[key])
            if "listen" in params:
                self._sync_listen(bool(params["listen"]))
            if "model_dir" in params:
                self._refresh_ready()

        def _sync_listen(self, value):
            if self.btn_listen.isChecked() == value:
                return
            with QSignalBlocker(self.btn_listen):
                self.btn_listen.setChecked(value)

        # -- telemetry ---------------------------------------------------
        def _poll_monitor(self):
            telemetry = getattr(self.proxy, "monitor_queue", None)
            if telemetry is None:
                return
            data = telemetry.pop_latest()
            if data:
                self._apply(data)

        def on_telemetry(self, data: dict):
            if data:
                self._apply(data)

        def _apply(self, data: dict):
            self.lbl_status.setText(str(data.get("status", "Idle")))
            if "ready" in data:
                self._set_ready(bool(data["ready"]))
            self._render(data.get("text") or "", data.get("pending") or "")

        def _render(self, text, pending):
            from html import escape

            html = escape(text)
            if pending:
                html += ('<br><span style="color:#8a8a8a; '
                         'font-style:italic">%s</span>' % escape(pending))
            # Change-gate: the browser only re-renders when the text actually
            # moved, so a 10 Hz telemetry tick does not reset the scroll
            # position or re-lay-out the document.
            if html != self._rendered:
                self._rendered = html
                self.browser.setHtml(html or "&nbsp;")

        def _refresh_ready(self):
            node = self._node()
            if node is None:
                return
            try:
                self._set_ready(assets_ready(node._resolve_model_dir()))
            except OSError:
                self._set_ready(False)

        def _set_ready(self, ready):
            self._ready = ready
            self.btn_download.setEnabled(not ready)
            if ready:
                self.btn_download.setToolTip(
                    "Whistle model and engine are already installed.")
            else:
                self.btn_download.setToolTip(
                    "Whistle is not installed yet (~17 MB + the engine wheel). "
                    "Press Download, or run: python tools/whistle/fetch.py")

        def _node(self):
            engine = getattr(self.proxy.controller, "engine", None)
            if engine is None or not engine.graph:
                return None
            return engine.graph.node_map.get(self.proxy.node_id)

else:  # pragma: no cover - exercised only without PySide6

    class WhistleStreamerWidget:  # noqa: D401 - placeholder for headless docs
        IS_NODE_UI = False
        NODE_CLASS_NAME = "WhistleStreamer"