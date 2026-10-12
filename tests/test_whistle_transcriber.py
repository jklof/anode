"""Tests for the Whistle live-transcription node.

Everything here must pass with ``cactus-needle`` and ``soxr`` NOT installed:
the real engine is never imported, the worker is stubbed with the same
``_FakeWorker`` duck type ``test_swift_f0.py`` uses, and the pre-flight
failure paths are exercised instead.
"""

import gc
import json
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE

import whistle_backend as wb


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("WhistleStreamer")
    assert cls is not None, "WhistleStreamer not registered"
    return cls()


def widget_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.get_ui_class("WhistleStreamer")
    assert cls is not None and getattr(cls, "IS_NODE_UI", False)
    return cls


def node_module():
    """Live top-level plugin module object (see test_qwen3_asr_transcriber)."""
    import sys
    plugin_system.load_plugins("plugins")
    return sys.modules["whistle_transcriber"]


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    inst = QCoreApplication.instance()
    if inst is None:
        return QApplication([])
    if isinstance(inst, QApplication):
        return inst
    pytest.skip("a bare QCoreApplication is active; QWidget tests cannot run")


def process_block(node, tensor):
    node.inp.get_tensor = lambda t=tensor: t
    node.process()


def sine_block(freq=440.0, amp=0.5):
    n = np.arange(BLOCK_SIZE)
    tone = (amp * np.sin(2 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    return torch.from_numpy(np.tile(tone, (CHANNELS, 1)))


class _FakeWorker:
    """Duck-typed stand-in for the worker thread handle."""

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


# ----------------------------------------------------------------------
# backend helpers (pure, no engine)
# ----------------------------------------------------------------------
def test_language_menu_and_codes():
    menu = wb.whistle_language_menu()
    assert menu[0] == "Auto (detect)"
    assert menu[1:] == ["EN", "DE", "FR", "ES", "IT", "NL", "PL"]
    assert len(menu) == len(wb.WHISTLE_LANGUAGES) + 1
    for index, code in enumerate(wb.WHISTLE_LANGUAGES, start=1):
        assert wb.language_code(index) == code
    assert wb.language_code(0) is None
    assert wb.language_code(-1) is None
    assert wb.language_code(len(menu)) is None
    assert wb.language_code("nope") is None


def test_parse_keywords_accepts_commas_and_newlines():
    assert wb.parse_keywords("") == []
    assert wb.parse_keywords(None) == []
    assert wb.parse_keywords("  anode , whistle  ") == ["anode", "whistle"]
    assert wb.parse_keywords("anode\nwhistle") == ["anode", "whistle"]
    assert wb.parse_keywords("a,,b,") == ["a", "b"]


def test_fetch_specs_cover_weights_and_pinned_engine():
    specs = wb.whistle_fetch_specs(Path("models/Whistle"))
    labels = [s.label for s in specs]
    assert "Whistle: whistle.cact" in labels
    assert "Whistle: LICENSE" in labels
    weights = next(s for s in specs if s.dest.name == "whistle.cact")
    assert weights.size == 16919407
    assert weights.sha256 == (
        "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb")
    # LICENSE is a plain git blob upstream: size-checked, no content hash.
    license_spec = next(s for s in specs if s.dest.name == "LICENSE")
    assert license_spec.size == 11358 and license_spec.sha256 is None
    # The engine wheel, on Windows x64.
    wheel = next((s for s in specs if s.dest.suffix == ".whl"), None)
    if wb.engine_platform_tag() == "win_amd64":
        assert wheel is not None
        assert wheel.dest.parent.name == "bin"
        assert wheel.size == 699518
        assert wheel.sha256 == (
            "4e50c5fa983c8737de200d9e96af937dd8135371ca557fe8c7c9eb1adba39f5a")
        assert wb.ENGINE_LIB_SIZE == 1531392


def test_check_assets_reports_missing(tmp_path):
    ok, missing = wb.check_assets(tmp_path)
    assert ok is False
    assert "whistle.cact" in missing
    assert wb.assets_ready(tmp_path) is False


def test_extract_engine_wheel_verifies_and_cleans_up(tmp_path):
    import io
    import zipfile
    import hashlib

    # Wrong member name -> nothing is written.
    wheel = wb.engine_wheel_path(tmp_path)
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("needle/not-the-engine.bin", b"x" * 16)
    with pytest.raises(wb.DownloadError):
        wb.extract_engine_wheel(tmp_path)
    assert not wb.engine_library_path(tmp_path).exists()

    # Correct member but wrong bytes -> verification rejects it and leaves
    # no .part behind (a corrupt DLL would otherwise fail inside ctypes).
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"needle/{wb.engine_library_name()}", b"y" * 32)
    with pytest.raises(wb.DownloadError):
        wb.extract_engine_wheel(tmp_path)
    lib = wb.engine_library_path(tmp_path)
    assert not lib.exists()
    assert not lib.with_name(lib.name + ".part").exists()

    # Correct bytes -> installed. Built by patching the pinned size/hash so
    # the test never needs a real 1.5 MB engine blob.
    payload = b"z" * 4096
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"needle/{wb.engine_library_name()}", payload)
    original = (wb.ENGINE_LIB_SIZE, wb.ENGINE_LIB_SHA256)
    wb.ENGINE_LIB_SIZE = len(payload)
    wb.ENGINE_LIB_SHA256 = hashlib.sha256(payload).hexdigest()
    try:
        out = wb.extract_engine_wheel(tmp_path)
    finally:
        wb.ENGINE_LIB_SIZE, wb.ENGINE_LIB_SHA256 = original
    assert out == wb.engine_library_path(tmp_path)
    assert out.read_bytes() == payload


# ----------------------------------------------------------------------
# registration / documentation contract
# ----------------------------------------------------------------------
def test_registration_and_metadata():
    node = make_node()
    assert node.category == "Voice & Pitch"
    assert node.label == "Whistle Live Transcriber"
    assert len(node.description) >= 15
    # The brief's key claims must actually appear in the palette help text.
    for token in ("Apache-2.0", "16 kHz", "word timestamps", "keyword",
                  "second", "cannot be unloaded"):
        assert token in node.description, token
    # Not an offline job node: it is real-time and has no pulse output.
    assert getattr(node, "is_offline", False) is False
    assert not hasattr(node, "PULSE_OUTPUT_NAME")
    assert "done" not in node.outputs


def test_documentation_contract():
    plugin_system.load_plugins("plugins")
    doc = plugin_system.get_node_documentation("WhistleStreamer")
    assert doc["inputs"]["in"]["help"]
    assert doc["inputs"]["in"]["slot_type"] == "audio"

    assert doc["outputs"]["out"]["slot_type"] == "audio"
    assert doc["outputs"]["out"]["channels"] == CHANNELS
    assert doc["outputs"]["out"]["help"]
    for uri_port in ("transcript_uri", "words_uri"):
        assert doc["outputs"][uri_port]["slot_type"] == "uri"
        assert doc["outputs"][uri_port]["help"]

    for name in ("listen", "language", "keywords", "model_dir", "download",
                 "last_txt", "last_words"):
        assert doc["params"][name]["help"], name
        assert isinstance(doc["params"][name]["unit"], str)

    assert doc["params"]["listen"]["value"] is False
    assert doc["is_offline"] is False

    ui_cls = plugin_system.get_ui_class("WhistleStreamer")
    assert ui_cls is not None
    assert ui_cls.IS_NODE_UI is True
    assert ui_cls.NODE_CLASS_NAME == "WhistleStreamer"


# ----------------------------------------------------------------------
# audio thread
# ----------------------------------------------------------------------
def test_pass_through_and_mono_buffer_shape_guard():
    """A (1, BLOCK_SIZE) input must broadcast, never shrink out.buffer."""
    node = make_node()
    mono = torch.full((1, BLOCK_SIZE), 0.35, dtype=DTYPE)

    process_block(node, mono)

    assert node.out.buffer.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.allclose(node.out.buffer[0], mono[0])
    assert torch.allclose(node.out.buffer[1], mono[0])


def test_pass_through_is_bit_exact_stereo():
    node = make_node()
    blk = sine_block(220.0)
    process_block(node, blk)
    assert node.out.buffer.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(node.out.buffer, blk)


def test_start_zeroes_output_and_clears_state():
    node = make_node()
    node.out.buffer.fill_(0.9)
    node._transcript = "stale"
    node._pending = "tail"
    node.start()
    assert node.out.buffer.abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert node._transcript == ""
    assert node._pending == ""
    node.stop()


def test_lifecycle_no_thread_unless_listening():
    node = make_node()
    node.start()
    try:
        assert node._worker_thread is None
    finally:
        node.stop()

    node._restage("listen", True)
    node.start()
    try:
        assert node._worker_thread is not None
    finally:
        node.stop()
    assert node._worker_thread is None


def test_stop_joins_worker_without_engine():
    node = make_node()
    started = []
    node._worker_loop = lambda gen: started.append(gen) or _sleep_forever(
        node._stop_event)
    node._restage("listen", True)
    node.start()
    assert node._worker_thread is not None
    node.stop()
    node._worker_thread = None  # stop() already detached it
    assert node._stop_event.is_set()


def _sleep_forever(stop_event):
    while not stop_event.wait(0.01):
        pass


def test_pool_index_handoff():
    """process() hands the worker a pool *index*, never a fresh numpy copy."""
    node = make_node()
    blk = sine_block(440.0)
    process_block(node, blk)  # warm-up (no worker: push path skipped)

    node._worker_thread = _FakeWorker()
    try:
        for _ in range(10):
            process_block(node, blk)

        indices = []
        while True:
            payload, ok = node._audio_queue.try_pop()
            if not ok:
                break
            indices.append(payload)

        assert node._pool_seq >= 10
        assert len(indices) == 10
        assert all(isinstance(i, int) for i in indices)
        assert len(set(indices)) == 10, "indices must be unique within a cycle"
        assert all(0 <= i < len(node._mono_pool) for i in indices)

        # Stereo inputs are averaged into the referenced slot.
        expected = (blk[0].numpy() + blk[1].numpy()) * 0.5
        for i in indices:
            assert node._mono_pool[i].shape == (BLOCK_SIZE,)
            assert np.allclose(node._mono_pool[i], expected, atol=1e-7)
    finally:
        node._worker_thread = None


def test_zero_net_allocation():
    node = make_node()
    blk = sine_block(440.0)
    node._worker_thread = _FakeWorker()

    for _ in range(5):
        process_block(node, blk)

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        process_block(node, blk)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    node._worker_thread = None

    assert growth < 64 * 1024, f"net allocation {growth} bytes over 50 blocks"


def test_backpressure_never_blocks_or_raises():
    """A worker that never drains must not stall or break the audio thread."""
    node = make_node()
    node._worker_thread = _FakeWorker()
    blk = sine_block(440.0)
    try:
        for _ in range(node._audio_queue._cap + 40):
            process_block(node, blk)
        # Ring full: pushes now return False and blocks are dropped.
        assert node._audio_queue.try_push(0) is False
        process_block(node, blk)
        assert node.out.buffer.shape == (CHANNELS, BLOCK_SIZE)
        assert node.error_msg is None
    finally:
        node._worker_thread = None


# ----------------------------------------------------------------------
# generation guard
# ----------------------------------------------------------------------
def test_stale_generation_payload_is_dropped():
    node = make_node()
    node._worker_generation = 7
    node._status = "Listening"
    node._transcript = "kept"

    node._results_queue.try_push({
        "gen": node._worker_generation - 1,
        "text": "from a previous session",
        "pending": "stale tail",
        "language": "de",
        "recovered": True,
    })
    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE))

    assert node._transcript == "kept"
    assert node._pending == ""
    assert node._language == ""
    assert node._status == "Listening"


def test_current_generation_payload_is_applied():
    node = make_node()
    node._results_queue.try_push({
        "gen": node._worker_generation,
        "text": "turn off the lights",
        "pending": "and the",
        "language": "en",
        "received": 3.0,
        "pass_ms": 240.0,
        "recovered": True,
    })
    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE))

    assert node._transcript == "turn off the lights"
    assert node._pending == "and the"
    assert node._language == "en"
    assert node._status == "Listening"
    assert node.error_msg is None
    assert "3.0 s" in node._status_detail


def test_error_payload_sets_error_msg():
    node = make_node()
    node._results_queue.try_push({
        "gen": node._worker_generation,
        "error": "Whistle stream failed: boom",
    })
    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE))
    assert node.error_msg == "Whistle stream failed: boom"
    assert node._status == "Error"


# ----------------------------------------------------------------------
# telemetry
# ----------------------------------------------------------------------
def test_monitor_queue_round_trip():
    """A worker-shaped payload reaches the widget through monitor_queue."""
    node = make_node()
    node._results_queue.try_push({
        "gen": node._worker_generation,
        "text": "hello there",
        "pending": "genera",
        "language": "en",
        "received": 2.0,
        "pass_ms": 180.0,
        "recovered": True,
    })
    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE))

    telemetry = node.get_telemetry()
    assert telemetry["status"] == "Listening"
    assert telemetry["text"] == "hello there"
    assert telemetry["pending"] == "genera"
    assert telemetry["language"] == "en"
    assert telemetry["busy"] is False
    assert "ready" in telemetry

    frame = node.monitor_queue.pop_latest()
    assert frame is not None
    assert frame["text"] == "hello there"
    assert frame["pending"] == "genera"
    # The widget consumes exactly these keys.
    for key in ("status", "text", "pending", "ready", "busy"):
        assert key in frame
    assert node.monitor_queue.pop_latest() is None


def test_telemetry_preview_is_capped():
    node = make_node()
    node._transcript = "x" * 20000
    data = node.get_telemetry()
    assert data["detail"].endswith(node._transcript[-200:])
    # Every transcript-shaped value is capped, not just the detail string:
    # get_telemetry() runs at ~10 Hz, so an uncapped "text" would be rebuilt
    # and copied into the telemetry ring on every tick of a long session.
    assert data["text"] == node._transcript[-node_module().PREVIEW_CHARS:]
    assert len(data["text"]) <= node_module().PREVIEW_CHARS


def test_engine_env_opts_out_of_upstream_telemetry(monkeypatch):
    """The node promises the network is never touched, so it must actually opt out.

    cactus-needle's usage counters are ON by default and _track() is called from
    needle.stream(); HF_HUB_OFFLINE does not cover that path.
    """
    import os

    mod = node_module()
    for name in ("NEEDLE3_LIB_PATH", "NEEDLE_WHISTLE_WEIGHTS",
                 "HF_HUB_OFFLINE", "NEEDLE_TELEMETRY", "DO_NOT_TRACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mod, "_ENV_READY", False)

    mod._configure_engine_env("models/Whistle")

    assert os.environ["NEEDLE_TELEMETRY"] == "0"
    assert os.environ["DO_NOT_TRACK"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["NEEDLE3_LIB_PATH"].endswith(wb.engine_library_name())
    assert os.environ["NEEDLE_WHISTLE_WEIGHTS"].endswith("whistle.cact")


def test_chunk_seconds_is_clamped_to_the_supported_range():
    node = make_node()
    mod = node_module()
    for raw, expected in ((0, mod.CHUNK_SECONDS_MIN), (-5, mod.CHUNK_SECONDS_MIN),
                          (999, mod.CHUNK_SECONDS_MAX), (4, 4)):
        node._restage("chunk_seconds", raw)
        assert node._chunk_samples_48k() == expected * SAMPLE_RATE
    mod = node_module()
    assert mod.DEFAULT_CHUNK_SECONDS == 4


def test_pool_is_large_enough_to_hold_a_whole_chunk():
    """The pool must outlast one chunk, or the worker decodes corrupted audio.

    _chunks() holds a reference to every block of the chunk it is
    accumulating. The audio thread cycles the pool round-robin, so a pool
    smaller than one chunk means slots get overwritten while the worker is
    still reading them — silently wrong audio, not just a dropped block.
    """
    mod = node_module()
    blocks_per_max_chunk = (mod.CHUNK_SECONDS_MAX * SAMPLE_RATE) // BLOCK_SIZE
    assert mod.POOL_SIZE > blocks_per_max_chunk

    node = make_node()
    assert len(node._mono_pool) == mod.POOL_SIZE
    assert node._audio_queue._cap == mod.POOL_SIZE


def test_pool_absorbs_a_decode_pass_without_dropping():
    """One full decode pass has to fit in the ring, or audio arriving during
    it is lost. SwiftF0's 64 slots cover its ~10 ms pass many times over; a
    Whistle pass costs seconds, so the pool has to be sized from that."""
    mod = node_module()
    # A 3 s pass is mid-range for this model on one core.
    pass_blocks = (3 * SAMPLE_RATE) // BLOCK_SIZE
    # Worst-case chunk plus one pass, with margin, still fits.
    assert mod.POOL_SIZE > pass_blocks
    assert mod.POOL_SIZE >= (mod.CHUNK_SECONDS_MAX + 3) * SAMPLE_RATE / BLOCK_SIZE


def test_full_ring_counts_dropped_blocks_on_the_audio_thread():
    """Audio the worker never receives must be counted, not silently lost.

    A decode pass costs far more than the 10.7 ms of audio arriving during
    it, so the ring fills and process() drops the excess. The transcript then
    reads as one continuous utterance with a hole in it, which is why the
    count has to reach the status instead of staying a debug number.
    """
    node = make_node()
    node._worker_thread = _FakeWorker()

    # Far more blocks than the pool/ring can hold.
    blk = sine_block()
    for _ in range(len(node._mono_pool) + 200):
        process_block(node, blk)
    node._worker_thread = None

    assert node._dropped_blocks >= 200
    dropped_s = node._dropped_blocks * BLOCK_SIZE / SAMPLE_RATE
    assert dropped_s > 0.5

    detail = node._describe({"received": 10.0, "pass_ms": 900,
                             "dropped_s": dropped_s})
    assert "DROPPED" in detail and "chunk_seconds" in detail
    # A clean run never claims a loss.
    assert "DROPPED" not in node._describe({"received": 10.0, "pass_ms": 900,
                                            "dropped_s": 0.0})


def test_chunks_yield_chunk_samples_and_terminate_on_stop():
    node = make_node()
    resampler = SimpleNamespace(
        resample_chunk=lambda buf, **kw: np.zeros(len(buf) // 3, np.float32))
    gen = node._chunks(resampler, BLOCK_SIZE * 2)

    for i in range(4):
        idx = i % len(node._mono_pool)
        node._mono_pool[idx].fill(0.1)
        node._audio_queue.try_push(idx)
    # Two blocks make one chunk; the next pull hits the empty ring.
    first = next(gen)
    assert len(first) == BLOCK_SIZE * 2 // 3

    node._stop_event.set()
    # The fake resampler returns nothing for the empty flush, so the generator
    # finishes rather than yielding a tail.
    with pytest.raises(StopIteration):
        next(gen)


def test_flush_records_dropped_seconds_in_words_json(tmp_path):
    node = make_node()
    inbox = []
    node._push_nrt = lambda tag, ok, payload, epoch: inbox.append((tag, payload))
    node._flush(str(tmp_path), ["hello there"], [], "en", node._nrt_epoch,
                dropped_s=3.5)
    payload = inbox[0][1]
    assert json.loads(Path(payload["words"]).read_text())["dropped_seconds"] == 3.5
    assert Path(payload["txt"]).read_text().strip() == "hello there"


# ----------------------------------------------------------------------
# NRT inbox
# ----------------------------------------------------------------------
def test_flush_publishes_uris_and_last_params():
    node = make_node()
    node.error_msg = "old failure"
    node._status = "Listening"
    node._pending = "tail"

    node.on_nrt_complete("flush", True, {
        "txt": str(Path("models/Whistle/outputs/x/transcript.txt")),
        "words": str(Path("models/Whistle/outputs/x/words.json")),
        "text": "one two three",
        "n_words": 3,
    })

    assert node.outputs["transcript_uri"].uri.endswith("transcript.txt")
    assert node.outputs["words_uri"].uri.endswith("words.json")
    assert node.params["last_txt"].value.endswith("transcript.txt")
    assert node.params["last_words"].value.endswith("words.json")
    assert node._transcript == "one two three"
    assert node._pending == ""
    assert node.error_msg is None
    assert node._status == "Ready"


def test_unknown_and_none_tags_are_ignored():
    node = make_node()
    before = (node._status, node.error_msg, node.outputs["transcript_uri"].uri)
    # stop_stream teardown delivers completions with tag=None.
    node.on_nrt_complete(None, True, None)
    node.on_nrt_complete("something_else", False, RuntimeError("boom"))
    node.on_nrt_complete("fetch_progress", False, None)
    assert (node._status, node.error_msg,
            node.outputs["transcript_uri"].uri) == before


def test_on_nrt_discarded_drops_flush_payload():
    node = make_node()
    payload = {"txt": "a", "words": "b", "text": "hi"}
    node.on_nrt_discarded("flush", True, payload)
    assert payload == {}
    # Non-flush traffic is not ours to touch.
    other = {"anything": 1}
    node.on_nrt_discarded(None, True, other)
    assert other == {"anything": 1}


def test_worker_pushes_actionable_status_without_engine(tmp_path):
    """No cactus-needle/soxr installed: the worker must say so, not crash."""
    node = make_node()
    node._restage("listen", True)
    node._restage("model_dir", str(tmp_path))
    gen = node._worker_generation

    node._worker_loop(gen)

    epoch, tag, ok, payload = node._nrt_inbox.get_nowait()
    assert epoch == node._nrt_epoch
    assert tag == "worker_status" and ok is False
    assert payload, "the worker must explain itself instead of dying quietly"
    assert any(token in payload for token in
               ("cactus-needle", "soxr", "tools/whistle/fetch.py"))

    node.on_nrt_complete(tag, ok, payload)
    assert node.error_msg is not None
    assert node._status == "Error"


def test_worker_reports_missing_assets(tmp_path, monkeypatch):
    """With a stubbed engine import but no assets, the message names the fix."""
    import sys
    import types as pytypes

    stub = pytypes.ModuleType("needle")
    stub.Whistle = object  # never constructed: check_assets fails first
    monkeypatch.setitem(sys.modules, "needle", stub)
    monkeypatch.setitem(sys.modules, "soxr", pytypes.ModuleType("soxr"))

    node = make_node()
    node._restage("listen", True)
    node._restage("model_dir", str(tmp_path))

    node._worker_loop(node._worker_generation)

    _epoch, tag, ok, payload = node._nrt_inbox.get_nowait()
    assert tag == "worker_status" and ok is False
    assert "tools/whistle/fetch.py" in payload


# ----------------------------------------------------------------------
# end-to-end worker path with a stubbed engine (no cactus-needle needed)
# ----------------------------------------------------------------------
def _stub_engine(monkeypatch, node, n_chunks, model=None):
    """Install a fake `needle` + `soxr` and return a record of what was seen."""
    import sys
    import types as pytypes

    import whistle_transcriber as wt

    seen = {"in_lengths": [], "chunk_sizes": [], "options": None}

    class _FakeResampler:
        def __init__(self, rate_in, rate_out, channels, dtype=None, quality=None):
            seen["rates"] = (rate_in, rate_out, channels, quality)

        def resample_chunk(self, data, last=False):
            seen["in_lengths"].append(len(data))
            # 3:1, like the real 48k -> 16k conversion.
            return np.asarray(data[:len(data) // 3], dtype=np.float32)

    class _FakeModel:
        def stream(self, chunks, language=None, keywords=None):
            seen["options"] = (language, keywords)
            for index, chunk in enumerate(chunks, start=1):
                seen["chunk_sizes"].append(len(chunk))
                if index >= n_chunks:
                    node._stop_event.set()
                yield {
                    "text": f"word{index}",
                    "words": [{"word": f"word{index}", "start": 0.0,
                               "end": 1.0, "probability": 0.9}],
                    "pending": "par",
                    "language": language or "en",
                    "received": float(index),
                    "pass_ms": 210.0,
                }

    soxr_stub = pytypes.ModuleType("soxr")
    soxr_stub.ResampleStream = _FakeResampler
    monkeypatch.setitem(sys.modules, "needle", pytypes.ModuleType("needle"))
    monkeypatch.setitem(sys.modules, "soxr", soxr_stub)
    monkeypatch.setattr(wt, "check_assets", lambda model_dir: (True, []))
    monkeypatch.setattr(wt, "_configure_engine_env", lambda model_dir: None)
    monkeypatch.setattr(wt, "_load_model",
                        lambda weights: (model() if model else _FakeModel()))
    # One block == one Whistle chunk keeps the test to a handful of pushes
    # (the pool is 64 slots and must not be recycled under the generator).
    # The chunk length is a node parameter now, so drive it through the node
    # rather than a module constant.
    monkeypatch.setattr(node, "_chunk_samples_48k", lambda: BLOCK_SIZE)
    return seen


def test_worker_streams_resamples_and_flushes_artifacts(tmp_path, monkeypatch):
    node = make_node()
    node._restage("listen", True)
    node._restage("model_dir", str(tmp_path))
    node._restage("language", 2)          # menu index 2 == DE
    node._restage("keywords", "anode, whistle")
    seen = _stub_engine(monkeypatch, node, n_chunks=3)

    # Three mono blocks standing in for three engine seconds.
    for _ in range(3):
        node._mono_pool[node._pool_seq % len(node._mono_pool)].fill(0.25)
        node._audio_queue.try_push(node._pool_seq % len(node._mono_pool))
        node._pool_seq += 1

    node._worker_loop(node._worker_generation)

    # 48 kHz -> 16 kHz conversion with the pinned rates and stateful soxr.
    assert seen["rates"][:3] == (SAMPLE_RATE, 16000, 1)
    assert seen["in_lengths"][:3] == [BLOCK_SIZE] * 3
    assert seen["chunk_sizes"][:3] == [BLOCK_SIZE // 3] * 3
    # Language and keyword biasing are forwarded verbatim.
    assert seen["options"] == ("de", ["anode", "whistle"])

    # The worker reports through the ring, the audio thread applies it.
    applied = []
    while True:
        payload, ok = node._results_queue.try_pop()
        if not ok:
            break
        applied.append(payload)
    assert [p["text"] for p in applied] == ["word1", "word1 word2",
                                            "word1 word2 word3"]
    assert applied[0]["pending"] == "par"
    assert applied[0]["received"] == pytest.approx(1.0)

    # The terminal flush lands on the control thread.
    _epoch, tag, ok, payload = node._nrt_inbox.get_nowait()
    assert tag == "flush" and ok is True
    txt, words = Path(payload["txt"]), Path(payload["words"])
    assert txt.read_text(encoding="utf-8").strip() == "word1 word2 word3"
    saved = json.loads(words.read_text(encoding="utf-8"))
    assert saved["language"] == "de"
    assert [w["word"] for w in saved["words"]] == ["word1", "word2", "word3"]
    assert saved["words"][0]["probability"] == pytest.approx(0.9)

    node.on_nrt_complete(tag, ok, payload)
    assert node.outputs["transcript_uri"].uri == str(txt)
    assert node.outputs["words_uri"].uri == str(words)
    assert node._transcript == "word1 word2 word3"
    assert node._status == "Ready"


def test_worker_survives_unusable_step_and_keeps_streaming(tmp_path, monkeypatch):
    node = make_node()
    node._restage("listen", True)
    node._restage("model_dir", str(tmp_path))

    class _FlakyModel:
        def stream(self, chunks, language=None, keywords=None):
            for index, chunk in enumerate(chunks, start=1):
                if index == 2:
                    yield "not a dict"          # must be skipped, not fatal
                    continue
                if index >= 4:
                    node._stop_event.set()
                yield {
                    "text": f"word{index}",
                    "words": [],
                    "pending": "",
                    "language": "en",
                    "received": float(index),
                    "pass_ms": 100.0,
                }

    _stub_engine(monkeypatch, node, n_chunks=4, model=_FlakyModel)

    for _ in range(4):
        idx = node._pool_seq % len(node._mono_pool)
        node._mono_pool[idx].fill(0.1)
        node._audio_queue.try_push(idx)
        node._pool_seq += 1

    node._worker_loop(node._worker_generation)

    errors, texts = [], []
    while True:
        payload, ok = node._results_queue.try_pop()
        if not ok:
            break
        if payload.get("error"):
            errors.append(payload["error"])
        else:
            texts.append(payload["text"])
    assert len(errors) == 1 and "unusable step" in errors[0]
    assert texts == ["word1", "word1 word3", "word1 word3 word4"]
    _epoch, tag, ok, payload = node._nrt_inbox.get_nowait()
    assert tag == "flush" and ok is True


# ----------------------------------------------------------------------
# save / load
# ----------------------------------------------------------------------
def test_load_state_forces_listen_false_and_clamps_language(tmp_path):
    node = make_node()
    keep = tmp_path / "outputs"
    keep.mkdir(parents=True)
    txt = keep / "transcript.txt"
    txt.write_text("earlier take", encoding="utf-8")
    words = keep / "words.json"
    words.write_text("{}", encoding="utf-8")

    node.load_state({
        "params": {
            "listen": True,
            "download": True,
            "language": 99,
            "last_txt": str(txt),
            "last_words": str(words),
        },
    })

    assert node.params["listen"].value is False
    assert node.params["download"].value is False
    assert node.params["language"].value == len(wb.whistle_language_menu()) - 1
    assert node.outputs["transcript_uri"].uri == str(txt)
    assert node.outputs["words_uri"].uri == str(words)
    assert node._status == "Ready"


def test_load_state_missing_transcript_falls_back():
    node = make_node()
    node.load_state({
        "params": {
            "listen": True,
            "language": -5,
            "last_txt": str(Path("does/not/exist.txt")),
            "last_words": "",
        },
    })
    assert node.params["listen"].value is False
    assert node.params["language"].value == 0
    assert node.outputs["transcript_uri"].uri == ""
    assert node.outputs["words_uri"].uri == ""
    assert "missing" in node._status_detail


def test_save_load_round_trip_keeps_artifact_relink():
    node = make_node()
    node._restage("language", 2)
    node._restage("keywords", "anode, whistle")
    data = node.to_dict()

    restored = make_node()
    restored.load_state(data)
    assert restored.params["language"].value == 2
    assert restored.params["keywords"].value == "anode, whistle"
    assert restored.params["listen"].value is False


# ----------------------------------------------------------------------
# widget
# ----------------------------------------------------------------------
def _param_widget(commit_hook):
    """A real QWidget the layout can hold, with the two hooks the node's
    widgets use (StringParamWidget._commit_text / update_from_backend)."""
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    widget.update_from_backend = lambda value: None
    widget._commit_text = commit_hook
    return widget


def test_widget_renders_committed_and_pending(qapp):
    cls = widget_cls()

    class _Proxy:
        node_id = "n1"
        monitor_queue = None
        controller = SimpleNamespace(engine=None)

        def create_param_widget(self, name):
            return _param_widget(lambda: None)

        def set_parameter(self, name, value):
            pass

    widget = cls(_Proxy())
    widget.on_telemetry({"status": "Listening", "text": "hello",
                         "pending": "wor", "ready": True})

    html = widget.browser.toHtml()
    assert "hello" in html
    assert "wor" in html
    assert "font-style:italic" in html or "font-style: italic" in html
    assert widget.lbl_status.text() == "Listening"
    # Ready -> Download disabled with no install hint.
    assert widget.btn_download.isEnabled() is False
    assert "already installed" in widget.btn_download.toolTip() \
        or "installed" in widget.btn_download.toolTip()

    # Change-gate: identical payload must not rebuild the document.
    before = widget._rendered
    widget.on_telemetry({"status": "Listening", "text": "hello",
                         "pending": "wor", "ready": True})
    assert widget._rendered == before


def test_widget_listen_toggle_commits_keywords_first(qapp):
    cls = widget_cls()
    order = []

    class _Proxy:
        node_id = "n1"
        monitor_queue = None
        controller = SimpleNamespace(engine=None)

        def create_param_widget(self, name):
            return _param_widget(lambda: order.append("commit"))

        def set_parameter(self, name, value):
            order.append((name, value))

    widget = cls(_Proxy())
    widget.btn_listen.setChecked(True)
    assert order[0] == "commit"
    assert order[1] == ("listen", True)


def test_widget_unready_download_button_explains(qapp):
    cls = widget_cls()

    class _Proxy:
        node_id = "n1"
        monitor_queue = None
        controller = SimpleNamespace(engine=None)

        def create_param_widget(self, name):
            return _param_widget(lambda: None)

        def set_parameter(self, name, value):
            pass

    widget = cls(_Proxy())
    widget.on_telemetry({"status": "Idle", "text": "", "pending": "",
                         "ready": False})
    assert widget.btn_download.isEnabled() is True
    assert "tools/whistle/fetch.py" in widget.btn_download.toolTip()