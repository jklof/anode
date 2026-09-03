"""Regression tests for the concurrency / RT-safety / DSP stabilization plan.

Covers:
1. DeleteNodeCommand undo before the queued 'del' has been processed
   (FIFO ordering + pre-captured type metadata; no audio-thread cls()).
2. TelemetryDictRingBuffer copies the slot BEFORE advancing the consumer
   tail (producer may immediately reuse the slot).
3. NRTExecutor tracks in-flight jobs across node deletion (discard ->
   drain_discarded -> on_nrt_discarded), then retires tracking.
4. MonoToStereo equal-power pan law.
5. SineOscillator.start() resets the phase accumulator (AGENTS.md §7).
6. WaveShaper soft-clip saturates at ±2/3 with no boolean-mask sync.
7. MediaPlayer EOF auto-stop requests the param change through the engine
   command queue, exactly once (one-shot guard).
"""

import queue as queue_mod
import threading
import time

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, TelemetryDictRingBuffer
from core import Engine, NRTExecutor
from commands import DeleteNodeCommand
from plugins.basic_nodes import MonoToStereo, SineOscillator
from plugins.waveshaper import WaveShaper


# ==============================================================================
# 1. DeleteNodeCommand undo with an unprocessed queued 'del'
# ==============================================================================


def test_delete_undo_before_del_processed_restores_via_fifo():
    """Undo may run while the engine still has the ('del', id, holder)
    queued. FIFO ordering guarantees the holder is filled by the time the
    'restore' is applied; the pre-captured type metadata avoids
    instantiating the node class on the audio thread."""
    plugin_system.load_plugins("plugins")
    engine = Engine()
    engine.running = True  # async path: commands sit in the queue

    cls = plugin_system.NODE_REGISTRY["Gain"]
    for nid in ("a", "x"):
        n = cls()
        n.id = nid
        engine.push_command(("restore", (n.to_dict(), n)))
    engine.push_command(("conn", "a", "out", "x", "in"))

    # Controller emulating the real AppController contract: get_node_data
    # reads the latest UI snapshot cache. The command pre-captures the
    # immutable type/name metadata from it.
    ctl = type("C", (), {"engine": engine,
                         "get_node_data": staticmethod(
                             lambda nid: {"id": "x", "type": "Gain", "name": ""})})()
    delete_cmd = DeleteNodeCommand(ctl, "x")
    delete_cmd.execute()  # queues del; holder empty
    assert delete_cmd.node_data is None

    # Undo is invoked BEFORE the executor has drained the 'del'. The undo
    # must still enqueue 'restore' (no early return) using the
    # pre-captured metadata.
    assert delete_cmd.node_type == "Gain"
    delete_cmd.undo()

    # Now drain everything in FIFO order.
    while not engine.command_queue.empty():
        _, cmd = engine.command_queue.get_nowait()
        engine._apply_command(cmd)

    assert "x" in engine.graph.node_map
    assert engine.graph.node_map["x"].params["vol"].value == 1.0
    assert engine.graph.node_map["x"].inputs["in"].connected_outputs

    # The restored connection must have been announced to the UI: when the
    # engine is running, no full snapshot follows this command, so without
    # a "connected" event the wire stays invisible after undo.
    events = []
    while True:
        try:
            events.append(engine.output_queue.get_nowait())
        except queue_mod.Empty:
            break
    connected = [e for e in events if e.get("type") == "connected"]
    assert {"src_id": "a", "src_port": "out", "dst_id": "x", "dst_port": "in"} in \
        [{k: e[k] for k in ("src_id", "src_port", "dst_id", "dst_port")} for e in connected]


# ==============================================================================
# 2. TelemetryDictRingBuffer snapshot-before-tail-advance
# ==============================================================================


def test_telemetry_ring_pop_copies_before_tail_advance():
    rb = TelemetryDictRingBuffer(capacity=4)
    assert rb.push({"a": 1})
    out = rb.pop_latest()
    assert out == {"a": 1}

    # Producer overwrites the freed slot immediately after the consumer
    # pops; the previously returned dict must be unaffected.
    rb.push({"a": 1})
    first = rb.pop_latest()
    rb.push({"a": 999, "b": 2})
    assert first == {"a": 1}

    rb.push({"c": 3})
    item, ok = rb.try_pop()
    assert ok and item == {"c": 3}


# ==============================================================================
# 3. NRTExecutor in-flight tracking across node deletion
# ==============================================================================


class _DiscardSpyNode:
    def __init__(self):
        self.id = "spy"
        self.name = "spy"
        self._nrt_epoch = 0
        self._nrt_inbox = None
        self.completed = []
        self.discarded = []

    def on_nrt_complete(self, tag, ok, payload):
        self.completed.append((tag, ok, payload))

    def on_nrt_discarded(self, tag, ok, payload):
        self.discarded.append((tag, ok, payload))


def test_nrt_discard_drains_inflight_results():
    nrt = NRTExecutor(max_workers=2)
    node = _DiscardSpyNode()

    release = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        release.wait(timeout=5.0)
        return "payload"

    nrt.submit(node, slow, (), tag="job")
    assert started.wait(timeout=5.0)

    # Delete the node while the job is still running.
    nrt.discard(node)

    # Simulate the node being gone from the graph: _drain_nrt_all must
    # still reach it via drain_discarded().
    release.set()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        nrt.drain_discarded()
        if node.discarded:
            break
        time.sleep(0.01)

    assert node.discarded == [("job", True, "payload")]
    assert node.completed == []
    # Tracking must be retired once quiescent (bounded references).
    assert node not in nrt._discarded_nodes
    nrt._pool.shutdown(wait=True)


def test_nrt_discard_without_pending_work_does_not_track():
    nrt = NRTExecutor(max_workers=2)
    node = _DiscardSpyNode()
    nrt.discard(node)
    assert node not in nrt._discarded_nodes
    nrt._pool.shutdown(wait=True)


# ==============================================================================
# 4. MonoToStereo equal-power pan law
# ==============================================================================


class _FakeOutput:
    """Minimal stand-in for an OutputSlot: get_tensor() only touches .buffer."""

    def __init__(self, value, channels=1):
        self.buffer = torch.full((channels, BLOCK_SIZE), value, dtype=torch.float32)


def _feed_mono(node, value):
    # Connect a mono source output; an unconnected non-param-bound input
    # returns a zeroed scratch by design.
    node.inp.connect(_FakeOutput(value, channels=1))


def test_mono_to_stereo_equal_power_center():
    n = MonoToStereo()
    _feed_mono(n, 1.0)
    n.process()
    expected = np.cos(np.pi / 4)
    assert torch.allclose(n.out.buffer[0], torch.full((BLOCK_SIZE,), expected))
    assert torch.allclose(n.out.buffer[1], torch.full((BLOCK_SIZE,), expected))


def test_mono_to_stereo_equal_power_hard_pans():
    n = MonoToStereo()
    _feed_mono(n, 1.0)
    n.params["pan"].set(1.0)
    n.params["pan"].sync()
    n.process()
    assert torch.allclose(n.out.buffer[0], torch.zeros(BLOCK_SIZE))
    assert torch.allclose(n.out.buffer[1], torch.ones(BLOCK_SIZE))


# ==============================================================================
# 5. SineOscillator start() resets phase
# ==============================================================================


def test_sine_start_resets_phase():
    n = SineOscillator()
    # freq/amp are param-bound slots; unconnected they return the parameter
    # constant cache.
    n.params["freq"].set(440.0)
    n.params["freq"].sync()
    n.params["amp"].set(1.0)
    n.params["amp"].sync()
    n.process()
    assert n.phase != 0.0
    n.start()
    assert n.phase == 0.0


# ==============================================================================
# 6. WaveShaper soft clip saturation
# ==============================================================================


def test_waveshaper_soft_clip_saturates_without_mask():
    n = WaveShaper()
    n.inp.connect(_FakeOutput(3.0, channels=1))
    for name, val in (("mode", 1), ("drive", 1.0), ("bias", 0.0),
                      ("mix", 1.0), ("output_level", 1.0)):
        n.params[name].set(val)
        n.params[name].sync()
    n.process()
    # Driven = 3.0 (|x| > 1): saturated value must be sign(x) * 2/3.
    expected = 2.0 / 3.0
    assert torch.allclose(
        n.out.buffer, torch.full((CHANNELS, BLOCK_SIZE), expected), atol=1e-6
    )
    # Output buffer must keep its full (CHANNELS, BLOCK_SIZE) shape.
    assert n.out.buffer.shape == (CHANNELS, BLOCK_SIZE)


# ==============================================================================
# 7. MediaPlayer EOF auto-stop via engine command queue, once
# ==============================================================================


class _StubParam:
    def __init__(self, value):
        self.value = value
        self.set_calls = []

    def set(self, v):
        self.set_calls.append(v)
        self.value = v


class _StubEngine:
    def __init__(self):
        self.commands = []

    def push_command(self, cmd):
        self.commands.append(cmd)


class _StubGraph:
    def __init__(self):
        self.engine = _StubEngine()


def test_media_player_eof_auto_stop_uses_command_queue_once():
    from plugins.media_player import MediaPlayerNode

    node = MediaPlayerNode()
    node.graph = _StubGraph()
    # Stub the 'playing' param so we can observe direct mutation attempts.
    node.params["playing"] = _StubParam(True)
    node.queue = queue_mod.Queue()  # underrun branch
    node.worker = None
    node.eof = True
    node._eof_reported = False

    node.process()
    node.process()
    node.process()

    # Exactly one command queued (one-shot guard), no direct param.set().
    assert node.graph.engine.commands == [("param", node.id, "playing", False)]
    assert node.params["playing"].set_calls == []
    # The flag latches even though EOF persists.
    assert node._eof_reported is True

    # Playback restart resets the guard.
    node._handle_worker_event("seeked", 0.0)
    assert node._eof_reported is False
