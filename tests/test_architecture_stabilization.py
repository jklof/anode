"""Regression tests for architecture stabilization work:

1. NAM native-state ownership (NRT prepares an independent DSP state)
2. FileRecorder writer-side file ownership
3. Command identity / connect_rejected association
4. Authoritative delete memento capture (no stale undo)
5. Save as coherent synchronization point
8. FFINode dirty-flag native parameter sync
9. Channel compatibility at connect time
"""

import os
import queue
import threading
import time
import wave

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE, OutputSlot
from core import Engine, Graph
from commands import ConnectCommand, DeleteNodeCommand


# ==============================================================================
# Fakes
# ==============================================================================


class FakeLib:
    """Stand-in for a standard ANode C-ABI library."""

    def __init__(self, load_ok=True):
        self.load_ok = load_ok
        self.created = []
        self.destroyed = []
        self.loaded_handles = []
        self.set_param_calls = []
        self.reset_calls = []
        self._next = 1000

    def create(self):
        self._next += 1
        self.created.append(self._next)
        return self._next

    def destroy(self, handle):
        self.destroyed.append(handle)

    def set_samplerate(self, handle, sr):
        pass

    def load_model_sync(self, handle, path, sr, block):
        self.loaded_handles.append(handle)
        return 1 if self.load_ok else 0

    def set_param(self, handle, pid, val):
        self.set_param_calls.append((handle, pid, val))

    def reset(self, handle):
        self.reset_calls.append(handle)

    def process(self, handle, in_ptr, out_ptr, ch, frames):
        pass


def make_ffi_node(lib=None):
    from ffi_base import FFINode

    class _Node(FFINode):
        LIB_NAME = ""
        PARAM_MAP = {"vol": 0}

        def __init__(self, name=""):
            super().__init__(name)
            self.add_float_param("vol", 1.0, 0.0, 4.0)
            self.add_output("out")

    n = _Node("ffi")
    n.lib = lib if lib is not None else FakeLib()
    n.dsp_handle = n.lib.create()
    return n



# ==============================================================================
# 8. FFINode dirty-flag parameter sync
# ==============================================================================


def test_ffi_params_pushed_once_then_only_on_change():
    lib = FakeLib()
    node = make_ffi_node(lib)
    node.process()  # initial push
    initial = len(lib.set_param_calls)
    assert initial == 1  # one PARAM_MAP entry

    for _ in range(5):
        node.process()
    assert len(lib.set_param_calls) == initial  # unchanged -> no repeated pushes

    node.params["vol"].set(2.0)
    node.params["vol"].sync()
    node.process()
    assert len(lib.set_param_calls) == initial + 1
    assert lib.set_param_calls[-1][2] == 2.0


def test_ffi_reset_and_load_state_force_resync():
    lib = FakeLib()
    node = make_ffi_node(lib)
    node.process()
    n0 = len(lib.set_param_calls)

    node.start()
    assert lib.reset_calls  # native reset happened
    node.process()
    assert len(lib.set_param_calls) > n0  # re-push after reset

    n1 = len(lib.set_param_calls)
    node.load_state({"params": {"vol": 3.0}})
    node.process()
    assert len(lib.set_param_calls) > n1


# ==============================================================================
# 1. NAM native-state ownership
# ==============================================================================


def make_nam_node(load_ok=True):
    from types import SimpleNamespace
    from plugins.neural_amp import NamNode

    class _TestNam(NamNode):
        LIB_NAME = ""  # do not load the real native library

    node = _TestNam("nam")
    node.lib = FakeLib(load_ok=load_ok)
    node.dsp_handle = node.lib.create()
    node._load_epoch = 0

    # Capture NRT submissions so tests can verify work is deferred to the
    # background pool instead of running on the calling (engine/audio) thread.
    submitted = []

    class FakeNRT:
        def submit(self, n, fn, args, tag):
            submitted.append((fn, args, tag))

        def spawn_stream(self, target, *args):
            return None

    node.graph = SimpleNamespace(engine=SimpleNamespace(nrt=FakeNRT()))
    node._submitted_nrt = submitted
    return node


def test_nam_load_builds_new_handle_and_retires_old_off_audio_path():
    node = make_nam_node()
    old = node.dsp_handle

    node._load_epoch = 1  # as if on_ui_param_change submitted this load
    result = node._load_blocking("/tmp/fake.nam", epoch=1)
    new_handle, filename, epoch = result

    # The live handle was never passed to load_model_sync
    assert node.lib.loaded_handles == [new_handle]
    assert old not in node.lib.loaded_handles

    node.on_nrt_complete("load_model", True, result)
    assert node.dsp_handle == new_handle
    # Old handle retirement is DEFERRED to the NRT pool: it must not be
    # destroyed synchronously on the calling (engine/audio) thread...
    assert old not in node.lib.destroyed
    deferred = [s for s in node._submitted_nrt if s[2] == "cleanup_old_handle"]
    assert deferred and deferred[0][1] == (old,)
    # ...and executing the deferred job on the NRT thread retires it.
    deferred[0][0](*deferred[0][1])
    assert old in node.lib.destroyed
    assert node._status == "Active"


def test_nam_failed_load_destroys_prepared_handle_keeps_live():
    node = make_nam_node(load_ok=False)
    old = node.dsp_handle
    with pytest.raises(RuntimeError):
        node._load_blocking("/tmp/bad.nam", epoch=1)
    # prepared handle destroyed, live handle untouched
    assert node.lib.created[-1] in node.lib.destroyed
    assert node.dsp_handle == old


def test_nam_stale_result_is_rejected_and_destroyed():
    node = make_nam_node()
    old = node.dsp_handle
    result = node._load_blocking("/tmp/fake.nam", epoch=1)
    node._load_epoch = 2  # a newer load superseded this one
    node.on_nrt_complete("load_model", True, result)
    assert node.dsp_handle == old  # live state unchanged
    assert result[0] in node.lib.destroyed


def test_nam_two_overlapping_loads_install_latest():
    node = make_nam_node()
    node._load_epoch = 1
    r1 = node._load_blocking("/tmp/a.nam", epoch=1)
    node._load_epoch = 2
    r2 = node._load_blocking("/tmp/b.nam", epoch=2)
    node.on_nrt_complete("load_model", True, r1)  # stale arrives first
    node.on_nrt_complete("load_model", True, r2)
    assert node.dsp_handle == r2[0]
    assert r1[0] in node.lib.destroyed



# ==============================================================================
# 2. FileRecorder writer-side file ownership
# ==============================================================================


def test_file_recorder_opens_closes_on_writer_thread(tmp_path):
    import plugins.extended_nodes as ext

    open_threads = []
    real_open = ext.wave.open

    def tracking_open(*a, **kw):
        open_threads.append(threading.get_ident())
        return real_open(*a, **kw)

    ext.wave.open = tracking_open
    try:
        rec = ext.FileRecorder("rec")
        fname = str(tmp_path / "out.wav")
        rec.params["filename"].set(fname)
        rec.params["record"].set(True)
        rec.on_ui_param_change("record")

        # wait until the writer thread finished opening
        deadline = time.time() + 2.0
        while not rec._recording and time.time() < deadline:
            time.sleep(0.005)
        assert rec._recording, "writer never became ready"
        assert open_threads and open_threads[0] != threading.get_ident()

        # Audio path: enqueue a few blocks; must not block or touch files
        tone = torch.ones((CHANNELS, BLOCK_SIZE), dtype=torch.float32) * 0.25
        rec.inp.get_tensor = lambda: tone
        for _ in range(3):
            rec.process()

        rec.params["record"].set(False)
        rec.on_ui_param_change("record")  # engine not running -> joins writer
        deadline = time.time() + 2.0
        while rec._writer_thread is not None and time.time() < deadline:
            time.sleep(0.005)

        with wave.open(fname, "rb") as w:
            assert w.getnframes() == 3 * BLOCK_SIZE
            assert w.getnchannels() == CHANNELS
    finally:
        ext.wave.open = real_open


def test_file_recorder_start_stop_never_blocks_audio_path(tmp_path):
    import plugins.extended_nodes as ext

    rec = ext.FileRecorder("rec2")
    rec.params["filename"].set(str(tmp_path / "b.wav"))
    rec.params["record"].set(True)
    rec.on_ui_param_change("record")
    deadline = time.time() + 2.0
    while not rec._recording and time.time() < deadline:
        time.sleep(0.005)

    tone = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    rec.inp.get_tensor = lambda: tone
    t0 = time.perf_counter()
    for _ in range(200):
        rec.process()  # must return immediately even with a slow writer
    assert (time.perf_counter() - t0) < 0.5

    rec.params["record"].set(False)
    rec.on_ui_param_change("record")


# ==============================================================================
# 3. Command identity
# ==============================================================================


class _Ctl:
    def __init__(self):
        self.engine = Engine()


def test_push_command_returns_unique_ids():
    eng = Engine()
    ids = [eng.push_command(("snapshot",)) for _ in range(5)]
    assert len(set(ids)) == 5
    assert ids == sorted(ids)


def test_connect_rejected_carries_originating_cmd_id():
    plugin_system.load_plugins("plugins")
    ctl = _Ctl()
    g = ctl.engine.graph

    n1 = plugin_system.NODE_REGISTRY["Gain"]()
    n1.id = "a"
    n2 = plugin_system.NODE_REGISTRY["Gain"]()
    n2.id = "b"
    g.add_node(n1)
    g.add_node(n2)

    cmd = ConnectCommand(ctl, "a", "out", "b", "in")
    cmd.execute()
    good_id = cmd.cmd_id

    bad = ConnectCommand(ctl, "a", "out", "a", "in")  # self-loop -> rejected
    bad.execute()

    # Engine stopped: commands already applied; rejection message is queued
    msgs = []
    while not ctl.engine.output_queue.empty():
        msgs.append(ctl.engine.output_queue.get_nowait())
    rejected = [m for m in msgs if m["type"] == "connect_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["cmd_id"] == bad.cmd_id
    assert rejected[0]["cmd_id"] != good_id

    # History removal targets exactly the rejected command
    from controller import CommandHistory

    hist = CommandHistory()
    hist.push(cmd)
    hist.push(bad)
    assert hist.remove_by_cmd_id(rejected[0]["cmd_id"])
    assert len(hist.undo_stack) == 1
    assert hist.undo_stack[0] is cmd



# ==============================================================================
# 4. Authoritative delete memento capture
# ==============================================================================


def test_delete_captures_memento_at_executor_not_request_time():
    plugin_system.load_plugins("plugins")
    engine = Engine()
    engine.running = True  # force async path so we control executor ordering

    cls = plugin_system.NODE_REGISTRY["Gain"]
    for nid in ("a", "b", "c"):
        n = cls()
        n.id = nid
        engine.push_command(("restore", (n.to_dict(), n)))
    engine.push_command(("conn", "a", "out", "b", "in"))
    engine.push_command(("conn", "b", "out", "c", "in"))

    # Queue a move, then request a delete immediately (the race scenario)
    engine.push_command(("move", "b", 9.0, 9.0))
    ctl = type("C", (), {"engine": engine})()
    delete_cmd = DeleteNodeCommand(ctl, "b")
    delete_cmd.execute()  # queues ("del", "b", holder); nothing captured yet
    assert delete_cmd.node_data is None  # not captured at request time

    # Emulate the authoritative executor draining the queue in order
    while not engine.command_queue.empty():
        _, cmd = engine.command_queue.get_nowait()
        engine._apply_command(cmd)

    assert delete_cmd.node_data is not None
    assert tuple(delete_cmd.node_data["pos"]) == (9.0, 9.0)  # post-move state
    assert len(delete_cmd.connections) == 2

    engine.running = False
    delete_cmd.undo()
    assert "b" in engine.graph.node_map
    assert tuple(engine.graph.node_map["b"].pos) == (9.0, 9.0)
    b = engine.graph.node_map["b"]
    assert len(b.inputs["in"].connected_outputs) == 1
    assert b.inputs["in"].connected_outputs[0].parent.id == "a"


def test_delete_undo_restores_param_state():
    plugin_system.load_plugins("plugins")
    engine = Engine()
    ctl = type("C", (), {"engine": engine})()
    n = plugin_system.NODE_REGISTRY["Gain"]()
    n.id = "x"
    engine.push_command(("restore", (n.to_dict(), n)))
    engine.push_command(("param", "x", "vol", 0.75))

    delete_cmd = DeleteNodeCommand(ctl, "x")
    delete_cmd.execute()
    assert "x" not in engine.graph.node_map
    assert delete_cmd.node_data["params"]["vol"]["value"] == 0.75

    delete_cmd.undo()
    assert "x" in engine.graph.node_map
    assert engine.graph.node_map["x"].params["vol"].value == 0.75



# ==============================================================================
# 5. Save as coherent synchronization point
# ==============================================================================


def test_save_serializes_pending_param_value(tmp_path):
    plugin_system.load_plugins("plugins")
    engine = Engine()
    n = plugin_system.NODE_REGISTRY["Gain"]()
    n.id = "g1"
    engine.push_command(("restore", (n.to_dict(), n)))
    fname = str(tmp_path / "patch.json")

    # Coherence path: pending param command queued BEFORE save command
    engine.push_command(("param", "g1", "vol", 0.42))
    engine.push_command(("save", fname))
    deadline = time.time() + 2.0
    while not os.path.exists(fname) and time.time() < deadline:
        time.sleep(0.01)

    import json

    with open(fname) as f:
        data = json.load(f)
    node_data = [x for x in data["nodes"] if x["id"] == "g1"][0]
    assert node_data["params"]["vol"] == pytest.approx(0.42)


def test_controller_save_flushes_pending_params(tmp_path):
    # NOTE: must be a QApplication (GUI app), not a bare QCoreApplication.
    # Qt only allows one app instance per process; if a bare QCoreApplication
    # lingers, any later test that constructs a QWidget aborts with qFatal.
    # The offscreen platform keeps this headless-safe.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    import controller as ctrl_mod

    c = ctrl_mod.AppController()
    plugin_system.load_plugins("plugins")
    n = plugin_system.NODE_REGISTRY["Gain"]()
    engine_pushed = []
    c.engine.push_command(("restore", (n.to_dict(), n)))
    orig_push = c.engine.push_command

    def spy_push(cmd):
        engine_pushed.append(cmd[0])
        return orig_push(cmd)

    c.engine.push_command = spy_push
    c.set_parameter(n.id, "vol", 0.33)
    fname = str(tmp_path / "p2.json")
    c.save(fname)

    # The param command must be pushed before the save command
    assert engine_pushed[:2] == ["param", "save"]
    assert not c._pending_params
    deadline = time.time() + 2.0
    while not os.path.exists(fname) and time.time() < deadline:
        time.sleep(0.01)
    import json

    with open(fname) as f:
        data = json.load(f)
    node_data = [x for x in data["nodes"] if x["id"] == n.id][0]
    assert node_data["params"]["vol"] == pytest.approx(0.33)


# ==============================================================================
# 9. Channel compatibility at connect time
# ==============================================================================


class _FakeInput:
    def __init__(self, parent):
        self.name = "in"
        self.parent = parent
        self.param_name = None
        self.connected_outputs = []

    def connect(self, output):
        self.connected_outputs.append(output)

    def disconnect(self, target=None):
        if target is None:
            self.connected_outputs = []
        elif target in self.connected_outputs:
            self.connected_outputs.remove(target)


class _FakeOutput:
    def __init__(self, parent, channels):
        self.name = "out"
        self.parent = parent
        self.buffer = torch.zeros((channels, BLOCK_SIZE), dtype=torch.float32)


class _ChanNode:
    def __init__(self, name, out_channels):
        self.id = name
        self.name = name
        self.pos = (0, 0)
        self.error_msg = None
        self.params = {}
        self.inputs = {"in": _FakeInput(self)}
        self.outputs = {"out": _FakeOutput(self, out_channels)}
        self.monitor_queue = None

    def sync(self):
        pass

    def process(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def on_ui_param_change(self, param_name):
        pass

    def to_dict(self):
        return {"id": self.id, "type": "X", "name": self.name, "params": {}, "pos": self.pos}


def test_output_slot_rejects_impossible_channels():
    with pytest.raises(ValueError):
        OutputSlot("bad", None, channels=0)
    with pytest.raises(ValueError):
        OutputSlot("bad", None, channels=-1)


def test_connect_rejects_invalid_channel_output():
    g = Graph()
    bad = _ChanNode("bad", out_channels=0)
    ok = _ChanNode("ok", out_channels=2)
    g.add_node(bad)
    g.add_node(ok)
    assert g.connect("bad", "out", "ok", "in") is False
    assert not ok.inputs["in"].connected_outputs


def test_connect_allows_mono_to_stereo():
    g = Graph()
    mono = _ChanNode("mono", out_channels=1)
    stereo = _ChanNode("stereo", out_channels=2)
    g.add_node(mono)
    g.add_node(stereo)
    assert g.connect("mono", "out", "stereo", "in") is True
    assert len(stereo.inputs["in"].connected_outputs) == 1


def test_connect_rejected_carries_cmd_id_while_engine_running():
    """AGENTS.md §8: when the running audio loop drains the command queue,
    connect_rejected must still carry the originating command id so history
    can remove exactly the rejected command. Regression: the worker unpacked
    the queued entry as `_, cmd` and dropped the id."""
    plugin_system.load_plugins("plugins")
    engine = Engine()
    g = engine.graph
    n1 = plugin_system.NODE_REGISTRY["Gain"]()
    n1.id = "a"
    n2 = plugin_system.NODE_REGISTRY["Gain"]()
    n2.id = "b"
    g.add_node(n1)
    g.add_node(n2)

    engine.start()
    try:
        bad_id = engine.push_command(("conn", "a", "out", "a", "in"))
        rejected = None
        deadline = time.time() + 5.0
        while time.time() < deadline and rejected is None:
            try:
                msg = engine.output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(msg, dict) and msg.get("type") == "connect_rejected":
                rejected = msg
        assert rejected is not None, "no connect_rejected message while engine running"
        assert rejected["cmd_id"] == bad_id
        assert rejected["cmd_id"] is not None
    finally:
        engine.stop()


def test_nrt_discarded_hook_receives_superseded_payloads():
    """When a newer submit() supersedes an in-flight job, drain() must route
    the completed-but-stale payload to on_nrt_discarded() (so resource-owning
    payloads — native DSP handles, open streams, worker bundles — can be
    released) instead of dropping it silently."""
    from base import Node

    completed, discarded = [], []

    class _Probe(Node):
        def on_nrt_complete(self, tag, ok, payload):
            completed.append((tag, ok, payload))

        def on_nrt_discarded(self, tag, ok, payload):
            discarded.append((tag, ok, payload))

    engine = Engine()
    probe = _Probe("probe")
    probe.id = "probe"
    engine.graph.add_node(probe)

    probe.submit_nrt(lambda: "first", tag="job")
    probe.submit_nrt(lambda: "second", tag="job")

    # Wait for both jobs to finish and land in the node's inbox.
    deadline = time.time() + 5.0
    while time.time() < deadline and probe._nrt_inbox.qsize() < 2:
        time.sleep(0.01)
    assert probe._nrt_inbox.qsize() >= 2

    engine._drain_nrt_all()
    assert ("job", True, "second") in completed
    assert discarded == [("job", True, "first")]


def test_connect_rejects_outputs_wider_than_engine_channels():
    """Outputs declaring more channels than the global engine format are
    rejected at connect time: InputSlot scratch buffers are sized for
    CHANNELS, and channel adaptation must stay explicit and deterministic
    (a wider copy would raise a shape RuntimeError mid-block)."""
    plugin_system.load_plugins("plugins")
    g = Graph()
    wide = _ChanNode("wide", out_channels=CHANNELS + 2)
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    gain.id = "g"
    g.add_node(wide)
    g.add_node(gain)

    assert g.connect("wide", "out", "g", "in") is False
    assert gain.inputs["in"].connected_outputs == []

    # Stereo outputs remain connectable.
    assert g.connect("g", "out", "wide", "in") is True
    assert len(wide.inputs["in"].connected_outputs) == 1



def test_audio_device_param_change_does_not_start_stream_when_stopped():
    """Regression: changing the device with the engine stopped and no active
    stream must not open the hardware. The stream opens on engine start via
    node.start(); an already-active stream is a live device swap and still
    restarts."""
    from core import Engine as _Engine

    plugin_system.load_plugins("plugins")
    eng = _Engine()
    node = plugin_system.NODE_REGISTRY["AudioDeviceInput"]()
    node.id = "ain"
    eng.graph.add_node(node)

    node.params["device_index"].set(3)
    node.sync()

    calls = []

    def _fake_start():
        calls.append("start")

    original_start = node.start
    node.start = _fake_start

    try:
        # Engine stopped, no active stream: device change must NOT start.
        assert eng.running is False
        node.on_ui_param_change("device_index")
        assert calls == [], \
            "device change while stopped must not start the stream"

        # Positive control: engine running -> a start is requested.
        eng.running = True
        try:
            node.on_ui_param_change("device_index")
        finally:
            eng.running = False
        assert calls == ["start"], \
            "device change while running must request a stream restart"

        # Positive control 2: engine stopped but a stream is already active
        # (live device swap continues to work).
        node.stream = object()
        calls.clear()
        node.on_ui_param_change("device_index")
        assert calls == ["start"], \
            "device swap with an active stream must still restart"
        node.stream = None
    finally:
        node.start = original_start


def test_audio_input_callback_mono_upmix_and_no_allocation():
    """The PortAudio callback must upmix mono hardware inputs to the stereo
    ring without per-callback heap allocation (np.hstack/np.zeros were called
    every block on the callback thread)."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["AudioDeviceInput"]()

    mono = np.zeros((BLOCK_SIZE, 1), dtype=np.float32)
    mono[:, 0] = 0.25
    node._callback(mono, BLOCK_SIZE, None, None)

    scratch = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)
    assert node.ring_buffer.read(scratch), "callback block must land in the ring"
    assert np.allclose(scratch[:, 0], scratch[:, 1]), \
        "mono hardware input must be duplicated to both ring channels"
    assert np.allclose(scratch[:, 0], 0.25)

    # Multi-channel clamping path: 4-channel hardware into 2-channel ring,
    # repeated twice so stale scratch content from the first callback would
    # leak into the second if the unused region were not cleared.
    quad = np.zeros((BLOCK_SIZE, 4), dtype=np.float32)
    quad[:, :2] = 0.5
    quad[:, 2:] = 0.9   # would leak into ring channels if not cleared
    for _ in range(2):
        node._callback(quad, BLOCK_SIZE, None, None)
        assert node.ring_buffer.read(scratch)
        assert np.allclose(scratch[:, :2], 0.5)
        assert np.allclose(scratch[:, 2:], 0.0), \
            "unused ring channels must be zero, not stale scratch content"

    # Steady-state callbacks must not allocate on the callback thread.
    import gc
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(20):
        node._callback(mono, BLOCK_SIZE, None, None)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 16 * 1024, f"callback allocated {growth} bytes over 20 blocks"

