"""Tests for ValuePlotterNode — pass-through + telemetry + zero allocation."""

import tracemalloc

import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("ValuePlotterNode")
    assert cls is not None, "ValuePlotterNode not registered"
    return cls()


def process_block(node, blk):
    node.inputs["in"].get_tensor = lambda b=blk: b
    node.process()
    return node.outputs["out"].buffer.clone()


def test_value_plotter_bit_exact_passthrough():
    node = make_node()

    # Stereo input: output must be bit-exact and keep shape (2, 512).
    stereo = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, stereo)
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(out, stereo)

    # Mono input: broadcast to stereo, shape stays (2, 512), both rows equal.
    mono = torch.randn(1, BLOCK_SIZE, dtype=DTYPE)
    out = process_block(node, mono)
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(out[0], mono[0])
    assert torch.equal(out[1], mono[0])


def test_value_plotter_telemetry_and_drop():
    node = make_node()
    sig = torch.randn(1, BLOCK_SIZE, dtype=DTYPE)

    # Push many blocks; the ring never blocks and pop_all returns frames.
    for _ in range(50):
        process_block(node, sig)

    frames = node.monitor_queue.pop_all()
    assert isinstance(frames, list)
    for f in frames:
        assert f.shape == (1, 8)
    # The queue should not have raised or blocked regardless of count.


def test_value_plotter_zero_steady_state_allocation():
    node = make_node()
    sig = torch.randn(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    process_block(node, sig)  # warm up allocators

    import gc
    gc.collect()
    tracemalloc.start()
    for _ in range(50):
        process_block(node, sig)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"
def test_value_plotter_decimate_columns_math():
    """_decimate_columns is the draw-time decimator: one min/max pair per
    screen column, output bounded by the widget width, envelope preserved."""
    from plugins.visualization_value_plotter import _decimate_columns

    # Dense -> one vertical beat per column with correct envelope pixels.
    # w is the widget width in pixels; 6 samples bucket into 3 columns, so the
    # beats land at the pixel centers 0.5 / 1.5 / 2.5 of a 3px-wide widget.
    xs, y_top, y_bot = _decimate_columns([0, 1, 4, 5, 8, 9], w=3, h=100,
                                         min_v=0, max_v=10)
    assert np.allclose(xs, [0.5, 1.5, 2.5])
    assert np.allclose(y_top, [86.8, 50.0, 13.2])      # column max pixels
    assert np.allclose(y_bot, [96.0, 59.2, 22.4])      # column min pixels

    # Sparse (fewer points than columns): one beat per point, min == max so
    # the point is still drawn.
    xs2, t2, b2 = _decimate_columns([0.5, 0.6], w=10, h=100, min_v=0, max_v=1)
    assert np.allclose(xs2, [2.5, 7.5])
    assert np.allclose(t2, b2)
    assert np.allclose(t2, [50.0, 40.8])

    # Output size is bounded by the widget width regardless of input length.
    big = np.sin(np.linspace(0, 40, 10000))
    xs3, t3, b3 = _decimate_columns(big, w=200, h=100, min_v=-1, max_v=1)
    assert len(xs3) == 200 and len(t3) == 200 and len(b3) == 200
    assert (t3 <= 100.0 + 1e-9).all() and (t3 >= -1e-9).all()
    assert (b3 <= 100.0 + 1e-9).all() and (b3 >= -1e-9).all()


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
    # A bare QCoreApplication created by an earlier test module is still alive;
    # Qt cannot attach GUI support to it afterwards. Skip rather than abort.
    pytest.skip("a bare QCoreApplication is active; QWidget tests cannot run")


def test_value_plotter_widget_poll_consumes_all_frames(qapp):
    """The widget must plot every frame returned by pop_all(): the UI poll
    (~30 FPS) is slower than the audio block rate (~94/s), so taking only
    frames[-1] dropped ~2-3 of every 4 sampled points and distorted the time
    axis (missed short transients)."""
    from base import TelemetryRingBuffer
    from plugins.visualization_value_plotter import ValuePlotterWidget

    q = TelemetryRingBuffer(capacity=16, shape=(1, 8), dtype=np.float32)
    q.push(np.full((1, 8), 0.25, dtype=np.float32))
    q.push(np.full((1, 8), 0.75, dtype=np.float32))

    class _FakeProxy:
        def __init__(self, queue):
            self.monitor_queue = queue

    w = ValuePlotterWidget(_FakeProxy(q))
    w.poll()

    history = list(w._history)
    assert history.count(0.25) == 8, "first frame's 8 points were dropped"
    assert history.count(0.75) == 8, "later frames must append after earlier ones"
    assert w._current == pytest.approx(0.75)
    assert w._has_data


def test_value_plotter_paint_dense_history_is_bounded(qapp):
    """A dense (multi-thousand-point) history must paint via decimation without
    error, and the decimated envelope must preserve peaks/valleys. The drawing
    cost is bounded by the widget width, not the sample count — this is the
    regression that made the consume-all-frames change too slow."""
    from collections import deque

    from plugins.visualization_value_plotter import _decimate_columns, ValuePlotterWidget

    class _FakeProxy:
        node_item = None
        monitor_queue = None

    w = ValuePlotterWidget(_FakeProxy())
    w.resize(320, 120)

    # Trace far denser than the widget width, with a narrow spike.
    vals = np.linspace(-0.5, 0.5, 600)
    vals[-10] = 5.0
    w._history = deque(vals.tolist(), maxlen=w.HISTORY_LEN)
    assert len(w._history) > w.width()
    w._has_data = True
    w._current = 5.0

    # Smoke: the decimating paint path must run without error.
    w.paintEvent(None)

    # Envelope preserved: with h=120, min=-1, max=6, span=7:
    #   y(5.0)  = 116 - (6/7)*112 = 20   (the spike's column max)
    #   y(-0.5) = 116 - (0.5/7)*112 = 108 (valley column min)
    xs_d, y_top_d, y_bot_d = _decimate_columns(vals, w.width(), w.height(),
                                               -1.0, 6.0)
    assert len(xs_d) <= w.width()
    assert float(np.min(y_top_d)) == pytest.approx(20.0, abs=0.5), \
        "spike peak must survive decimation"
    assert float(np.max(y_bot_d)) == pytest.approx(108.0, abs=0.5), \
        "valley must survive decimation"