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


def test_value_plotter_auto_range_param_save_load():
    """auto_range defaults ON, round-trips through save/load, and legacy
    patches without the key keep the ON default (generic load_state only
    sets keys that are present)."""
    node = make_node()
    assert node.params["auto_range"].value is True
    assert node.to_dict()["params"]["auto_range"] is True

    node.params["auto_range"].set(False)
    node.params["auto_range"].sync()
    snap = node.to_dict()["params"]

    restored = make_node()
    restored.load_state({"params": dict(snap), "pos": (0, 0)})
    assert restored.params["auto_range"].value is False
    assert restored.params["min_val"].value == pytest.approx(-1.0)

    legacy = make_node()
    legacy.load_state({"params": {"min_val": -2.0, "max_val": 2.0}, "pos": (0, 0)})
    assert legacy.params["auto_range"].value is True
    assert legacy.params["min_val"].value == pytest.approx(-2.0)


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


def _make_widget_proxies(queue=None, params=None):
    """Fake proxy trio for widget tests: a stub smart-widget factory (mimics
    FloatParamWidget's update_from_backend/setEnabled interface), a
    set_parameter recorder that also stages into node_item (like the real
    controller snapshot round-trip), and NodeItem-style params."""
    from types import SimpleNamespace

    from PySide6.QtWidgets import QWidget

    if params is None:
        params = {
            "min_val": {"value": -1.0},
            "max_val": {"value": 1.0},
            "auto_range": {"value": True},
        }

    class _StubParamWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.backend_values = []

        def update_from_backend(self, value):
            self.backend_values.append(float(value))

    class _FakeProxy:
        def __init__(self):
            self.monitor_queue = queue
            self.node_item = SimpleNamespace(params={k: dict(v) for k, v in params.items()})
            self.set_calls = []
            self.created = []

        def set_parameter(self, name, value):
            self.set_calls.append((name, value))
            entry = self.node_item.params.get(name)
            if isinstance(entry, dict):
                entry["value"] = value

        def create_param_widget(self, pname):
            w = _StubParamWidget()
            w.setObjectName(pname)
            self.created.append(pname)
            return w

    return _FakeProxy()


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
    # Auto-range seeds from the first data and covers it (no proxy params ->
    # AUTO defaults ON).
    assert w._auto_min is not None and w._auto_max is not None
    assert w._auto_min <= 0.25 and w._auto_max >= 0.75


def test_value_plotter_widget_range_controls(qapp):
    """The compact row under the plot exposes min/max editors plus AUTO, and
    toggling AUTO routes through proxy.set_parameter() and dims the manual
    editors while auto drives the view."""
    from plugins.visualization_value_plotter import ValuePlotterWidget

    proxy = _make_widget_proxies()
    w = ValuePlotterWidget(proxy)

    assert proxy.created == ["min_val", "max_val"]
    assert w._min_widget is not None and w._max_widget is not None
    assert w._auto_box.isChecked(), "AUTO is ON by default"
    assert not w._min_widget.isEnabled()
    assert not w._max_widget.isEnabled()

    # User disables AUTO: staged via the controller path, editors re-enable.
    w._auto_box.setChecked(False)
    assert ("auto_range", False) in proxy.set_calls
    assert w._min_widget.isEnabled() and w._max_widget.isEnabled()

    w._auto_box.setChecked(True)
    assert ("auto_range", True) in proxy.set_calls
    assert not w._min_widget.isEnabled()


def test_range_presets_fit_param_bounds():
    """Every preset must survive the min_val/max_val clamp (Parameter.set
    clamps to meta) — otherwise a preset would silently land elsewhere."""
    from plugins.visualization_value_plotter import RANGE_PRESETS

    node = make_node()
    min_meta = node.params["min_val"].meta
    max_meta = node.params["max_val"].meta
    assert len(RANGE_PRESETS) >= 4
    for name, lo, hi in RANGE_PRESETS:
        assert lo < hi, name
        assert min_meta["min"] <= lo <= min_meta["max"], name
        assert max_meta["min"] <= hi <= max_meta["max"], name


def test_apply_preset_stages_range_and_disables_auto(qapp):
    """Right-click preset: exact min/max + auto_range False go through the
    controller staging path, and the AUTO box visibly unchecks at once."""
    from plugins.visualization_value_plotter import ValuePlotterWidget

    proxy = _make_widget_proxies()
    w = ValuePlotterWidget(proxy)
    assert w._auto_box.isChecked()

    assert w._apply_preset("Control 0–1") is True
    assert ("min_val", 0.0) in proxy.set_calls
    assert ("max_val", 1.0) in proxy.set_calls
    assert ("auto_range", False) in proxy.set_calls
    assert not w._auto_box.isChecked()
    assert w._min_widget.isEnabled() and w._max_widget.isEnabled()

    assert w._apply_preset("No such range") is False
    assert len(proxy.set_calls) == 3


def test_value_plotter_widget_update_from_params(qapp):
    """Backend snapshots sync the embedded controls (NodeItem only forwards
    to param_controls, so custom-embedded widgets need explicit forwarding)."""
    from plugins.visualization_value_plotter import ValuePlotterWidget

    proxy = _make_widget_proxies()
    w = ValuePlotterWidget(proxy)

    w.update_from_params({"auto_range": False, "min_val": -2.0, "max_val": 3.0})
    assert not w._auto_box.isChecked()
    assert w._min_widget.backend_values[-1] == pytest.approx(-2.0)
    assert w._max_widget.backend_values[-1] == pytest.approx(3.0)
    assert w._min_widget.isEnabled()

    w.update_from_params({"auto_range": True, "min_val": -2.0, "max_val": 3.0})
    assert w._auto_box.isChecked()
    assert not w._min_widget.isEnabled()


def test_value_plotter_auto_peak_hold_attack_and_release(qapp):
    """AUTO peak-hold: instant attack on spikes, exponential release toward a
    shrunken signal, and display-only (manual params never rewritten)."""
    from base import TelemetryRingBuffer
    from plugins.visualization_value_plotter import ValuePlotterWidget

    proxy = _make_widget_proxies()
    q = TelemetryRingBuffer(capacity=16, shape=(1, 8), dtype=np.float32)
    proxy.monitor_queue = q
    w = ValuePlotterWidget(proxy)

    q.push(np.full((1, 8), 5.0, dtype=np.float32))
    w.poll()
    assert w._auto_max >= 5.0, "spike must attack immediately"

    # Signal collapses to 0.1: the hold releases downward over polls.
    prev = w._auto_max
    for _ in range(100):
        q.push(np.full((1, 8), 0.1, dtype=np.float32))
        w.poll()
        assert w._auto_max <= prev + 1e-9, "release must not grow without new peaks"
        prev = w._auto_max
    assert w._auto_max < 1.0, "hold must converge toward the shrunken signal"
    assert w._auto_max > 0.1

    # A new spike re-attacks on the very next poll.
    q.push(np.full((1, 8), 5.0, dtype=np.float32))
    w.poll()
    assert w._auto_max >= 5.0

    # Display-only: the staged manual parameters are untouched by tracking.
    assert proxy.node_item.params["min_val"]["value"] == pytest.approx(-1.0)
    assert proxy.node_item.params["max_val"]["value"] == pytest.approx(1.0)
    lo, hi = w._display_range()
    assert lo < 5.0 < hi


def test_value_plotter_auto_flat_signal_and_inverted_manual(qapp):
    """Flat (DC) signals get a finite, ordered window; an inverted manual
    range is normalized instead of painting a degenerate mapping."""
    from base import TelemetryRingBuffer
    from plugins.visualization_value_plotter import ValuePlotterWidget

    proxy = _make_widget_proxies()
    q = TelemetryRingBuffer(capacity=16, shape=(1, 8), dtype=np.float32)
    proxy.monitor_queue = q
    w = ValuePlotterWidget(proxy)
    w.resize(320, 160)

    q.push(np.zeros((1, 8), dtype=np.float32))
    w.poll()
    lo, hi = w._display_range()
    assert lo < 0.0 < hi, "flat silence must still bracket the line"
    w.paintEvent(None)

    # Inverted manual range with AUTO off: ordered, paintable.
    proxy.node_item.params["auto_range"]["value"] = False
    proxy.node_item.params["min_val"]["value"] = 5.0
    proxy.node_item.params["max_val"]["value"] = -5.0
    w.update_from_params({"auto_range": False, "min_val": 5.0, "max_val": -5.0})
    lo2, hi2 = w._display_range()
    assert (lo2, hi2) == (-5.0, 5.0)
    w.paintEvent(None)


def test_value_plotter_zero_line_tracks_scale(qapp):
    """The dashed guide must sit on the trace's true zero, not the widget
    center: with an asymmetric manual range it moves, and it hides when 0 is
    outside the view window."""
    from plugins.visualization_value_plotter import _zero_line_y, ValuePlotterWidget

    h = 160.0
    # Symmetric: zero is centered (plot area, 4px margins).
    assert _zero_line_y(h, -1.0, 1.0) == pytest.approx(h / 2.0)
    # Asymmetric: min=0/max=10 puts zero at the bottom margin line.
    assert _zero_line_y(h, 0.0, 10.0) == pytest.approx(h - 4.0)
    # Asymmetric mid: min=-1/max=3 -> y = h-4 - (1/4)*(h-8).
    assert _zero_line_y(h, -1.0, 3.0) == pytest.approx(h - 4.0 - 0.25 * (h - 8.0))
    # Zero outside the window: no guide (None), not a misleading edge line.
    assert _zero_line_y(h, 5.0, 10.0) is None
    assert _zero_line_y(h, -10.0, -5.0) is None
    # Inverted manual entry is normalized first.
    assert _zero_line_y(h, 10.0, 0.0) == pytest.approx(h - 4.0)

    # Paint smoke with an asymmetric manual range (exercises the guide path).
    proxy = _make_widget_proxies(params={
        "min_val": {"value": 0.0},
        "max_val": {"value": 10.0},
        "auto_range": {"value": False},
    })
    w = ValuePlotterWidget(proxy)
    w.resize(320, 160)
    w.show()
    qapp.processEvents()
    from collections import deque
    import numpy as np
    w._history = deque(np.linspace(0.0, 10.0, 300).tolist(), maxlen=w.HISTORY_LEN)
    w._has_data = True
    w.paintEvent(None)


def test_value_plotter_paint_dense_history_is_bounded(qapp):
    """A dense (multi-thousand-point) history must paint via decimation without
    error, and the decimated envelope must preserve peaks/valleys. The drawing
    cost is bounded by the widget width, not the sample count — this is the
    regression that made the consume-all-frames change too slow."""
    from collections import deque

    from plugins.visualization_value_plotter import _decimate_columns, ValuePlotterWidget

    proxy = _make_widget_proxies(params={
        "min_val": {"value": -1.0},
        "max_val": {"value": 6.0},
        "auto_range": {"value": False},
    })
    w = ValuePlotterWidget(proxy)
    w.resize(320, 160)
    w.show()
    qapp.processEvents()
    plot_h = w._plot_height()
    assert 0 < plot_h <= w.height()

    # Trace far denser than the widget width, with a narrow spike.
    vals = np.linspace(-0.5, 0.5, 600)
    vals[-10] = 5.0
    w._history = deque(vals.tolist(), maxlen=w.HISTORY_LEN)
    assert len(w._history) > w.width()
    w._has_data = True
    w._current = 5.0

    # Smoke: the decimating paint path must run without error.
    w.paintEvent(None)

    # Envelope preserved in the plot area with min=-1, max=6, span=7:
    #   y(5.0)  = plot_h-4 - (6/7)*(plot_h-8)  (the spike's column max)
    #   y(-0.5) = plot_h-4 - (0.5/7)*(plot_h-8) (valley column min)
    xs_d, y_top_d, y_bot_d = _decimate_columns(vals, w.width(), plot_h,
                                               -1.0, 6.0)
    assert len(xs_d) <= w.width()
    assert float(np.min(y_top_d)) == pytest.approx(
        plot_h - 4 - (6.0 / 7.0) * (plot_h - 8), abs=0.5), \
        "spike peak must survive decimation"
    assert float(np.max(y_bot_d)) == pytest.approx(
        plot_h - 4 - (0.5 / 7.0) * (plot_h - 8), abs=0.5), \
        "valley must survive decimation"


def _orange_pixels(image):
    """Count trace-orange pixels (the #ff9900 trace + its glow halo)."""
    from PySide6.QtGui import QColor

    n = 0
    for y in range(image.height()):
        for x in range(image.width()):
            c = QColor(image.pixel(x, y))
            if c.red() > 150 and 80 < c.green() < 180 and c.blue() < 80:
                n += 1
    return n


def test_value_plotter_flat_signal_renders_solid_line(qapp):
    """A constant value must paint as a solid horizontal, not vanish: dense
    decimation turns flats into zero-height column beats (isolated dots), so
    the envelope midline is stroked to keep them visible."""
    from collections import deque

    from plugins.visualization_value_plotter import ValuePlotterWidget

    proxy = _make_widget_proxies(params={
        "min_val": {"value": 0.0},
        "max_val": {"value": 1.0},
        "auto_range": {"value": False},
    })
    w = ValuePlotterWidget(proxy)
    w.resize(240, 160)
    w.show()
    qapp.processEvents()
    w._history = deque([0.7] * 1024, maxlen=w.HISTORY_LEN)
    assert len(w._history) > w.width()  # steady-state dense path
    w._has_data = True
    w._current = 0.7

    image = w.grab().toImage()
    # A solid row across the width: glow (4px) + core over ~240 columns is
    # ~1000+ orange pixels; the old dots-only rendering measured ~144.
    assert _orange_pixels(image) >= 500, "flat 0.7 must render a solid row"