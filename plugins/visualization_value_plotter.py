"""
ValuePlotterNode — real-time scrolling line plot for CV / envelopes / peaks (Visual).

Real-time notes:
- Audio pass-through is bit-exact: the input is copied to the stereo output
  untouched (a mono input is broadcast to both output channels without
  resizing the output buffer, per the MonoToStereo convention).
- Analysis runs on a private copy: channel 0 is isolated, wall-clamped to the
  display bounds, and downsampled 512 -> 8 points.
- Telemetry travels through a bounded, non-blocking SPSC ring buffer. The UI
  poll consumes every frame that arrived since the last poll (only a stalled UI
  that fills the 16 slots drops frames). paintEvent decimates the trace to one
  min/max pair per screen column, so a dense history stays cheap to draw
  (AGENTS.md section 10).
- Display range is user-configurable: ``min_val``/``max_val`` sliders in a
  compact row under the plot, plus an ``AUTO`` mode (on by default) that tracks
  the signal with a UI-side peak-hold envelope (fast attack, ~1 s release).
  Auto-range is display-only: it never writes the manual parameters and never
  touches the audio thread. Right-clicking the plot offers one-click range
  presets per signal family (audio, control, MIDI note, pitch Hz), which
  stage min/max and switch AUTO off.
- Zero heap allocation on the audio thread.
"""

import logging
import math

import numpy as np
import torch

from base import Node, BLOCK_SIZE, CHANNELS, DTYPE, TelemetryRingBuffer

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QMenu
    from PySide6.QtCore import Qt, QTimer, QRect, QRectF, QPointF, QLineF, QSignalBlocker
    from PySide6.QtGui import QPainter, QPen, QColor, QFont

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


# Manual range presets (right-click the plot): one per signal family found
# in the graph — bipolar audio, unipolar control (gates/envelopes/
# confidence), MIDI note numbers, and pitch-track Hz (SwiftF0 spans
# ~47–2094 Hz). Applying a preset stages min/max and switches AUTO off so
# the choice takes visible effect immediately.
RANGE_PRESETS = (
    ("Audio ±1", -1.0, 1.0),
    ("Control 0–1", 0.0, 1.0),
    ("MIDI note 0–127", 0.0, 127.0),
    ("Pitch 0–2000 Hz", 0.0, 2000.0),
)


def _decimate_columns(values, w, h, min_v, max_v):
    """Min/max decimation at draw time (pure numpy, Qt-free).

    Splits ``values`` into ``min(w, len(values))`` horizontal buckets — one per
    screen column — and returns parallel arrays (xs, y_top, y_bot) forming one
    vertical beat per column, where y_top is the pixel of the column max and
    y_bot the pixel of the column min. The outputs have at most ``w`` entries,
    so drawing cost is bounded by the widget width no matter how many samples
    the trace holds.
    """
    n = len(values)
    if n <= 0 or w <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    cols = max(1, min(int(w), n))
    arr = np.asarray(values, dtype=np.float64)
    span = max(1e-6, float(max_v) - float(min_v))
    edges = np.linspace(0, n, cols + 1).astype(np.int64)
    col_min = np.minimum.reduceat(arr, edges[:-1])
    col_max = np.maximum.reduceat(arr, edges[:-1])
    xs = (np.arange(cols) + 0.5) * (float(w) / cols)
    y_top = float(h) - 4.0 - (col_max - float(min_v)) / span * (float(h) - 8.0)
    y_bot = float(h) - 4.0 - (col_min - float(min_v)) / span * (float(h) - 8.0)
    np.clip(y_top, 0.0, float(h), out=y_top)
    np.clip(y_bot, 0.0, float(h), out=y_bot)
    return xs, y_top, y_bot


def _zero_line_y(h, min_v, max_v):
    """Pixel y of value 0 under the plot's scale mapping, or None when 0 is
    outside the [min_v, max_v] view window (no misleading edge line then).

    Uses the same ``h - 4 - (v - min)/span * (h - 8)`` mapping as the trace so
    the dashed guide always sits on the trace's true zero.
    """
    min_v = float(min_v)
    max_v = float(max_v)
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    if not (min_v <= 0.0 <= max_v):
        return None
    span = max(1e-6, max_v - min_v)
    y = float(h) - 4.0 - (0.0 - min_v) / span * (float(h) - 8.0)
    return min(max(y, 0.0), float(h))


# ==============================================================================
# DSP Node Logic (zero allocation on audio thread)
# ==============================================================================
class ValuePlotterNode(Node):
    category = "Visual & Analysis"
    label = "Value Plotter"
    description = (
        "Real-time scrolling line plot for control voltages, envelopes, and "
        "audio peaks with zero audio-thread allocation."
    )

    VISUAL_WIDTH = 8
    RING_CAPACITY = 16

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in",
                                  help="Signal to visualize (analysis plots channel 0 of a private copy).")
        self.out = self.add_output("out", channels=CHANNELS,
                                   help="Pass-through copy of the input, unaltered.")

        self.add_float_param("min_val", -1.0, -5000.0, 5000.0, unit="",
                             help="Bottom scale bound (manual mode). Right-click the plot for range presets.")
        self.add_float_param("max_val", 1.0, -5000.0, 5000.0, unit="",
                             help="Top scale bound (manual mode). Right-click the plot for range presets.")
        self.add_bool_param("auto_range", True,
                            help="Automatically fit the display range to the "
                                 "signal (UI-side peak-hold, display only).")

        # Bounded SPSC telemetry ring buffer (16 slots; overflow drops frames).
        self.monitor_queue = TelemetryRingBuffer(
            capacity=self.RING_CAPACITY, shape=(1, self.VISUAL_WIDTH), dtype=np.float32
        )
        # Pre-allocated analysis buffers.
        self._analysis_buf = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)
        self._downsampled = torch.zeros((1, self.VISUAL_WIDTH), dtype=DTYPE)

    def process(self):
        sig = self.inp.get_tensor()

        # 1. STRICT BIT-EXACT PASS-THROUGH (copy_ broadcasts mono without
        #    resizing the output buffer).
        self.out.buffer.copy_(sig)

        # 2. Analysis on a private copy: isolate channel 0.
        self._analysis_buf[0].copy_(sig[0])

        # 3. Subsample 512 frames -> 8 points (step = 64).
        step = BLOCK_SIZE // self.VISUAL_WIDTH
        # 4. Sanitize (NaN/inf) into the pre-allocated downsampled buffer.
        torch.nan_to_num(
            self._analysis_buf[0, ::step],
            nan=0.0, posinf=500.0, neginf=-500.0,
            out=self._downsampled[0],
        )

        # 5. Dispatch to the UI via the lock-free ring buffer (overflow is free).
        self.monitor_queue.push(self._downsampled.numpy())


# ==============================================================================
# Qt custom UI
# ==============================================================================
if GUI_AVAILABLE:

    class ValuePlotterWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "ValuePlotterNode"

        # Bounded rolling trace. Points enter at block rate (~94 blocks/s x 8
        # points = ~750/s); painting decimates to one min/max pair per screen
        # column, so a larger window costs no extra drawing time.
        HISTORY_LEN = 1024

        # Automatic range detection (UI-side peak-hold envelope, display only).
        # Fast attack on new extrema, exponential release toward the recent
        # window. RELEASE_K assumes the ~30 ms poll cadence (≈1 s tau).
        AUTO_RELEASE_K = 0.03
        AUTO_PAD_FRAC = 0.10
        AUTO_PAD_EPS = 1e-3

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(240, 150)

            self.bg_color = QColor("#141414")
            self.guide_color = QColor("#323232")
            self.line_color = QColor("#ff9900")
            self.text_color = QColor(180, 180, 180, 200)
            self.label_font = QFont("Monospace", 8, QFont.Bold)

            self._history = None  # set on first data
            self._current = 0.0
            self._has_data = False

            # Peak-hold state (UI thread only; None until first data).
            self._auto_min = None
            self._auto_max = None

            self.timer = QTimer(self)
            self.timer.setInterval(30)
            self.timer.timeout.connect(self.poll)
            self.timer.start()

            # -- Compact range-controls row under the plot ------------------
            # A custom UI REPLACES the node's auto-generated parameter panel
            # (ui_system builds generic editors only when there is no custom
            # widget), so min/max must be embedded here explicitly. All edits
            # go through proxy.set_parameter() (controller staging path);
            # this widget never touches engine DSP state directly.
            self._controls = QWidget(self)
            row = QHBoxLayout(self._controls)
            row.setContentsMargins(4, 2, 4, 2)
            row.setSpacing(6)

            self._min_widget = None
            self._max_widget = None
            create = getattr(self.proxy, "create_param_widget", None)
            if callable(create):
                for attr in ("_min_widget", "_max_widget"):
                    pname = "min_val" if attr == "_min_widget" else "max_val"
                    try:
                        setattr(self, attr, create(pname))
                        row.addWidget(getattr(self, attr))
                    except Exception:
                        logging.exception(f"param widget '{pname}' failed")
                        setattr(self, attr, None)

            self._auto_box = QCheckBox("AUTO")
            self._auto_box.setToolTip(
                "Automatically fit the display range to the signal "
                "(peak-hold, display only; manual values are kept)."
            )
            self._auto_box.setChecked(bool(self._param_value("auto_range", True)))
            self._auto_box.toggled.connect(self._on_auto)
            row.addWidget(self._auto_box)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            outer.addStretch(1)
            outer.addWidget(self._controls)

            self._apply_auto_enabled(self._auto_box.isChecked())

        # -- parameters (UI-side snapshot reads) ---------------------------
        def _param_value(self, name, default):
            node_item = getattr(self.proxy, "node_item", None)
            params = getattr(node_item, "params", None) or {}
            entry = params.get(name)
            if isinstance(entry, dict) and "value" in entry:
                try:
                    return float(entry["value"])
                except (TypeError, ValueError):
                    return default
            return default

        def _auto_enabled(self):
            return bool(self._param_value("auto_range", True))

        def _apply_auto_enabled(self, auto_on):
            """Dim the manual editors while AUTO drives the view window."""
            if self._min_widget is not None:
                self._min_widget.setEnabled(not auto_on)
            if self._max_widget is not None:
                self._max_widget.setEnabled(not auto_on)

        def _on_auto(self, checked):
            setter = getattr(self.proxy, "set_parameter", None)
            if callable(setter):
                try:
                    setter("auto_range", bool(checked))
                except Exception:
                    logging.exception("set_parameter('auto_range') failed")
            self._apply_auto_enabled(bool(checked))

        def _apply_preset(self, name):
            """Stage a named manual range and switch AUTO off.

            Returns True when the preset exists. Goes through
            proxy.set_parameter() like every other edit (controller staging
            path); the backend snapshot round-trip re-syncs the editors.
            """
            for preset_name, lo, hi in RANGE_PRESETS:
                if preset_name != name:
                    continue
                setter = getattr(self.proxy, "set_parameter", None)
                if callable(setter):
                    try:
                        setter("min_val", float(lo))
                        setter("max_val", float(hi))
                        setter("auto_range", False)
                    except Exception:
                        logging.exception(f"preset '{name}' failed")
                with QSignalBlocker(self._auto_box):
                    self._auto_box.setChecked(False)
                self._apply_auto_enabled(False)
                return True
            return False

        def contextMenuEvent(self, event):
            menu = QMenu(self)
            for preset_name, _lo, _hi in RANGE_PRESETS:
                action = menu.addAction(preset_name)
                action.triggered.connect(
                    lambda _checked=False, n=preset_name: self._apply_preset(n))
            menu.addSeparator()
            auto_action = menu.addAction("Auto-range")
            auto_action.setCheckable(True)
            auto_action.setChecked(self._auto_box.isChecked())
            auto_action.triggered.connect(
                lambda checked=False: self._auto_box.setChecked(checked))
            menu.exec(event.globalPos())
            event.accept()

        def update_from_params(self, simple_params: dict):
            """Keep embedded controls in sync with backend values.

            Called by NodeItem.update_from_snapshot(); embedded smart widgets
            are NOT in NodeItem.param_controls, so they must be forwarded here
            explicitly (same pattern as NamNode/ReverbWidget).
            """
            try:
                if "auto_range" in simple_params:
                    with QSignalBlocker(self._auto_box):
                        self._auto_box.setChecked(bool(simple_params["auto_range"]))
                if "min_val" in simple_params and self._min_widget is not None:
                    updater = getattr(self._min_widget, "update_from_backend", None)
                    if callable(updater):
                        updater(float(simple_params["min_val"]))
                if "max_val" in simple_params and self._max_widget is not None:
                    updater = getattr(self._max_widget, "update_from_backend", None)
                    if callable(updater):
                        updater(float(simple_params["max_val"]))
            except (TypeError, ValueError):
                logging.exception("ValuePlotter update_from_params failed")
            auto_on = bool(simple_params.get("auto_range", self._auto_box.isChecked()))
            self._apply_auto_enabled(auto_on)

        # -- display range --------------------------------------------------
        def _display_range(self):
            """Effective (min, max) view window for painting.

            AUTO uses the UI-side peak-hold envelope (display-only; the manual
            parameters are never modified). Manual mode uses min_val/max_val
            with an inverted-range guard.
            """
            if self._auto_enabled() and self._auto_min is not None \
                    and self._auto_max is not None:
                lo, hi = float(self._auto_min), float(self._auto_max)
            else:
                lo = float(self._param_value("min_val", -1.0))
                hi = float(self._param_value("max_val", 1.0))
                if hi < lo:
                    lo, hi = hi, lo
            if not (math.isfinite(lo) and math.isfinite(hi)):
                lo, hi = -1.0, 1.0
            if hi - lo < 1e-6:
                c = 0.5 * (lo + hi)
                lo, hi = c - 5e-7, c + 5e-7
            return lo, hi

        def _update_auto_range(self, chunk_min, chunk_max):
            """Peak-hold envelope step from one poll's new-point extrema.

            Attack is immediate (with padding from the current hold span so the
            trace never touches the rail); release relaxes exponentially toward
            the padded chunk window. Runs on the UI thread only.
            """
            chunk_min = float(chunk_min)
            chunk_max = float(chunk_max)
            if not (math.isfinite(chunk_min) and math.isfinite(chunk_max)):
                return
            if chunk_max < chunk_min:
                chunk_min, chunk_max = chunk_max, chunk_min
            if self._auto_min is None or self._auto_max is None:
                span = chunk_max - chunk_min
                pad = span * self.AUTO_PAD_FRAC + self.AUTO_PAD_EPS
                self._auto_min = chunk_min - pad
                self._auto_max = chunk_max + pad
                return
            span = max(1e-9, float(self._auto_max) - float(self._auto_min))
            pad = span * self.AUTO_PAD_FRAC + self.AUTO_PAD_EPS
            k = self.AUTO_RELEASE_K
            if chunk_min < self._auto_min:
                self._auto_min = chunk_min - pad
            else:
                self._auto_min += ((chunk_min - pad) - self._auto_min) * k
            if chunk_max > self._auto_max:
                self._auto_max = chunk_max + pad
            else:
                self._auto_max += ((chunk_max + pad) - self._auto_max) * k
            if self._auto_max - self._auto_min < 1e-6:
                c = 0.5 * (self._auto_min + self._auto_max)
                self._auto_min, self._auto_max = c - 5e-7, c + 5e-7

        def _plot_height(self):
            """Paintable plot height: widget height minus the controls row."""
            h = self.height()
            controls = getattr(self, "_controls", None)
            if controls is not None:
                try:
                    ch = controls.height()
                except RuntimeError:
                    ch = 0
                if 0 < ch < h:
                    return h - ch
            return h

        def poll(self):
            queue = getattr(self.proxy, "monitor_queue", None)
            if not queue:
                return
            frames = queue.pop_all()
            if not frames:
                return
            if self._history is None:
                from collections import deque
                self._history = deque(maxlen=self.HISTORY_LEN)
            # Consume EVERY frame since the last poll: the ring is pushed at
            # block rate (~94/s) while the UI poll runs at ~30 FPS, so taking
            # only the newest frame would drop ~2-3 of every 4 sampled points
            # and distort the time axis. paintEvent decimates the trace, so a
            # denser history stays cheap to draw.
            chunk_min = math.inf
            chunk_max = -math.inf
            for f in frames:
                # f is shape (1, 8); extend the rolling trace with the 8 points.
                for v in f[0]:
                    fv = float(v)
                    self._history.append(fv)
                    if math.isfinite(fv):
                        if fv < chunk_min:
                            chunk_min = fv
                        if fv > chunk_max:
                            chunk_max = fv
            self._current = float(frames[-1][0, -1])
            self._has_data = True
            if chunk_min <= chunk_max:
                self._update_auto_range(chunk_min, chunk_max)
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            plot_h = self._plot_height()

            min_v, max_v = self._display_range()
            span = max(1e-6, max_v - min_v)

            painter.fillRect(0, 0, w, plot_h, self.bg_color)

            # Dashed zero guide at the true mapped position of 0 (hidden when
            # 0 is outside the view window).
            zero_y = _zero_line_y(plot_h, min_v, max_v)
            if zero_y is not None:
                painter.setPen(QPen(self.guide_color, 1, Qt.DashLine))
                painter.drawLine(0, int(round(zero_y)), w, int(round(zero_y)))

            plot_rect = QRect(0, 0, w, plot_h)
            if not self._has_data or not self._history:
                painter.setPen(self.text_color)
                painter.drawText(plot_rect, Qt.AlignCenter, "No Signal")
                return

            # Draw at a cost bounded by the widget width:
            # - Sparse trace (fits in the width): connect the points directly
            #   with a polyline (~w points max — cheap by construction).
            # - Dense trace: decimate to one min/max pair per screen column and
            #   draw vertical beats, so a multi-thousand-point history still
            #   paints in O(width) instead of O(n).
            n = len(self._history)
            values = np.fromiter(self._history, dtype=np.float64, count=n)
            if n > w:
                xs, y_top, y_bot = _decimate_columns(values, w, plot_h, min_v, max_v)
                lines = [QLineF(float(xi), float(yt), float(xi), float(yb))
                         for xi, yt, yb in zip(xs, y_top, y_bot)]
                # Envelope midline: a flat signal decimates to zero-height
                # column beats (isolated dots, nearly invisible); stroking
                # the midline renders flats as a solid horizontal while
                # tracing the mean through a varying envelope.
                y_mid = 0.5 * (y_top + y_bot)
                mid = [QPointF(float(xi), float(ym)) for xi, ym in zip(xs, y_mid)]
                if lines:
                    painter.setPen(QPen(QColor(255, 153, 0, 60), 4.0,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawLines(lines)
                    painter.setPen(QPen(self.line_color, 1.6,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawLines(lines)
                if len(mid) > 1:
                    painter.setPen(QPen(QColor(255, 153, 0, 60), 4.0,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPolyline(mid)
                    painter.setPen(QPen(self.line_color, 1.6,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPolyline(mid)
                elif mid:
                    painter.setPen(QPen(self.line_color, 1.6,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPoints(mid)
            else:
                ys = plot_h - 4 - (values - min_v) / span * (plot_h - 8)
                np.clip(ys, 0.0, float(plot_h), out=ys)
                xs = np.arange(n) * (w / max(1, n))
                pts = [QPointF(float(xi), float(yi)) for xi, yi in zip(xs, ys)]
                if pts:
                    painter.setPen(QPen(QColor(255, 153, 0, 60), 4.0,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPolyline(pts)
                    painter.drawPoints(pts)
                    painter.setPen(QPen(self.line_color, 1.6,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPolyline(pts)
                    painter.drawPoints(pts)

            # Current value badge (top-right of the plot area).
            painter.setFont(self.label_font)
            badge_text = f"{self._current:+.3f}"
            fm = painter.fontMetrics()
            b_w = fm.horizontalAdvance(badge_text) + 12
            margin = 4
            bg_rect = QRectF(w - b_w - margin, margin, b_w, 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(bg_rect, 3, 3)
            painter.setPen(self.line_color)
            painter.drawText(bg_rect, Qt.AlignCenter, badge_text)