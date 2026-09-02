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
- Zero heap allocation on the audio thread.
"""

import numpy as np
import torch

from base import Node, BLOCK_SIZE, CHANNELS, DTYPE, TelemetryRingBuffer

try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QLineF
    from PySide6.QtGui import QPainter, QPen, QColor, QFont

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


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


# ==============================================================================
# DSP Node Logic (zero allocation on audio thread)
# ==============================================================================
class ValuePlotterNode(Node):
    category = "Visual"
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

        self.add_float_param("min_val", -1.0, -100.0, 100.0, unit="",
                             help="Bottom scale bound.")
        self.add_float_param("max_val", 1.0, -100.0, 100.0, unit="",
                             help="Top scale bound.")

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
            nan=0.0, posinf=100.0, neginf=-100.0,
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

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(200, 100)

            self.bg_color = QColor("#141414")
            self.guide_color = QColor("#323232")
            self.line_color = QColor("#ff9900")
            self.text_color = QColor(180, 180, 180, 200)
            self.label_font = QFont("Monospace", 8, QFont.Bold)

            self._history = None  # set on first data
            self._current = 0.0
            self._has_data = False

            self.timer = QTimer(self)
            self.timer.setInterval(30)
            self.timer.timeout.connect(self.poll)
            self.timer.start()

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
            for f in frames:
                # f is shape (1, 8); extend the rolling trace with the 8 points.
                self._history.extend(float(v) for v in f[0])
            self._current = float(frames[-1][0, -1])
            self._has_data = True
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()

            painter.fillRect(self.rect(), self.bg_color)

            min_v = self._param_value("min_val", -1.0)
            max_v = self._param_value("max_val", 1.0)
            span = max(1e-6, max_v - min_v)

            # Dashed center guide.
            center_y = h / 2.0
            painter.setPen(QPen(self.guide_color, 1, Qt.DashLine))
            painter.drawLine(0, int(center_y), w, int(center_y))

            if not self._has_data or not self._history:
                painter.setPen(self.text_color)
                painter.drawText(self.rect(), Qt.AlignCenter, "No Signal")
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
                xs, y_top, y_bot = _decimate_columns(values, w, h, min_v, max_v)
                lines = [QLineF(float(xi), float(yt), float(xi), float(yb))
                         for xi, yt, yb in zip(xs, y_top, y_bot)]
                if lines:
                    painter.setPen(QPen(QColor(255, 153, 0, 60), 4.0,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawLines(lines)
                    painter.setPen(QPen(self.line_color, 1.6,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawLines(lines)
            else:
                ys = h - 4 - (values - min_v) / span * (h - 8)
                np.clip(ys, 0.0, float(h), out=ys)
                xs = np.arange(n) * (w / max(1, n))
                pts = [QPointF(float(xi), float(yi)) for xi, yi in zip(xs, ys)]
                if pts:
                    painter.setPen(QPen(QColor(255, 153, 0, 60), 4.0,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPolyline(pts)
                    painter.setPen(QPen(self.line_color, 1.6,
                                        Qt.SolidLine, Qt.RoundCap))
                    painter.drawPolyline(pts)

            # Current value badge (top-right).
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