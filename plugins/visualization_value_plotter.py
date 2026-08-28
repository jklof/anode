"""
ValuePlotterNode — real-time scrolling line plot for CV / envelopes / peaks (Visual).

Real-time notes:
- Audio pass-through is bit-exact: the input is copied to the stereo output
  untouched (a mono input is broadcast to both output channels without
  resizing the output buffer, per the MonoToStereo convention).
- Analysis runs on a private copy: channel 0 is isolated, wall-clamped to the
  display bounds, and downsampled 512 -> 8 points.
- Telemetry travels through a bounded, non-blocking SPSC ring buffer. Pushing
  ~94 blocks/sec into 16 slots causes intentional frame drops when the UI poll
  (~30 FPS) is slower than the audio thread (AGENTS.md section 10).
- Zero heap allocation on the audio thread.
"""

import numpy as np
import torch

from base import Node, BLOCK_SIZE, CHANNELS, DTYPE, TelemetryRingBuffer

try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
    from PySide6.QtGui import QPainter, QPen, QColor, QFont

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


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

        HISTORY_LEN = 256

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
            latest = frames[-1]
            # latest is shape (1, 8); extend the rolling trace with the 8 points.
            self._history.extend(float(v) for v in latest[0])
            self._current = float(latest[0, -1])
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

            if not self._has_data or self._history is None:
                painter.setPen(self.text_color)
                painter.drawText(self.rect(), Qt.AlignCenter, "No Signal")
                return

            # Normalize history into plot Y coordinates.
            n = len(self._history)
            hist = list(self._history)
            y_coords = []
            for i, v in enumerate(hist):
                norm = (v - min_v) / span
                y = h - 4 - norm * (h - 8)
                y_coords.append((i, float(np.clip(y, 0, h))))

            # Glowing polyline.
            painter.setPen(QPen(QColor(255, 153, 0, 60), 4.0))
            glow = [QPointF(x * w / max(1, n), y) for x, y in y_coords]
            if glow:
                painter.drawPolyline(glow)
            painter.setPen(QPen(self.line_color, 1.6))
            if glow:
                painter.drawPolyline(glow)

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