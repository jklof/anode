import numpy as np
import torch
from base import Node, BLOCK_SIZE, CHANNELS, TelemetryRingBuffer


class WaveformDisplay(Node):
    category = "Visual"
    label = "Oscilloscope"
    VISUAL_WIDTH = 128

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out")
        # Pre-allocated SPSC telemetry buffer owning 4 isolated slots
        self.monitor_queue = TelemetryRingBuffer(
            capacity=4, shape=(CHANNELS, self.VISUAL_WIDTH), dtype=np.float32
        )
        # Pre-allocated analysis buffer for visualization downsampling
        self._analysis_buf = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
        # Pre-allocated downsampled buffer
        self._downsampled = torch.zeros((CHANNELS, self.VISUAL_WIDTH), dtype=torch.float32)

    def process(self):
        sig = self.inp.get_tensor()

        # 1. STRICT BIT-EXACT PASS-THROUGH: Copy input directly to output
        # No clamping, no sanitization on the main audio path
        self.out.buffer.copy_(sig)

        # 2. Visualization: Perform analysis on private buffer copy
        # Downsample for the visual trace
        num_samples = sig.shape[-1]
        step = max(1, num_samples // self.VISUAL_WIDTH)

        # Copy to analysis buffer (private, can be sanitized)
        self._analysis_buf.copy_(sig)
        # Sanitize ONLY the analysis buffer
        self._analysis_buf.clamp_(-1.0, 1.0)
        torch.nan_to_num(self._analysis_buf, nan=0.0, posinf=1.0, neginf=-1.0, out=self._analysis_buf)

        # Downsample
        self._downsampled.copy_(self._analysis_buf[..., ::step])

        # 3. Push to telemetry ring buffer (copies into ring slot; zero allocation)
        self.monitor_queue.push(self._downsampled.numpy())


try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
    from PySide6.QtGui import QPainter, QPen, QColor, QFont

    class WaveformWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "WaveformDisplay"

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(250, 150)
            self.data = None
            self.shape_text = f"{CHANNELS} Ch x {BLOCK_SIZE}"

            # Pre-allocate colors
            self.bg_color = QColor(20, 20, 20)
            self.grid_color = QColor(50, 50, 50)
            self.channel_colors = [QColor("#00ff00"), QColor("#00ccff")]
            self.text_color = QColor(150, 150, 150)
            self.debug_text_color = QColor("#00ff00")
            self.overlay_bg = QColor(0, 0, 0, 160)

            self.timer = QTimer(self)
            self.timer.setInterval(33)  # ~30 FPS
            self.timer.timeout.connect(self.poll)
            self.timer.start()

            self._cached_x = None
            self._last_width = 0

        def poll(self):
            q = getattr(self.proxy, "monitor_queue", None)
            if q is not None:
                latest = q.pop_latest()
                if latest is not None:
                    self.data = latest
                    self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), self.bg_color)

            if self.data is None:
                painter.setPen(self.text_color)
                painter.drawText(self.rect(), Qt.AlignCenter, "No Signal")
                return

            num_channels, num_samples = self.data.shape
            w, h = self.width(), self.height()
            center_y = h / 2.0
            scale_y = center_y * 0.9

            # --- 1. Draw Grid ---
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
            painter.drawLine(0, int(center_y), w, int(center_y))

            # --- 2. Draw Waveforms ---
            if w != self._last_width or self._cached_x is None or len(self._cached_x) != num_samples:
                self._cached_x = np.linspace(0, w, num=num_samples)
                self._last_width = w

            for ch in range(min(num_channels, 2)):
                painter.setPen(QPen(self.channel_colors[ch % 2], 1.5))
                chan_data = self.data[ch]
                y_coords = np.clip(center_y - (chan_data * scale_y), 0, h)
                points = [QPointF(x, y) for x, y in zip(self._cached_x, y_coords) if np.isfinite(y)]
                painter.drawPolyline(points)

            # --- 3. Draw Debug Shape Overlay ---
            if self.shape_text:
                painter.setFont(QFont("Monospace", 8, QFont.Bold))
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(self.shape_text)
                text_height = metrics.height()
                margin = 8
                bg_rect = QRectF(w - text_width - (margin * 2), margin, text_width + margin, text_height + 4)

                painter.setPen(Qt.NoPen)
                painter.setBrush(self.overlay_bg)
                painter.drawRoundedRect(bg_rect, 4, 4)

                painter.setPen(self.debug_text_color)
                painter.drawText(bg_rect, Qt.AlignCenter, self.shape_text)

except ImportError:
    pass
