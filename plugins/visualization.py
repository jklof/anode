import numpy as np
import queue
import torch
from base import Node


class WaveformDisplay(Node):
    category = "Visual"
    label = "Oscilloscope"
    VISUAL_WIDTH = 128

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out")
        # Keep queue small to drop frames gracefully
        self.monitor_queue = queue.Queue(maxsize=1)

    def process(self):
        sig = self.inp.get_tensor()

        # 1. Sanitize signal to prevent UI freezing
        sig = torch.clamp(sig, -1.0, 1.0)
        sig = torch.nan_to_num(sig, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. Pass-through audio efficiently (In-place copy to output buffer)
        self.out.buffer.copy_(sig)

        # 3. Handle Visualization Queue
        if not self.monitor_queue.full():
            # Downsample for the visual trace
            num_samples = sig.shape[-1]
            step = max(1, num_samples // self.VISUAL_WIDTH)
            downsampled_sig = sig[..., ::step]

            # PACKAGE PAYLOAD:
            # We include the shape metadata.
            # Note: sig.shape is a torch.Size (tuple), which is allocation-efficient.
            payload = {"samples": downsampled_sig.cpu().numpy().copy(), "shape": sig.shape}  # e.g., (2, 512)
            self.monitor_queue.put_nowait(payload)


try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
    from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF, QFont

    class WaveformWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "WaveformDisplay"

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(250, 150)
            self.data = None
            self.data_shape = (0, 0)
            self.shape_text = ""

            # Pre-allocate colors
            self.bg_color = QColor(20, 20, 20)
            self.grid_color = QColor(50, 50, 50)
            self.channel_colors = [QColor("#00ff00"), QColor("#00ccff")]
            self.text_color = QColor(150, 150, 150)
            self.debug_text_color = QColor("#00ff00")
            self.overlay_bg = QColor(0, 0, 0, 160)

            self.timer = QTimer(self)
            self.timer.interval = 33  # ~30 FPS
            self.timer.timeout.connect(self.poll)
            self.timer.start()

            self._cached_x = None
            self._last_width = 0

        def poll(self):
            try:
                latest = None
                q = getattr(self.proxy, "monitor_queue", None)
                if q:
                    while not q.empty():
                        latest = q.get_nowait()

                if latest is not None:
                    self.data = latest["samples"]
                    # Performance: Only re-format string if shape changes
                    if self.data_shape != latest["shape"]:
                        self.data_shape = latest["shape"]
                        # Format: "[Channels] Ch x [Frames]"
                        self.shape_text = f"{self.data_shape[0]} Ch x {self.data_shape[1]}"

                    self.update()
            except queue.Empty:
                pass

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Fill Background
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

                # Calculate metrics for the background "pill"
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(self.shape_text)
                text_height = metrics.height()

                # Position: Top Right with margin
                margin = 8
                bg_rect = QRectF(w - text_width - (margin * 2), margin, text_width + margin, text_height + 4)

                # Draw Background Pill
                painter.setPen(Qt.NoPen)
                painter.setBrush(self.overlay_bg)
                painter.drawRoundedRect(bg_rect, 4, 4)

                # Draw Shape Text
                painter.setPen(self.debug_text_color)
                painter.drawText(bg_rect, Qt.AlignCenter, self.shape_text)

except ImportError:
    pass
