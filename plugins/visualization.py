import torch
import numpy as np
import queue
from core import Node, BLOCK_SIZE, CHANNELS, DTYPE


class WaveformDisplay(Node):
    CUSTOM_UI = "WaveformWidget"

    def __init__(self, name="Scope"):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out")
        self.monitor_queue = queue.Queue(maxsize=1)

    def process(self):
        sig = self.inp.get_tensor()
        self.out.buffer.copy_(sig)

        if not self.monitor_queue.full():
            snapshot = sig.cpu().numpy().copy()
            try:
                self.monitor_queue.put_nowait(snapshot)
            except:
                pass


try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath

    class WaveformWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "WaveformDisplay"

        def __init__(self, node_logic):
            super().__init__()
            self.node = node_logic
            self.setMinimumSize(250, 150)
            self.data = None
            self.channel_colors = [QColor("#00ff00"), QColor("#00ccff"), QColor("#ff00ff"), QColor("#ffff00")]

            self.timer = QTimer(self)
            self.timer.interval = 33
            self.timer.timeout.connect(self.poll)
            self.timer.start()

        def poll(self):
            try:
                self.data = self.node.monitor_queue.get_nowait()
                self.update()
            except queue.Empty:
                pass

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(20, 20, 20))
            if self.data is None:
                painter.setPen(QColor(100, 100, 100))
                painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for Signal...")
                return

            num_channels, num_samples = self.data.shape
            if num_channels == 0 or num_samples == 0:
                return

            w = self.width()
            h = self.height()
            center_y = h / 2.0

            painter.setPen(QPen(QColor(50, 50, 50), 1, Qt.DashLine))
            painter.drawLine(0, int(center_y), w, int(center_y))

            for ch in range(num_channels):
                color = self.channel_colors[ch % len(self.channel_colors)]
                painter.setPen(QPen(color, 1.5))
                path = QPainterPath()
                step = max(1, num_samples // w)
                val_start = np.clip(self.data[ch, 0], -1.0, 1.0)
                path.moveTo(0, center_y - (val_start * center_y * 0.9))

                for i in range(step, num_samples, step):
                    x = (i / num_samples) * w
                    val = np.clip(self.data[ch, i], -1.0, 1.0)
                    y = center_y - (val * center_y * 0.9)
                    path.lineTo(x, y)
                painter.drawPath(path)
                painter.setPen(color)
                painter.drawText(5, 15 + (ch * 15), f"Ch {ch+1}")

except ImportError:
    pass
