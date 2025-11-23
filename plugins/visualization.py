import numpy as np
import queue
from base import Node


class WaveformDisplay(Node):
    CUSTOM_UI = "WaveformWidget"

    def __init__(self, name="Scope"):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out")
        # Keep queue small to drop frames gracefully
        self.monitor_queue = queue.Queue(maxsize=1)

    def process(self):
        sig = self.inp.get_tensor()
        # Pass-through audio efficiently
        self.out.buffer.copy_(sig)

        # OPTIMIZATION: Check queue BEFORE converting data.
        # This prevents allocating numpy arrays that will just be thrown away.
        if not self.monitor_queue.full():
            # We must copy() to ensure thread safety (detach from DSP memory)
            # using .numpy() on CPU tensor is cheap, the .copy() is the cost.
            snapshot = sig.cpu().numpy().copy()
            self.monitor_queue.put_nowait(snapshot)


try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, QPointF
    from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF

    class WaveformWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "WaveformDisplay"

        def __init__(self, node_logic):
            super().__init__()
            self.node = node_logic
            self.setMinimumSize(250, 150)
            self.data = None

            # Pre-allocate colors to avoid creating objects in paintEvent
            self.bg_color = QColor(20, 20, 20)
            self.grid_color = QColor(50, 50, 50)
            self.channel_colors = [QColor("#00ff00"), QColor("#00ccff")]
            self.text_color = QColor(100, 100, 100)

            self.timer = QTimer(self)
            # 30 FPS is sufficient for a scope.
            # 33ms interval = ~30fps.
            self.timer.interval = 33
            self.timer.timeout.connect(self.poll)
            self.timer.start()

        def poll(self):
            # Drain queue to get the LATEST frame, discarding older ones if any
            try:
                latest = None
                while not self.node.monitor_queue.empty():
                    latest = self.node.monitor_queue.get_nowait()

                if latest is not None:
                    self.data = latest
                    self.update()  # Schedule repaint
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
            if num_samples < 2:
                return

            w = self.width()
            h = self.height()
            center_y = h / 2.0
            scale_y = center_y * 0.9  # Leave some margin

            # Draw Center Line
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
            painter.drawLine(0, int(center_y), w, int(center_y))

            # OPTIMIZATION: Downsampling (LOD)
            # If we have 512 samples but width is 250px, step by 2.
            step = max(1, num_samples // w)

            for ch in range(min(num_channels, 2)):  # Limit to 2 channels for viz
                pen_color = self.channel_colors[ch % 2]
                painter.setPen(QPen(pen_color, 1.5))

                # Get channel data view
                chan_data = self.data[ch]

                # OPTIMIZATION: Fast Point Mapping
                # Create X coordinates (0 to w)
                x_coords = np.linspace(0, w, num=len(chan_data[::step]))

                # Create Y coordinates (inverted and scaled)
                y_coords = center_y - (chan_data[::step] * scale_y)

                # Combine into QPointF objects
                points = [QPointF(x, y) for x, y in zip(x_coords, y_coords)]

                # Draw the whole line at once
                painter.drawPolyline(points)

                # Draw Channel Label
                painter.drawText(5, 15 + (ch * 15), f"Ch {ch+1}")

except ImportError:
    pass
