"""
SpectrumDisplay — stereo spectrum analyzer with overlaid L/R curves (Visual).

Sibling of SpectrogramDisplay: same per-channel sliding-FFT analysis, but
renders instantaneous spectra (with display ballistics) instead of a waterfall.

Audio-thread notes:
- Pass-through is bit-exact; the audio path never modifies the signal.
- Mono inputs are analysed and passed through duplicated on both channels.
- Magnitudes are normalized against window_sum/2 so a full-scale sine reads
  ~0 dBFS (defaults: [-70, +6] dB).
- The rfft output is the single documented per-block transient (no out=
  variant); everything else runs through pre-allocated buffers with out=.
"""

import queue

import numpy as np
import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    from PySide6.QtCore import Qt, QTimer, QRectF
    from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath, QLinearGradient, QFont

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


FFT_SIZE = 2048
DISPLAY_BINS = 256
MIN_FREQ = 20.0
MAX_FREQ = 20000.0


# ==============================================================================
# DSP Node Logic (per-channel sliding FFT, zero net allocations)
# ==============================================================================
class SpectrumDisplay(Node):
    category = "Visual"
    label = "Spectrum Visualizer"

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=CHANNELS)

        # Display parameters (min/max consumed by the node, smoothing by the UI)
        self.add_float_param("min_db", -70.0, -120.0, -20.0)
        self.add_float_param("max_db", 6.0, -10.0, 24.0)
        self.add_float_param("smoothing", 0.65, 0.0, 0.95)

        # Thread-safe IPC to the UI widget (frames may be dropped freely)
        self.monitor_queue = queue.Queue(maxsize=2)

        self._fft_bins = FFT_SIZE // 2 + 1

        self._ring = torch.zeros((CHANNELS, FFT_SIZE), dtype=DTYPE)
        self._write_pos = 0
        self._unwrapped = torch.zeros((CHANNELS, FFT_SIZE), dtype=DTYPE)
        self._window = torch.hann_window(FFT_SIZE, dtype=DTYPE)
        self._windowed = torch.zeros((CHANNELS, FFT_SIZE), dtype=DTYPE)
        self._mag = torch.zeros((CHANNELS, self._fft_bins), dtype=DTYPE)
        self._db = torch.zeros((CHANNELS, self._fft_bins), dtype=DTYPE)
        self._display_points = torch.zeros((CHANNELS, DISPLAY_BINS), dtype=DTYPE)

        # Full-scale sine (amplitude 1.0) peaks near window_sum/2 -> 0 dBFS
        self._db_ref = max(1e-9, float(self._window.sum()) / 2.0)

        # Logarithmic frequency mapping: linear bin -> display bin indices
        bin_freqs = torch.linspace(0, SAMPLE_RATE / 2.0, steps=self._fft_bins)
        targets = torch.tensor(
            np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), num=DISPLAY_BINS),
            dtype=DTYPE,
        )
        self._log_indices = torch.searchsorted(bin_freqs, targets).clamp_(0, self._fft_bins - 1)

    def start(self):
        # Transport restart must not smear the previous session's audio back in
        self._ring.zero_()
        self._write_pos = 0

    def process(self):
        sig = self.inp.get_tensor()
        in_ch = sig.shape[0]

        # 1. Bit-exact pass-through. Mono broadcasts to both channels.
        self.out.buffer.copy_(sig)

        # 2. Ring write. Safe slice: FFT_SIZE % BLOCK_SIZE == 0.
        wp = self._write_pos
        seg = self._ring[:, wp:wp + BLOCK_SIZE]
        if in_ch >= CHANNELS:
            seg.copy_(sig[:CHANNELS])
        else:
            seg[0].copy_(sig[0])
            seg[1].copy_(sig[0])
        self._write_pos = (wp + BLOCK_SIZE) % FFT_SIZE

        # 3. Unwrap circular -> chronological [oldest ... newest]
        tail = FFT_SIZE - self._write_pos
        self._unwrapped[:, :tail].copy_(self._ring[:, self._write_pos:])
        self._unwrapped[:, tail:].copy_(self._ring[:, :self._write_pos])

        # 4. Window + FFT (rfft has no out=; documented transient)
        torch.mul(self._unwrapped, self._window, out=self._windowed)
        spectrum = torch.fft.rfft(self._windowed, n=FFT_SIZE, dim=1)

        # 5. Magnitude -> dBFS
        torch.abs(spectrum, out=self._mag)
        self._mag.div_(self._db_ref).clamp_(min=1e-9)
        torch.log10(self._mag, out=self._db)
        self._db.mul_(20.0)

        # 6. Log-frequency resample + normalize into [0, 1]
        torch.index_select(self._db, 1, self._log_indices, out=self._display_points)
        min_db = self.params["min_db"].value
        max_db = self.params["max_db"].value
        db_range = max(1.0, max_db - min_db)
        self._display_points.sub_(min_db).div_(db_range).clamp_(0.0, 1.0)

        # 7. Dispatch to UI (checked before copying: overflow is free)
        if not self.monitor_queue.full():
            self.monitor_queue.put_nowait(
                self._display_points.numpy().copy()  # shape (CHANNELS, DISPLAY_BINS)
            )


# ==============================================================================
# UI Widget: overlaid translucent stereo curves
# ==============================================================================
if GUI_AVAILABLE:

    class SpectrumWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "SpectrumDisplay"

        LANDMARK_FREQS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
        DB_GRID_LEVELS = [6, 0, -20, -40, -60]
        DEFAULT_MIN_DB = -70.0
        DEFAULT_MAX_DB = 6.0
        DEFAULT_SMOOTHING = 0.65

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(260, 150)

            self.bg_color = QColor(18, 18, 18)
            self.grid_pen = QPen(QColor(255, 255, 255, 35), 1, Qt.DashLine)
            self.text_color = QColor(180, 180, 180, 200)
            self.label_font = QFont("Monospace", 7, QFont.Bold)
            self.overlay_bg = QColor(0, 0, 0, 160)

            # Channel styling matching the Oscilloscope palette
            self.channel_colors = [QColor("#00ff00"), QColor("#00ccff")]
            self.peak_pens = [
                QPen(QColor(0, 255, 100, 230), 1.5),
                QPen(QColor(0, 204, 255, 230), 1.5),
            ]

            self._smoothed = np.zeros((CHANNELS, DISPLAY_BINS), dtype=np.float32)
            self._has_data = False

            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            self.timer = QTimer(self)
            self.timer.setInterval(25)
            self.timer.timeout.connect(self.poll_queue)
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

        def poll_queue(self):
            q = getattr(self.proxy, "monitor_queue", None)
            if not q or q.empty():
                return

            latest = None
            while not q.empty():
                try:
                    latest = q.get_nowait()
                except queue.Empty:
                    break

            if not (isinstance(latest, np.ndarray) and latest.shape == (CHANNELS, DISPLAY_BINS)):
                return

            smoothing = self._param_value("smoothing", self.DEFAULT_SMOOTHING)
            if not self._has_data:
                self._smoothed[:] = latest
                self._has_data = True
            else:
                # Ballistic smoothing toward the newest frame
                self._smoothed *= smoothing
                self._smoothed += latest * (1.0 - smoothing)
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            rect = self.rect()
            painter.fillRect(rect, self.bg_color)

            pad_left, pad_right, pad_top, pad_bottom = 6, 6, 8, 16
            plot_rect = QRectF(
                pad_left,
                pad_top,
                rect.width() - pad_left - pad_right,
                rect.height() - pad_top - pad_bottom,
            )
            if plot_rect.width() <= 0 or plot_rect.height() <= 0:
                return

            min_db = self._param_value("min_db", self.DEFAULT_MIN_DB)
            max_db = self._param_value("max_db", self.DEFAULT_MAX_DB)
            db_range = max(1.0, max_db - min_db)

            # --- 1. Horizontal dB grid ---
            painter.setFont(self.label_font)
            fm = painter.fontMetrics()
            for db in self.DB_GRID_LEVELS:
                if not (min_db <= db <= max_db):
                    continue
                norm_y = (db - min_db) / db_range
                y_pixel = plot_rect.bottom() - norm_y * plot_rect.height()
                painter.setPen(self.grid_pen)
                painter.drawLine(QRectF(plot_rect.left(), y_pixel, plot_rect.width(), 0).topLeft(),
                                 QRectF(plot_rect.left(), y_pixel, plot_rect.width(), 0).bottomRight())
                label = f"{db:+d}dB" if db != 0 else "0dB"
                painter.setPen(self.text_color)
                painter.drawText(int(plot_rect.right() - fm.horizontalAdvance(label) - 2),
                                 int(y_pixel - 2), label)

            # --- 2. Vertical frequency grid + bottom labels ---
            log_min = np.log10(MIN_FREQ)
            log_range = np.log10(MAX_FREQ) - log_min
            for freq in self.LANDMARK_FREQS:
                norm_x = (np.log10(freq) - log_min) / log_range
                x_pixel = plot_rect.left() + norm_x * plot_rect.width()
                painter.setPen(self.grid_pen)
                painter.drawLine(int(x_pixel), int(plot_rect.top()),
                                 int(x_pixel), int(plot_rect.bottom()))
                text = f"{freq // 1000}k" if freq >= 1000 else f"{freq}"
                painter.setPen(self.text_color)
                painter.drawText(int(x_pixel - fm.horizontalAdvance(text) / 2),
                                 int(rect.bottom() - 3), text)

            # --- 3. Overlaid translucent channel curves ---
            if self._has_data:
                x_step = plot_rect.width() / float(DISPLAY_BINS - 1)

                gradients = [
                    QLinearGradient(0, plot_rect.top(), 0, plot_rect.bottom()),
                    QLinearGradient(0, plot_rect.top(), 0, plot_rect.bottom()),
                ]
                gradients[0].setColorAt(0.0, QColor(0, 255, 0, 110))
                gradients[0].setColorAt(0.5, QColor(0, 200, 50, 60))
                gradients[0].setColorAt(1.0, QColor(0, 40, 10, 15))
                gradients[1].setColorAt(0.0, QColor(0, 204, 255, 110))
                gradients[1].setColorAt(0.5, QColor(0, 120, 220, 60))
                gradients[1].setColorAt(1.0, QColor(0, 20, 50, 15))

                for ch in range(min(CHANNELS, self._smoothed.shape[0])):
                    pts = self._smoothed[ch]
                    fill_path = QPainterPath()
                    line_path = QPainterPath()
                    start_y = plot_rect.bottom() - float(pts[0]) * plot_rect.height()
                    fill_path.moveTo(plot_rect.left(), plot_rect.bottom())
                    fill_path.lineTo(plot_rect.left(), start_y)
                    line_path.moveTo(plot_rect.left(), start_y)
                    for i in range(1, DISPLAY_BINS):
                        px = plot_rect.left() + i * x_step
                        py = plot_rect.bottom() - float(pts[i]) * plot_rect.height()
                        fill_path.lineTo(px, py)
                        line_path.lineTo(px, py)
                    fill_path.lineTo(plot_rect.right(), plot_rect.bottom())
                    fill_path.closeSubpath()

                    painter.setPen(Qt.NoPen)
                    painter.setBrush(gradients[ch])
                    painter.drawPath(fill_path)

                    painter.setPen(self.peak_pens[ch])
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(line_path)

            # --- 4. Stereo legend pill (top-right) ---
            legend_w, legend_h = 60, 16
            legend_rect = QRectF(plot_rect.right() - legend_w - 4, plot_rect.top() + 4,
                                 legend_w, legend_h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.overlay_bg)
            painter.drawRoundedRect(legend_rect, 3, 3)

            f = QFont(self.label_font)
            f.setBold(True)
            f.setPointSize(8)
            painter.setFont(f)
            baseline = int(legend_rect.bottom() - 3)
            painter.setPen(self.channel_colors[0])
            painter.drawText(int(legend_rect.left() + 6), baseline, "L")
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(int(legend_rect.left() + 24), baseline, "|")
            painter.setPen(self.channel_colors[1])
            painter.drawText(int(legend_rect.left() + 42), baseline, "R")

            # --- 5. Outer border ---
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(plot_rect, 2, 2)
