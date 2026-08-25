"""
SpectrogramDisplay — real-time scrolling waterfall spectrogram (Visual node).

Native-analysis notes (audio thread):
- Pass-through is bit-exact: the input is copied to the output untouched.
- Stereo analysis: each channel gets its own 2048-point sliding FFT; the
  widget renders L/R waterfalls side by side. Mono inputs are analysed and
  passed through duplicated on both channels (MonoToStereo convention).
- Pre-allocated buffers everywhere; magnitudes are normalized by 2/window_sum
  so a full-scale sine reads ~0 dBFS against the [-80, 0] dB defaults.
- The only deliberate per-block transient is the rfft output (no out= variant,
  same exception as convolution_reverb/filters). Queue dispatch is non-blocking
  and checked BEFORE copying the payload, so overflow costs nothing.
"""

import queue

import numpy as np
import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox
    from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
    from PySide6.QtGui import QPainter, QColor, QPen, QImage, QFont

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


# ==============================================================================
# 1. Color lookup tables (256-entry ARGB uint32, 0xAARRGGBB little-endian safe)
# ==============================================================================
def _generate_colormap_lut(stops):
    """Interpolates RGB stops into a 256-element uint32 array."""
    lut = np.zeros(256, dtype=np.uint32)
    pos = [s[0] for s in stops]
    r_vals = [s[1][0] for s in stops]
    g_vals = [s[1][1] for s in stops]
    b_vals = [s[1][2] for s in stops]

    x = np.linspace(0, 1, 256)
    r = np.clip(np.interp(x, pos, r_vals), 0, 255).astype(np.uint32)
    g = np.clip(np.interp(x, pos, g_vals), 0, 255).astype(np.uint32)
    b = np.clip(np.interp(x, pos, b_vals), 0, 255).astype(np.uint32)

    lut[:] = (0xFF << 24) | (r << 16) | (g << 8) | b
    return lut


COLORMAPS = {
    "Inferno": _generate_colormap_lut([
        (0.00, (0, 0, 4)),
        (0.20, (40, 11, 84)),
        (0.40, (101, 21, 110)),
        (0.60, (186, 54, 85)),
        (0.80, (249, 140, 10)),
        (1.00, (252, 255, 164)),
    ]),
    "Viridis": _generate_colormap_lut([
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]),
    "Turbo": _generate_colormap_lut([
        (0.00, (48, 18, 59)),
        (0.20, (70, 134, 251)),
        (0.40, (27, 229, 181)),
        (0.60, (164, 252, 60)),
        (0.80, (251, 153, 44)),
        (1.00, (122, 4, 3)),
    ]),
}


FFT_SIZE = 2048
DISPLAY_BINS = 128
MIN_FREQ = 20.0
MAX_FREQ = 20000.0


# ==============================================================================
# 2. Node DSP Logic (per-channel sliding FFT, zero net allocations)
# ==============================================================================
class SpectrogramDisplay(Node):
    category = "Visual"
    label = "Spectrogram"

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=CHANNELS)

        self.add_menu_param("colormap", list(COLORMAPS.keys()), initial_idx=0)
        self.add_float_param("min_db", -80.0, -120.0, -20.0)
        self.add_float_param("max_db", 0.0, -20.0, 20.0)

        # Thread-safe IPC to the UI widget (frames may be dropped freely)
        self.monitor_queue = queue.Queue(maxsize=2)

        self._fft_bins = FFT_SIZE // 2 + 1

        # Per-channel sliding buffer & analysis buffers
        self._ring = torch.zeros((CHANNELS, FFT_SIZE), dtype=DTYPE)
        self._write_pos = 0
        self._unwrapped = torch.zeros((CHANNELS, FFT_SIZE), dtype=DTYPE)
        self._window = torch.hann_window(FFT_SIZE, dtype=DTYPE)
        # Normalize so a full-scale sine peaks at ~0 dBFS (Hann sums to ~N/2)
        self._norm = 2.0 / float(self._window.sum())
        self._windowed = torch.zeros((CHANNELS, FFT_SIZE), dtype=DTYPE)
        self._mag = torch.zeros((CHANNELS, self._fft_bins), dtype=DTYPE)
        self._db = torch.zeros((CHANNELS, self._fft_bins), dtype=DTYPE)
        self._column = torch.zeros((CHANNELS, DISPLAY_BINS), dtype=DTYPE)

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

        # 1. Bit-exact pass-through. Mono duplicates to both channels
        # (MonoToStereo convention) so downstream routing never ghosts.
        self.out.buffer.copy_(sig)

        # 2. Write this block into the per-channel sliding buffers.
        # Safe because FFT_SIZE (2048) is a multiple of BLOCK_SIZE (512):
        # the [wp : wp+BLOCK] slice can never straddle the ring wrap.
        wp = self._write_pos
        seg = self._ring[:, wp:wp + BLOCK_SIZE]
        if in_ch >= CHANNELS:
            seg.copy_(sig[:CHANNELS])
        else:
            # Mono (or fewer channels): duplicate channel 0 for analysis
            seg[:, :in_ch].copy_(sig[:in_ch])
            for c in range(in_ch, CHANNELS):
                seg[:, c].copy_(sig[0])
        self._write_pos = (wp + BLOCK_SIZE) % FFT_SIZE

        # 3. Unwrap circular -> chronological [oldest ... newest]
        tail = FFT_SIZE - self._write_pos
        self._unwrapped[:, :tail].copy_(self._ring[:, self._write_pos:])
        self._unwrapped[:, tail:].copy_(self._ring[:, :self._write_pos])

        # 4. Window + FFT (rfft has no out=; this transient is the one
        # documented per-block allocation)
        torch.mul(self._unwrapped, self._window, out=self._windowed)
        spectrum = torch.fft.rfft(self._windowed, n=FFT_SIZE, dim=1)

        # 5. Magnitude -> dBFS with normalization
        torch.abs(spectrum, out=self._mag)
        self._mag.mul_(self._norm).clamp_(min=1e-9)
        torch.log10(self._mag, out=self._db)
        self._db.mul_(20.0)

        # 6. Log-frequency resample + normalize into [0, 1]
        torch.index_select(self._db, 1, self._log_indices, out=self._column)
        min_db = self.params["min_db"].value
        max_db = self.params["max_db"].value
        db_range = max(1.0, max_db - min_db)
        self._column.sub_(min_db).div_(db_range).clamp_(0.0, 1.0)

        # 7. Dispatch to UI (checked before copying: overflow is free)
        if not self.monitor_queue.full():
            self.monitor_queue.put_nowait(
                self._column.numpy().copy()  # shape (CHANNELS, DISPLAY_BINS)
            )


# ==============================================================================
# 3. UI Widget: dual-panel scrolling waterfall with log-frequency grid
# ==============================================================================
if GUI_AVAILABLE:

    class SpectrogramWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "SpectrogramDisplay"

        LANDMARK_FREQS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
        PANEL_GAP = 8

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(280, 170)

            self.bg_color = QColor(18, 18, 18)
            self.grid_pen = QPen(QColor(255, 255, 255, 45), 1, Qt.DashLine)
            self.text_color = QColor(210, 210, 210, 220)
            self.label_font = QFont("Monospace", 7, QFont.Bold)
            self.overlay_bg = QColor(0, 0, 0, 140)

            self._colormap_names = list(COLORMAPS.keys())
            self._current_lut = COLORMAPS["Inferno"]

            self._image_width = 240
            self._image_height = DISPLAY_BINS
            self._wf = [
                self._make_image(),
                self._make_image(),
            ]

            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(4, 4, 4, 4)
            main_layout.setSpacing(4)

            top_row = QHBoxLayout()
            self.combo_map = QComboBox()
            self.combo_map.addItems(self._colormap_names)
            self.combo_map.currentIndexChanged.connect(self._on_colormap_changed)
            self.combo_map.setFixedHeight(22)
            top_row.addWidget(self.combo_map)
            top_row.addStretch()
            main_layout.addLayout(top_row)
            main_layout.addStretch()

            self.timer = QTimer(self)
            self.timer.setInterval(25)
            self.timer.timeout.connect(self.poll_queue)
            self.timer.start()

        def _make_image(self):
            img = QImage(self._image_width, self._image_height, QImage.Format_RGB32)
            img.fill(0xFF000000)
            return img

        def _on_colormap_changed(self, idx):
            self._current_lut = COLORMAPS[self._colormap_names[idx]]
            self.proxy.set_parameter("colormap", idx)

        def _shift_and_paint(self, img, columns):
            """Scroll img left by len(columns) px and paint them on the right."""
            w, hh = img.width(), img.height()
            n = len(columns)
            if n <= 0:
                return
            if n >= w:
                columns = columns[-w:]
                n = w
            shifted = img.copy(n, 0, w - n, hh)
            p = QPainter(img)
            p.drawImage(0, 0, shifted)
            p.end()
            lut = self._current_lut
            for i, col in enumerate(columns):
                x = w - n + i
                # Column rows run 20 Hz -> 20 kHz; flip so highs render on top
                indices = (np.asarray(col, dtype=np.float32)[::-1] * 255.0).astype(np.uint8)
                colors = lut[indices]
                for y in range(hh):
                    img.setPixel(x, y, int(colors[y]))

        def poll_queue(self):
            q = getattr(self.proxy, "monitor_queue", None)
            if not q or q.empty():
                return

            frames = []
            while not q.empty():
                try:
                    frames.append(q.get_nowait())
                except queue.Empty:
                    break

            valid = [f for f in frames if isinstance(f, np.ndarray) and f.shape == (2, DISPLAY_BINS)]
            if not valid:
                return

            self._shift_and_paint(self._wf[0], [f[0] for f in valid])
            self._shift_and_paint(self._wf[1], [f[1] for f in valid])
            self.update()

        def update_from_params(self, params):
            if "colormap" in params:
                idx = int(params["colormap"])
                if idx != self.combo_map.currentIndex():
                    self.combo_map.setCurrentIndex(idx)
                self._current_lut = COLORMAPS[self._colormap_names[idx]]

        def resizeEvent(self, event):
            super().resizeEvent(event)
            new_w = max(100, self.width() - 12)
            if abs(new_w - self._image_width) > 40:
                self._image_width = new_w
                for i, img in enumerate(self._wf):
                    scaled = img.scaled(self._image_width, self._image_height)
                    fresh = self._make_image()
                    p = QPainter(fresh)
                    p.drawImage(0, 0, scaled)
                    p.end()
                    self._wf[i] = fresh

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            rect = self.rect()
            painter.fillRect(rect, self.bg_color)

            plot_rect = QRectF(6, 30, rect.width() - 12, rect.height() - 36)
            pw = plot_rect.width()
            panel_w = (pw - self.PANEL_GAP) / 2.0
            panels = [
                QRectF(plot_rect.left(), plot_rect.top(), panel_w, plot_rect.height()),
                QRectF(plot_rect.left() + panel_w + self.PANEL_GAP, plot_rect.top(),
                       panel_w, plot_rect.height()),
            ]

            painter.drawImage(panels[0], self._wf[0])
            painter.drawImage(panels[1], self._wf[1])

            # Frequency grid + labels (label pill drawn on the left panel only)
            painter.setFont(self.label_font)
            fm = painter.fontMetrics()
            log_min = np.log10(MIN_FREQ)
            log_range = np.log10(MAX_FREQ) - log_min

            for freq in self.LANDMARK_FREQS:
                norm_y = (np.log10(freq) - log_min) / log_range
                y_pixel = plot_rect.bottom() - norm_y * plot_rect.height()
                if not (plot_rect.top() + 8 <= y_pixel <= plot_rect.bottom() - 4):
                    continue

                painter.setPen(self.grid_pen)
                for pr in panels:
                    painter.drawLine(QPointF(pr.left(), y_pixel), QPointF(pr.right(), y_pixel))

                text = f"{freq // 1000}k" if freq >= 1000 else f"{freq}"
                txt_w = fm.horizontalAdvance(text) + 6
                txt_h = fm.height() + 2
                pill = QRectF(panels[0].left() + 2, y_pixel - txt_h / 2, txt_w, txt_h)
                painter.setPen(Qt.NoPen)
                painter.setBrush(self.overlay_bg)
                painter.drawRoundedRect(pill, 2, 2)
                painter.setPen(self.text_color)
                painter.drawText(pill, Qt.AlignCenter, text)

            # Channel tags
            for tag, pr in (("L", panels[0]), ("R", panels[1])):
                pill = QRectF(pr.right() - 16, pr.top() + 3, 14, 13)
                painter.setPen(Qt.NoPen)
                painter.setBrush(self.overlay_bg)
                painter.drawRoundedRect(pill, 2, 2)
                painter.setPen(self.text_color)
                painter.drawText(pill, Qt.AlignCenter, tag)

            # Panel borders
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            painter.setBrush(Qt.NoBrush)
            for pr in panels:
                painter.drawRoundedRect(pr, 2, 2)
