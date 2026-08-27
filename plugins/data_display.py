"""
DataDisplay — real-time signal statistics HUD (Visual node).

Native-analysis notes (audio thread):
- Pass-through is bit-exact: the input is copied to the output untouched
  (mono inputs broadcast to both output channels, MonoToStereo convention).
- Statistics are computed once every UPDATE_INTERVAL_BLOCKS blocks (~23 Hz)
  and cover the whole buffer globally (all channels pooled); per-channel
  breakdowns are deliberately not computed to keep the compact grid.
- Signal classification order matters: silence (peak below -120 dBFS) is
  detected FIRST so an all-zero/disconnected input reads "Silent" instead of
  being misreported as a constant DC value (zeros are trivially constant).
- Pre-allocated scratch buffer for the squared-signal reduction; telemetry
  dict is mutated in place (never swapped) so UI readers holding a reference
  via get_telemetry() always see a consistent snapshot.
- Lock-free SPSC ring buffer for telemetry transport (never blocks audio thread).
"""

import math

import torch

from base import Node, BLOCK_SIZE, SAMPLE_RATE, CHANNELS, DTYPE, TelemetryDictRingBuffer

try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, QTimer, QRectF
    from PySide6.QtGui import QPainter, QColor, QPen, QFont

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


# ==============================================================================
# 1. DSP Node Logic Class (Zero-Allocation on Audio Thread)
# ==============================================================================
class DataDisplayNode(Node):
    category = "Visual"
    label = "Data Display"

    # Analyze every N blocks: ~23 Hz at 48 kHz / 512 samples per block.
    UPDATE_INTERVAL_BLOCKS = 4
    # Peak below this linear amplitude (-120 dBFS) classifies as silence.
    SILENCE_FLOOR = 1e-6
    # max - min below this (and not silent) classifies as a constant value.
    CONSTANT_EPSILON = 1e-6

    def __init__(self, name=""):
        super().__init__(name)
        self.inp = self.add_input("in")
        self.out = self.add_output("out", channels=CHANNELS)

        # Pre-allocated SPSC telemetry buffer for dict statistics
        self.monitor_queue = TelemetryDictRingBuffer(capacity=4)

        # Scratch for the RMS reduction (avoids sig.pow(2) temp allocation)
        self._squared = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)

        self._block_counter = 0

        # Telemetry is mutated in place; never rebind this dict.
        self._cached_telemetry = {
            "type": "None",
            "shape": "N/A",
            "dtype": "N/A",
            "is_constant": False,
            "constant_val": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "peak": 0.0,
            "peak_db": -120.0,
            "rms": 0.0,
            "rms_db": -120.0,
            "crest_factor": 1.0,
            "crest_db": 0.0,
        }

    def start(self):
        # Transport restart must reset analysis state: no stale counter phase
        # or leftover stats from the previous session.
        self._block_counter = 0
        t = self._cached_telemetry
        t["type"] = "None"
        t["shape"] = "N/A"
        t["dtype"] = "N/A"
        t["is_constant"] = False
        t["constant_val"] = 0.0
        t["min"] = 0.0
        t["max"] = 0.0
        t["mean"] = 0.0
        t["std"] = 0.0
        t["peak"] = 0.0
        t["peak_db"] = -120.0
        t["rms"] = 0.0
        t["rms_db"] = -120.0
        t["crest_factor"] = 1.0
        t["crest_db"] = 0.0

    def process(self):
        sig = self.inp.get_tensor()

        # 1. Pass-through audio untouched. copy_ broadcasts a mono (1, B)
        # input into both output channels (MonoToStereo convention).
        self.out.buffer.copy_(sig)

        # 2. Rate-limited Analysis
        self._block_counter += 1
        if self._block_counter < self.UPDATE_INTERVAL_BLOCKS:
            return
        self._block_counter = 0

        min_val = float(torch.min(sig).item())
        max_val = float(torch.max(sig).item())
        mean_val = float(torch.mean(sig).item())
        std_val = float(torch.std(sig).item())

        abs_max = max(abs(min_val), abs(max_val))
        # copy_ broadcasts a mono (1, B) input into the fixed (CHANNELS, B)
        # scratch without resizing it (an out= reduction would shrink it).
        self._squared.copy_(sig)
        self._squared.pow_(2)
        rms_val = math.sqrt(max(0.0, float(torch.mean(self._squared).item())))

        # Classification order: silence first — an all-zero buffer satisfies
        # the constant test trivially but must NOT read as "Constant".
        if abs_max < self.SILENCE_FLOOR:
            data_type = "Silent"
            is_const = False
            const_val = 0.0
        elif (max_val - min_val) < self.CONSTANT_EPSILON:
            data_type = "Constant"
            is_const = True
            const_val = mean_val
        else:
            data_type = "Tensor"
            is_const = False
            const_val = 0.0

        peak_db = 20.0 * math.log10(max(1e-9, abs_max))
        rms_db = 20.0 * math.log10(max(1e-9, rms_val))
        crest = abs_max / (rms_val + 1e-9)
        crest_db = 20.0 * math.log10(max(1.0, crest))

        # In-place mutation: readers via get_telemetry() keep a valid snapshot
        t = self._cached_telemetry
        t["type"] = data_type
        t["shape"] = f"{tuple(sig.shape)}"
        t["dtype"] = str(sig.dtype).replace("torch.", "")
        t["is_constant"] = is_const
        t["constant_val"] = const_val
        t["min"] = min_val
        t["max"] = max_val
        t["mean"] = mean_val
        t["std"] = std_val
        t["peak"] = abs_max
        t["peak_db"] = peak_db
        t["rms"] = rms_val
        t["rms_db"] = rms_db
        t["crest_factor"] = crest
        t["crest_db"] = crest_db

        # Dispatch to UI via lock-free SPSC ring buffer
        self.monitor_queue.push(t)

    def get_telemetry(self) -> dict:
        return self._cached_telemetry


# ==============================================================================
# 2. UI Widget: Technical Monospace HUD
# ==============================================================================
if GUI_AVAILABLE:

    class DataDisplayWidget(QWidget):
        IS_NODE_UI = True
        NODE_CLASS_NAME = "DataDisplayNode"

        def __init__(self, proxy):
            super().__init__()
            self.proxy = proxy
            self.setMinimumSize(330, 150)

            # Style Constants
            self.bg_color = QColor(20, 20, 20)
            self.border_color = QColor(45, 45, 45)
            self.text_primary = QColor(230, 230, 230)
            self.text_secondary = QColor(140, 140, 140)
            self.cyan_color = QColor("#00ccff")
            self.green_color = QColor("#00ff66")
            self.orange_color = QColor("#ff9900")
            self.red_color = QColor("#ff4444")

            self.font_badge = QFont("Monospace", 8, QFont.Bold)
            self.font_data = QFont("Monospace", 8)
            self.font_data_bold = QFont("Monospace", 8, QFont.Bold)

            self._data = None

            # Polling Timer (~30 FPS)
            self.timer = QTimer(self)
            self.timer.setInterval(33)
            self.timer.timeout.connect(self.poll_queue)
            self.timer.start()

        def poll_queue(self):
            telemetry = getattr(self.proxy, "monitor_queue", None)
            if not telemetry:
                return

            latest = telemetry.pop_latest()
            if latest:
                self._data = latest
                self.update()

        def on_telemetry(self, data: dict):
            if data:
                self._data = data
                self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            w, h = self.width(), self.height()
            rect = self.rect()

            # Main Background
            painter.fillRect(rect, self.bg_color)
            painter.setPen(QPen(self.border_color, 1))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

            if not self._data or self._data.get("type") == "None":
                painter.setFont(self.font_data)
                painter.setPen(self.text_secondary)
                painter.drawText(rect, Qt.AlignCenter, "No Signal / Disconnected")
                return

            d = self._data
            margin = 8
            y_cursor = margin + 14

            # --- 1. Top Type Badge & Header ---
            badge_color = self.cyan_color
            if d["type"] == "Silent":
                badge_text = " SILENT "
                badge_color = self.text_secondary
            elif d["type"] == "Constant":
                badge_text = " CONSTANT FLOAT "
                badge_color = self.orange_color
            else:
                badge_text = f" TENSOR: {d['shape']} {d['dtype']} "

            painter.setFont(self.font_badge)
            fm_badge = painter.fontMetrics()
            badge_w = fm_badge.horizontalAdvance(badge_text) + 8
            badge_rect = QRectF(w - badge_w - margin, margin, badge_w, 18)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(badge_color.red(), badge_color.green(), badge_color.blue(), 35))
            painter.drawRoundedRect(badge_rect, 3, 3)

            painter.setPen(badge_color)
            painter.drawText(badge_rect, Qt.AlignCenter, badge_text)

            painter.setFont(self.font_data_bold)
            painter.setPen(self.text_secondary)
            painter.drawText(margin, y_cursor, "SIGNAL INSPECTOR")

            # Divider line
            y_cursor += 10
            painter.setPen(QPen(self.border_color, 1))
            painter.drawLine(margin, y_cursor, w - margin, y_cursor)
            y_cursor += 14

            # --- 2. Constant / Scalar Display Mode ---
            if d["type"] == "Constant":
                val = d.get("constant_val", 0.0)
                val_rect = QRectF(margin, y_cursor, w - 2 * margin, h - y_cursor - margin)

                painter.setFont(QFont("Monospace", 14, QFont.Bold))
                painter.setPen(self.text_primary)
                painter.drawText(val_rect, Qt.AlignCenter, f"{val:+.6f}")
                return

            # --- 3. Dynamic Multi-Channel Tensor Stats Grid ---
            col1_x = margin + 2
            col2_x = (w / 2.0) + 4
            row_h = 17

            # Formatting helper
            def draw_stat(x, y, label, val_str, val_color=self.text_primary):
                painter.setFont(self.font_data)
                painter.setPen(self.text_secondary)
                painter.drawText(int(x), int(y), label)

                label_w = painter.fontMetrics().horizontalAdvance(label)
                painter.setFont(self.font_data_bold)
                painter.setPen(val_color)
                painter.drawText(int(x + label_w + 4), int(y), val_str)

            # Peak color warning logic
            peak_db = d.get("peak_db", -120.0)
            peak_color = (
                self.red_color if peak_db >= -0.1
                else self.orange_color if peak_db >= -6.0
                else self.green_color
            )

            # Row 1: Peak & Min
            draw_stat(col1_x, y_cursor, "Peak:", f"{peak_db:+.2f} dBFS ({d['peak']:.3f})", peak_color)
            draw_stat(col2_x, y_cursor, "Min:", f"{d['min']:+.4f}")
            y_cursor += row_h

            # Row 2: RMS & Max
            draw_stat(col1_x, y_cursor, "RMS: ", f"{d['rms_db']:+.2f} dBFS ({d['rms']:.3f})", self.cyan_color)
            draw_stat(col2_x, y_cursor, "Max:", f"{d['max']:+.4f}")
            y_cursor += row_h

            # Row 3: Crest & Mean (DC)
            draw_stat(col1_x, y_cursor, "Crest:", f"{d['crest_db']:+.1f} dB ({d['crest_factor']:.2f}x)")
            draw_stat(col2_x, y_cursor, "Mean:", f"{d['mean']:+.4f} (DC)")
            y_cursor += row_h

            # Row 4: Std Dev & Dynamic Headroom
            headroom = max(0.0, -peak_db)
            draw_stat(col1_x, y_cursor, "Headroom:", f"{headroom:.1f} dB")
            draw_stat(col2_x, y_cursor, "Std Dev:", f"{d['std']:.4f}")
