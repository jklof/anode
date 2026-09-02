"""
On-screen MIDI keyboard node.
"""

from base import Node, SPSCRingBuffer, TelemetryDictRingBuffer
try:
    from midi_core import is_note_on, is_note_off
except ImportError:
    from plugins.midi_core import is_note_on, is_note_off

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPen

try:
    import mido
except ImportError:
    mido = None


class PianoWidget(QWidget):
    noteOn = Signal(int, int)   # note, velocity
    noteOff = Signal(int)       # note

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_note = 48  # C3
        self.num_octaves = 2
        self.active_notes = set()
        self._last_note = -1
        self.setMinimumSize(240, 70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_active_notes(self, active_set):
        if self.active_notes != active_set:
            self.active_notes = active_set
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        num_white_keys = self.num_octaves * 7
        kw = w / max(1, num_white_keys)
        bw = kw * 0.6
        bh = h * 0.6

        white_classes = [0, 2, 4, 5, 7, 9, 11]
        black_classes = [1, 3, 6, 8, 10]
        total_keys = self.num_octaves * 12

        # 1. White Keys
        white_idx = 0
        for i in range(total_keys):
            note = self.start_note + i
            if (i % 12) in white_classes:
                rect = QRectF(white_idx * kw, 0, kw, h)
                brush = QColor("#ffaa00") if note in self.active_notes else QColor("#ffffff")
                painter.setBrush(brush)
                painter.setPen(QPen(QColor("#222222"), 1))
                painter.drawRect(rect)
                white_idx += 1

        # 2. Black Keys
        white_idx = 0
        for i in range(total_keys):
            note = self.start_note + i
            p_class = i % 12
            if p_class in white_classes:
                white_idx += 1
            elif p_class in black_classes:
                x = (white_idx * kw) - (bw / 2.0)
                rect = QRectF(x, 0, bw, bh)
                brush = QColor("#ff8800") if note in self.active_notes else QColor("#111111")
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)
                painter.drawRect(rect)

    def mousePressEvent(self, event):
        note = self._note_at_pos(event.position().x(), event.position().y())
        if note >= 0:
            self._last_note = note
            self.noteOn.emit(note, 100)

    def mouseReleaseEvent(self, event):
        if self._last_note >= 0:
            self.noteOff.emit(self._last_note)
            self._last_note = -1

    def _note_at_pos(self, x, y):
        num_white_keys = self.num_octaves * 7
        kw = self.width() / max(1, num_white_keys)
        bw = kw * 0.6
        bh = self.height() * 0.6
        white_classes = [0, 2, 4, 5, 7, 9, 11]
        black_classes = [1, 3, 6, 8, 10]
        total_keys = self.num_octaves * 12

        # Check Black Keys first
        if y <= bh:
            white_idx = 0
            for i in range(total_keys):
                note = self.start_note + i
                p_class = i % 12
                if p_class in white_classes:
                    white_idx += 1
                elif p_class in black_classes:
                    kx = (white_idx * kw) - (bw / 2.0)
                    if kx <= x <= kx + bw:
                        return note

        # Check White Keys
        white_idx = int(x // kw)
        curr_w = 0
        for i in range(total_keys):
            note = self.start_note + i
            if (i % 12) in white_classes:
                if curr_w == white_idx:
                    return note
                curr_w += 1

        # No key matched (e.g. x < 0 or x >= width): report "no note" instead of
        # an implicit None so mousePressEvent's `if note >= 0` cannot crash.
        return -1


class MIDIKeyboardWidget(QWidget):
    IS_NODE_UI = True
    NODE_CLASS_NAME = "MIDIKeyboardNode"

    def __init__(self, proxy):
        super().__init__()
        self.proxy = proxy
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.btn_oct_down = QPushButton("<")
        self.btn_oct_up = QPushButton(">")
        self.lbl_octave = QLabel("Octave: C3")
        self.lbl_octave.setAlignment(Qt.AlignCenter)
        row.addWidget(self.btn_oct_down)
        row.addWidget(self.lbl_octave)
        row.addWidget(self.btn_oct_up)
        layout.addLayout(row)

        self.piano = PianoWidget()
        layout.addWidget(self.piano)

        self.piano.noteOn.connect(self._on_ui_note_on)
        self.piano.noteOff.connect(self._on_ui_note_off)
        self.btn_oct_down.clicked.connect(lambda: self._shift_octave(-1))
        self.btn_oct_up.clicked.connect(lambda: self._shift_octave(1))

    def _shift_octave(self, delta):
        start = self.piano.start_note + (delta * 12)
        start = max(12, min(start, 96))
        self.proxy.set_parameter("start_note", start)

    def _on_ui_note_on(self, note, vel):
        self.proxy.push_custom_event(("note_on", note, vel))

    def _on_ui_note_off(self, note):
        self.proxy.push_custom_event(("note_off", note, 0))

    def update_from_params(self, params):
        if "start_note" in params:
            start = int(params["start_note"])
            self.piano.start_note = start
            self.lbl_octave.setText(f"Octave: C{start // 12 - 1}")
        if "octaves" in params:
            self.piano.num_octaves = int(params["octaves"])
        self.piano.update()

    def on_telemetry(self, data):
        if "active_notes" in data:
            self.piano.set_active_notes(set(data["active_notes"]))


class MIDIKeyboardNode(Node):
    category = "MIDI"
    label = "MIDI Keyboard"
    description = (
        "On-screen virtual keyboard with press/release keys that merges UI performance "
        "with incoming MIDI."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.midi_in = self.add_midi_input("midi_in", help="Pass-through MIDI input.")
        self.midi_out = self.add_midi_output("midi_out", help="Combined MIDI output stream.")

        self.add_int_param("start_note", 48, 12, 96, help="MIDI root note for the first key (48 = C3).")
        self.add_int_param("octaves", 2, 1, 5, help="Number of displayed keyboard octaves.")

        self._ui_queue = SPSCRingBuffer(capacity=128)
        self.monitor_queue = TelemetryDictRingBuffer(capacity=4)
        self._active_notes = set()
        self._notes_dirty = False

    def start(self):
        self._active_notes.clear()
        self._notes_dirty = True

    def process(self):
        # Enforce output packet clearing contract
        self.midi_out.packet.messages.clear()

        # 1. Pass-through incoming messages
        in_pkt = self.midi_in.get_packet()
        if in_pkt.messages:
            self.midi_out.packet.messages.extend(in_pkt.messages)
            for offset, msg in in_pkt.messages:
                if is_note_on(msg):
                    self._active_notes.add(msg.note)
                    self._notes_dirty = True
                elif is_note_off(msg):
                    self._active_notes.discard(msg.note)
                    self._notes_dirty = True

        # 2. Drain UI events
        while True:
            item, ok = self._ui_queue.try_pop()
            if not ok:
                break
            ev_type, note, vel = item
            if mido is not None:
                msg = mido.Message(ev_type, note=note, velocity=vel)
                self.midi_out.packet.messages.append((0, msg))
            if ev_type == "note_on" and vel > 0:
                self._active_notes.add(note)
                self._notes_dirty = True
            else:
                self._active_notes.discard(note)
                self._notes_dirty = True

        # 3. Sort merged messages chronologically
        if len(self.midi_out.packet.messages) > 1:
            self.midi_out.packet.messages.sort(key=lambda x: x[0])

        # 4. Push active note mask to telemetry buffer only on change
        #    (zero allocation in the steady state: no dict/list per block
        #    when nothing changed).
        if self._notes_dirty:
            self.monitor_queue.push({"active_notes": list(self._active_notes)})
            self._notes_dirty = False

    def get_telemetry(self) -> dict:
        latest, ok = self.monitor_queue.try_pop()
        return latest if ok else {}