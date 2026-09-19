"""
ABCPlayer — score-to-MIDI auditioning player (MIDI).

Parses an ABC score (file param or wired URI, e.g. from
SheetSage2Transcriber) on an NRT worker and renders its melody as MIDI
note on/off packets in real time, so a transcribed or cover score can be
auditioned through any synth before spending GPU time on a YuE2 render.

Architecture (mirrors SamplePlayer):
- Score loading + parsing run on NRTExecutor (submit_nrt /
  on_nrt_complete); the audio thread never touches disk.
- Playback is block-granular event scheduling: a beat cursor advances by
  the score tempo every block and due note on/offs are emitted with
  sample offsets. Trigger detection is block-granular by design (same as
  SamplePlayer): one rising edge restarts once per block.
- Chord symbols ("Bm") are harmony labels, not clusters: only melody
  notes (plus [CEG] clusters) sound. Staves merge sequentially in file
  order (see abc_score).
- Loading a score does NOT auto-play; playback starts on the first
  rising edge. A wired uri_in path overrides score_file: trigger edges
  load changed paths in the background and restart them once ready.
"""

from base import Node, BLOCK_SIZE, SAMPLE_RATE

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

try:
    import mido
    _MIDO_AVAILABLE = True
except ImportError:
    mido = None
    _MIDO_AVAILABLE = False


class ABCPlayerWidget(QWidget):
    """Custom UI: param rows (a custom widget replaces the generic panel,
    so every param must be embedded explicitly) plus a live status line
    and progress bar fed by on_telemetry."""

    IS_NODE_UI = True
    NODE_CLASS_NAME = "ABCPlayer"

    PARAM_KEYS = ("score_file", "channel", "velocity", "tempo_scale", "loop")

    def __init__(self, node_proxy):
        super().__init__()
        self.proxy = node_proxy
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self.param_widgets = {}
        for key in self.PARAM_KEYS:
            widget = self.proxy.create_param_widget(key)
            self.param_widgets[key] = widget
            layout.addWidget(widget)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar)

    def on_telemetry(self, data: dict):
        status = str(data.get("status", ""))
        section = str(data.get("section", "") or "")
        self.lbl_status.setText(
            status + (f" · {section}" if section else ""))
        try:
            pos = float(data.get("pos", 0.0) or 0.0)
            total = float(data.get("total", 0.0) or 0.0)
        except (TypeError, ValueError):
            pos, total = 0.0, 0.0
        if total > 0:
            self.bar.setValue(max(0, min(1000, int(1000.0 * pos / total))))
            self.bar.setFormat(f"beat {pos:.0f}/{total:.0f}")
        else:
            self.bar.setValue(0)
            self.bar.setFormat("no score")

    def update_from_params(self, params):
        for key, widget in self.param_widgets.items():
            if key in params:
                widget.update_from_backend(params[key])


class ABCPlayer(Node):
    category = "MIDI"
    label = "ABC Score Player"
    description = (
        "Plays an ABC score as MIDI (melody notes + clusters; chord symbols "
        "are labels and stay silent). Scores load and parse on a background "
        "NRT worker; playback schedules note on/off packets per block at the "
        "score tempo. Starts on the first rising edge of the trigger input. "
        "A wired uri_in path overrides score_file: trigger edges load changed "
        "paths in the background and restart them once ready. Requires the "
        "'mido' package for MIDI output."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge above 0 restarts playback from the start.")
        self.add_uri_input("uri_in",
                           help="Wired score path (e.g. from SheetSage2Transcriber). While connected and "
                                "non-empty it overrides score_file: a trigger edge loads a changed "
                                "path on a background worker and restarts it once ready.")
        self.midi_out = self.add_midi_output("midi_out",
                                             help="MIDI note on/off stream for the score melody.")

        self.add_file_param("score_file", "", filter="ABC Files (*.abc);;All Files (*.*)",
                            help="ABC score file to play; loading/parsing happens on a background worker.")
        self.add_int_param("channel", 0, 0, 15,
                           help="MIDI channel for emitted notes.")
        self.add_int_param("velocity", 96, 1, 127,
                           help="MIDI velocity for emitted note-ons.")
        self.add_float_param("tempo_scale", 1.0, 0.25, 4.0, unit="x",
                             help="Playback tempo multiplier (1.0 = score tempo).")
        self.add_bool_param("loop", False,
                            help="Loop the score continuously instead of stopping at the end.")

        self._score = None  # abc_score.Score, set off-thread
        self._beat_pos = 0.0
        self._note_idx = 0  # next unplayed note in _score.notes
        self._active = []  # [(midi, off_beat)] hanging note-offs
        self._is_playing = False
        self._last_trig = 0.0
        self._current_path = ""
        self._loaded_source = None
        self._submitted_source = None
        self._pending_restart = False  # set by uri-triggered loads only
        self._status = "Idle"
        self._status_detail = "No score loaded"

    def start(self):
        self._beat_pos = 0.0
        self._note_idx = 0
        self._active = []
        self._is_playing = False
        self._last_trig = 0.0
        self.midi_out.clear_packet()

    # ------------------------------------------------------------------
    # NRT load path
    # ------------------------------------------------------------------
    def on_ui_param_change(self, param_name):
        if param_name != "score_file":
            return
        path = self.params["score_file"].get_staging_safe()
        if path and path != self._current_path:
            self._current_path = path
            self._pending_restart = False  # param loads wait for a trigger
            self._submitted_source = path
            self.submit_nrt(self._load_score_nrt, path, tag="load")

    def load_state(self, data):
        super().load_state(data)
        if "score_file" in self.params:
            path = self.params["score_file"].value
            if path:
                self._current_path = path
                self._pending_restart = False  # param loads wait for a trigger
                self._submitted_source = path
                self.submit_nrt(self._load_score_nrt, path, tag="load")

    @staticmethod
    def _load_score_nrt(path):
        from abc_score import load_score
        return load_score(path)  # ValueError when noteless, OSError when unreadable

    def on_nrt_complete(self, tag, ok, result):
        if tag != "load":
            return
        if ok:
            self._score = result
            self._loaded_source = self._submitted_source
            self.error_msg = None
            total = result.total_seconds
            self._status = "Ready"
            self._status_detail = (f"{len(result.notes)} notes, "
                                   f"{total:.1f} s @ {result.bpm:.0f} BPM")
            if self._pending_restart:
                # Trigger-initiated URI load: start at once instead of
                # waiting for a second edge that will never come.
                self._pending_restart = False
                self._restart()
            else:
                self._beat_pos = 0.0  # wait for a trigger
                self._is_playing = False
        else:
            self._pending_restart = False
            self._score = None
            self._status = "Error"
            self._status_detail = f"Score load failed: {result}"
            self.error_msg = f"Score load failed: {result}"

    # ------------------------------------------------------------------
    # Audio thread: block-granular MIDI scheduling (no file I/O here)
    # ------------------------------------------------------------------
    def _restart(self):
        self._beat_pos = 0.0
        self._note_idx = 0
        self._active = []
        self._is_playing = True

    def _all_off(self):
        """Emit note-offs for hanging notes at end of block and forget them."""
        if not _MIDO_AVAILABLE:
            self._active = []
            return
        channel = int(self.params["channel"].value)
        offset = BLOCK_SIZE - 1
        for midi_note, _off in self._active:
            self.midi_out.packet.messages.append(
                (offset, mido.Message("note_off", note=midi_note, velocity=0,
                                 channel=channel)))
        self._active = []
        self.midi_out.packet.messages.sort(key=lambda m: m[0])

    def _on_trigger_edge(self):
        """Rising gate edge. With a wired, non-empty URI this loads a
        changed path in the background (restarting on completion) and
        restarts an already-loaded one; otherwise legacy restart. File I/O
        itself stays on the NRT worker — only the submit happens here, and
        submit_nrt never blocks."""
        uri_in = self.inputs.get("uri_in")
        uri = ""
        if uri_in is not None and uri_in.connected_outputs:
            uri = uri_in.get_uri()
        if uri:
            if uri != self._loaded_source or self._score is None:
                self._current_path = uri
                self._pending_restart = True
                self._submitted_source = uri
                self.submit_nrt(self._load_score_nrt, uri, tag="load")
            else:
                self._all_off()
                self._restart()
        elif uri_in is None or not uri_in.connected_outputs:
            # Unwired: legacy restart of the param-loaded score.
            self._all_off()
            self._restart()
        # Connected-but-empty URI (upstream not ready yet): ignore the edge.

    def _section_at(self, beat):
        name = ""
        if self._score is not None:
            for section in self._score.sections:
                if section.start <= beat:
                    name = section.name
                else:
                    break
        return name

    def process(self):
        self.midi_out.clear_packet()  # anti-ghost: stale MIDI must never replay
        trig = self.inputs["trigger_in"].get_tensor()[0]

        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            self._on_trigger_edge()
        self._last_trig = float(trig[-1].item())

        score = self._score
        if score is None or not self._is_playing:
            return
        if not _MIDO_AVAILABLE:
            if self.error_msg is None:
                self.error_msg = "'mido' package is required for MIDI output"
            return
        if score.total_beats <= 0:
            self._is_playing = False
            return

        channel = int(self.params["channel"].value)
        velocity = int(self.params["velocity"].value)
        tempo_scale = float(self.params["tempo_scale"].value)
        beats_per_block = (BLOCK_SIZE / SAMPLE_RATE
                           * (score.bpm * tempo_scale / 60.0))
        if beats_per_block <= 0:
            return
        pos, end = self._beat_pos, self._beat_pos + beats_per_block
        notes = score.notes

        def _offset(beat):
            return max(0, min(BLOCK_SIZE - 1,
                              int((beat - pos) / beats_per_block * BLOCK_SIZE)))

        # Hanging note-offs due this block first (standard offs-before-ons).
        still_active = []
        for midi_note, off_beat in self._active:
            if off_beat < end:
                self.midi_out.packet.messages.append(
                    (_offset(max(off_beat, pos)),
                     mido.Message("note_off", note=midi_note, velocity=0,
                                  channel=channel)))
            else:
                still_active.append((midi_note, off_beat))
        self._active = still_active

        # Note-ons (and immediately-due offs) in this block's window.
        while self._note_idx < len(notes) and notes[self._note_idx].start < end:
            note = notes[self._note_idx]
            self._note_idx += 1
            if note.start < pos:
                continue
            self.midi_out.packet.messages.append(
                (_offset(note.start),
                 mido.Message("note_on", note=note.midi, velocity=velocity,
                              channel=channel)))
            off_beat = note.start + max(note.dur, 0.0)
            if off_beat < end:
                self.midi_out.packet.messages.append(
                    (_offset(off_beat),
                     mido.Message("note_off", note=note.midi, velocity=0,
                                  channel=channel)))
            else:
                self._active.append((note.midi, off_beat))

        self._beat_pos = end
        if end >= score.total_beats:
            if self.params["loop"].value and score.total_beats > 0:
                self._all_off()
                # Rewind to exactly 0 (not end - total): a modulo overshoot
                # would push pos past the downbeat and the start < pos guard
                # would swallow it. Cost is < 1 block of loop timing slack.
                self._beat_pos = 0.0
                self._note_idx = 0
            else:
                self._all_off()
                self._is_playing = False

    def get_telemetry(self) -> dict:
        detail = self._status_detail
        pos, total, section = 0.0, 0.0, ""
        if self._score is not None:
            total = self._score.total_beats
        if self._is_playing and self._score is not None:
            pos = self._beat_pos
            section = self._section_at(pos)
            detail = (f"beat {pos:.0f}/{total:.0f}"
                      + (f" ({section})" if section else ""))
        status = "Playing" if self._is_playing else self._status
        return {"status": status, "audio": detail, "pos": pos,
                "total": total, "section": section}
