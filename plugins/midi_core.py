"""
MIDI-to-CV conversion and merging nodes.

Real-time notes:
- All steady-state buffers are pre-allocated in ``__init__``; the processing
  path performs zero heap allocation (except the packet message lists, which
  are cleared/reused rather than re-created).
- MIDI input slots expose ``get_packet()`` which aggregates a ``MIDIPacket``
  (list of ``(sample_offset, mido.Message)``) from all connected MIDI outputs.
- MIDI output slots must clear their ``packet.messages`` at the top of
  ``process()`` to uphold the anti-ghosting contract (no stale messages across
  block boundaries).
"""

import math
import torch
from base import Node, BLOCK_SIZE, DTYPE, SAMPLE_RATE


def midi_to_hz(note_number: int) -> float:
    """Exact equal-temperament frequency: f = 440 * 2^((n - 69) / 12)."""
    return 440.0 * (2.0 ** ((float(note_number) - 69.0) / 12.0))


def is_note_on(msg) -> bool:
    return getattr(msg, "type", "") == "note_on" and getattr(msg, "velocity", 0) > 0


def is_note_off(msg) -> bool:
    return getattr(msg, "type", "") == "note_off" or (
        getattr(msg, "type", "") == "note_on" and getattr(msg, "velocity", 0) == 0
    )


class MIDINoteToCV(Node):
    category = "MIDI"
    label = "MIDI Note to CV"
    description = (
        "Converts MIDI Note-On / Note-Off messages into monophonic CV tensors "
        "(pitch in Hz, binary gate, and linear velocity) with last-note priority. "
        "Outputs are (1, BLOCK_SIZE) float32 tensors ready for direct modulation. "
        "Note events are evaluated per audio block: sub-block sample offsets are "
        "coalesced at block boundaries, so gate/velocity transitions are "
        "block-quantized (up to ~10.7 ms at 48 kHz)."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.midi_in = self.add_midi_input("midi_in", help="Incoming MIDI stream.")
        self.pitch_out = self.add_output("pitch_out", channels=1, help="Pitch CV in Hz (f = 440 * 2^((n-69)/12)).")
        self.gate_out = self.add_output("gate_out", channels=1, help="Gate CV: 1.0 when a note is active, 0.0 when released.")
        self.velocity_out = self.add_output("velocity_out", channels=1, help="Velocity CV: linear 0.0 to 1.0.")

        self.add_float_param("glide_ms", 0.0, 0.0, 500.0, unit="ms",
                             help="Portamento glide time between pitches.")

        self._note_stack = []        # Priority stack of (note_number, velocity_float)
        self._last_pitch_hz = 440.0  # Seed value for block glide continuity
        self._target_pitch_hz = 440.0
        self._current_gate = 0.0
        self._current_velocity = 0.0

        # Pre-allocated CV buffers
        self._pitch_buf = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)
        self._gate_buf = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)
        self._vel_buf = torch.zeros((1, BLOCK_SIZE), dtype=DTYPE)

    def start(self):
        self._note_stack.clear()
        self._current_gate = 0.0
        self._current_velocity = 0.0

    def process(self):
        packet = self.midi_in.get_packet()

        # Update note stack
        for offset, msg in packet.messages:
            if is_note_on(msg):
                self._note_stack = [item for item in self._note_stack if item[0] != msg.note]
                self._note_stack.append((msg.note, msg.velocity / 127.0))
            elif is_note_off(msg):
                self._note_stack = [item for item in self._note_stack if item[0] != msg.note]

        # Last-note priority evaluation
        if self._note_stack:
            active_note, active_vel = self._note_stack[-1]
            self._target_pitch_hz = midi_to_hz(active_note)
            self._current_gate = 1.0
            self._current_velocity = active_vel
        else:
            self._current_gate = 0.0

        # In-place write for Gate and Velocity
        self._gate_buf.fill_(self._current_gate)
        self._vel_buf.fill_(self._current_velocity)

        # Multi-block continuous portamento glide
        glide_ms = self.params["glide_ms"].value
        if glide_ms <= 1.0 or abs(self._last_pitch_hz - self._target_pitch_hz) < 1e-3:
            self._pitch_buf.fill_(self._target_pitch_hz)
            self._last_pitch_hz = self._target_pitch_hz
        else:
            block_dur = BLOCK_SIZE / SAMPLE_RATE
            alpha = 1.0 - math.exp(-block_dur / (glide_ms / 1000.0))
            end_pitch = self._last_pitch_hz + alpha * (self._target_pitch_hz - self._last_pitch_hz)
            torch.linspace(self._last_pitch_hz, end_pitch, BLOCK_SIZE, out=self._pitch_buf[0])
            self._last_pitch_hz = end_pitch

        self.pitch_out.buffer.copy_(self._pitch_buf)
        self.gate_out.buffer.copy_(self._gate_buf)

        self.velocity_out.buffer.copy_(self._vel_buf)
class MIDIControlChange(Node):
    category = "MIDI"
    label = "MIDI CC to CV"
    description = "Extracts a specific MIDI CC number and converts it to a normalized 0.0 to 1.0 CV tensor."

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.midi_in = self.add_midi_input("midi_in", help="Incoming MIDI stream.")
        self.cv_out = self.add_output("cv_out", channels=1, help="Normalized CC CV tensor in [0.0, 1.0].")

        self.add_int_param("cc_number", 1, 0, 127, help="MIDI CC index to extract (e.g. 1 = Mod Wheel).")
        self.add_float_param("default_val", 0.0, 0.0, 1.0, help="Value output prior to receiving CC messages.")
        self._current_val = 0.0

    def start(self):
        self._current_val = self.params["default_val"].value

    def process(self):
        target_cc = self.params["cc_number"].value
        packet = self.midi_in.get_packet()
        for offset, msg in packet.messages:
            if getattr(msg, "type", "") == "control_change" and getattr(msg, "control", -1) == target_cc:
                self._current_val = float(msg.value) / 127.0
        self.cv_out.buffer[0].fill_(self._current_val)


class MIDIPitchBend(Node):
    category = "MIDI"
    label = "MIDI Pitch Bend"
    description = "Converts 14-bit pitch wheel messages (-8192..+8191) to a bipolar [-1.0, +1.0] CV tensor."

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.midi_in = self.add_midi_input("midi_in", help="Incoming MIDI stream.")
        self.cv_out = self.add_output("cv_out", channels=1, help="Bipolar pitch bend CV in [-1.0, +1.0].")
        self._bend_val = 0.0

    def start(self):
        self._bend_val = 0.0

    def process(self):
        packet = self.midi_in.get_packet()
        for offset, msg in packet.messages:
            if getattr(msg, "type", "") == "pitchwheel":
                # mido signed pitch range: -8192 .. +8191, center 0
                pitch = getattr(msg, "pitch", 0)
                self._bend_val = max(-1.0, min(1.0, float(pitch) / 8192.0))
        self.cv_out.buffer[0].fill_(self._bend_val)


class MIDIMerge(Node):
    category = "MIDI"
    label = "MIDI Merge"
    description = "Combines multiple MIDI input streams and sorts merged messages chronologically by sample offset."

    def __init__(self, name=""):
        super().__init__(name)
        self.error_msg = None
        self.in_a = self.add_midi_input("in_a", help="First MIDI stream.")
        self.in_b = self.add_midi_input("in_b", help="Second MIDI stream.")
        self.out = self.add_midi_output("out", help="Merged and sorted MIDI stream.")

    def process(self):
        # Enforce output packet clearing contract
        self.out.packet.messages.clear()

        pkt_a = self.in_a.get_packet()
        pkt_b = self.in_b.get_packet()

        if pkt_a.messages:
            self.out.packet.messages.extend(pkt_a.messages)
        if pkt_b.messages:
            self.out.packet.messages.extend(pkt_b.messages)

        if len(self.out.packet.messages) > 1:
            self.out.packet.messages.sort(key=lambda item: item[0])