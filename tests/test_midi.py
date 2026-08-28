import pytest
import torch
import plugin_system
from base import BLOCK_SIZE, MIDIPacket
from core import Graph


class FakeMidoMessage:
    def __init__(self, type, note=60, velocity=64, control=1, value=0, pitch=0):
        self.type = type
        self.note = note
        self.velocity = velocity
        self.control = control
        self.value = value
        self.pitch = pitch  # signed range -8192..+8191, center 0


class FakeMidiOut:
    """A stand-in MIDI output slot (carries a packet, marked slot_type='midi')."""

    def __init__(self, messages=None):
        self.slot_type = "midi"
        self.packet = MIDIPacket(messages=list(messages or []))
        self.help = ""


def _make_midi_out(*messages):
    return FakeMidiOut(messages)


def test_get_node_documentation_midi_nodes():
    """Verify that get_node_documentation() parses MIDI nodes without eager getattr crash."""
    plugin_system.load_plugins("plugins")
    doc = plugin_system.get_node_documentation("MIDINoteToCV")
    assert doc["type"] == "MIDINoteToCV"
    assert "midi_in" in doc["inputs"]
    assert doc["inputs"]["midi_in"]["slot_type"] == "midi"
    assert "pitch_out" in doc["outputs"]
    assert doc["outputs"]["pitch_out"]["slot_type"] == "audio"


def test_midi_packet_lifecycle_anti_ghosting():
    """Verify output packet is cleared at top of process(), preventing cross-block message duplication."""
    plugin_system.load_plugins("plugins")
    merge = plugin_system.NODE_REGISTRY["MIDIMerge"]()
    out_a = _make_midi_out((100, FakeMidoMessage("note_on", 60, 100)))
    merge.in_a.connected_outputs = [out_a]

    # Block 0: Contains 1 note-on message
    merge.process()
    assert len(merge.out.packet.messages) == 1

    # Block 1: Upstream slot cleared. Output packet MUST be completely empty.
    out_a.packet.messages.clear()
    merge.process()
    assert len(merge.out.packet.messages) == 0, "Stale MIDI message leaked across block boundary!"


def test_graph_connect_rejects_audio_to_midi_mismatch():
    """Verify Graph.connect() rejects connections between mismatched slot types."""
    g = Graph()
    plugin_system.load_plugins("plugins")
    sine = plugin_system.NODE_REGISTRY["WaveformOscillator"]()
    note_to_cv = plugin_system.NODE_REGISTRY["MIDINoteToCV"]()
    g.add_node(sine)
    g.add_node(note_to_cv)

    assert g.connect(sine.id, "signal", note_to_cv.id, "midi_in") is False
    assert len(note_to_cv.inputs["midi_in"].connected_outputs) == 0


def test_midi_note_to_cv_accuracy_and_priority_stack():
    """Verify Hz calculation, last-note priority stack, and velocity scaling."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["MIDINoteToCV"]()

    # Note 69 -> A4 (440.0 Hz)
    fake_out = _make_midi_out((0, FakeMidoMessage("note_on", 69, 127)))
    node.midi_in.connected_outputs = [fake_out]
    node.process()

    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(440.0, abs=1e-3)
    assert node.gate_out.buffer[0, 0].item() == 1.0
    assert node.velocity_out.buffer[0, 0].item() == pytest.approx(1.0, abs=1e-3)

    # Legato Note 60 -> C4 (261.6255 Hz)
    fake_out.packet.messages = [(0, FakeMidoMessage("note_on", 60, 64))]
    node.process()
    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(261.6255, abs=0.01)
    assert node.velocity_out.buffer[0, 0].item() == pytest.approx(64 / 127.0, abs=1e-3)

    # Release Note 60 -> Priority falls back to Note 69 (A4, 440 Hz)
    fake_out.packet.messages = [(0, FakeMidoMessage("note_off", 60, 0))]
    node.process()
    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(440.0, abs=1e-3)
    assert node.gate_out.buffer[0, 0].item() == 1.0

    # Release Note 69 -> Gate drops to 0.0, pitch holds 440 Hz
    fake_out.packet.messages = [(0, FakeMidoMessage("note_off", 69, 0))]
    node.process()
    assert node.gate_out.buffer[0, 0].item() == 0.0
    assert node.pitch_out.buffer[0, 0].item() == pytest.approx(440.0, abs=1e-3)


def test_midi_glide_continuity():
    """Verify multi-block portamento trajectory continuity across block seams."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["MIDINoteToCV"]()
    node.params["glide_ms"].set(50.0)
    node.sync()

    # Play Note 60 (261.6255 Hz)
    fake_out = _make_midi_out((0, FakeMidoMessage("note_on", 60, 100)))
    node.midi_in.connected_outputs = [fake_out]
    node.process()

    # Step to Note 72 (523.251 Hz) and process two blocks
    fake_out.packet.messages = [(0, FakeMidoMessage("note_on", 72, 100))]
    node.process()
    block1_tail = node.pitch_out.buffer[0, -1].item()

    fake_out.packet.messages.clear()
    node.process()
    block2_head = node.pitch_out.buffer[0, 0].item()

    # Continuous glide trajectory across the block boundary
    assert block2_head == pytest.approx(block1_tail, abs=1e-2)
    assert block1_tail < block2_head + 1.0 < 523.251


def test_midi_control_change_mapping():
    """Verify CC numeric mapping (0 -> 0.0, 127 -> 1.0) and default fallback."""
    plugin_system.load_plugins("plugins")
    cc_node = plugin_system.NODE_REGISTRY["MIDIControlChange"]()
    cc_node.params["default_val"].set(0.25)
    cc_node.sync()
    cc_node.start()

    # Default value before any message
    fake_out = _make_midi_out()
    cc_node.midi_in.connected_outputs = [fake_out]
    cc_node.process()
    assert cc_node.cv_out.buffer[0, 0].item() == pytest.approx(0.25, abs=1e-3)

    # CC 1 (Mod Wheel) to 127
    fake_out.packet.messages = [(0, FakeMidoMessage("control_change", control=1, value=127))]
    cc_node.process()
    assert cc_node.cv_out.buffer[0, 0].item() == pytest.approx(1.0, abs=1e-3)

    # CC 1 to 0
    fake_out.packet.messages = [(0, FakeMidoMessage("control_change", control=1, value=0))]
    cc_node.process()
    assert cc_node.cv_out.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-3)


def test_midi_pitch_bend_signed_range():
    """Verify mido signed pitch wheel mapping (-8192 -> -1.0, 0 -> 0.0, 8191 -> +1.0)."""
    plugin_system.load_plugins("plugins")
    pb_node = plugin_system.NODE_REGISTRY["MIDIPitchBend"]()

    # Center position (0)
    fake_out = _make_midi_out((0, FakeMidoMessage("pitchwheel", pitch=0)))
    pb_node.midi_in.connected_outputs = [fake_out]
    pb_node.process()
    assert pb_node.cv_out.buffer[0, 0].item() == pytest.approx(0.0, abs=1e-3)

    # Max pitch bend up (+8191)
    fake_out.packet.messages = [(0, FakeMidoMessage("pitchwheel", pitch=8191))]
    pb_node.process()
    assert pb_node.cv_out.buffer[0, 0].item() == pytest.approx(1.0, abs=1e-3)

    # Max pitch bend down (-8192)
    fake_out.packet.messages = [(0, FakeMidoMessage("pitchwheel", pitch=-8192))]
    pb_node.process()
    assert pb_node.cv_out.buffer[0, 0].item() == pytest.approx(-1.0, abs=1e-3)


def test_midi_merge_chronological_sorting():
    """Verify MIDIMerge sorts messages ascending by sample_offset."""
    plugin_system.load_plugins("plugins")
    merge = plugin_system.NODE_REGISTRY["MIDIMerge"]()
    out_a = _make_midi_out((300, FakeMidoMessage("note_on", 64)))
    out_b = _make_midi_out((50, FakeMidoMessage("note_on", 60)))
    merge.in_a.connected_outputs = [out_a]
    merge.in_b.connected_outputs = [out_b]

    merge.process()
    assert len(merge.out.packet.messages) == 2
    assert merge.out.packet.messages[0][0] == 50
    assert merge.out.packet.messages[1][0] == 300


def test_cv_buffer_pointer_stability():
    """Verify that MIDINoteToCV maintains in-place buffer data_ptr stability."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["MIDINoteToCV"]()
    pitch_ptr = node.pitch_out.buffer.data_ptr()
    gate_ptr = node.gate_out.buffer.data_ptr()

    fake_out = _make_midi_out((0, FakeMidoMessage("note_on", 60, 100)))
    node.midi_in.connected_outputs = [fake_out]

    for _ in range(50):
        node.process()
        assert node.pitch_out.buffer.data_ptr() == pitch_ptr
        assert node.gate_out.buffer.data_ptr() == gate_ptr
        assert node.pitch_out.buffer.shape == (1, BLOCK_SIZE)


def test_midi_save_load_json_roundtrip():
    """Verify graph serialization and deserialization preserves MIDI nodes and connections."""
    import json
    plugin_system.load_plugins("plugins")
    g = Graph()

    k = plugin_system.NODE_REGISTRY["MIDIKeyboardNode"]()
    k.id = "k1"
    cv = plugin_system.NODE_REGISTRY["MIDINoteToCV"]()
    cv.id = "cv1"
    g.add_node(k)
    g.add_node(cv)
    assert g.connect(k.id, "midi_out", cv.id, "midi_in") is True

    json_str = g.to_json()
    data = json.loads(json_str)

    fresh_g = Graph()
    for n_data in data["nodes"]:
        cls = plugin_system.NODE_REGISTRY[n_data["type"]]
        node = cls(n_data["name"])
        node.id = n_data["id"]
        fresh_g.add_node(node)
        node.load_state(n_data)

    for c in data["connections"]:
        fresh_g.connect(c["src_id"], c["src_port"], c["dst_id"], c["dst_port"])

    assert len(fresh_g.nodes) == 2
    assert len(fresh_g.node_map["cv1"].inputs["midi_in"].connected_outputs) == 1
    assert fresh_g.node_map["cv1"].inputs["midi_in"].connected_outputs[0].slot_type == "midi"


def test_midi_devices_nrt_epoch_rejection_on_stop():
    """Verify stopping a MIDIInputNode increments epoch and discards in-flight port open requests."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["MIDIInputNode"]()
    node.params["device_name"].set("VirtualPort1")
    node.sync()

    node.start()
    epoch_at_start = node._device_epoch

    node.stop()
    assert node._device_epoch > epoch_at_start

    # Simulate late NRT completion from start()
    fake_port = type("FakePort", (), {"close": lambda self: None})()
    node.on_nrt_complete("open_input", True, (fake_port, "Active: VirtualPort1", epoch_at_start))
    assert node._inport is None
    assert node._status == "Stopped"

def test_engine_startup_reset_skips_midi_slots():
    """The engine's startup cleanup must zero only audio buffers; MIDI outputs
    carry a packet (no tensor buffer) and must not crash the worker thread."""
    from core import Engine
    plugin_system.load_plugins("plugins")

    engine = Engine()
    midi_node = plugin_system.NODE_REGISTRY["MIDIInputNode"]()
    osc = plugin_system.NODE_REGISTRY["WaveformOscillator"]()
    engine.graph.add_node(midi_node)
    engine.graph.add_node(osc)

    # MIDI output must not carry a tensor buffer (that's what tripped engine).
    assert not hasattr(midi_node.msg_out, "buffer")
    assert hasattr(midi_node.msg_out, "packet")

    # Pre-fill an audio buffer, then run the same reset `_worker` performs at
    # startup. Previously this raised AttributeError on the MIDI output slot.
    osc.outputs["signal"].buffer.fill_(0.5)
    engine._reset_audio_buffers()

    assert float(osc.outputs["signal"].buffer[0, 0]) == 0.0
    # MIDI packet untouched (still empty).
    assert midi_node.msg_out.packet.messages == []
