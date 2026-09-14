"""Per-node enabled/bypass switch (AGENTS.md §2, §4, §5).

When ``enabled == False`` the engine skips ``process()`` and runs
``apply_bypass()`` instead: audio inputs pass through with no DSP cost,
sources go silent, sinks consume nothing, MIDI is dropped.
"""
import json

import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, apply_bypass
from core import Engine, Graph


def _make_graph(*nodes):
    g = Graph()
    for n in nodes:
        g.add_node(n)
    return g


def test_enabled_param_defaults_true():
    plugin_system.load_plugins("plugins")
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    assert "enabled" in gain.params
    assert gain.params["enabled"].value is True
    assert gain.is_enabled() is True


def test_effect_bypass_passes_audio_through_bit_exact():
    """A disabled effect copies its input dry, ignoring its own DSP."""
    plugin_system.load_plugins("plugins")
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    sig = torch.full((CHANNELS, BLOCK_SIZE), 0.25, dtype=torch.float32)
    gain.inp.get_tensor = lambda: sig
    gain.params["vol"].set(0.0)  # would mute if DSP ran
    gain.params["vol"].sync()
    gain.params["enabled"].set(False)
    gain.params["enabled"].sync()

    apply_bypass(gain)
    assert torch.equal(gain.out.buffer, sig)
    assert gain.out.buffer.shape == (CHANNELS, BLOCK_SIZE)


def test_bypass_ignores_modulation_inputs():
    """Gain's param-bound 'mod' input must not leak into the dry copy."""
    plugin_system.load_plugins("plugins")
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    dry = torch.full((CHANNELS, BLOCK_SIZE), 0.5, dtype=torch.float32)
    mod = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    gain.inp.get_tensor = lambda: dry
    gain.gain_mod.get_tensor = lambda: mod
    gain.params["enabled"].set(False)
    gain.params["enabled"].sync()

    apply_bypass(gain)
    assert torch.equal(gain.out.buffer, dry)


def test_bypass_mono_input_keeps_stereo_shape():
    """Mono source into a stereo output must broadcast, never shrink the
    pre-allocated buffer (PyTorch out= shrinkage regression guard)."""
    plugin_system.load_plugins("plugins")
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    mono = torch.full((1, BLOCK_SIZE), 0.7, dtype=torch.float32)
    gain.inp.get_tensor = lambda: mono

    apply_bypass(gain)
    assert gain.out.buffer.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.equal(gain.out.buffer[0], mono[0])
    assert torch.equal(gain.out.buffer[1], mono[0])


def test_bypass_generator_outputs_silence():
    """Sources have no true-audio input: bypass == silence (never a copy of
    a modulation CV)."""
    plugin_system.load_plugins("plugins")
    sine = plugin_system.NODE_REGISTRY["SineOscillator"]()
    sine.out_sig.buffer.fill_(0.9)  # stale audio must not survive
    apply_bypass(sine)
    assert sine.out_sig.buffer.shape[0] == 1
    assert torch.equal(sine.out_sig.buffer, torch.zeros_like(sine.out_sig.buffer))


def test_bypass_zeros_extra_outputs_and_clears_midi():
    """SignalAnalyzer: main 'out' passes through, CV outs go silent (a dry
    audio copy must never fire a downstream gate detector)."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["SignalAnalyzer"]()
    sig = torch.full((CHANNELS, BLOCK_SIZE), 0.3, dtype=torch.float32)
    node.inp.get_tensor = lambda: sig
    for name, slot in node.outputs.items():
        if getattr(slot, "slot_type", "audio") == "audio":
            slot.buffer.fill_(0.9)
    apply_bypass(node)
    assert torch.equal(node.outputs["out"].buffer, sig)
    for name, slot in node.outputs.items():
        if name != "out" and getattr(slot, "slot_type", "audio") == "audio":
            assert torch.equal(slot.buffer, torch.zeros_like(slot.buffer)), name


def test_bypass_cv_only_node_outputs_silence():
    """EnvelopeFollower has no 'out'/'signal' port: bypass == silence."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["EnvelopeFollower"]()
    sig = torch.full((CHANNELS, BLOCK_SIZE), 0.4, dtype=torch.float32)
    node.inputs["in"].get_tensor = lambda: sig
    for slot in node.outputs.values():
        if getattr(slot, "slot_type", "audio") == "audio":
            slot.buffer.fill_(0.9)
    apply_bypass(node)
    for name, slot in node.outputs.items():
        if getattr(slot, "slot_type", "audio") == "audio":
            assert torch.equal(slot.buffer, torch.zeros_like(slot.buffer)), name


def test_bypass_splitter_without_main_out_goes_silent():
    """ChannelSplitter (left/right, no 'out') cannot be generically
    bypassed: silence is the safe documented fallback."""
    plugin_system.load_plugins("plugins")
    node = plugin_system.NODE_REGISTRY["ChannelSplitter"]()
    sig = torch.full((CHANNELS, BLOCK_SIZE), 0.4, dtype=torch.float32)
    node.inp.get_tensor = lambda: sig
    apply_bypass(node)
    for name, slot in node.outputs.items():
        assert torch.equal(slot.buffer, torch.zeros_like(slot.buffer)), name


def test_bypass_drops_midi_never_forwards():
    """MIDIMerge bypass drops messages instead of forwarding them."""
    plugin_system.load_plugins("plugins")
    from test_midi import FakeMidoMessage, _make_midi_out

    merge = plugin_system.NODE_REGISTRY["MIDIMerge"]()
    merge.in_a.connected_outputs = [_make_midi_out((10, FakeMidoMessage("note_on", 60, 100)))]
    merge.in_b.connected_outputs = [_make_midi_out((20, FakeMidoMessage("note_on", 64, 100)))]
    apply_bypass(merge)
    assert merge.out.packet.messages == []


def test_bypass_sink_does_nothing_but_not_crash():
    """FileRecorder (audio in, no audio out): bypass skips recording."""
    plugin_system.load_plugins("plugins")
    rec = plugin_system.NODE_REGISTRY["FileRecorder"]()
    sig = torch.full((CHANNELS, BLOCK_SIZE), 0.4, dtype=torch.float32)
    rec.inp.get_tensor = lambda: sig
    apply_bypass(rec)  # must not raise; records nothing


def test_engine_skips_process_when_disabled():
    """The engine dispatch must not call process() on disabled nodes."""
    plugin_system.load_plugins("plugins")
    engine = Engine()
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    calls = []
    orig_process = gain.process
    gain.process = lambda: (calls.append(1), orig_process())
    sig = torch.full((CHANNELS, BLOCK_SIZE), 0.2, dtype=torch.float32)
    gain.inp.get_tensor = lambda: sig
    gain.params["vol"].set(0.0)
    gain.params["vol"].sync()
    gain.params["enabled"].set(False)
    gain.params["enabled"].sync()

    engine._process_plan_node(gain)
    assert calls == []
    assert torch.equal(gain.out.buffer, sig)

    gain.params["enabled"].set(True)
    gain.params["enabled"].sync()
    engine._process_plan_node(gain)
    assert calls == [1]
    assert torch.equal(gain.out.buffer, torch.zeros_like(gain.out.buffer))


def test_engine_treats_paramless_test_doubles_as_enabled():
    """Foreign test doubles without is_enabled() keep running (back-compat)."""
    engine = Engine()

    class _Double:
        def process(self):
            self.ran = True

    d = _Double()
    engine._process_plan_node(d)
    assert d.ran is True


def test_bypass_no_ghosting_after_reenable_cycle():
    """Stale audio from an enabled block must not survive a bypass block."""
    plugin_system.load_plugins("plugins")
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    loud = torch.full((CHANNELS, BLOCK_SIZE), 0.8, dtype=torch.float32)
    silent = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    gain.inp.get_tensor = lambda: loud
    gain.params["vol"].set(1.0)
    gain.params["vol"].sync()
    gain.process()
    assert torch.equal(gain.out.buffer, loud)

    gain.inp.get_tensor = lambda: silent
    apply_bypass(gain)
    assert torch.equal(gain.out.buffer, silent)


def test_enabled_persists_through_save_load_roundtrip():
    plugin_system.load_plugins("plugins")
    g = Graph()
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    gain.id = "g1"
    g.add_node(gain)
    gain.params["enabled"].set(False)
    gain.params["enabled"].sync()

    data = json.loads(g.to_json())
    assert data["nodes"][0]["params"]["enabled"] is False

    fresh = Graph()
    for n_data in data["nodes"]:
        cls = plugin_system.NODE_REGISTRY.get(n_data["type"])
        node = cls(n_data["name"])
        node.id = n_data["id"]
        node.load_state(n_data)
        fresh.add_node(node)
    assert fresh.node_map["g1"].params["enabled"].value is False
    assert fresh.node_map["g1"].is_enabled() is False


def test_load_state_without_enabled_key_stays_enabled():
    """Old patches pre-dating the switch load as enabled (back-compat)."""
    plugin_system.load_plugins("plugins")
    gain = plugin_system.NODE_REGISTRY["Gain"]()
    assert gain.is_enabled() is True
    gain.load_state({"pos": (0, 0), "params": {"vol": 0.5}})
    assert gain.is_enabled() is True
    assert gain.params["vol"].value == pytest.approx(0.5)
