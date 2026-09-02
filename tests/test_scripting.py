import pytest
import torch
import plugin_system
from base import BLOCK_SIZE, CHANNELS
from core import Graph


def test_script_node_compilation_and_ports():
    plugin_system.load_plugins("plugins")

    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")
    assert ScriptClass is not None

    node = ScriptClass()
    # Verify defaults
    assert "audio_in" in node.inputs
    assert "gain" in node.inputs
    assert "audio_out" in node.outputs
    assert node.error_msg is None

    # Update code to define different ports
    new_code = """
inputs = ['left', 'right', 'factor']
outputs = ['stereo_out']
stereo_out = (left + right) * factor
"""
    node.params["code"].set(new_code)
    node.sync()
    node.on_ui_param_change("code")

    # Verify ports were updated dynamically
    assert "left" in node.inputs
    assert "right" in node.inputs
    assert "factor" in node.inputs
    assert "stereo_out" in node.outputs
    assert "audio_in" not in node.inputs


def test_script_node_processing():
    plugin_system.load_plugins("plugins")
    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")

    graph = Graph()
    node = ScriptClass()
    node.id = "script_node"
    graph.add_node(node)

    # Input signals
    audio_in = torch.ones(2, 512)
    gain_in = torch.full((2, 512), 0.5)

    node.inputs["audio_in"].get_tensor = lambda: audio_in
    node.inputs["gain"].get_tensor = lambda: gain_in

    # Execute
    node.process()

    # Output should be (1.0 * 0.5) = 0.5
    expected = torch.full((2, 512), 0.5)
    assert torch.allclose(node.outputs["audio_out"].buffer, expected)
    assert node.error_msg is None


def test_script_node_compilation_error():
    plugin_system.load_plugins("plugins")
    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")
    node = ScriptClass()

    # Intentionally broken syntax
    broken_code = """
inputs = ['audio_in']
outputs = ['audio_out']
if True
    audio_out = audio_in
"""
    node.params["code"].set(broken_code)
    node.sync()
    node.on_ui_param_change("code")

    assert node.compiled_code is None
    assert node.error_msg is not None
    assert node.error_line == 4


def test_script_node_mono_value_broadcast_to_stereo_output():
    """A mono (1, B) value assigned to a stereo script output must broadcast to
    both channels (AGENTS.md §2 channel adaptation), not mute channel 1."""
    plugin_system.load_plugins("plugins")
    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")
    node = ScriptClass()

    code = """
inputs = ['audio_in']
outputs = ['stereo_out']
stereo_out = audio_in
"""
    node.params["code"].set(code)
    node.sync()
    node.on_ui_param_change("code")

    mono = torch.full((1, BLOCK_SIZE), 0.25, dtype=torch.float32)
    node.inputs["audio_in"].get_tensor = lambda: mono
    node.process()

    out = node.outputs["stereo_out"].buffer
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.allclose(out[0], mono[0])
    assert torch.allclose(out[1], mono[0])
    assert node.error_msg is None


def test_script_node_unassigned_output_zero_filled():
    """Script outputs that are never assigned must be zero-filled every block
    (anti-ghosting), while assigned ones keep their value."""
    plugin_system.load_plugins("plugins")
    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")
    node = ScriptClass()

    code = """
inputs = ['a']
outputs = ['used_out', 'unused_out']
used_out = a
"""
    node.params["code"].set(code)
    node.sync()
    node.on_ui_param_change("code")

    sig = torch.full((2, BLOCK_SIZE), 0.5, dtype=torch.float32)
    node.inputs["a"].get_tensor = lambda: sig
    node.process()

    assert torch.allclose(node.outputs["used_out"].buffer, sig)
    assert node.outputs["unused_out"].buffer.abs().max().item() == 0.0


def test_script_partial_frames_zero_fill_trailing_samples():
    """Anti-ghosting (AGENTS.md §2): a script output with fewer frames than
    BLOCK_SIZE must leave no stale samples in the trailing part of the
    output buffer."""
    plugin_system.load_plugins("plugins")
    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")
    node = ScriptClass()

    code = """
outputs = ['short_out']
short_out = torch.ones(1, 256)
"""
    node.params["code"].set(code)
    node.sync()
    node.on_ui_param_change("code")
    assert "short_out" in node.outputs

    # Pre-fill with stale data so ghosting would be visible.
    node.outputs["short_out"].buffer.fill_(0.75)
    node.process()

    out = node.outputs["short_out"].buffer
    assert torch.equal(out[0, :256], torch.ones(256, dtype=torch.float32)), \
        "first 256 samples must carry the script value"
    assert float(out[0, 256:].abs().max()) == 0.0, \
        "trailing samples of a short script output must be zeroed (no ghosting)"


def test_script_partial_frames_trailing_channels_cleared():
    """A short output (fewer frames AND fewer channels than the buffer) must
    zero both the trailing-frame and trailing-channel regions."""
    plugin_system.load_plugins("plugins")
    ScriptClass = plugin_system.NODE_REGISTRY.get("ScriptNode")
    node = ScriptClass()

    code = """
outputs = ['short_out']
short_out = torch.full((1, 128), 0.5)
"""
    node.params["code"].set(code)
    node.sync()
    node.on_ui_param_change("code")
    assert "short_out" in node.outputs

    node.outputs["short_out"].buffer.fill_(0.75)
    node.process()

    out = node.outputs["short_out"].buffer
    assert torch.equal(out[0, :128], torch.full((128,), 0.5, dtype=torch.float32))
    assert float(out[0, 128:].abs().max()) == 0.0
