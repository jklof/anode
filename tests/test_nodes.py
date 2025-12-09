import pytest
import torch
from unittest.mock import patch
import plugin_system


def test_gain():
    # Load plugins
    plugin_system.load_plugins("plugins")

    gain_cls = plugin_system.NODE_REGISTRY.get("Gain")
    gain = gain_cls()

    # Set input tensor to all 1.0s
    input_tensor = torch.ones(2, 512, dtype=torch.float32)  # channels=2, BLOCK_SIZE=512
    gain.inp.connected_outputs = []  # clear connections
    gain.inp.get_tensor = lambda: input_tensor  # mock get_tensor

    # Set 'vol' param to 0.5
    gain.params["vol"].set(0.5)
    gain.sync()  # apply staged to value

    # Run process
    gain.process()

    # Assert output buffer contains 0.5s
    expected = torch.full_like(gain.out.buffer, 0.5)
    assert torch.allclose(gain.out.buffer, expected)


def test_sine_oscillator():
    # Load plugins
    plugin_system.load_plugins("plugins")

    sine_cls = plugin_system.NODE_REGISTRY.get("SineOscillator")
    sine = sine_cls()

    # Run process
    sine.process()

    # Assert output values are within [-1.0, 1.0]
    output = sine.out_sig.buffer[0]  # shape (512,)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)
    # Also check not all zeros (since it's an oscillator)
    assert not torch.allclose(output, torch.zeros_like(output))


def test_stereo_to_mono():
    # Load plugins
    plugin_system.load_plugins("plugins")

    stereo_to_mono_cls = plugin_system.NODE_REGISTRY.get("StereoToMono")
    stm = stereo_to_mono_cls()

    # Test 1: Stereo input
    stereo_input = torch.ones(2, 512, dtype=torch.float32)  # channels=2, BLOCK_SIZE=512
    stm.inp.connected_outputs = []  # clear connections
    stm.inp.get_tensor = lambda: stereo_input  # mock get_tensor

    # First, pollute buffer[0] with stale data to simulate ghosting from previous run
    stm.out.buffer[0].fill_(0.9)

    # Run process - buffer zeroing should ensure clean output
    stm.process()

    # Assert buffer[0] contains averaged stereo input:
    # (1.0 + 1.0) / 2 = 1.0 (not 0.5!)
    expected_mono = torch.full_like(stm.out.buffer[0], 1.0)
    assert torch.allclose(stm.out.buffer[0], expected_mono)

    # Test 2: Mono input
    mono_input = torch.ones(1, 512, dtype=torch.float32)  # channels=1, BLOCK_SIZE=512
    stm.inp.get_tensor = lambda: mono_input  # mock get_tensor

    # Pollute buffer[0] again with stale data
    stm.out.buffer[0].fill_(0.7)

    # Run process - should properly overwrite with mono input
    stm.process()

    # Assert buffer[0] contains mono input (1.0s)
    expected_mono = torch.full_like(stm.out.buffer[0], 1.0)
    assert torch.allclose(stm.out.buffer[0], expected_mono)

    # Test 3: Test buffer structure - ensure it's truly single channel
    assert stm.out.buffer.shape[0] == 1  # Only 1 channel
    assert stm.out.buffer.shape[1] == 512  # BLOCK_SIZE

    # Test 4: Test different stereo values
    different_stereo = torch.ones(2, 512, dtype=torch.float32)
    different_stereo[0] = 2.0  # Left channel = 2.0
    different_stereo[1] = 4.0  # Right channel = 4.0

    stm.inp.get_tensor = lambda: different_stereo
    stm.out.buffer[0].fill_(0.5)  # Pollute again

    stm.process()

    # Expected: (2.0 + 4.0) / 2 = 3.0
    expected_mono = torch.full_like(stm.out.buffer[0], 3.0)
    assert torch.allclose(stm.out.buffer[0], expected_mono)


def test_reverb_mix_logic():
    """
    Test the dry/wet mixing math in ConvolutionReverb.
    """
    plugin_system.load_plugins("plugins")
    Reverb = plugin_system.NODE_REGISTRY.get("ConvolutionReverb")
    node = Reverb()
    
    # Mock input: Constant 1.0
    input_tensor = torch.ones(2, 512)
    node.inputs["in"].get_tensor = lambda: input_tensor
    
    # 1. Test Full Dry (Mix = 0.0)
    node.params["mix"].set(0.0)
    node.sync()
    node.process()
    # Output should be exactly input (1.0)
    assert torch.allclose(node.outputs["out"].buffer, input_tensor)
    
    # 2. Test Full Wet (Mix = 1.0)
    # Since we haven't loaded an IR, the convolution result is 0.0 (silence)
    # So output should be 0.0
    node.params["mix"].set(1.0)
    node.sync()
    node.process()
    assert torch.allclose(node.outputs["out"].buffer, torch.zeros_like(input_tensor))
    
    # 3. Test 50% Mix
    # Expected: (1.0 * 0.5) + (0.0 * 0.5) = 0.5
    node.params["mix"].set(0.5)
    node.sync()
    node.process()
    expected = torch.full_like(node.outputs["out"].buffer, 0.5)
    assert torch.allclose(node.outputs["out"].buffer, expected)

def test_nam_parameter_mapping():
    """
    Verify NAM parameters 'drive' and 'level' are registered correctly.
    """
    plugin_system.load_plugins("plugins")
    Nam = plugin_system.NODE_REGISTRY.get("NamNode")
    node = Nam()

    assert "drive" in node.params
    assert "level" in node.params
    assert node.params["drive"].value == 1.0
    assert node.params["level"].value == 1.0


def test_split_join_nodes():
    # Load plugins
    plugin_system.load_plugins("plugins")

    # Instantiate ChannelSplitter and ChannelJoiner
    splitter_cls = plugin_system.NODE_REGISTRY.get("ChannelSplitter")
    splitter = splitter_cls()

    joiner_cls = plugin_system.NODE_REGISTRY.get("ChannelJoiner")
    joiner = joiner_cls()

    # Test Splitter:
    # Mock input tensor: shape (2, 512). Ch0=1.0, Ch1=2.0.
    splitter_input = torch.ones(2, 512, dtype=torch.float32)
    splitter_input[0].fill_(1.0)
    splitter_input[1].fill_(2.0)
    splitter.inp.connected_outputs = []  # clear connections
    splitter.inp.get_tensor = lambda: splitter_input  # mock get_tensor

    # Process
    splitter.process()

    # Assert outputs_list[0] buffer contains 1.0
    expected_left = torch.full_like(splitter.outputs_list[0].buffer[0], 1.0)
    assert torch.allclose(splitter.outputs_list[0].buffer[0], expected_left)

    # Assert outputs_list[1] buffer contains 2.0
    expected_right = torch.full_like(splitter.outputs_list[1].buffer[0], 2.0)
    assert torch.allclose(splitter.outputs_list[1].buffer[0], expected_right)

    # Test Joiner:
    # Mock input 1 tensor: shape (1, 512). Val=0.5.
    left_input = torch.full((1, 512), 0.5, dtype=torch.float32)
    joiner.inputs_list[0].connected_outputs = []  # clear connections
    joiner.inputs_list[0].get_tensor = lambda: left_input  # mock get_tensor

    # Mock input 2 tensor: shape (1, 512). Val=0.8.
    right_input = torch.full((1, 512), 0.8, dtype=torch.float32)
    joiner.inputs_list[1].connected_outputs = []  # clear connections
    joiner.inputs_list[1].get_tensor = lambda: right_input  # mock get_tensor

    # Process
    joiner.process()

    # Assert output buffer Ch0 contains 0.5
    expected_ch0 = torch.full_like(joiner.out.buffer[0], 0.5)
    assert torch.allclose(joiner.out.buffer[0], expected_ch0)

    # Assert output buffer Ch1 contains 0.8
    expected_ch1 = torch.full_like(joiner.out.buffer[1], 0.8)
    assert torch.allclose(joiner.out.buffer[1], expected_ch1)
