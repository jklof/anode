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
