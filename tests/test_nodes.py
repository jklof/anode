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
