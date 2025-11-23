import pytest
import torch
import os
import wave
import numpy as np
import plugin_system
from core import Graph


def test_file_recorder():
    # Load plugins
    plugin_system.load_plugins("plugins")

    # Create graph
    graph = Graph()

    # Create SineOscillator and FileRecorder
    sine_cls = plugin_system.NODE_REGISTRY.get("SineOscillator")
    rec_cls = plugin_system.NODE_REGISTRY.get("FileRecorder")

    sine = sine_cls()
    sine.id = "sine"
    recorder = rec_cls()
    recorder.id = "rec"

    # Add to graph
    graph.add_node(sine)
    graph.add_node(recorder)

    # Connect sine output to recorder input
    graph.connect("sine", "signal", "rec", "in")

    # Set recorder as master clock
    graph.set_master_clock(recorder)

    # Set filename and start recording
    temp_filename = "test_output.wav"
    recorder.params["filename"].set(temp_filename)
    recorder.params["record"].set(True)
    recorder.sync()  # apply staged to value
    recorder.on_ui_param_change("record")

    # Manually step through 10 process loops
    for _ in range(10):
        for node in graph.execution_order:
            try:
                node.process()
            except Exception as e:
                print(f"Process error in {node.name}: {e}")
                raise

    # Stop recording
    recorder.params["record"].set(False)
    recorder.sync()
    recorder.on_ui_param_change("record")
    recorder.stop()

    try:
        # Verify the WAV file exists and contains non-silent audio
        assert os.path.exists(temp_filename), "WAV file was not created"

        # Read the file and check it has data
        with wave.open(temp_filename, "rb") as wav_file:
            assert wav_file.getnchannels() == 2, "Expected stereo audio"  # CHANNELS is 2
            assert wav_file.getsampwidth() == 2, "Expected 16-bit"
            assert wav_file.getframerate() == 48000, "Expected 48000 Hz sample rate"  # SAMPLE_RATE
            frames = wav_file.getnframes()
            assert frames > 0, "No frames written"

            # Read some data
            data = wav_file.readframes(min(1024, frames))
            int_data = np.frombuffer(data, dtype=np.int16)
            # Check if max absolute value > 0 (not silent)
            assert np.max(np.abs(int_data)) > 0, "Audio appears silent"
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
