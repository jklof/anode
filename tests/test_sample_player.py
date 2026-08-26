import os

import numpy as np
import pytest
import soundfile as sf
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


@pytest.fixture(scope="module")
def player_cls():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("SamplePlayer")
    assert cls is not None
    return cls


def make_node(player_cls):
    return player_cls()


def trigger_rise(node, prev_silent=True):
    """Feed a block containing a rising gate edge."""
    trig = torch.ones((CHANNELS, BLOCK_SIZE), dtype=DTYPE) if not prev_silent \
        else torch.cat([torch.zeros((CHANNELS, BLOCK_SIZE // 2), dtype=DTYPE),
                        torch.ones((CHANNELS, BLOCK_SIZE // 2), dtype=DTYPE)], dim=1)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def feed_gate(node, level):
    trig = torch.full((CHANNELS, BLOCK_SIZE), level, dtype=DTYPE)
    node.inputs["trigger_in"].get_tensor = lambda t=trig: t
    node.process()


def test_sample_player_registration(player_cls):
    assert player_cls.category == "Sources"
    assert player_cls.label == "Sample Player"


def test_idle_output_is_exact_zero(player_cls):
    """No sample loaded: output stays exactly zero even when polluted."""
    node = make_node(player_cls)
    node.out.buffer.fill_(0.99)
    feed_gate(node, 0.0)
    assert torch.all(node.out.buffer == 0.0)


def test_load_nrt_result_does_not_autoplay(player_cls):
    node = make_node(player_cls)
    data = torch.full((CHANNELS, 48000), 0.5, dtype=DTYPE)
    node.on_nrt_complete("load", True, data)
    assert node._audio_data is not None
    assert node._is_playing is False
    feed_gate(node, 0.0)
    assert torch.all(node.out.buffer == 0.0)


def test_trigger_starts_playback_and_retrigger_restarts(player_cls):
    node = make_node(player_cls)
    data = torch.linspace(0, 1, 48000, dtype=DTYPE).unsqueeze(0).repeat(CHANNELS, 1).contiguous()
    node.on_nrt_complete("load", True, data)

    trigger_rise(node)
    assert node._is_playing
    pos_after_edge = node._read_pos
    assert pos_after_edge > 0

    # Let it run, then retrigger: read position restarts from ~0
    for _ in range(20):
        feed_gate(node, 1.0)
    assert node._read_pos > pos_after_edge * 20

    node._last_trig = 0.0   # simulate a fresh gate edge
    trigger_rise(node)
    assert node._read_pos <= BLOCK_SIZE + 1.0, "retrigger must reset read position"


def test_pitch_doubles_read_speed(player_cls):
    node = make_node(player_cls)
    data = torch.zeros((CHANNELS, SAMPLE_RATE // 2), dtype=DTYPE)
    node.on_nrt_complete("load", True, data)
    node.params["pitch"].set(12.0)   # speed = 2
    node.sync()

    trigger_rise(node)
    pos_one = node._read_pos
    feed_gate(node, 1.0)
    delta = node._read_pos - pos_one
    assert delta == pytest.approx(2.0 * BLOCK_SIZE, rel=0.01)


def test_constant_sample_interpolates_exactly(player_cls):
    node = make_node(player_cls)
    data = torch.full((CHANNELS, 44100), 0.37, dtype=DTYPE)
    node.on_nrt_complete("load", True, data)
    node.params["pitch"].set(-3.0)   # fractional speed < 1
    node.sync()

    trigger_rise(node)
    out = node.out.buffer.clone()
    assert torch.allclose(out, torch.full_like(out, 0.37))


def test_non_loop_end_mutes_to_exact_zeros(player_cls):
    node = make_node(player_cls)
    short = torch.randn(CHANNELS, 1024, dtype=DTYPE)
    node.on_nrt_complete("load", True, short)

    trigger_rise(node)
    for _ in range(5):
        feed_gate(node, 1.0)
    assert node._is_playing is False
    assert torch.all(node.out.buffer == 0.0)


def test_loop_wrap_plays_indefinitely(player_cls):
    node = make_node(player_cls)
    one_block_tone = (0.5 * np.sin(2 * np.pi * 1000.0 * np.arange(512) / SAMPLE_RATE)).astype(np.float32)
    loop_data = torch.from_numpy(np.tile(one_block_tone, (CHANNELS, 1))).contiguous()
    node.on_nrt_complete("load", True, loop_data)
    node.params["loop"].set(True)
    node.sync()

    trigger_rise(node)
    last_max = 0.0
    for _ in range(30):
        feed_gate(node, 1.0)
        last_max = float(node.out.buffer.abs().max())
    assert node._is_playing
    assert last_max > 0.1, "looping sample must keep producing audio past its length"


def test_load_file_nrt_mono_wav_duplicates_to_stereo(player_cls, tmp_path):
    path = str(tmp_path / "mono.wav")
    sr = 24000   # deliberately != engine rate to exercise resampling
    sf.write(path, (0.3 * np.sin(2 * np.pi * 500.0 * np.arange(sr) / sr)).astype(np.float32), sr)

    node = make_node(player_cls)
    result = node._load_file_nrt(path)
    assert result.shape[0] == CHANNELS
    assert result.shape[1] == pytest.approx(SAMPLE_RATE, rel=0.01)
    assert torch.allclose(result[0], result[1])
    assert result.is_contiguous()


def test_load_failure_sets_error_msg(player_cls):
    node = make_node(player_cls)
    node.on_nrt_complete("load", False, RuntimeError("missing file"))
    assert node.error_msg and "Sample load failed" in node.error_msg


def test_start_resets_transport_state(player_cls):
    node = make_node(player_cls)
    data = torch.zeros((CHANNELS, 48000), dtype=DTYPE)
    node.on_nrt_complete("load", True, data)
    trigger_rise(node)
    assert node._is_playing

    node.start()
    assert node._is_playing is False
    assert node._read_pos == 0.0
    assert node._last_trig == 0.0


def test_sample_player_no_net_allocation_while_playing(player_cls):
    node = make_node(player_cls)
    data = torch.zeros((CHANNELS, SAMPLE_RATE), dtype=DTYPE)
    node.on_nrt_complete("load", True, data)
    trigger_rise(node)

    import gc
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    for _ in range(50):
        feed_gate(node, 1.0)
    growth, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert growth < 128 * 1024, f"net allocation {growth} bytes over 50 blocks"
