"""Regression tests for FileRecorder real-time-path allocation behavior."""

import tracemalloc

import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, DTYPE


def _make_recorder_active():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("FileRecorder")
    node = cls()
    # Simulate an active recording without spawning the writer thread:
    # process() enqueues into the pre-allocated pool either way.
    node._recording = True
    return node


def test_file_recorder_process_is_allocation_free_in_steady_state():
    """Regression: pool_slot[:, ch] = temp.astype(np.int16, copy=False)
    allocated a fresh NumPy array every block. The in-place np.copyto cast
    must keep the steady-state heap allocation at zero."""
    node = _make_recorder_active()

    stereo = torch.zeros(CHANNELS, BLOCK_SIZE, dtype=DTYPE)
    mono = torch.zeros(1, BLOCK_SIZE, dtype=DTYPE)

    # Warmup (lazy imports, first-touch allocations)
    node.inp.get_tensor = lambda: stereo
    node.process()
    node.inp.get_tensor = lambda: mono
    node.process()

    tracemalloc.start()
    for _ in range(50):
        node.inp.get_tensor = lambda: stereo
        node.process()
        node.inp.get_tensor = lambda: mono
        node.process()
    growth, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert growth < 8192, f"net heap growth {growth} bytes over 100 blocks"


def test_file_recorder_mono_input_duplicates_channel():
    node = _make_recorder_active()
    mono = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
    node.inp.get_tensor = lambda: mono
    node.process()
    slot = node._block_pool[node._write_index - 1]
    assert (slot[:, 1] == slot[:, 0]).all()
    assert abs(int(slot[0, 0])) > 10000  # 0.5 -> ~16383
