import numpy as np
import pytest
import torch

import plugin_system
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

LATENCY = 9216
SETTLE_BLOCKS = (LATENCY // BLOCK_SIZE) + 2


def make_node():
    plugin_system.load_plugins("plugins")
    cls = plugin_system.NODE_REGISTRY.get("StudioVocalTransformer")
    assert cls is not None, "StudioVocalTransformer not registered (library build missing?)"
    node = cls()
    assert node.error_msg is None, f"native library failed to load: {node.error_msg}"
    return node


def process_block(node, blk):
    node.inp.get_tensor = lambda b=blk: b
    node.process()
    return node.out.buffer.clone()


def set_params(node, **kw):
    for k, v in kw.items():
        node.params[k].set(v)
    node.sync()
    node._sync_params_to_cpp()


def tone_blocks(freq, n_blocks, amp=0.4):
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = (amp * np.sin(2.0 * np.pi * freq * n / SAMPLE_RATE)).astype(np.float32)
    t2 = np.tile(t, (CHANNELS, 1))
    return [torch.from_numpy(t2[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(n_blocks)]


def saw_blocks(freq, n_blocks, n_harm=30, amp=0.3):
    """Continuous-phase bandlimited sawtooth (voice-like harmonic stack)."""
    n = np.arange(n_blocks * BLOCK_SIZE, dtype=np.float64)
    t = np.zeros_like(n)
    for h in range(1, n_harm + 1):
        if freq * h >= SAMPLE_RATE / 2:
            break
        t += (1.0 / h) * np.sin(2.0 * np.pi * freq * h * n / SAMPLE_RATE)
    t = (t / np.max(np.abs(t)) * amp).astype(np.float32)
    t2 = np.tile(t, (CHANNELS, 1))
    return [torch.from_numpy(t2[:, i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE].copy())
            for i in range(n_blocks)]


class _FakeMidiOut:
    """Minimal fake MIDI output slot (slot_type='midi') with a message list."""
    def __init__(self, messages):
        self.slot_type = "midi"
        self.packet = type("P", (), {"messages": messages})()


class _NoteOn:
    def __init__(self, note, velocity=100):
        self.type = "note_on"
        self.note = note
        self.velocity = velocity


class _NoteOff:
    def __init__(self, note):
        self.type = "note_off"
        self.note = note
        self.velocity = 0


def test_studio_vocal_transformer_instantiation():
    node = make_node()
    assert "in" in node.inputs
    assert "midi_in" in node.inputs
    assert node.inputs["midi_in"].slot_type == "midi"
    assert "out" in node.outputs
    assert node.outputs["out"].buffer.shape[0] == CHANNELS
    assert "retune_speed" in node.params
    assert "scale_root" in node.params
    assert "scale_type" in node.params
    assert "correction_enable" in node.params
    assert "vibrato_depth" in node.params


def test_studio_vocal_transformer_zero_allocation_audio_block():
    node = make_node()
    audio_in = torch.randn((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    out = process_block(node, audio_in)

    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_studio_vocal_transformer_mono_channel_adaptation():
    node = make_node()
    mono = torch.randn((1, BLOCK_SIZE), dtype=torch.float32)
    out = process_block(node, mono)
    # Mono source must expand to the full stereo output, never shrink it.
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.isfinite(out).all()


def test_studio_vocal_transformer_scale_snapping():
    """Design-spec scale test: hard snap on a C-Major in-scale tone must stay
    finite, non-zero, and bounded through the whole pipeline."""
    node = make_node()
    set_params(node, correction_enable=1.0, retune_speed=0.0, mix=1.0)
    node.params["scale_root"].set(0)  # C
    node.params["scale_type"].set(1)  # Major
    node.sync()

    for b in tone_blocks(440.0, SETTLE_BLOCKS + 8, amp=0.4):
        process_block(node, b)
    out = process_block(node, tone_blocks(440.0, 1, amp=0.4)[0])

    assert float(out.abs().sum()) > 0.0
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) < 4.0


def test_studio_vocal_transformer_scale_params_pushed():
    """Menu params map to the native scale_root / scale_mask without a KeyError
    (scale_mask is derived, not a Parameter, so it must NOT be in PARAM_MAP)."""
    node = make_node()
    calls = []
    node.lib.set_param = lambda h, pid, v: calls.append((pid, float(v)))

    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32))

    # Switch root to D (index 2) and scale to Minor (index 2).
    node.params["scale_root"].set(2)
    node.params["scale_type"].set(2)
    node.sync()
    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32))

    node._sync_params_to_cpp()  # must not raise (no 'scale_mask' Parameter)
    scale_root_calls = [v for pid, v in calls if pid == 1]
    scale_mask_calls = [v for pid, v in calls if pid == 2]
    assert scale_root_calls and scale_root_calls[-1] == pytest.approx(2.0), \
        f"scale_root should map D -> 2.0: {scale_root_calls}"
    from plugins.studio_vocal_transformer import SCALES
    assert scale_mask_calls and scale_mask_calls[-1] == pytest.approx(float(SCALES["Minor"])), \
        f"scale_mask should map Minor index 2: {scale_mask_calls}"


def test_studio_vocal_transformer_midi_target():
    node = make_node()
    calls = []
    node.lib.set_param = lambda h, pid, v: calls.append((pid, float(v)))

    # note_on 72 (C5) -> midi_mode must be 1, target_note 72.
    node.midi_in.connected_outputs = [_FakeMidiOut([(0, _NoteOn(72, 100))])]
    audio = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    node.inp.get_tensor = lambda: audio
    node.process()
    mode = [v for pid, v in calls if pid == 12]
    target = [v for pid, v in calls if pid == 13]
    assert mode and mode[-1] == 1.0, f"midi_mode not set: {mode}"
    assert target and target[-1] == pytest.approx(72.0), f"target note not set: {target}"

    # note_off 72 clears the target -> midi_mode returns to 0.
    calls.clear()
    node.midi_in.connected_outputs = [_FakeMidiOut([(0, _NoteOff(72))])]
    node.process()
    mode = [v for pid, v in calls if pid == 12]
    assert mode and mode[-1] == 0.0, f"midi_mode not cleared: {mode}"


def test_studio_vocal_transformer_preset_bounded():
    node = make_node()
    preset = node.PRESETS["Male -> Female Pop Lead"]
    set_params(node, **preset)

    for b in saw_blocks(220.0, SETTLE_BLOCKS, amp=0.3):
        process_block(node, b)
    out = process_block(node, saw_blocks(220.0, 1, amp=0.3)[0])

    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) < 4.0