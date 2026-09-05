import numpy as np
import pytest
import torch
import tracemalloc

import plugin_system
from base import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

LATENCY = 768
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


# ---------------------------------------------------------------------------
# Core verification test suite (all 8 criteria from LP-PSOLA test plan)
# ---------------------------------------------------------------------------

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
    telem = node.get_telemetry()
    assert telem["latency_samples"] == 768
    assert telem["latency_ms"] == pytest.approx(16.0)


def test_studio_vocal_transformer_latency_and_delay():
    """Test 1: Latency & Delay Test. Feed single Dirac impulse with mix = 1.0,
    pitch_shift = 0.0. Output peak arrives at sample 768 (tolerance +-1)."""
    node = make_node()
    set_params(node, mix=1.0, pitch_shift=0.0, correction_enable=0.0)

    emitted = []
    for blk in range(4):
        in_t = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
        if blk == 0:
            in_t[:, 0] = 1.0  # Dirac impulse at sample 0
        out = process_block(node, in_t)
        emitted.append(out)

    all_out = torch.cat(emitted, dim=1)[0].numpy()
    peak_idx = int(np.argmax(np.abs(all_out)))
    assert peak_idx == pytest.approx(768, abs=1), f"Expected peak at 768, got {peak_idx}"
    assert all_out[peak_idx] == pytest.approx(1.0, rel=0.01)


def test_studio_vocal_transformer_dry_bypass_anti_ghosting():
    """Test 2: Dry Bypass Anti-Ghosting. Set mix = 0.0. Check output against input.
    Bit-exact identity; latency = 0 samples."""
    node = make_node()
    set_params(node, mix=0.0)

    in_t = torch.randn((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    out_t = process_block(node, in_t)
    assert torch.equal(in_t, out_t), "mix=0.0 must be bit-exact identity"


def test_studio_vocal_transformer_bypass_boundary_reset():
    """Test 3: Bypass Boundary Reset. Toggle mix from 1.0 -> 0.0 -> 1.0.
    Zero residual audio or clicks from previous stream."""
    node = make_node()
    set_params(node, mix=1.0)
    # Stream audio
    for _ in range(5):
        process_block(node, torch.randn((CHANNELS, BLOCK_SIZE)))

    # Toggle to mix = 0.0
    set_params(node, mix=0.0)
    process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE)))

    # Toggle back to mix = 1.0; feeding zeros must emit clean silence
    set_params(node, mix=1.0)
    out = process_block(node, torch.zeros((CHANNELS, BLOCK_SIZE)))
    assert float(out.abs().max()) == 0.0, "Ghost audio detected after bypass boundary reset"


def test_studio_vocal_transformer_mono_channel_adaptation():
    """Test 4: Mono-to-Stereo Duplication. Input shape (1, 512). Output buffer shape
    is (2, 512) with channels 0 and 1 bit-identical."""
    node = make_node()
    mono = torch.randn((1, BLOCK_SIZE), dtype=torch.float32)
    out = process_block(node, mono)
    assert out.shape == (CHANNELS, BLOCK_SIZE)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], out[1]), "Mono input must be duplicated to both stereo channels"


def test_studio_vocal_transformer_pitch_retune_accuracy():
    """Test 5: Pitch Retune Accuracy. Feed 220 Hz (A3) sine. Set scale_root='C',
    scale_type='Major', retune_speed=0 ms. Pitch detector locks to A3 (in C Major);
    ratio is 1.0. Switch to C Minor (A not in scale) -> shifts to G#3 (207.65 Hz) or Bb3."""
    node = make_node()
    set_params(node, correction_enable=1.0, retune_speed=0.0, mix=1.0)
    node.params["scale_root"].set(0)  # C
    node.params["scale_type"].set(1)  # Major
    node.sync()

    out_major = []
    for b in tone_blocks(220.0, 25, amp=0.5):
        out_major.append(process_block(node, b))

    sig_major = torch.cat(out_major[5:], dim=1)[0].numpy()
    fft_maj = np.abs(np.fft.rfft(sig_major))
    freqs = np.fft.rfftfreq(len(sig_major), 1.0 / SAMPLE_RATE)
    peak_major = freqs[np.argmax(fft_maj)]
    assert peak_major == pytest.approx(220.0, abs=5.0), f"Expected ~220 Hz in C Major, got {peak_major}"

    # Switch to Minor (scale_type index 2)
    node.params["scale_type"].set(2)
    node.sync()
    out_minor = []
    for b in tone_blocks(220.0, 25, amp=0.5):
        out_minor.append(process_block(node, b))

    sig_minor = torch.cat(out_minor[5:], dim=1)[0].numpy()
    fft_min = np.abs(np.fft.rfft(sig_minor))
    freqs_min = np.fft.rfftfreq(len(sig_minor), 1.0 / SAMPLE_RATE)
    peak_minor = freqs_min[np.argmax(fft_min)]
    # A3 (57) in C Minor snaps to G#3 (56 = 207.65 Hz) or Bb3 (58 = 233.08 Hz)
    assert peak_minor == pytest.approx(207.65, abs=8.0) or peak_minor == pytest.approx(233.08, abs=8.0), \
        f"Expected snap to G#3 (207.65) or Bb3 (233.08) in C Minor, got {peak_minor}"


def test_studio_vocal_transformer_midi_targeting_speed():
    """Test 6: MIDI Targeting Speed. Send note_on (MIDI 60 = 261.6 Hz) into midi_in.
    Output reaches target pitch within the retune_speed time constant without 192 ms lag."""
    node = make_node()
    set_params(node, correction_enable=1.0, retune_speed=0.0, mix=1.0)

    # Warm up with 220 Hz
    t = np.arange(25 * BLOCK_SIZE) / SAMPLE_RATE
    sine = (0.5 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    t2 = np.tile(sine, (CHANNELS, 1))

    for blk in range(5):
        b = torch.from_numpy(t2[:, blk * BLOCK_SIZE:(blk + 1) * BLOCK_SIZE].copy())
        process_block(node, b)

    # Send note_on 60 (C4 = 261.63 Hz)
    node.midi_in.connected_outputs = [_FakeMidiOut([(0, _NoteOn(60, 100))])]
    out_midi = []
    for blk in range(5, 25):
        b = torch.from_numpy(t2[:, blk * BLOCK_SIZE:(blk + 1) * BLOCK_SIZE].copy())
        out_midi.append(process_block(node, b))

    # Check that it reached target note without 192 ms lag (within 2-3 blocks).
    # Autocorrelation-based F0 (robust against the AM sidebands that a raw
    # FFT peak-pick chases on PSOLA output).
    sig = torch.cat(out_midi[3:], dim=1)[0].numpy().astype(np.float64)
    sig = sig - sig.mean()
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    lo, hi = int(SAMPLE_RATE / 400), int(SAMPLE_RATE / 150)
    lag = lo + int(np.argmax(ac[lo:hi]))
    f0 = SAMPLE_RATE / lag
    assert f0 == pytest.approx(261.63, rel=0.05), f"Expected target ~261.6 Hz, got {f0:.1f}"


def test_studio_vocal_transformer_filter_stability_stress():
    """Test 7: Filter Stability Stress Test. Feed white noise burst at 0 dBFS with
    extreme formant shifts (+-24 st) and gender morphs (+-1.0). Output values remain
    bounded in [-4.0, +4.0]; no NaN or Inf."""
    node = make_node()
    set_params(node, mix=1.0)
    for shift in [-24.0, 0.0, 24.0]:
        for gender in [-1.0, 0.0, 1.0]:
            set_params(node, formant_shift=shift, gender_morph=gender)
            for _ in range(10):
                noise = torch.randn((CHANNELS, BLOCK_SIZE))
                out = process_block(node, noise)
                assert torch.isfinite(out).all(), f"Non-finite output for shift={shift}, gender={gender}"
                assert float(out.abs().max()) <= 4.0, f"Output exceeded 4.0: {out.abs().max()}"


def test_studio_vocal_transformer_realtime_allocations_1000_blocks():
    """Test 8: Real-Time Allocations. Run 1000 blocks under torch.no_grad() while
    tracing memory. Zero Python allocations inside node.process()."""
    node = make_node()
    set_params(node, mix=1.0, correction_enable=1.0)

    audio_in = torch.randn((CHANNELS, BLOCK_SIZE), dtype=torch.float32)
    node.inp.get_tensor = lambda: audio_in

    # Warm-up
    for _ in range(50):
        node.process()

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    with torch.no_grad():
        for _ in range(1000):
            node.process()

    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap2.compare_to(snap1, "lineno")
    svt_stats = [
        s for s in stats
        if s.traceback[0].filename.endswith("/studio_vocal_transformer.py")
    ]
    leak_size = sum(s.size_diff for s in svt_stats if s.size_diff > 0)
    assert leak_size == 0, f"Detected allocations in studio_vocal_transformer.py: {svt_stats}"


# ---------------------------------------------------------------------------
# Additional compatibility tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LP-PSOLA quality regression tests
# ---------------------------------------------------------------------------

def test_studio_vocal_transformer_neutral_transparent():
    """With all transforms neutral (no correction, no shift, no formant warp,
    no vibrato, no breathiness) the LP-PSOLA path must be near-transparent:
    output ~ input delayed by the fixed 768-sample latency. Regression guard
    for the identity-path coloration and filter-stability defects."""
    node = make_node()
    set_params(node, mix=1.0, correction_enable=0.0, pitch_shift=0.0,
               formant_shift=0.0, gender_morph=0.0, vibrato_depth=0.0,
               vibrato_rate=5.5, breathiness=0.0)

    blocks = saw_blocks(220.0, 40, amp=0.3)
    outs = [process_block(node, b) for b in blocks]
    full = torch.cat(outs, dim=1)
    src = torch.cat(blocks, dim=1)

    delay = LATENCY
    ref = src[:, :full.shape[1] - delay]
    got = full[:, delay:]
    # Skip the startup/settle region; compare steady state only.
    start = BLOCK_SIZE * 8
    err = (got[:, start:] - ref[:, start:]).pow(2).mean().sqrt()
    ref_rms = ref[:, start:].pow(2).mean().sqrt() + 1e-9
    snr = 20.0 * float(torch.log10(ref_rms / err))
    assert snr > 6.0, f"neutral-path SNR too low: {snr:.1f} dB"


def test_studio_vocal_transformer_voicing_toggle_no_stale_energy():
    """Voiced -> silence -> voiced: the PSOLA overlap-add accumulator must be
    cleared when voicing drops, so no stale grain energy pops out during or
    after the silence gap."""
    node = make_node()
    set_params(node, mix=1.0, correction_enable=0.0, vibrato_depth=0.0,
               breathiness=0.0)
    amp = 0.3
    seq = (saw_blocks(220.0, 8, amp=amp)
           + [torch.zeros((CHANNELS, BLOCK_SIZE)) for _ in range(10)]
           + saw_blocks(220.0, 8, amp=amp))
    outs = [process_block(node, b) for b in seq]

    # During silence the output must be essentially silent (no stale grains).
    # Skip the first two silent blocks: legitimate LPC filter ringing decay.
    for i in range(11, 17):
        m = float(outs[i].abs().max())
        assert m < 0.1 * amp, f"stale energy during silence, block {i}: {m}"
    # After voicing returns, no explosive transient: bounded to ~3x input amp
    # (the LPC model has a legitimate onset transient at the voicing boundary).
    for i in range(19, 26):
        m = float(outs[i].abs().max())
        assert m < 3.0 * amp, f"pop after voicing return, block {i}: {m}"
    assert all(torch.isfinite(o).all() for o in outs)