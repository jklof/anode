"""Tests for the shared audio_io decode helpers (no GPU, no Qt)."""
import numpy as np
import pytest
import soundfile as sf

import audio_io
from audio_io import decode_audio_file, write_wav_file


def _tone(seconds=0.5, sr=48000, channels=2):
    n = int(seconds * sr)
    t = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([t * (i + 1) / channels for i in range(channels)])


@pytest.mark.parametrize("ext", ["wav", "flac", "ogg"])
def test_decode_lossless_formats(tmp_path, ext):
    if ext == "ogg":
        pytest.importorskip("soundfile")  # ogg needs libsndfile vorbis support
    p = tmp_path / f"s.{ext}"
    try:
        sf.write(str(p), _tone().T, 48000)
    except Exception as e:
        pytest.skip(f"format not writable here: {e}")
    audio, sr = decode_audio_file(p)
    assert sr == 48000
    assert audio.shape == (2, 24000)
    assert audio.dtype == np.float32


def test_decode_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        decode_audio_file(tmp_path / "gone.wav")


def test_decode_garbage_file(tmp_path):
    p = tmp_path / "junk.wav"
    p.write_bytes(b"not audio at all" * 100)
    with pytest.raises(RuntimeError, match="could not decode"):
        decode_audio_file(p)


def test_orient_channels():
    from audio_io import _orient_channels
    mono = np.zeros(100, dtype=np.float32)
    assert _orient_channels(mono).shape == (1, 100)
    packed = np.zeros((1000, 2), dtype=np.float32)
    assert _orient_channels(packed).shape == (2, 1000)
    planar = np.zeros((2, 1000), dtype=np.float32)
    assert _orient_channels(planar).shape == (2, 1000)
    with pytest.raises(ValueError):
        _orient_channels(np.zeros((2, 3, 4)))


def test_write_wav_file_roundtrip(tmp_path):
    out = write_wav_file(tmp_path / "sub" / "o.wav", _tone(seconds=0.2), 48000)
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == 48000
    assert data.shape == (9600, 2)


def test_decode_mp3_roundtrip(tmp_path):
    pytest.importorskip("av")
    from audio_io import encode_mp3_file
    mp3 = tmp_path / "s.mp3"
    try:
        encode_mp3_file(mp3, np.zeros((2, 4410), dtype=np.float32), 44100)
    except RuntimeError as e:
        pytest.skip(f"mp3 encoder unavailable: {e}")
    audio, sr = decode_audio_file(mp3)
    assert sr == 44100
    assert audio.shape[0] == 2 and audio.shape[1] > 0


def test_encode_mp3_file_roundtrip(tmp_path):
    pytest.importorskip("av")
    from audio_io import encode_mp3_file
    tone = _tone(seconds=1.0)
    mp3 = tmp_path / "s.mp3"
    try:
        encode_mp3_file(mp3, tone, 48000)
    except RuntimeError as e:
        pytest.skip(f"mp3 encoder unavailable: {e}")
    assert mp3.stat().st_size > 1024
    audio, sr = decode_audio_file(mp3)
    assert sr == 48000 and audio.shape[0] == 2
    # Encoder padding shifts length slightly; content must correlate.
    n = min(audio.shape[1], tone.shape[1])
    assert float(np.corrcoef(audio[:, :n].flatten(),
                             tone[:, :n].flatten())[0, 1]) > 0.99


def test_encode_mp3_file_rejects_bad_shapes(tmp_path):
    pytest.importorskip("av")
    from audio_io import encode_mp3_file
    with pytest.raises((ValueError, RuntimeError)):
        encode_mp3_file(tmp_path / "s.mp3", np.zeros((3, 100), dtype=np.float32), 48000)
