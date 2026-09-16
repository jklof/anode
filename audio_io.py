"""Shared audio file decoding. Stdlib + numpy + optional decoders only.

Two entry points, both pure helpers (no node state) for NRT workers:

* :func:`decode_audio_file` — any readable audio file to a
  ``(channels, samples)`` float32 array plus its sample rate. WAV/FLAC/OGG
  go through soundfile; MP3 (and anything else soundfile rejects) falls
  back to PyAV. Frame layouts are normalized per frame: PyAV does not
  guarantee uniform planar/packed layout across frames.
* :func:`write_wav_file` — float array to a 16-bit PCM WAV file.

SamplePlayer and the SheetSage2 transcriber share this so a format that
decodes in one place decodes everywhere.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    sf = None
    _SF_AVAILABLE = False

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    av = None
    _AV_AVAILABLE = False


def _orient_channels(data):
    """Orient a decoded array to (channels, samples). Decoders disagree on
    axis order, so assume the small dim (<= 8) is channels."""
    data = np.asarray(data)
    if data.ndim == 1:
        return data[None, :]
    if data.ndim != 2:
        raise ValueError(f"unsupported decoded shape {data.shape}")
    if data.shape[0] <= 8:
        return data
    return data.T


def _decode_av(path):
    """Decode via PyAV. Returns (channels, samples) float32 + sample rate."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    sr = stream.sample_rate
    # Orient frame-by-frame: layouts are not guaranteed uniform (mixed
    # planar/packed or trailing mono flush frames break bulk concatenation).
    parts = []
    for frame in container.decode(audio=0):
        parts.append(_orient_channels(frame.to_ndarray()).astype(np.float32))
    if not parts:
        raise RuntimeError(f"no audio frames decoded from {path.name}")
    channels = max(p.shape[0] for p in parts)
    aligned = []
    for p in parts:
        if p.shape[0] == 1 and channels > 1:
            p = np.repeat(p, channels, axis=0)  # mono flush frame
        if p.shape[0] != channels:
            raise RuntimeError(
                f"inconsistent channel counts while decoding {path.name}")
        aligned.append(p)
    return np.concatenate(aligned, axis=-1), sr


def decode_audio_file(path):
    """Decode an audio file to ``(channels, samples)`` float32 + sample rate.

    Raises RuntimeError with a human-readable message when neither decoder
    is available or the file cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"input audio not found: {path}")
    if path.suffix.lower() != ".mp3" and _SF_AVAILABLE:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return _orient_channels(data.T).astype(np.float32), sr
        except Exception:
            pass  # fall through to PyAV
    if not _AV_AVAILABLE:
        raise RuntimeError(
            f"could not decode {path.name} with soundfile and PyAV "
            "is not installed (MP3 needs PyAV)"
        )
    try:
        return _decode_av(path)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"could not decode {path.name}: {e}")


def write_wav_file(path, audio, sample_rate):
    """Write a (channels, samples) float array as PCM WAV. Returns path."""
    if sf is None:
        raise RuntimeError("'soundfile' is required to write WAV files")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(audio.T), sample_rate)
    return path


def encode_mp3_file(path, audio, sample_rate, bitrate=320000):
    """Encode a (channels, samples) float array to MP3 (libmp3lame).

    Pure helper for NRT workers (seconds of CPU for minutes of audio).
    Returns path. Raises RuntimeError when PyAV or the MP3 encoder is
    unavailable.
    """
    if not _AV_AVAILABLE:
        raise RuntimeError("'av' (PyAV) is required to encode MP3 files")
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] < 1:
        raise ValueError(f"expected (1|2, N) audio, got {audio.shape}")
    layout = "mono" if audio.shape[0] == 1 else "stereo"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: channel count/layout are deliberately NOT assigned on the
    # stream: PyAV >= 18 exposes them read-only (the encoder derives them
    # from the first frame). The frame layout below is authoritative.
    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libmp3lame", rate=int(sample_rate))
        stream.bit_rate = int(bitrate)
        frame_size = 1152 * 16
        for start in range(0, audio.shape[1], frame_size):
            chunk = audio[:, start:start + frame_size]
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
            frame.sample_rate = int(sample_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as e:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"MP3 encode failed: {e}")
    return path
