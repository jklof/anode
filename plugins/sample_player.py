"""
SamplePlayer — RAM-cached one-shot / looping sampler (Sources).

Architecture (per docs/new_node_specifications.md §5):
- File decoding + resampling run on NRTExecutor (submit_nrt /
  on_nrt_complete); the audio thread never touches disk.
- Playback is a fully vectorized fractional-read kernel: pre-allocated
  int64 index tensors + gather(out=) linear interpolation. No per-sample
  Python loops, no per-sample setitem.
- Trigger detection is block-granular by design: a rising edge
  (prev_max <= 0 < cur_max) retriggers once per block; multiple edges
  inside one block collapse into one trigger. Gate-style sources are the
  intended drivers.
- Loading a sample does NOT auto-play; playback starts on the first
  rising edge. Load failures surface through node.error_msg.
"""

import numpy as np
import torch

from base import Node, SAMPLE_RATE, CHANNELS, BLOCK_SIZE, DTYPE

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    sf = None
    _SF_AVAILABLE = False

try:
    import resampy
    _RESAMPY_AVAILABLE = True
except ImportError:
    resampy = None
    _RESAMPY_AVAILABLE = False


class SamplePlayer(Node):
    category = "Sources"
    label = "Sample Player"
    description = (
        "RAM-cached one-shot/looping sampler. Files are decoded and resampled to "
        "48 kHz on a background NRT worker; playback uses a vectorized fractional-"
        "read kernel with linear interpolation. Playback starts on the first rising "
        "edge of the trigger input (block-granular detection). Loading does not "
        "auto-play; load failures surface via node error status."
    )

    def __init__(self, name=""):
        super().__init__(name)
        self.add_input("trigger_in",
                       help="Gate/trigger signal; a rising edge above 0 restarts playback from the start.")
        self.out = self.add_output("out", channels=CHANNELS,
                                   help="Stereo sample playback (silence while idle).")

        self.add_file_param("sample_path", "", filter="Audio Files (*.wav *.flac *.mp3 *.ogg)",
                            help="Audio file to load; decoding/resampling happens on a background worker.")
        self.add_float_param("pitch", 0.0, -24.0, 24.0, unit="st",
                             help="Playback pitch offset in semitones (speed = 2^(pitch/12)).")
        self.add_float_param("gain", 1.0, 0.0, 2.0, unit="x",
                             help="Output gain applied after interpolation.")
        self.add_bool_param("loop", False,
                            help="Loop the sample continuously instead of stopping at the end.")

        self._audio_data = None      # (2, N) contiguous float32, set off-thread
        self._read_pos = 0.0
        self._is_playing = False
        self._last_trig = 0.0
        self._current_path = ""

        # Vectorized fractional-read kernel scratches
        self._arange = torch.arange(BLOCK_SIZE, dtype=DTYPE)
        self._pos = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._floor = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._frac = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._frac_inv = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
        self._weight = torch.ones(BLOCK_SIZE, dtype=DTYPE)
        self._idx_a = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
        self._idx_b = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
        self._gidx = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.int64)
        self._ga = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._gb = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._tmp = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
        self._maskb = torch.zeros(BLOCK_SIZE, dtype=torch.bool)

    def start(self):
        self._read_pos = 0.0
        self._is_playing = False
        self._last_trig = 0.0

    # ------------------------------------------------------------------
    # NRT load path
    # ------------------------------------------------------------------
    def on_ui_param_change(self, param_name):
        if param_name != "sample_path" or not _SF_AVAILABLE:
            return
        path = self.params["sample_path"].get_staging_safe()
        if path and path != self._current_path:
            self._current_path = path
            self.submit_nrt(self._load_file_nrt, path, tag="load")

    def _load_file_nrt(self, path):
        if sf is None:
            raise RuntimeError(
                "SamplePlayer: 'soundfile' is required to load samples but is not installed"
            )
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.T[:CHANNELS].copy()
        if data.shape[0] == 1:
            data = np.vstack([data[0], data[0]])          # mono -> stereo dup
        if sr != SAMPLE_RATE:
            if resampy is None:
                raise RuntimeError(
                    "SamplePlayer: 'resampy' is required to resample "
                    f"{sr} Hz audio to {SAMPLE_RATE} Hz but is not installed"
                )
            data = resampy.resample(data, sr, SAMPLE_RATE, axis=-1)
        return torch.from_numpy(np.ascontiguousarray(data))

    def on_nrt_complete(self, tag, ok, result):
        if tag != "load":
            return
        if ok:
            self._audio_data = result
            self._read_pos = 0.0
            self._is_playing = False                      # wait for a trigger
        else:
            self.error_msg = f"Sample load failed: {result}"

    # ------------------------------------------------------------------
    # Audio thread
    # ------------------------------------------------------------------
    def process(self):
        out = self.out.buffer
        trig = self.inputs["trigger_in"].get_tensor()[0]

        t_max = float(trig.max().item())
        if self._last_trig <= 0.0 and t_max > 0.0:
            self._is_playing = True
            self._read_pos = 0.0
        self._last_trig = float(trig[-1].item())

        out.zero_()                                       # anti-ghost: idle silence
        data = self._audio_data
        if data is None or not self._is_playing or data.shape[1] < 2:
            return

        num = data.shape[1]
        speed = 2.0 ** (self.params["pitch"].value / 12.0)
        is_loop = bool(self.params["loop"].value)

        # Fractional read positions for this block
        torch.mul(self._arange, speed, out=self._pos).add_(self._read_pos)
        if is_loop:
            self._pos.remainder_(num)
        else:
            torch.le(self._pos, float(num - 1), out=self._maskb)
            self._weight.copy_(self._maskb)

        torch.floor(self._pos, out=self._floor)
        self._idx_a.copy_(self._floor)                    # float -> int64 cast
        self._idx_b.copy_(self._idx_a).add_(1)
        if is_loop:
            self._idx_b.remainder_(num)
        else:
            self._idx_b.clamp_(max=num - 1)
        self._frac.copy_(self._pos).sub_(self._floor)
        torch.sub(1.0, self._frac, out=self._frac_inv)

        # Linear interpolation via gather (indices broadcast over channels)
        self._gidx.copy_(self._idx_a)
        torch.gather(data, 1, self._gidx, out=self._ga)
        self._gidx.copy_(self._idx_b)
        torch.gather(data, 1, self._gidx, out=self._gb)

        self._tmp.copy_(self._ga).mul_(self._frac_inv)
        out.copy_(self._gb).mul_(self._frac).add_(self._tmp)
        out.mul_(self.params["gain"].value)
        if not is_loop:
            out.mul_(self._weight)                        # mute past end of file

        last = float(self._pos[-1].item())
        self._read_pos = last + speed
        if not is_loop and last >= num - 1:
            self._is_playing = False
