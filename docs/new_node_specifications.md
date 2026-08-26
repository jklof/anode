# ANode: Finalized Technical Implementation Specifications for Proposed Nodes

> **Status: FINAL — supersedes all earlier drafts.** Every issue from the spec
> review has been resolved: per-sample Python loops are eliminated
> (vectorized gather kernels or C++ FFI), modulation inputs use the
> param-binding fallback, `_compute_polyblep` is fully specified, the C-ABI
> wrappers and `restype`/`argtypes` rules are explicit, ChorusFlanger and
> NoiseGate specs are complete, and the test plan uses house conventions with
> exact-value ghosting asserts.

## Global Rules (apply to every node below)

1. **Constants**: `BLOCK_SIZE = 512`, `SAMPLE_RATE = 48000`, `CHANNELS = 2`,
   `DTYPE = torch.float32`.
2. **Zero allocations in `process()`**: every tensor listed in a "State &
   Buffers" section is pre-allocated in `__init__`. All writes use in-place
   ops (`copy_`, `mul_`, `add_`, `sub_`, `zero_`, `clamp_`, `remainder_`,
   `pow_`) or `out=` variants. Boolean masks are pre-allocated `torch.bool`
   tensors written via `torch.lt/gt/ge/le(..., out=mask)`; float masks via
   `float_mask.copy_(bool_mask)` (zero-allocation cast).
3. **Modulation-input convention**: any input that can modulate a parameter
   MUST be registered as `self.add_input(name, param_name)`
   (`base.py:170`). An unconnected slot then returns the parameter's constant
   tensor cache automatically (`base.py:98`) — never hand-check
   `if "x_in" in self.inputs` (always true) and never read a raw unconnected
   scratch (silently zeroes the signal, e.g. an oscillator going quiet).
4. **Anti-ghosting**: output buffers are written on every channel of every
   block. Mono inputs broadcast into stereo destinations via `copy_`; unused
   channels are explicitly zeroed when channel counts genuinely mismatch.
5. **`start()` reset**: every node resets phase/counters/state in `start()`
   so transport restarts never leak stale audio or state.
6. **Per-sample DSP placement**: per-sample feedback loops run behind the FFI
   in C++. Purely feed-forward per-sample math is acceptable in Python only
   if expressed as vectorized gather/index kernels (see SamplePlayer,
   Bitcrusher). No `.item()` inside loops; at most a handful of scalar
   syncs per block.
7. **FFI rule**: every additional export beyond `create` gets explicit
   `restype`/`argtypes`. Unannotated handles default to 32-bit `c_int` and
   silently truncate 64-bit pointers.

### File layout (definitive)

| Node(s) | Python file | Native | Test file |
|---|---|---|---|
| WaveformOscillator | `plugins/oscillators.py` (new) | — | `tests/test_oscillators.py` |
| WaveShaper | `plugins/waveshaper.py` (new) | — | `tests/test_waveshaper.py` |
| EnvelopeFollower | `plugins/envelope.py` (new) | `cpp/envelope.cpp` → `libenvelope.so` | `tests/test_envelope.py` |
| NoiseGate | append to `plugins/dynamics.py` | — | extend `tests/test_dynamics.py` |
| SamplePlayer | `plugins/sample_player.py` (new) | — | `tests/test_sample_player.py` |
| StereoPanner, MidSideEncoder, MidSideDecoder | `plugins/spatial.py` (new) | — | `tests/test_spatial.py` |
| Bitcrusher | `plugins/bitcrusher.py` (new) | — | `tests/test_bitcrusher.py` |
| ChorusFlanger | `plugins/chorus.py` (new) | `cpp/chorus.cpp` → `libchorus.so` | `tests/test_chorus.py` |
| MathOp | `plugins/math_op.py` (new) | — | `tests/test_math_op.py` |

---

## 1. WaveformOscillator (PolyBLEP Multi-Wave Generator)

### 1.1 Metadata
* Class `WaveformOscillator`, label `"Waveform Oscillator"`, category `"Sources"`.

### 1.2 DSP Formulation
Phase increment `dt[n] = clamp(f[n]/fs, 1e-5, 0.49)`; phase accumulates
mod 1.0. PolyBLEP residual centered on the wrap discontinuity at t = 0, with
`u = t/dt`, `v = (t-1)/dt`:

```
BLEP(t) = -(u - 1)^2        if 0    <= t <  dt      (== 2u - u^2 - 1)
        =  (v + 1)^2        if 1-dt <  t <= 1       (== 2v + v^2 + 1)
        =  0                otherwise
```

* **Sine**: `y = sin(2*pi*phi)`
* **Triangle** (naive — no step discontinuity; mild HF aliasing accepted):
  `y = 2*|2*phi - 1| - 1`
* **Sawtooth** (falling): naive `1 - 2*phi`, corrected `y += BLEP(phi)`
* **Square/Pulse**: naive `+1 if phi < pw else -1`, corrected
  `y += BLEP(phi) - BLEP((phi - pw) mod 1)`

### 1.3 Ports & Parameters
* Inputs: `freq_in` bound to `"freq"`; `pw_in` bound to `"pulse_width"`
  (param-binding rule #3 — both work unconnected).
* Output: `signal` (`channels=1`).
* Params: menu `waveform ["Sine","Triangle","Sawtooth","Square"]` idx 0;
  `freq` 440.0 [1, 20000] Hz; `amp` 0.5 [0, 1]; `pulse_width` 0.5 [0.01, 0.99].
* The hard-sync input from the draft is **dropped** (was never implemented;
  hard sync needs its own BLEP treatment and will be specced separately).

### 1.4 State & Buffers
```python
self.phase = 0.0                                    # reset in start()
self._dt         = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._phase_buf  = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._naive      = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._blep       = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._blep_a     = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._blep_b     = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._temp       = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._mask_a     = torch.zeros(BLOCK_SIZE, dtype=torch.bool)
self._mask_b     = torch.zeros(BLOCK_SIZE, dtype=torch.bool)
```

### 1.5 Implementation

```python
def start(self):
    self.phase = 0.0

def _compute_polyblep(self, phase, dt, out):
    """out = BLEP(phase, dt), fully vectorized, zero-allocation."""
    # Region A value: -(u-1)^2 with u = phase/dt  (dt clamped >= 1e-5: safe)
    torch.div(phase, dt, out=self._blep_a)
    self._blep_a.sub_(1.0).pow_(2).neg_()
    # Region B value: (v+1)^2 with v = (phase-1)/dt
    torch.sub(phase, 1.0, out=self._blep_b)
    torch.div(self._blep_b, dt, out=self._blep_b)
    self._blep_b.add_(1.0).pow_(2)
    # Region B threshold is a TENSOR because dt varies under FM
    torch.sub(1.0, dt, out=self._temp)
    torch.gt(phase, self._temp, out=self._mask_b)
    torch.lt(phase, dt, out=self._mask_a)
    out.zero_()
    out.copy_(self._blep_a).mul_(self._mask_a)   # bool->float cast via copy_
    # add region B masked: reuse _blep_b
    self._blep_b.mul_(self._mask_b)
    out.add_(self._blep_b)

def process(self):
    freq_sig = self.inputs["freq_in"].get_tensor()[0]   # param cache if unconnected
    pw_sig   = self.inputs["pw_in"].get_tensor()[0]
    amp      = self.params["amp"].value
    wave     = int(self.params["waveform"].value)
    out      = self.outputs["signal"].buffer[0]

    torch.mul(freq_sig, 1.0 / SAMPLE_RATE, out=self._dt)
    self._dt.clamp_(min=1e-5, max=0.49)
    self._phase_buf.copy_(self._dt).cumsum_(dim=0).add_(self.phase).remainder_(1.0)
    self.phase = float(self._phase_buf[-1].item())      # single sync per block

    if wave == 0:      # Sine
        torch.mul(self._phase_buf, 2.0 * np.pi, out=self._temp)
        torch.sin(self._temp, out=out)
    elif wave == 1:    # Triangle (naive)
        torch.mul(self._phase_buf, 2.0, out=self._temp)
        self._temp.sub_(1.0).abs_().mul_(2.0).sub_(1.0)
        out.copy_(self._temp)
    elif wave == 2:    # Sawtooth + PolyBLEP at wrap
        torch.mul(self._phase_buf, -2.0, out=self._naive).add_(1.0)
        self._compute_polyblep(self._phase_buf, self._dt, self._blep)
        torch.add(self._naive, self._blep, out=out)
    else:              # Square/Pulse + PolyBLEP at 0 and pw
        torch.lt(self._phase_buf, pw_sig, out=self._mask_a)
        self._naive.copy_(self._mask_a).mul_(2.0).sub_(1.0)     # +1/-1
        self._compute_polyblep(self._phase_buf, self._dt, self._blep)
        self._naive.add_(self._blep)
        torch.sub(self._phase_buf, pw_sig, out=self._temp).remainder_(1.0)
        self._compute_polyblep(self._temp, self._dt, self._blep)
        self._naive.sub_(self._blep)
        out.copy_(self._naive)

    out.mul_(amp)
```

Calibration invariants for tests: sine/saw/square peak ≈ amp (±0.02),
triangle peak ≈ amp; saw/square max inter-sample delta ≤ `8 * max(dt)` +
epsilon (continuity ⇒ no Nyquist splatter); DC offset < 0.01 after warmup.

---

## 2. WaveShaper / Saturator

### 2.1 Metadata
Class `WaveShaper`, label `"WaveShaper / Saturation"`, category `"Effects"`.

### 2.2 DSP
Conditioning `x_d = drive*x + bias`; transfer functions:

* Tanh: `f(x) = tanh(x)`
* Soft clip: `x - x^3/3` for `|x| <= 1`, else `sign(x)*2/3`
  (C1-continuous at |x|=1)
* Hard clip: `clamp(x, -1, 1)`
* Foldback: `sin(x)`
* Asymmetric tube: `x/(1+x)` for `x >= 0`; `x/(1-x) - 0.1x^2` for `x < 0`
  (denominators bounded away from 0 in both branches)

Mix: `y = output_level * ((1-mix)*dry + mix*f(x_d))`.
Note: bias + asymmetric modes produce DC by design; downstream nodes can
remove it (Data Display shows Mean (DC); MathOp Subtract works as a crude
blocker). A per-mode DC blocker is explicitly out of scope.

### 2.3 Ports & Parameters
* Inputs: `in`; `drive_mod` **bound to `"drive"`** (rule #3).
* Output: `out` (`channels=2`). Mono in → stereo out via `copy_` broadcast.
* Params: menu `mode ["Tanh (Tape)","Soft Clip","Hard Clip","Wavefolder","Asymmetric Tube"]`;
  `drive` 1.0 [0.1, 20]; `bias` 0.0 [-1, 1]; `mix` 1.0 [0, 1];
  `output_level` 1.0 [0, 2].

### 2.4 State & Buffers
```python
self._driven = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._shaped = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._dry    = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._tmp    = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
```

### 2.5 Implementation sketch (zero-alloc)
```python
sig = self.inp.get_tensor()
drive = drive_mod tensor row 0            # always a tensor (binding rule)
mode, mix, bias, lvl = params...

self._dry.copy_(sig)                       # broadcasts mono -> stereo
torch.mul(sig, drive_row_view, out=self._driven)   # drive_row = drive_mod[0].unsqueeze? use broadcasting mul
self._driven.add_(bias)

if mode == 0:   self._shaped.copy_(self._driven).tanh_()
elif mode == 1: # soft clip: cubic where |x|<=1 else sign*2/3
    self._shaped.copy_(self._driven).clamp_(-1.0, 1.0)          # x_c
    self._tmp.copy_(self._shaped).pow_(3)
    self._shaped.sub_(self._tmp.mul_(1.0/3.0))                  # x - x^3/3
    torch.gt(self._driven.abs(), 1.0, out=self._maskb)          # maskb bool buf
    self._mask.copy_(self._maskb)
    self._tmp.copy_(self._driven).sign_().mul_(2.0/3.0)
    self._shaped.copy_(torch.where(self._mask, self._tmp, self._shaped))
elif mode == 2: self._shaped.copy_(self._driven).clamp_(-1.0, 1.0)
elif mode == 3: self._shaped.copy_(self._driven).sin_()
else:           # asymmetric tube via where on x >= 0
    pos = self._driven.div(self._driven.add(1.0))               # temp; or scratch pair
    neg = self._driven.div(self._tmp.copy_(self._driven).neg_().add_(1.0)).sub_(self._sq_term)
    self._shaped.copy_(torch.where(self._ge_mask, pos, neg))    # ge mask precomputed
```
(Implementer may restructure branch temporaries across `_tmp`/extra
pre-allocated scratches; constraint: no new tensors.) Then:
```python
out.copy_(self._dry).mul_(1.0 - mix).add_(self._shaped, alpha=mix).mul_(lvl)
```

---

## 3. EnvelopeFollower (Dynamics & Audio-Rate CV Extractor)

### 3.1 Metadata
Class `EnvelopeFollower`, label `"Envelope Follower"`, category `"Utilities"`.
Native: `cpp/envelope.cpp` → `libenvelope.so`.

### 3.2 DSP
One-pole ballistics with attack/release coefficient switching per sample
(C++ side):

```
alpha = 1 - exp(-1000 / (fs * tau_ms));  tau = att_ms if det > env else rel_ms
env'  = env + alpha * (det - env)
det   = gain * (peak: max_c |x[c,n]|  |  rms: sqrt(mean_c x^2))
```

**Gate with hysteresis** (chatter fix): gate opens when `env >= thresh`,
stays open until `env < thresh * 0.5`. State boolean persists between blocks.

CV output range: `[0, gain]` — **not clamped to 1.0** (gain up to 10×).
Clamp downstream with MathOp mode 8 if a strict [0,1] CV is needed.

### 3.3 Ports & Parameters
* Input: `in`. Outputs: `cv_out` (channels=1), `gate_out` (channels=1, {0,1}).
* Params → PARAM_MAP ids: `mode` 0 (["Peak","RMS"]), `attack_ms` 10.0 [0.1,500],
  `release_ms` 100.0 [1,2000], `gain` 1.0 [0.1,10], `gate_thresh` 0.1 [0,1].

### 3.4 C++ (`cpp/envelope.cpp`)
Internal class as drafted, plus gate hysteresis state, plus the standard
ANode C-ABI wrapper:

```cpp
extern "C" {
    void* create();
    void  destroy(void* h);
    void  set_samplerate(void* h, float sr);
    void  set_param(void* h, int id, float v);
    void  reset(void* h);          // env_=0, gate_open_=false
    void  process(void* h, const float* in, float* cv, float* gate,
                  int channels, int frames);   // planar in; mono planar outs
}
```
Detection loops index `in[c*frames + i]` (planar). RMS divides by actual
`channels` passed in — never by a hardcoded 2 (sidechain-channel-count rule).

### 3.5 Python wrapper (`plugins/envelope.py`)
`class EnvelopeFollower(_CppParamMixin, FFINode)` replicating the lazy
param-sync mixin pattern from `filters.py:27` (covers UI edits, load_state,
and direct Parameter.set()+sync()). `LIB_NAME="envelope"`.
Because there are two outputs, override `process()` entirely (do not use the
single-output FFINode.process) and bind the extended export manually:

```python
def _bind_functions(self):
    super()._bind_functions()
    # MANDATORY annotations — unannotated handles truncate pointers
    self.lib.process.restype = None
    self.lib.process.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)]*3 \
                                + [ctypes.c_int, ctypes.c_int]
    self._bind_reset()   # reset: restype=None, argtypes=[c_void_p]

def process(self):
    sig = self.inputs["in"].get_tensor()
    cv   = self.outputs["cv_out"].buffer[0]
    gate = self.outputs["gate_out"].buffer[0]
    self.lib.process(self.dsp_handle,
                     ctypes.cast(sig.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                     ctypes.cast(cv.data_ptr(),   ctypes.POINTER(ctypes.c_float)),
                     ctypes.cast(gate.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                     int(sig.shape[0]), BLOCK_SIZE)

def start(self):
    self._call_reset(); self._cpp_param_state = None
```
Contiguity: slot buffers and their rows are contiguous; assert like
`ffi_base.py:135` does.

---

## 4. NoiseGate (Downward Expander)

### 4.1 Metadata
Class `NoiseGate`, label `"Noise Gate"`, category `"Effects"`. Appended to
`plugins/dynamics.py`; pure-Python, **block-rate detector + linear gain
ramp** — fully vectorized, no per-sample loop.

### 4.2 Architecture (replaces undefined "lookahead")
* `LOOKAHEAD = 64` samples: main audio passes through a pre-allocated ring
  `(CHANNELS, LOOKAHEAD)`; the smoothed gain trajectory is applied to the
  delayed signal, aligning gain changes ahead of transients.
* `DECIM = 16`: detector evaluates one RMS level per block over
  `sc[:, ::16]` (slice view — no copy).

Per block:
```
level_db = 10*log10(mean(sc_decim^2) + 1e-9)
open            = level_db >= thresh_db
target_gr_db    = 0                                  if open or holding
                = max(-range_db, (level_db-thresh_db)*(ratio-1))  otherwise
hold_left      = hold_samples   if open  else max(0, hold_left - BLOCK)
coeff          = att_block_coeff if target_gr > gr_db else rel_block_coeff
gr_db         += coeff * (target_gr_db - gr_db)      # freeze while holding
g_start, g_end = 10^(gr_prev_db/20), 10^(gr_db/20)
ramp           = linspace(g_start, g_end, BLOCK)     # into scratch
delayed        = concat(ring_tail, sig[:, :BLOCK-LOOKAHEAD])
out            = delayed * ramp
```
Block-rate coefficients: `att_c = 1 - exp(-(BLOCK/SR)/att_s)`, same form for
release. While `hold_left > 0`, `target_gr_db` pins to 0 dB so a re-opened
gate ramps shut-inhibit with the attack coefficient (no snap).

### 4.3 Ports & Parameters
* Inputs: `in` (stereo); `sidechain` (**unbound** optional key input — if
  `sidechain.connected_outputs` is empty, fall back to `in` itself; binding
  to a param would be wrong here since silence must gate, not pass).
* Output: `out` (channels=2). Sidechain may be mono `(1,BLOCK)` — reductions
  run over all dims, never indexing by CHANNELS.
* Params: `thresh` −40 dB [−80, 0]; `ratio` 10 [1, 50]; `attack` 1 ms
  [0.1, 50]; `hold` 50 ms [0, 500]; `release` 100 ms [5, 1000];
  `range` 60 dB [0, 90].

### 4.4 State & Buffers
`gr_db=0.0`, `hold_left=0`, `att_block_c/rel_block_c` recomputed on param
sync; `_ring (CHANNELS, LOOKAHEAD)`, `_delayed (CHANNELS, BLOCK)`,
`_ramp (BLOCK,)`, `_sq (CHANNELS, BLOCK)`, masks. `start()` zeroes ring,
`gr_db`, `hold_left`.

---

## 5. SamplePlayer (RAM-Cached One-Shot & Looping Sampler)

### 5.1 Metadata
Class `SamplePlayer`, label `"Sample Player"`, category `"Sources"`.

### 5.2 Architecture (rework: replaces the per-sample loop)
The draft's 512-iteration Python interpolation loop (~12 ms/block measured
for dispatcher setitem loops) is replaced by a **vectorized fractional-read
kernel** using pre-allocated integer index tensors and `torch.gather(out=)`.
Loading runs on `NRTExecutor` exactly as drafted.

Trigger semantics (documented honestly): rising edges are detected at block
granularity — `trig.max() > 0 and last_trig <= 0` retriggers once per block;
multiple edges inside one block collapse into one trigger. Gate-style
sources are the intended drivers.

Playback policy: loading a sample does **not** auto-play; playback starts on
the first rising edge. Load failures surface through `node.error_msg`.

### 5.3 Ports & Parameters
As drafted: `trigger_in` input; `out` stereo; file param `sample_path`
(filter `"Audio Files (*.wav *.flac *.mp3 *.ogg)"`), `pitch` [−24, 24],
`gain` [0, 2], `loop` bool.

### 5.4 State & Buffers
```python
self._audio_data = None          # (2, N) contiguous float32, set off-thread
self._read_pos = 0.0; self._is_playing = False; self._last_trig = 0.0
self._current_path = ""
# kernel scratches (all pre-allocated):
self._arange   = torch.arange(BLOCK_SIZE, dtype=DTYPE)
self._pos      = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._floor    = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._frac     = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._frac_inv = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._weight   = torch.ones(BLOCK_SIZE, dtype=DTYPE)
self._idx_a    = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
self._idx_b    = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
self._gidx     = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=torch.int64)
self._ga       = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._gb       = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._tmp      = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._maskb    = torch.zeros(BLOCK_SIZE, dtype=torch.bool)
```

### 5.5 Implementation
```python
def process(self):
    out = self.outputs["out"].buffer
    data = self._audio_data
    trig = self.inputs["trigger_in"].get_tensor()[0]
    t_max = float(trig.max().item())
    if self._last_trig <= 0.0 and t_max > 0.0:
        self._is_playing, self._read_pos = True, 0.0
    self._last_trig = float(trig[-1].item())
    out.zero_()                                   # anti-ghost: silent when idle
    if data is None or not self._is_playing:
        return

    num = data.shape[1]
    speed = 2.0 ** (self.params["pitch"].value / 12.0)
    is_loop = bool(self.params["loop"].value)

    torch.mul(self._arange, speed, out=self._pos).add_(self._read_pos)
    if is_loop:
        self._pos.remainder_(num)
    else:
        torch.le(self._pos, float(num - 1), out=self._maskb)
        self._weight.copy_(self._maskb)

    torch.floor(self._pos, out=self._floor)
    self._idx_a.copy_(self._floor)                          # float->int64 cast
    self._idx_b.copy_(self._idx_a).add_(1)
    if is_loop: self._idx_b.remainder_(num)
    else:       self._idx_b.clamp_(max=num - 1)
    self._frac.copy_(self._pos).sub_(self._floor)
    torch.sub(1.0, self._frac, out=self._frac_inv)

    self._gidx.copy_(self._idx_a)                           # (B,) -> (2,B) bcast
    torch.gather(data, 1, self._gidx, out=self._ga)
    self._gidx.copy_(self._idx_b)
    torch.gather(data, 1, self._gidx, out=self._gb)

    self._tmp.copy_(self._ga).mul_(self._frac_inv)
    out.copy_(self._gb).mul_(self._frac).add_(self._tmp)
    out.mul_(self.params["gain"].value)
    if not is_loop:
        out.mul_(self._weight)                              # mute past end

    last = float(self._pos[-1].item())
    self._read_pos = last + speed
    if not is_loop and last >= num - 1:
        self._is_playing = False
```

### 5.6 NRT load path
```python
def on_ui_param_change(self, param_name):
    if param_name != "sample_path": return
    path = self.params["sample_path"].get_staging_safe()
    if path and path != self._current_path:
        self._current_path = path
        self.submit_nrt(self._load_file_nrt, path, tag="load")

def _load_file_nrt(self, path):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.T[:CHANNELS].copy()
    if data.shape[0] == 1:
        data = np.vstack([data[0], data[0]])       # mono -> stereo dup
    if sr != SAMPLE_RATE:
        data = resampy.resample(data, sr, SAMPLE_RATE, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(data))

def on_nrt_complete(self, tag, ok, result):
    if tag != "load": return
    if ok:
        self._audio_data, self._read_pos, self._is_playing = result, 0.0, False
    else:
        self.error_msg = f"Sample load failed: {result}"
```
`soundfile`/`resampy` imports sit inside `_load_file_nrt` (NRT thread);
module-top guarded imports follow `media_player.py` convention if shared.

Tests must include: genuine-mono-file path, loop wrap correctness (idx_b
wrap sample continuity), non-loop end mutates to exact zeros, retrigger
resets read position, tracemalloc net-growth over 50 blocks.

---

## 6. StereoPanner, MidSideEncoder, MidSideDecoder

### 6.1 Metadata
All category `"Utilities"` in `plugins/spatial.py`.

### 6.2 DSP (unchanged — verified correct)
* Pan law: `theta = pi/4 * (pan+1)`; `gL = sqrt(2)*cos(theta)`,
  `gR = sqrt(2)*sin(theta)` (constant power: gL^2+gR^2 == 2; unity center).
* Width: `M=(L+R)/2`, `S=(R-L)/2`; `L_w = M - S*w`, `R_w = M + S*w`
  (exact reconstruction at w=1).
* Mid/Side (orthonormal): `Mid=(L+R)/sqrt2`, `Side=(L-R)/sqrt2`; inverse
  symmetric.

### 6.3 Notes
* `pan_mod` **bound to `"pan"`** (works unconnected). Width is param-only.
* Mono input: `r = sig[0]` fallback; both output channels always written
  (anti-ghost verified: w=1, pan=0 reduces to identity within float error).
* Buffers: `_mid`, `_side` `(BLOCK,)` only.
* Encoder/Decoder: plain matrix ops with `out=` writes; encoder accepts mono
  (treats R=L); decoder expects the encoder's stereo output. No state, no
  `start()` needed.

---

## 7. Bitcrusher (Sample Rate & Bit Depth Reducer)

### 7.1 Architecture (rework: replaces per-sample loops)
Global hold grid: sample `n` (global index) is a hold point iff `n % D == 0`.
Within a block starting at global offset `g0`, each local sample reads the
most recent hold point `k = floor(n/D)*D`; `k` can reach up to `D-1 < 64`
samples into the previous block, handled by an extended gather domain
`[prev tail | current block]`. Fully vectorized; zero `.item()` calls.

### 7.2 Parameters & math
`bits` int [1,16] default 8 → `steps = 2^(bits-1)`; quantize
`round(x*steps)/steps` (the post-clamp from the draft is dead code and is
removed); `downsample` int [1,64] default 1; `mix` [0,1] default 1.

### 7.3 State & Buffers
```python
DECIM_MAX = 64
self._g0   = 0                                            # reset in start()
self._ext  = torch.zeros((CHANNELS, BLOCK_SIZE + DECIM_MAX), dtype=DTYPE)
self._tail = torch.zeros((CHANNELS, DECIM_MAX), dtype=DTYPE)
self._held = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._g    = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
self._kh   = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
self._jidx = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
self._mask = torch.zeros(BLOCK_SIZE, dtype=torch.bool)
```

### 7.4 Implementation
```python
sig = self.inp.get_tensor()
D = int(self.params["downsample"].value)
steps = 2.0 ** (int(self.params["bits"].value) - 1)
mix = self.params["mix"].value
out = self.outputs["out"].buffer

self._ext[:, :BLOCK_SIZE].copy_(sig)          # broadcasts mono -> stereo
self._ext[:, BLOCK_SIZE:].copy_(self._tail)

torch.arange(self._g0, self._g0 + BLOCK_SIZE, out=self._g)
torch.div(self._g, D, rounding_mode="floor", out=self._kh).mul_(D)
self._jidx.copy_(self._kh).sub_(self._g0).add_(DECIM_MAX)   # ext-domain index
for c in range(CHANNELS):
    torch.gather(self._ext[c], 0, self._jidx, out=self._held[c])
self._tail.copy_(self._ext[:, BLOCK_SIZE:])
self._g0 += BLOCK_SIZE

self._held.mul_(steps).round_().div_(steps)                 # quantize
out.copy_(sig)                                              # dry (bcast mono)
out.mul_(1.0 - mix).add_(self._held, alpha=mix)             # NO temp alloc
```
Note the mix fix vs the draft: `add_(..., alpha=mix)` instead of
`(held * mix)` which allocated every block.

---

## 8. ChorusFlanger (Quadrature-Modulated Delay Line) — COMPLETE SPEC

### 8.1 Metadata
Class `ChorusFlanger`, label `"Chorus / Flanger"`, category `"Effects"`.
Native: `cpp/chorus.cpp` → `libchorus.so`; wrapper `plugins/chorus.py`.

### 8.2 Ports & Parameters (PARAM_MAP ids in brackets)
* Input `in` (mono duplicated to stereo internally in C++), output `out` stereo.
* `rate` [0]: LFO Hz, default 0.6, range [0.05, 8.0]
* `depth_ms` [1]: default 3.0, range [0.0, 8.0]
* `base_delay_ms` [2]: default 5.0, range [0.0, 20.0]
* `feedback` [3]: default 0.3, range [0.0, 0.9]
* `spread` [4]: 0.0 = mono LFO, 1.0 = quadrature (+90° on R); default 1.0
* `mix` [5]: default 0.5, range [0.0, 1.0]

### 8.3 DSP (per channel, C++)
```
phase_c  advances by rate/fs per sample (member state; survives blocks)
d[n]     = fs*(base_ms + depth_ms*(0.5 + 0.5*sin(2*pi*phase_c + spread*pi/2*ch))) / 1000
wet[n]   = hermite_read(ring_c, write_pos, d[n])
feed_in  = x[n] + tanh(feedback * wet[n])       # soft-saturated feedback
ring push; out[n] = (1-mix)*x[n] + mix*wet[n]
```
Hermite interpolation on the circular buffer (standard 4-point Catmull-Rom
form; fractional delay wrapped with modulo buffer length).
Buffer sizing: `CAP = ceil(SR * MAX_DELAY_MS/1000) + 4` with
`MAX_DELAY_MS = 30` (base 20 + depth 8 + margin), allocated once in the
constructor — never resized.

### 8.4 C ABI & bindings
Standard exports: `create`, `destroy`, `set_samplerate`, `set_param`,
`reset` (clears rings, phases, feedback state), `process(handle, in, out,
channels, frames)` (planar, honors actual channel count).
Python side: `class ChorusFlanger(_CppParamMixin, FFINode)`, `LIB_NAME="chorus"`;
all extended exports annotated per global rule #7.

### 8.5 Build integration (append to existing cpp/CMakeLists.txt)
```cmake
# --- Target: Chorus / Flanger ---
add_library(chorus SHARED chorus.cpp)
target_compile_features(chorus PRIVATE cxx_std_17)
copy_target_to_directory(chorus "${PROJECT_SOURCE_DIR}/../plugins")
```
(`copy_target_to_directory` already exists at `cpp/CMakeLists.txt:20`.)

Test invariants: static delay (rate=0) passes a click with measurable group
delay ≈ base_delay; output stays bounded for any settings (tanh feedback);
wet-only mix at rate=0 equals delayed input; zero net allocation across the
Python wrapper (tracemalloc).

---

## 9. MathOp (Vectorized Arithmetic & CV Conditioner)

### 9.1 Fixes vs draft
* **No `torch.tensor()` construction in `process()`**: scalar Min/Max use
  the clamp identity (`min(A,s) == A.clamp(max=s)`,
  `max(A,s) == A.clamp(min=s)`).
* Divide epsilon made **sign-correct** to match the operation table:
  `A / (B + sign(B)*1e-6)`, with exact zeros in B treated as positive
  (`sign(0) == 0` would otherwise leave a zero denominator).
* Documented polarity: result channel count follows the widest operand
  (torch broadcast into the stereo `out` buffer). Mono A × stereo B →
  stereo; that is intended behavior, covered by a test.
* Modes 6–8 are unary (B ignored); mode 9 uses only `scalar`/`offset`
  params (B ignored) — stated in the UI label help.

### 9.2 Implementation (branch bodies)
```python
sig_a = self.in_a.get_tensor()
op = int(self.params["op"].value)
b_conn = bool(self.in_b.connected_outputs)
sig_b = self.in_b.get_tensor() if b_conn else None
scalar = self.params["scalar"].value
out = self.outputs["out"].buffer

if op == 0:   torch.add(sig_a, sig_b if b_conn else scalar, out=out)
elif op == 1: torch.sub(sig_a, sig_b if b_conn else scalar, out=out)
elif op == 2: torch.mul(sig_a, sig_b if b_conn else scalar, out=out)
elif op == 3:
    if b_conn:
        torch.sign(sig_b, out=self._tmp)          # pre-allocated (CHANNELS, BLOCK)
        self._tmp.mul_(1e-6).add_(sig_b)
        torch.div(sig_a, self._tmp, out=out)
    else:
        d = scalar + (1e-6 if scalar >= 0 else -1e-6)
        torch.div(sig_a, d, out=out)
elif op == 4:
    if b_conn: torch.minimum(sig_a, sig_b, out=out)
    else:      out.copy_(sig_a).clamp_(max=scalar)
elif op == 5:
    if b_conn: torch.maximum(sig_a, sig_b, out=out)
    else:      out.copy_(sig_a).clamp_(min=scalar)
elif op == 6: torch.neg(sig_a, out=out)
elif op == 7: torch.abs(sig_a, out=out)
elif op == 8: out.copy_(sig_a).clamp_(0.0, 1.0)
elif op == 9: out.copy_(sig_a).mul_(scalar).add_(offset)
```
State: `self._tmp = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)` only.

---

## 10. Automated Test Acceptance Plan (house conventions)

Common helpers replicate `tests/test_spectrogram.py`: `make_node(class_name)`,
`stream(node, tensor, blocks)` draining `monitor_queue` where applicable,
and genuine `(1, BLOCK)` mono fixtures everywhere a mono path exists.

### 10.1 Metadata (complete list)
```python
NODES = ["WaveformOscillator", "WaveShaper", "EnvelopeFollower", "NoiseGate",
         "SamplePlayer", "StereoPanner", "MidSideEncoder", "MidSideDecoder",
         "Bitcrusher", "ChorusFlanger", "MathOp"]
```
Each: registered, `category` in {"Sources","Utilities","Effects","I/O","Visual"},
non-empty label.

### 10.2 Zero net allocation (per node, house style)
```python
gc.collect()
tracemalloc.start()
before, _ = tracemalloc.get_traced_memory()
for _ in range(50):
    node.process()
growth, _ = tracemalloc.get_traced_memory()
tracemalloc.stop()
assert growth < 128 * 1024
```
(Snapshot/compare_to drafting replaced — this matches test_spectrogram.py.)

### 10.3 Anti-ghosting — EXACT-value asserts (the draft's isfinite check was
vacuous: a stale-filled buffer is finite whether or not the node wrote it)
```python
mono = torch.full((1, BLOCK_SIZE), 0.5, dtype=DTYPE)
node.inp.get_tensor = lambda: mono
node.out.buffer.fill_(0.99)          # pollute BOTH channels
node.process()
assert torch.allclose(node.out.buffer[0], expected_first_channel)
assert torch.allclose(node.out.buffer[1], expected_second_channel)  # e.g. == ch0
```
Required per node: WaveShaper (mono→stereo), StereoPanner, MidSide pair,
Bitcrusher, MathOp (mono A × stereo B polarity), SamplePlayer idle zeros,
NoiseGate mono-sidechain reduction.

### 10.4 Node-specific calibration minimums
* Oscillator: waveform peaks ≈ amp; saw/square continuity bound
  `max|Δ| <= 8*max(dt)+eps`; triangle/sine spectra sanity via Data Display
  conventions not required — numeric peak/RMS suffice; FM sweep changes
  zero-crossing count; PW=0.25 square duty measurably shifts mean.
* WaveShaper: tanh(2·0.5) calibration point; soft-clip continuity at |x|=1
  (both branches give 2/3·sign); mix=0 bit-exact passthrough; wavefolder
  folds ±1.5 into known range; tube asymmetry shifts DC (mean ≠ 0).
* EnvelopeFollower: peak mode tracks half-wave rectified sine envelope with
  att/rel time constants (measure time-to-63% ≈ τ); gate hysteresis holds
  open between thresh and thresh/2; RMS mode reads 0.7071·amp for sine.
* NoiseGate: closed-loop attenuation ≥ range−6 dB below threshold; opens
  within attack+lookahead samples of a step; hold prevents chatter for
  pulsed input at threshold; bypassed (range=0) is near-unity.
* SamplePlayer: pitch doubling reads ~2× samples per block; loop wrap is
  continuous across the seam (max delta bounded); end-of-file mutes to
  exact zeros; retrigger restarts from 0; load failure sets error_msg.
* Bitcrusher: bits=16 ≈ passthrough (error < 2^-15); downsample=8 produces
  stair plateaus of width ≥ 7; mix=0 bit-exact; hold grid continuity across
  block boundaries (no double-triggered plateau edge at n=512).
* ChorusFlanger: per §8.5 invariants.
* MathOp: table-driven truth tests for all 10 ops incl. divide-by-near-zero
  sign behavior and scalar Min/Max clamp identities.

Every test file ends with the node's tracemalloc growth test; native-backed
nodes additionally verify `error_msg is None` after construction (library
loaded) so CI catches missing .so builds.
