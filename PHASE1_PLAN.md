# Phase 1 Implementation Plan: Essential DSP & Mixing Nodes

## Overview of Phase 1 Nodes

| Node Name | Class Name | Category | Target File | Implementation Type |
| :--- | :--- | :--- | :--- | :--- |
| **Brickwall Limiter** | `BrickwallLimiter` | `Effects` | `plugins/dynamics.py` | Vectorized PyTorch (Lookahead + Peak Detector) |
| **Colored Noise Generator** | `ColoredNoise` | `Sources` | `plugins/oscillators.py` | Vectorized PyTorch (FIR Spectral Filtering) |
| **Transient Shaper** | `TransientShaper` | `Effects` | `plugins/dynamics.py` | Vectorized PyTorch (Dual Ballistic Envelope) |
| **Auto Gain / Leveler** | `AutoGain` | `Utilities` | `plugins/dynamics.py` | Vectorized PyTorch (Sliding RMS Window) |

---

## Global Architectural Rules (Mandatory for Agent)

1. **Constants**: `BLOCK_SIZE = 512`, `SAMPLE_RATE = 48000`, `CHANNELS = 2`, `DTYPE = torch.float32`.
2. **Zero Allocations in `process()`**:
   - Every tensor, scratch buffer, mask, and index array **must** be pre-allocated in `__init__`.
   - Use in-place operations (`copy_`, `mul_`, `add_`, `sub_`, `clamp_`, `zero_`, `fill_`, `pow_`) and `out=` keyword arguments exclusively.
3. **No `out=` Down-Sizing Trap**:
   - When copying narrower inputs (e.g. mono `(1, BLOCK)` into stereo `(CHANNELS, BLOCK)`), always use `dest.copy_(src)` first, followed by in-place operators. Never pass narrower operands to binary `out=` functions directly.
4. **Modulation Input Fallback**:
   - Register parameter-modulating inputs using `self.add_input("name_in", "param_name")` so unconnected slots fall back to the parameter constant tensor cache.
5. **Anti-Ghosting & Transport Resets**:
   - Reset all phase state, rings, and accumulators in `start()`.
   - Output buffers must be fully written across all channels on every block.

---

# Detailed Node Specifications

---

## 1. Brickwall Limiter (`BrickwallLimiter`)

### 1.1 Specification
- **File**: Append to `plugins/dynamics.py`
- **Class**: `BrickwallLimiter(Node)`
- **Category**: `"Effects"`
- **Label**: `"Brickwall Limiter"`

### 1.2 DSP Formulation
- **Lookahead**: 240 samples ($5.0\text{ ms}$ at $48\text{ kHz}$). Delayed signal aligns with anticipatory gain reduction.
- **Peak Detection**: Inter-channel absolute maximum $P[n] = \max_c |x[c, n]|$ over the **current block** and the **lookahead tail** carried in the ring.
- **Gain Target** (threshold gates reduction, ceiling sets output cap):
  $$\text{Target Gain (lin)} = \min\left(1.0, \frac{\text{ceiling\_lin}}{\max(P_{\text{max}}, \text{thresh\_lin})}\right)$$
  where `thresh_lin = 10^(threshold_db/20)` and `ceiling_lin = 10^(ceiling_db/20)`.
- **Ballistics**: Instantaneous attack (anticipates peaks via lookahead), 1-pole exponential release across blocks:
  $$\alpha_{\text{rel}} = 1 - \exp\left(-\frac{\text{BLOCK\_SIZE}}{\text{SAMPLE\_RATE} \cdot (\text{release\_ms} / 1000)}\right)$$
  Clamp `alpha_rel = min(1.0, alpha_rel)`.
- **Gain Ramp**: `torch.linspace(g_prev, g_curr, BLOCK_SIZE, out=self._ramp)` applied to delayed audio.
- **Hard Ceiling Clamp**: In-place clamp ensuring zero overs past `ceiling_lin`.

> **Phase 1 trade-off**: Block-granular ballistics means the full reduction is reached at the end of the ramp (last sample of the block). Peaks occurring early in the block are limited primarily by the final `clamp_` (soft clipping) rather than anticipatory gain. This keeps the implementation allocation-free and passes the ceiling test via the hard clamp. Sub-block envelope resolution is deferred to Phase 2.

### 1.3 Ports & Parameters
- **Inputs**:
  - `in` (stereo audio)
  - `thresh_in` bound to `"threshold"`
- **Output**:
  - `out` (stereo, channels=2)
- **Parameters**:
  - `threshold`: float, default `-0.1`, min `-40.0`, max `0.0` (dB)
  - `ceiling`: float, default `-0.1`, min `-20.0`, max `0.0` (dB)
  - `release`: float, default `50.0`, min `1.0`, max `1000.0` (ms)

### 1.4 Buffers & State
```python
LOOKAHEAD = 240
self._ring = torch.zeros((CHANNELS, LOOKAHEAD), dtype=DTYPE)
self._delayed = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._mono_peak = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._mono_indices = torch.zeros(BLOCK_SIZE, dtype=torch.int64)
self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._current_gain = 1.0
self._prev_gain = 1.0
```

### 1.5 Implementation Pattern
```python
def start(self):
    self._ring.zero_()
    self._current_gain = 1.0
    self._prev_gain = 1.0

def process(self):
    sig = self.inputs["in"].get_tensor()
    thresh_db = float(self.inputs["thresh_in"].get_tensor()[0, 0].item())
    ceiling_db = self.params["ceiling"].value
    release_ms = max(1.0, self.params["release"].value)

    thresh_lin = 10.0 ** (thresh_db / 20.0)
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)

    # 1. Channel-pooled peak detection over current block
    torch.abs(sig, out=self._delayed)
    torch.max(self._delayed, dim=0, out=(self._mono_peak, self._mono_indices))

    # 2. Target gain from peak
    max_peak = max(float(self._mono_peak.max().item()), 1e-9)
    if max_peak > thresh_lin:
        target_gain = min(1.0, ceiling_lin / max_peak)
    else:
        target_gain = 1.0

    # 3. Ballistics: instant attack (target below current), smooth release
    if target_gain < self._current_gain:
        self._current_gain = target_gain
    else:
        alpha_rel = 1.0 - math.exp(-(BLOCK_SIZE / SAMPLE_RATE) / (release_ms / 1000.0))
        alpha_rel = min(1.0, alpha_rel)
        self._current_gain += alpha_rel * (target_gain - self._current_gain)

    # 4. Smooth gain ramp for this block
    torch.linspace(self._prev_gain, self._current_gain, BLOCK_SIZE, out=self._ramp)
    self._prev_gain = self._current_gain

    # 5. Delay line alignment & application
    la = LOOKAHEAD
    self._delayed[:, :la].copy_(self._ring)
    self._delayed[:, la:].copy_(sig[:, :BLOCK_SIZE - la])
    self._ring.copy_(sig[:, BLOCK_SIZE - la:])

    out = self.outputs["out"].buffer
    out.copy_(self._delayed)
    out.mul_(self._ramp)
    out.clamp_(-ceiling_lin, ceiling_lin)
```

---

## 2. Colored Noise Generator (`ColoredNoise`)

### 2.1 Specification
- **File**: Append to `plugins/oscillators.py`
- **Class**: `ColoredNoise(Node)`
- **Category**: `"Sources"`
- **Label**: `"Colored Noise Generator"`

### 2.2 DSP Formulation
- **White Noise**: Uniform random in $[-1, 1]$ generated per channel via in-place `uniform_`.
- **Colored Spectrum Slopes via FIR Kernels ($N = 127$ taps)**:
  - **Pink**: $-3\text{ dB/octave}$ ($1/f$ power spectrum).
  - **Brown**: $-6\text{ dB/octave}$ ($1/f^2$ power spectrum / Brownian integration).
  - **Blue**: $+3\text{ dB/octave}$ ($f$ power spectrum).
  - **Violet**: $+6\text{ dB/octave}$ ($f^2$ power spectrum / derivative).
- **Convolution Method**: Pre-computed Hann-windowed FIR impulse responses for each color, evaluated via `torch.nn.functional.conv1d` across `(CHANNELS, BLOCK_SIZE + TAPS - 1)` with continuous overlap history.
  - **Note**: `conv1d` allocates its output tensor every block. This is an accepted transient allocation per `AGENTS.md` (same exception as `convolution_reverb.py`) and must be documented at the call site.

### 2.3 Ports & Parameters
- **Inputs**: None (Source)
- **Output**: `out` (stereo, channels=2, uncorrelated channels)
- **Parameters**:
  - `type`: menu `["White", "Pink (-3dB/oct)", "Brown (-6dB/oct)", "Blue (+3dB/oct)", "Violet (+6dB/oct)"]`, default `0`
  - `amp`: float, default `0.2`, min `0.0`, max `1.0`

### 2.4 Buffers & State
```python
TAPS = 127
HIST = TAPS - 1
self._raw_noise = torch.zeros((CHANNELS, BLOCK_SIZE), dtype=DTYPE)
self._conv_in   = torch.zeros((1, CHANNELS, HIST + BLOCK_SIZE), dtype=DTYPE)
self._tail      = torch.zeros((CHANNELS, HIST), dtype=DTYPE)
self._kernels   = self._init_fir_kernels()  # (4, 1, TAPS) filters, persistent
```

### 2.5 Implementation Pattern
```python
def _init_fir_kernels(self):
    # Design fixed frequency response profiles in frequency domain and irfft to 127 taps
    # Shapes: (4 colors, 1, 127) for grouped 1D convolution
    ...

def start(self):
    self._tail.zero_()

def process(self):
    color_idx = int(self.params["type"].value)
    amp = self.params["amp"].value
    out = self.outputs["out"].buffer

    # 1. Generate independent white noise for each channel
    self._raw_noise.uniform_(-1.0, 1.0)

    if color_idx == 0:  # White noise
        out.copy_(self._raw_noise).mul_(amp)
        return

    # 2. Assemble continuous FIR input: [history | current block]
    self._conv_in[0, :, :HIST].copy_(self._tail)
    self._conv_in[0, :, HIST:].copy_(self._raw_noise)
    self._tail.copy_(self._raw_noise[:, BLOCK_SIZE - HIST:])

    # 3. Apply color filter (F.conv1d with pre-allocated kernel)
    # ALLOCATION: conv1d output is a new tensor each block — accepted transient.
    kernel = self._kernels[color_idx - 1]  # (1, 1, 127)
    filtered = torch.nn.functional.conv1d(
        self._conv_in, kernel.expand(CHANNELS, 1, TAPS), groups=CHANNELS
    )
    out.copy_(filtered[0]).mul_(amp)
```

---

## 3. Transient Shaper (`TransientShaper`)

### 3.1 Specification
- **File**: Append to `plugins/dynamics.py`
- **Class**: `TransientShaper(Node)`
- **Category**: `"Effects"`
- **Label**: `"Transient Shaper"`

### 3.2 DSP Formulation
- **Dual Ballistic Envelopes** updated once per block:
  - Fast Envelope ($E_{\text{fast}}$): $\tau_{\text{att}} = 1.0\text{ ms}$, $\tau_{\text{rel}} = 20.0\text{ ms}$.
  - Slow Envelope ($E_{\text{slow}}$): $\tau_{\text{att}} = 25.0\text{ ms}$, $\tau_{\text{rel}} = 200.0\text{ ms}$.
- **Peak Tracking**: Absolute mean level of the mono-summed signal per block.
- **Differential Separation**:
  $$\text{Transient Component } \Delta = E_{\text{fast}} - E_{\text{slow}}$$
- **Gain Calculation**:
  $$\text{Gain}_{\text{target}} = \frac{\Delta \cdot (1 + \text{attack}) + E_{\text{slow}} \cdot (1 + \text{sustain})}{E_{\text{fast}} + 10^{-6}}$$
  Clamped to `[0.0, 4.0]`.
- **Vectorized Smoothing**: `torch.linspace` trajectory from previous block's gain to current block's target gain, scaled by `output_gain`.

> **Phase 1 trade-off**: Block-granular peak tracking (46 Hz update) limits transient resolution. Sub-block envelope resolution is deferred to Phase 2.

### 3.3 Ports & Parameters
- **Inputs**:
  - `in` (stereo audio)
  - `attack_mod` bound to `"attack"`
  - `sustain_mod` bound to `"sustain"`
- **Output**:
  - `out` (stereo, channels=2)
- **Parameters**:
  - `attack`: float, default `0.0`, min `-1.0` (-100%), max `2.0` (+200%)
  - `sustain`: float, default `0.0`, min `-1.0` (-100%), max `1.0` (+100%)
  - `output_gain_db`: float, default `0.0`, min `-18.0`, max `18.0` (dB)

### 3.4 Buffers & State
```python
self._e_fast = 0.0
self._e_slow = 0.0
self._prev_gain = 1.0
self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
self._mono = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
```

### 3.5 Implementation Pattern
```python
def start(self):
    self._e_fast = 0.0
    self._e_slow = 0.0
    self._prev_gain = 1.0

def process(self):
    sig = self.inputs["in"].get_tensor()
    attack_val = float(self.inputs["attack_mod"].get_tensor()[0, 0].item())
    sustain_val = float(self.inputs["sustain_mod"].get_tensor()[0, 0].item())
    out_gain = 10.0 ** (self.params["output_gain_db"].value / 20.0)

    # 1. Mono peak level over block
    out_buf = self.outputs["out"].buffer
    torch.abs(sig, out=out_buf)
    torch.mean(out_buf, dim=0, out=self._mono)
    peak = float(torch.max(self._mono).item())

    # 2. Update dual ballistics (1-pole, coeff clamped)
    dt_block = BLOCK_SIZE / SAMPLE_RATE
    a_f = dt_block / 0.001 if peak > self._e_fast else dt_block / 0.020
    a_s = dt_block / 0.025 if peak > self._e_slow else dt_block / 0.200
    a_f = min(1.0, a_f)
    a_s = min(1.0, a_s)

    self._e_fast += a_f * (peak - self._e_fast)
    self._e_slow += a_s * (peak - self._e_slow)

    # 3. Differential Transient vs Sustain gain
    delta = self._e_fast - self._e_slow
    target_gain = ((delta * (1.0 + attack_val)) + (self._e_slow * (1.0 + sustain_val))) / (self._e_fast + 1e-6)
    target_gain = max(0.0, min(4.0, target_gain))

    # 4. Smooth ramp & in-place application
    torch.linspace(self._prev_gain, target_gain, BLOCK_SIZE, out=self._ramp)
    self._prev_gain = target_gain

    out_buf.copy_(sig)
    out_buf.mul_(self._ramp).mul_(out_gain)
```

---

## 4. Auto Gain / Leveler (`AutoGain`)

### 4.1 Specification
- **File**: Append to `plugins/dynamics.py`
- **Class**: `AutoGain(Node)`
- **Category**: `"Utilities"`
- **Label**: `"Auto Gain / Leveler"`

### 4.2 DSP Formulation
- **Loudness Tracker**: Circular FIFO of block RMS values storing up to $N = \text{int}(\text{window\_s} \cdot f_s / \text{BLOCK\_SIZE})$ blocks.
- **Silence Gate**: If measured input RMS is below `silence_gate_db` (e.g. $-50\text{ dBFS}$), freeze gain adjustments to avoid pumping background noise during speech pauses or musical rests.
- **Target Calculation**:
  $$\text{Gain}_{\text{dB}} = \text{clamp}\left(\text{target\_db} - \text{RMS}_{\text{long\_term\_dB}}, -\text{max\_gain\_db}, +\text{max\_gain\_db}\right)$$
- **Smoothing**: 1-pole smoothing coefficient ($\tau \approx 500\text{ ms}$) on the linear multiplier applied via `torch.linspace`.
- **Window Change**: On `window_s` change, shrink `hist_count = min(hist_count, new_N)` and keep pointer; no reallocation.

### 4.3 Ports & Parameters
- **Inputs**: `in` (stereo)
- **Output**: `out` (stereo)
- **Parameters**:
  - `target_db`: float, default `-14.0`, min `-40.0`, max `0.0` (dBFS)
  - `window_s`: float, default `2.0`, min `0.2`, max `10.0` (seconds)
  - `max_gain_db`: float, default `18.0`, min `0.0`, max `36.0` (dB)
  - `silence_gate_db`: float, default `-50.0`, min `-80.0`, max `-20.0` (dBFS)

### 4.4 Buffers & State
```python
MAX_BLOCKS = int(10.0 * SAMPLE_RATE / BLOCK_SIZE)  # 10s maximum buffer
self._rms_history = torch.zeros(MAX_BLOCKS, dtype=DTYPE)
self._hist_ptr = 0
self._hist_count = 0
self._current_gain_lin = 1.0
self._prev_gain_lin = 1.0
self._ramp = torch.zeros(BLOCK_SIZE, dtype=DTYPE)
```

### 4.5 Implementation Pattern
```python
def start(self):
    self._rms_history.zero_()
    self._hist_ptr = 0
    self._hist_count = 0
    self._current_gain_lin = 1.0
    self._prev_gain_lin = 1.0

def process(self):
    sig = self.inputs["in"].get_tensor()
    target_db = self.params["target_db"].value
    window_s = self.params["window_s"].value
    max_gain_db = self.params["max_gain_db"].value
    silence_gate_db = self.params["silence_gate_db"].value

    out = self.outputs["out"].buffer

    # 1. Block RMS (mono sum)
    torch.mean(torch.pow(sig, 2.0), dim=0, out=self._mono)
    rms = float(torch.sqrt(torch.mean(self._mono)).item())
    rms_db = 20.0 * math.log10(max(rms, 1e-9))

    # 2. Silence gate
    if rms_db < silence_gate_db:
        # Freeze gain
        torch.linspace(self._prev_gain_lin, self._current_gain_lin, BLOCK_SIZE, out=self._ramp)
        self._prev_gain_lin = self._current_gain_lin
        out.copy_(sig)
        out.mul_(self._ramp)
        return

    # 3. Push to circular history
    N = int(window_s * SAMPLE_RATE / BLOCK_SIZE)
    N = max(1, min(N, MAX_BLOCKS))
    self._rms_history[self._hist_ptr] = rms
    self._hist_ptr = (self._hist_ptr + 1) % MAX_BLOCKS
    self._hist_count = min(self._hist_count + 1, N)

    # 4. Long-term RMS
    hist_slice = self._rms_history[:self._hist_count]
    long_rms = float(torch.sqrt(torch.mean(torch.pow(hist_slice, 2.0))).item())
    long_rms_db = 20.0 * math.log10(max(long_rms, 1e-9))

    # 5. Target gain in dB, clamp, convert to linear
    gain_db = target_db - long_rms_db
    gain_db = max(-max_gain_db, min(max_gain_db, gain_db))
    target_gain = 10.0 ** (gain_db / 20.0)

    # 6. Smooth (500 ms 1-pole) and ramp
    alpha = 1.0 - math.exp(-(BLOCK_SIZE / SAMPLE_RATE) / 0.5)
    alpha = min(1.0, alpha)
    self._current_gain_lin += alpha * (target_gain - self._current_gain_lin)

    torch.linspace(self._prev_gain_lin, self._current_gain_lin, BLOCK_SIZE, out=self._ramp)
    self._prev_gain_lin = self._current_gain_lin

    out.copy_(sig)
    out.mul_(self._ramp)
```

---

# Test & Verification Plan (Unit Tests)

The agent must create/extend tests in `tests/test_dynamics.py` and `tests/test_oscillators.py` following house conventions (anti-ghosting exact asserts + `tracemalloc` memory assertions).

### Test Matrix

| Test Function | Verification Assertions |
| :--- | :--- |
| `test_limiter_never_exceeds_ceiling` | Feed loud sine ($+6\text{ dBFS}$); assert $\max(|\text{out}|) \le \text{ceiling\_lin} + 10^{-5}$. |
| `test_limiter_anti_ghosting_mono` | Feed genuine $(1, 512)$ mono block into polluted $(2, 512)$ output; assert both channels match exactly. |
| `test_colored_noise_spectral_slopes` | FFT 128 blocks of Pink vs Blue noise; assert spectral slope decreases for Pink ($\approx -3\text{ dB/oct}$) and increases for Blue ($\approx +3\text{ dB/oct}$). |
| `test_transient_shaper_attack_boost` | Feed step impulse / percussive transient at controlled block offset; assert peak output with `attack=1.0` is larger than with `attack=0.0`. Use low-amplitude step to avoid output clamp saturation. |
| `test_autogain_levels_quiet_signal` | Feed continuous $-24\text{ dBFS}$ tone; verify output converges toward target $-14\text{ dBFS}$ within $\pm 0.5\text{ dB}$. |
| `test_autogain_freezes_on_silence` | Feed silence below gate; verify gain does not ramp to `+max_gain_db`. |
| `test_<node>_no_net_allocation` | 50-block `tracemalloc` loop on each new node asserting `growth < 64 * 1024` bytes. |

---

# Step-by-Step Execution Sequence for the Agent

1. **Step 1 — Implement Colored Noise**:
   - Update `plugins/oscillators.py` with `ColoredNoise` class and FIR impulse response generator.
   - Run `pytest tests/test_oscillators.py -v`.
2. **Step 2 — Implement Brickwall Limiter**:
   - Add `BrickwallLimiter` to `plugins/dynamics.py`.
   - Add limiter unit tests to `tests/test_dynamics.py`.
   - Run `pytest tests/test_dynamics.py -k "limiter" -v`.
3. **Step 3 — Implement Transient Shaper & Auto Gain**:
   - Add `TransientShaper` and `AutoGain` to `plugins/dynamics.py`.
   - Add unit tests for transient shaping and leveling in `tests/test_dynamics.py`.
4. **Step 4 — Full Regression Suite**:
   - Run complete test suite: `pytest tests/ -v`.
   - Verify plugin discovery and UI registry in `plugin_system.py`.

> **No structural changes needed** — plugin discovery is automatic; no edits to `commands.py`, `controller.py`, or `plugin_system.py` are required.