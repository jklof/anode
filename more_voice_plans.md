## 1. Overall Project Architecture & System Evaluation

ANode is a hybrid Python/C++ digital audio workstation and DSP graph engine. Its primary design target is real-time and near-real-time audio processing using PyTorch CPU tensors, native C++ DSP via `ctypes` FFI, and a PySide6 (Qt) graphical node canvas.

```
┌───────────────────────────────────────────────────────────────────────┐
│                           UI / Control Thread                         │
│   (PySide6 GraphScene, NodeItem, Custom Widgets, CommandHistory)      │
└──────────────┬──────────────────────────────────────────▲─────────────┘
  Commands &   │                                          │ Snapshots &
  Staged Params│                                          │ Telemetry
               ▼                                          │
┌───────────────────────────────┐        ┌────────────────┴─────────────┐
│      Engine Audio Loop        │        │     NRT Task Executor        │
│  (Real-Time Audio Thread)     │◄───────┤ (ThreadPoolExecutor / Streams│
│  - Executes Active Plan (DAG) │  Swap  │  - File I/O, WAV decoding    │
│  - Blocks of 512 @ 48 kHz     │ Assets │  - Neural weight loading     │
│  - Zero-alloc in-place ops    │        │  - PortAudio stream setup    │
│  - Native C-ABI ctypes calls  │        └──────────────────────────────┘
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│     Audio Device Callback     │  (PortAudio / sounddevice)
│  SPSC Lock-Free Ring Buffer   │
└───────────────────────────────┘
```

### 1.1 Key Architectural Strengths

1. **Strict Real-Time Hygiene and Thread Ownership (`AGENTS.md`)**
   * **Separation of Concerns:** The engine cleanly separates the high-priority real-time audio thread from non-real-time (NRT) background execution (`NRTExecutor`) and the Qt GUI thread. Operations that block (e.g., disk I/O in `FileRecorder`, model instantiation in `NamNode`, neural inference in `SwiftF0Node`, or device querying in `audio_devices.py`) are strictly kept off the audio processing path.
   * **Zero Steady-State Allocation in Python:** Critical paths use pre-allocated tensors, `Tensor.copy_()`, and in-place arithmetic (`mul_()`, `add_()`). The code avoids dynamic tensor creation in steady-state `process()` calls.
   * **Lock-Free Telemetry (`TelemetryRingBuffer` & `TelemetryDictRingBuffer`):** Audio threads push monitoring data into lock-free Single-Producer Single-Consumer (SPSC) ring buffers. If the UI falls behind, frames are dropped cleanly without stalling or blocking the audio thread.

2. **Tensor Format Discipline & Anti-Shrinkage Policies**
   * The global audio format is invariant: `BLOCK_SIZE = 512`, `SAMPLE_RATE = 48000`, `CHANNELS = 2`, `torch.float32` CPU tensors.
   * **PyTorch `out=` Hazard Defense:** A recurring trap in PyTorch is that calling functional operations with `out=` (e.g., `torch.mul(a, b, out=buf)`) silently resizes `buf` to the broadcast shape if an input is mono `(1, 512)`. ANode systematically uses `buf.copy_(sig)` followed by in-place operators (`buf.mul_(...)`), guaranteeing that destination buffers retain their static `(2, 512)` geometry.

3. **Coherent Command Pattern and Undo/Redo Engine**
   * Topology mutations occur via discrete command objects (`AddNodeCommand`, `DeleteNodeCommand`, `ConnectCommand`, `MultiMoveNodeCommand`).
   * **Authoritative Memento Capture:** `DeleteNodeCommand` does not capture node state from an out-of-date UI snapshot. Instead, it passes an empty holder dict into the command queue; the engine thread fills this holder at the exact execution boundary when the node is detached. Undo therefore restores state consistent with the audio thread's actual execution timeline.
   * **Cycle Detection:** `Graph.connect()` executes a depth-first search (`_can_reach()`) to reject self-loops and directed cycles *before* topological execution orders are compiled.

4. **Robust Native C-ABI Interface (`FFINode`)**
   * Clean, predictable C-ABI contract across shared libraries: `create()`, `destroy()`, `process()`, `set_param()`, `set_samplerate()`, `reset()`.
   * Python side enforces explicit ctypes `argtypes` and `restype` declarations, preventing pointer truncation on 64-bit systems.
   * Staged parameters synchronize at block boundaries through dirty flags (`_native_params_dirty`), preventing redundant FFI calls when values have not moved.

---

### 1.2 System-Level Weaknesses & Technical Debt

1. **GIL-Bound Python Processing Loop**
   * Even though heavy DSP (filters, vocoders, stretchers, neural models) runs in compiled C++, the top-level dispatch loop in `Engine._worker()` is executed in pure Python. Python's Global Interpreter Lock (GIL) means high background thread activity (such as `SwiftF0` worker threads or Qt GUI polling) can introduce timing jitter into `_tick_semaphore.acquire()`, occasionally threatening the 10.67 ms block deadline (512 samples at 48 kHz).

2. **PyTorch Tensor / Ctypes Pointer Overhead**
   * Every block, `FFINode.process()` retrieves pointer addresses via `processed_tensor.data_ptr()`, executes `ctypes.cast()`, and calls C functions through ctypes wrappers. While ctypes is relatively fast, doing multiple ctypes function calls per node for dozens of nodes per block introduces measurable interpreter call overhead compared to a pure C++ graph runner.

3. **Block-Quantized Control & Modulation**
   * While audio signals run at 48 kHz, control voltage (CV) parameters and modulation sockets (such as `in_mod`, `freq_in`, `drive_mod`) are evaluated once per 512-sample block by taking the block's mean or initial sample (`sig[0, 0].item()`). True audio-rate modulation (per-sample parameter variation) is only supported when explicitly implemented inside custom C++ nodes.

---

## 2. Comprehensive Deep Dive: Vocal Transformers

ANode contains one of the most sophisticated vocal processing stacks seen in an open-source Python/C++ modular environment. It includes four distinct pitch/formant shifting paradigms:

1. **`VocalTransformer` (`cpp/vocal_transformer.cpp`, `plugins/vocal_transformer.py`)**: Custom studio-grade phase vocoder featuring True-Envelope spectral estimation, 3-zone anatomical VTLN, and retune logic.
2. **`WorldVoiceTransformer` (`cpp/world_transformer.cpp`, `plugins/world_voice_transformer.py`)**: Real-time integration of the WORLD vocoder (CheapTrick + D4C + Synthesis2).
3. **`SignalsmithVocal` (`cpp/signalsmith_vocal.cpp`, `plugins/signalsmith_vocal.py`)**: Sub-band transient-aware phase-locked pitch/formant shifter based on Signalsmith Stretch.
4. **`RubberBandPitchShifter` (`plugins/rubberband_pitch_shifter.py`)**: High-level integration of the Rubber Band Library R3 engine via `pylibrb`.

Supporting these are neural and classical pitch tracking engines:
* **`PitchTracker` (`cpp/pitch_tracker.h`)**: Shared native header-only normalized square difference function (NSDF) tracker.
* **`SwiftF0Node` (`plugins/swift_f0_node.py`)**: Asynchronous neural F0 tracking running via ONNX Runtime.

---

### 2.1 VocalTransformer: Architectural & DSP Deep Dive

`VocalTransformer` is an in-house, custom-built vocal restoration and transformation engine. It is designed to solve the classical problems of digital voice transformation: chipmunk effects, phase smearing, metallic ringing, formant peak collapse, and unnatural glottal source spectral slopes.

```
Input Audio
    │
    ├────────────────────────────────────────────────────────────────┐
    │ (Stride-1.0 Read on Unvoiced Frames)                          │
    ▼                                                                ▼
Resampled Analysis Frame Read (Stride = Ratio)               Unshifted Dry Frame
    │ (4-point Catmull-Rom Hermite interpolation)                    │
    ▼                                                                ▼
Real FFT (N=2048 or 1024, Hann Window)                       Real FFT (N=2048 or 1024)
    │                                                                │
    ├──► True-Envelope Estimation (Roebel-Rodet Cepstral Lifter)     │
    │      └─ PCHIP Monotonic Cubic Hermite Upper Hull               │
    │                                                                │
    ├──► Spectral Peak Picking & Main-Lobe Gating                    │
    │                                                                │
    ├──► Formant Warping (Single-Pass 3-Zone VTLN Map)               │
    │                                                                │
    ├──► Phase Vocoder Synthesis (Laroche-Dolson Peak Locking)       │
    │      └─ Transient-Gated Phase Reset (Spectral Flux Trigger)    │
    │                                                                │
    ├──► Glottal Source Reshaping (Spectral Tilt + H1/H2 Balance)    │
    │                                                                │
    ├──► Pitch-Synchronous Aspiration Noise Injection (1.5-7 kHz)    │
    │                                                                │
    ▼                                                                ▼
Magnitude / Phase Recombination ◄────────────────────── Sibilant Crossfade
    │                                                    (Voiced/Unvoiced Gated)
    ▼
Inverse Real FFT (IFFT) + Synthesis Window (Hann / COLA Unity)
    │
    ▼
Overlap-Add (OLA) Output Ring Buffer
    │
    ▼
Latency-Aligned Output Readout (Fixed Latency L) + Crossfade with Dry Path
```

#### A. Analysis Stride & Resampling
Rather than running time-stretching followed by sample-rate conversion, `VocalTransformer` reads input audio directly from an input history ring buffer (`in_ring_`) using a fractional stride proportional to the total pitch shift ratio:
$$\text{stride} = \text{ratio} = 2^{\frac{\Delta \text{pitch}}{12}}$$
Reading is performed via 4-point Catmull-Rom cubic Hermite interpolation. This effectively stretches or compresses the waveform into the analysis window, translating pitch shifting directly into the frequency domain coordinates.

#### B. True-Envelope Estimation via Monotonic PCHIP Upper Hull
Standard cepstral liftering on raw speech spectra causes severe dips (up to 50 dB) in the inter-harmonic spectral valleys, distorting the envelope when formants shift. `VocalTransformer` implements a robust version of the Roebel-Rodet True-Envelope algorithm:
1. **Harmonic Peak Detection:** Finds local maxima in the linear magnitude spectrum $\text{mag}[k]$.
2. **PCHIP Cubic Hermite Upper Hull:** Evaluates monotonic cubic Hermite interpolation between the peak log-magnitudes using Fritsch-Carlson slope limiting. This eliminates the slope discontinuities of piecewise-linear hulls and avoids Runge-type polynomial ripples.
3. **Iterative Symmetrical Cepstral Liftering:** Runs 4 iterations of forward/inverse FFT between log-magnitude spectrum and cepstrum:
   $$A_{i+1}(k) = \max(A_i(k), \text{IFFT}(\text{lifter}(q) \cdot \text{FFT}(A_i(k))))$$
   Crucially, liftering is applied symmetrically to both positive and negative quefrencies ($q$ and $N-q$), preventing imaginary leakage into the reconstructed envelope.
4. **Adaptive Lifter Cutoff for Gender Morphing:** As `gender_morph` increases towards +1.0 (feminine), vocal tract acoustic losses increase and formant bandwidths broaden. The lifter cutoff is dynamically reduced from 48 down to 36 points (in 2048-point mode), broadening the synthesized formant resonances.

#### C. Three-Zone Anatomical Vocal Tract Length Normalization (VTLN)
Uniform frequency scaling (multiplying all frequencies by a single factor) sounds unnatural because the human vocal tract does not scale uniformly across sexes. The male pharynx is roughly 20–25% longer than the female pharynx, while the oral cavity differs by only 8–10%.

`VocalTransformer` solves this using a custom piecewise 3-zone warping function ($f \to f_{\text{source}}$):
* **Zone 1 ($< 1000\text{ Hz}$, $F_1$ / Vowel Aperture):** Tapered warping ($0.6 + 0.4 \cdot \frac{f}{1000}$) to stabilize vowel identity and prevent vowel formant migration.
* **Zone 2 ($1000\text{–}2500\text{ Hz}$, $F_2$ / Oral Cavity & Tongue Body):** Warped at $0.30\times$ the base shift ($+2.1\text{ st}$ for a $+3.0\text{ st}$ nominal shift), matching oral cavity scaling.
* **Zone 3 ($2500\text{–}5000\text{ Hz}$, $F_3/F_4$ / Pharyngeal & Throat Length):** Warped at $1.50\times$ the base shift ($+2.75\text{ st}$ net), emulating anatomical pharyngeal variation.
* **Above $5000\text{ Hz}$:** Linear compression up to Nyquist, preserving anatomical air and sibilant boundaries.

All three warps are evaluated in a single algebraic mapping stage:
$$\text{pos} = \text{vtln\_warp\_bin}[k] \cdot \left(\frac{\text{ratio}}{r_f}\right)$$
This single-pass continuous evaluation avoids discrete re-sampling quantization and cascading interpolation low-pass filters.

#### D. Glottal Source Reshaping ($H_1/H_2$ Balance & Spectral Tilt)
Shifting vocal formants upward without altering the excitation produces a "pinched," buzzy timbre. `VocalTransformer` restores natural glottal acoustics:
* **Log-Octave Spectral Tilt:** Applies a logarithmic tilt anchored at 1 kHz ($-2.5\text{ dB/octave}$ for $+1.0$ gender morph, $+2.5\text{ dB/octave}$ for $-1.0$).
* **Dynamic $H_1$ Harmonic Boost & $H_2$ Attenuation:** For feminine transformations, female vocal cords exhibit less abrupt closure, resulting in a dominant fundamental relative to the second harmonic ($H_1 \gg H_2$). The algorithm tracks the fundamental $H_1$ bin (verified with an overtone search at $2H_1 \pm 2$ bins or guided by external F0) and applies up to $+6.8\text{ dB}$ on $H_1$ and $-4.5\text{ dB}$ on $H_2$.

#### E. Voiced/Unvoiced-Gated Sibilant Bypass
A major flaw in naive pitch shifters is that unvoiced consonants (/s/, /t/, /k/, /sh/) get pitch-shifted and formant-warped, sounding like metallic chirps.
* The processor calculates a composite voicing probability $P_{\text{voiced}} \in [0, 1]$ based on the low-to-high frequency energy ratio ($<1.5\text{ kHz}$ vs. $>3.5\text{ kHz}$) and spectral flatness.
* On unvoiced frames ($P_{\text{voiced}} \to 0$), the algorithm reads an **unshifted, true-stride dry frame** from `in_ring_` and blends its high-frequency spectrum ($3.5\text{–}5.5\text{ kHz}$ raised cosine transition) directly into the output complex spectrum.
* Because the dry sibilants are only injected when $P_{\text{voiced}} < 0.25$, vowels never comb-filter against the dry audio.

#### F. Pitch-Synchronous Glottally Modulated Aspiration Noise
Natural breathiness is not stationary tape hiss; vocal cord leakage pulses synchronously with glottal opening.
* In the $1.5\text{–}7.0\text{ kHz}$ tract band, uniform pseudo-random noise is generated via an inlined `xorshift32` generator and scaled by the True-Envelope.
* If the frame is voiced and $H_1$ is located at bin $k_0$, the noise spectrum $N(k)$ is blended with spectral sidebands:
  $$N_{\text{mod}}(k) = \frac{N(k) + \beta N(k - k_0) + \beta N(k + k_0)}{\sqrt{1 + 2\beta^2}}$$
  In the time domain, this frequency-domain convolution is equivalent to amplitude modulation:
  $$m(t) = 1 + 2\beta \cos(\omega_0 t)$$
  The breathiness naturally pulses with the glottal period.

#### G. Transient-Gated Phase Reset
Laroche-Dolson peak locking preserves phase coherence for stationary harmonics, but across plosives and percussive transients (/p/, /b/, /t/, /k/), phase integration across a 42.7 ms window smears energy over time.
* `VocalTransformer` monitors half-wave rectified relative spectral flux:
  $$\text{flux}_{\text{rel}} = \frac{\sum_{k \ge 4} \max(0, \text{mag}_m[k] - \text{mag}_{m-1}[k])}{\sum_{k \ge 4} \text{mag}_m[k] + \epsilon}$$
* When an onset occurs ($\text{flux}_{\text{rel}} > 0.4$ and sufficient energy), phase-vocoder phase accumulation is suspended for that frame. Synthesis phases are reset to the raw analysis phases ($\phi_{\text{syn}}[k] = \phi_{\text{ana}}[k]$), preserving crisp consonant attacks without dispersion.

---

### 2.2 WorldVoiceTransformer: WORLD Vocoder Integration

`WorldVoiceTransformer` wraps the canonical C++ implementation of WORLD (Morise et al.), a high-quality vocoder based on pitch-synchronous spectral analysis and synthesis:
* **CheapTrick:** Accurate spectral envelope estimation using a pitch-adaptive window that removes excitation ripple.
* **D4C:** Band-aperiodicity estimator evaluating the degree of turbulence across sub-bands.
* **Synthesis2:** Real-time synthesis combining minimum-phase impulse responses for the periodic component with band-filtered noise for aperiodicity.

```
Incoming Stereo Audio
    │
    ├──► Downmix to Mono
    │
    ▼
5 ms Analysis Cadence Generator (240 samples @ 48 kHz)
    │
    ├──► Silence Gate (RMS < 1e-4 -> output zero envelope)
    │
    ├──► F0 Pitch Detection (Internal NSDF Tracker or External CV)
    │
    ├──► CheapTrick Spectral Analysis (Single Frame, f0_length = 1)
    │
    ├──► D4C Aperiodicity Estimation (Evaluated at Half-Rate: every 10 ms)
    │
    ├──► Timbre Post-Processing:
    │      ├─ Bilinear All-Pass VTLN Formant Warp
    │      ├─ Glottal Spectral Tilt Multipliers
    │      ├─ Dynamic H1/H2 Glottal Peak Shaping
    │      └─ Aperiodicity Breathiness Injection (1.5 - 7 kHz)
    │
    ▼
WORLD Real-Time Synthesizer (Synthesis2)
    │  (Generates wet chunks into ring buffer)
    ▼
Fixed-Latency Readout (L = 1024 samples / 21.3 ms) + Crossfade with Dry Path
```

#### Notable Engineering Optimizations in `world_transformer.cpp`:
1. **Shared Mono Analysis Stream:** Rather than running CheapTrick and D4C on both channels (which would double CPU load), audio is downmixed to mono for analysis. The resulting parameters ($F_0$, $S_p$, $A_p$) drive two independent `WorldSynthesizer` instances with distinct random seeds. This produces natural stereo width while keeping CPU usage low. If the input is mono, output channels are bit-copied for consistency.
2. **Half-Rate D4C Execution:** Profiling revealed that D4C consumes over 60% of total frame compute time. Because band-aperiodicity changes much more slowly than the spectral envelope or fundamental frequency, D4C is computed every second hop (every 10 ms), caching $A_p$ across alternate frames. This reduces CPU usage by nearly 30% with negligible perceptual difference.
3. **Fixed Low-Latency Pipeline:** Latency is fixed at $L = 1024$ samples (21.3 ms @ 48 kHz). To allow an analysis window size of $N_{\text{FFT}} = 2048$ without starving the synthesizer, the $F_0$ floor is clamped to $71.0\text{ Hz}$.

---

### 2.3 SignalsmithVocal: Sub-Band Transient-Aware Stretcher

Based on the modern Signalsmith Stretch library (v1.3.2), `SignalsmithVocal` takes a fundamentally different mathematical approach from classical STFT phase vocoders and source-filter models:
* **Sub-Band Phase Vocoding with Modified Linear Transforms:** Instead of full-band FFTs, it uses multi-resolution sub-bands. Transients are detected and preserved within individual sub-bands, preventing the "phasiness" and transient smearing typical of traditional vocoders.
* **No $F_0$ Tracking Required:** Unlike WORLD or `VocalTransformer`, Signalsmith operates purely as a spectral transform without needing an explicit pitch tracker. As a result, it never suffers from octave jumps, tracking dropouts, or voicing detection failures.
* **True Comb-Free Wet/Dry Crossfade:** Because Signalsmith introduces internal algorithmic latency ($L = \text{inputLatency}() + \text{outputLatency}() = 5760\text{ samples}$ in Studio mode, $1920\text{ samples}$ in Live mode), the wrapper maintains an internal circular dry delay line (`dry_`). When the user adjusts `mix`, the wet output is blended with the *time-aligned* dry signal, eliminating comb-filtering artifacts across the entire crossfade range.

---

### 2.4 Comparative Trade-off Matrix: Vocal Transformation Engines

| Metric / Dimension | `VocalTransformer` | `WorldVoiceTransformer` | `SignalsmithVocal` | `RubberBandPitchShifter` |
| :--- | :--- | :--- | :--- | :--- |
| **Core DSP Topology** | Resampled Phase Vocoder + True-Envelope + VTLN | Source-Filter Vocoder (CheapTrick + D4C + Synthesis2) | Sub-band Transient Phase-Locked Stretcher | Multi-engine Phase Vocoder / Time-domain Hybrid (R3) |
| **Algorithmic Latency** | **Studio:** 192 ms (9216 spls)<br>**Live:** 53 ms (2560 spls)<br>**Ultra:** 42 ms (2048 spls) | **Fixed:** 21.3 ms (1024 spls)<br>(End-to-end $\approx 32.4\text{ ms}$) | **Studio:** 120 ms (5760 spls)<br>**Live:** 40 ms (1920 spls) | Variable start delay (~30–60 ms) + 2 blocks priming |
| **F0 Tracking Dependency** | Internal NSDF or External CV (`f0_in`). Can run tracking-free for manual shifts. | **Mandatory** (Internal NSDF or External CV). Unvoiced frames fall back to noise excitation. | **None** (Blind spectral sub-band transformation). | Internal pitch tracker for formant calculation (R3 engine). |
| **Formant Quality & VTLN** | **Highest customizability:** Decoupled F1, 3-zone anatomical warp, $H_1/H_2$ reshaping. | **High naturalness:** Bilinear all-pass warp, glottal tilt, $H_1/H_2$ boost. | **Very high:** Sub-band formant shifting + gender offset ($+3.5\text{ st/unit}$). | **Good:** Automatic formant tracking or explicit ratio scaling. |
| **Consonant / Sibilant Handling**| **Exceptional:** V/UV-gated dry sibilant bypass ($3.5\text{–}5.5\text{ kHz}$). Consonants stay crisp. | **Medium:** Unvoiced frames synthesized via white-noise excitation; can sound slightly raspy. | **Excellent:** Sub-band phase locking naturally preserves transients. | **Good:** Real-time transient preservation mode. |
| **Vocal Pitch Snapping / Autotune** | **Integrated:** Scale snap, MIDI target, exponential glide, vibrato. | None built-in (relies on upstream CV modulation). | None built-in (relies on upstream CV modulation). | None built-in (relies on upstream CV modulation). |
| **Best Use Case** | Lead vocal processing, hard-tune FX, extreme gender transformation, speech conversion. | Low-latency live vocal performance, broadcast voice masking, robotic/synthetic timbre FX. | Studio pitch transposition, backing vocals, natural formant shifting with zero tracking artifacts. | Offline/NRT time stretching and simple pitch shifting. |

---

## 3. Deep Technical & Code-Level Audit

During a granular line-by-line review of the native C++ codebases and Python plugin wrappers, several subtle numerical behaviors, edge cases, and design details were identified:

### 3.1 `cpp/vocal_transformer.cpp`

1. **Catmull-Rom Boundary Clamping in Frame Resampling**
   * *Implementation:*
     ```cpp
     const double fpos = std::fmod(static_cast<double>(hop_) * (double)frame_index_[c], (double)kRingSize);
     // ...
     const int i0 = (int)std::floor(pos);
     const float ym1 = in_ring_[c][(i0 - 1) & kRingMask];
     const float y0  = in_ring_[c][i0 & kRingMask];
     const float y1  = in_ring_[c][(i0 + 1) & kRingMask];
     const float y2  = in_ring_[c][(i0 + 2) & kRingMask];
     ```
   * *Assessment:* The modulo wrapping prior to integer casting avoids undefined behavior when casting large doubles to integers during long-running sessions. Bitwise masking with `kRingMask` (`16384 - 1`) handles negative sample indices (`i0 - 1`), preventing out-of-bounds reads.

2. **PCHIP Hermite Slope Limiting Stability**
   * *Implementation:*
     ```cpp
     if (dA * dB <= 0.0f) {
         d = 0.0f;
     } else {
         const float denom = dA + dB;
         d = (denom != 0.0f) ? (2.0f * dA * dB / denom) : 0.0f;
     }
     ```
   * *Assessment:* The check `dA * dB <= 0.0f` handles local extrema by setting the tangent to 0. The guard `denom != 0.0f` defends against floating-point anomalies where two small positive subnormal numbers might otherwise cause division by zero.

3. **Sub-Rumble Protection in $H_1/H_2$ Reshaping**
   * *Assessment:* Earlier vocoder revisions experienced issues where low-frequency sub-rumble (< 50 Hz) or microphone handling noise was incorrectly identified as the fundamental harmonic $H_1$, causing mud to be amplified by $+6.8\text{ dB}$ while attenuating the real fundamental. The current implementation protects against this with two constraints:
     * Candidates are restricted to the biological window ($80\text{–}400\text{ Hz}$).
     * A peak is only promoted to $H_1$ if an overtone exists at $2k \pm 2$ bins.

4. **Transient Phase Reset Threshold Tuning**
   * *Implementation:*
     ```cpp
     const float flux_rel = flux / (energy + 1.0e-6f * static_cast<float>(num_bins_));
     is_transient = (flux_rel > 0.4f) && (energy > 0.05f);
     ```
   * *Assessment:* Steady-state voiced speech exhibits a relative spectral flux between 0.002 and 0.02, whereas plosive bursts (/p/, /b/) exceed 0.55. The threshold of 0.4 provides a substantial margin of safety, preventing false-positive phase resets on voiced vowels while reliably catching unvoiced plosive onsets.

---

### 3.2 `cpp/world_transformer.cpp`

1. **Persistent Memory Ownership in `AddParameters`**
   * *Implementation:*
     ```cpp
     slot.sp_ptr = slot.sp.data();
     slot.ap_ptr = slot.ap.data();
     if (AddParameters(&slot.f0, 1, &slot.sp_ptr, &slot.ap_ptr, &st.synth) != 1) {
         return;
     }
     ```
   * *Assessment:* WORLD's `AddParameters` takes `double**` and **retains the pointer addresses internally** inside `synth->spectrogram[pointer]` to read across multiple future `Synthesis2()` hops. Passing pointers to local stack arrays causes dangling pointer bugs. `WorldTransformerProcessor` handles this correctly: `sp_ptr` and `ap_ptr` are persistent members of `FrameSlot`, allocated inside `an_.slots` which has a depth of 64 frames (320 ms), outliving the synthesizer's internal lookahead window.

2. **Aperiodicity Injection Headroom**
   * *Implementation:*
     ```cpp
     double a = slot.ap[k] + injection;
     if (a > 1.0) a = 1.0;
     slot.ap[k] = a;
     ```
   * *Assessment:* WORLD's synthesizer treats aperiodicity $A_p$ as a power-complementary split between periodic pulse excitation ($1 - A_p$) and noise excitation ($A_p$). Clamping $A_p \le 1.0$ guarantees that increasing `breathiness` never clips the synthesizer's excitation generator.

---

### 3.3 `cpp/pitch_tracker.h`

1. **Continuity Scoring & Harmonic Unwinding**
   * *Implementation:*
     ```cpp
     // Continuity scoring
     const float st = 12.0f * std::log2(f / last_accepted_f0_);
     const float pen = st > 0.0f ? 0.030f * st : -0.012f * st;
     const float score = v - pen;
     // Harmonic unwinding
     for (int m = 2; m <= 6; ++m) {
         const int cand = best_tau * m;
         // ...
     }
     ```
   * *Assessment:* The asymmetric penalty heavily penalizes sudden upward frequency jumps while remaining more tolerant of downward slides, matching typical vocal dynamics. Harmonic unwinding inspects integer multiples of the candidate lag ($m \in [2, 6]$); if a higher peak exists at a multiple lag, the tracker unwinds the fundamental, avoiding octave-high pitch estimation errors.

2. **Downsampling & Anti-Aliasing Architecture**
   * The tracker decimates by 4:1 (48 kHz $\to$ 12 kHz) using an internal second-order Butterworth lowpass filter with a cutoff at 1200 Hz. Because human singing and speech fundamentals rarely exceed 1000 Hz, filtering out higher harmonics before computing the autocorrelation simplifies the correlation surface and avoids octave-jumping artifacts.

---

### 3.4 `plugins/swift_f0_node.py`

1. **Alias-Free 3:1 Decimation Kernel**
   * *Implementation:*
     `swift_f0_node.py` replaces general-purpose resampling libraries with a 129-tap Blackman-windowed sinc FIR filter run through `torch.nn.functional.conv1d(..., stride=3)`.
   * *Performance & Accuracy:* This integer decimation (48 kHz $\to$ 16 kHz) is alias-free with unity DC gain and $>70\text{ dB}$ stopband attenuation above 8 kHz. It avoids OpenMP deadlocks with the audio thread and processes in sub-millisecond times on CPU.

2. **Audio-to-Worker Ring Buffer Pool Indexing**
   * The node transfers audio from the audio thread to the background ONNX inference worker using pre-allocated slots (`_mono_pool = [np.zeros(...) for _ in range(64)]`). The real-time thread pushes integer indices into `_audio_queue` rather than allocating and copying NumPy arrays, keeping the audio processing path allocation-free.

---

## 4. Specific Issues Identified & Recommended Fixes

While the codebase is stable and well-tested, the following edge cases and potential improvements were identified:

### Issue 1: High Pitch Ratio Frame Starvation in `VocalTransformer`

**Location:** `cpp/vocal_transformer.cpp`, lines 403–415
```cpp
const double span = static_cast<double>(fft_size_) * (double)std::max(ratio, 1.0f);
int frames_this_call = 0;
while (static_cast<double>(hop_) * (double)frame_index_[c] + span
           <= (double)total_received_[c] + 1e-6
        && frames_this_call < max_frames_per_call_) {
    process_frame(c, ratio);
    ++frames_this_call;
}
```

**Analysis:**
When pitch shifting upward by +24 semitones ($\text{ratio} = 4.0$), `span` equals $2048 \times 4 = 8192$ samples. The condition `frames_this_call < max_frames_per_call_` restricts the loop to a maximum of 4 frames per audio callback.
In steady state, an audio block of 512 samples advances `total_received_` by 512 samples. At $\text{hop} = 256$, exactly 2 frames *should* be processed per block ($512 / 256 = 2$).
However, if rapid pitch modulation swings upward and downward, `frame_index_` can fall behind. If `max_frames_per_call_` is capped at 4, it can take multiple blocks to drain a backlog, potentially causing the trailing OLA readout to read unaccumulated frames, leading to momentary dropouts.

**Recommended Fix:**
Dynamically calculate the burst limit based on the current lag:
```cpp
const int backlog = static_cast<int>(
    ((total_received_[c] - span) / hop_) - frame_index_[c]
);
const int burst_limit = std::max(max_frames_per_call_, std::min(backlog + 1, 8));
```

---

### Issue 2: Fixed $F_0$ Floor in `WorldVoiceTransformer` Restricts Bass Vocals

**Location:** `cpp/world_transformer.cpp`, line 49
```cpp
constexpr double kFoFloor = 71.0;
```

**Analysis:**
The $F_0$ floor is hardcoded to $71.0\text{ Hz}$ because WORLD requires an FFT size of at least:
$$N_{\text{FFT}} \ge 3 \cdot \frac{f_s}{f_{0,\text{floor}}} + 1$$
At $f_s = 48000\text{ Hz}$ and $f_{0,\text{floor}} = 71.0\text{ Hz}$, $3 \times 48000 / 71 \approx 2028$, fitting into $N_{\text{FFT}} = 2048$.
However, low male bass vocals regularly reach $C_2 \approx 65.4\text{ Hz}$ or $A_1 \approx 55\text{ Hz}$. When a vocalist sings below $71\text{ Hz}$, `WorldVoiceTransformer` flags the frame as unvoiced ($f_0 = 0.0$), causing the synthesizer to drop into unvoiced noise excitation and creating vocal dropouts.

**Recommended Fix:**
Provide an explicit "Bass Voicing" option or adapt $N_{\text{FFT}}$:
* When low tracking is desired, allow $N_{\text{FFT}} = 4096$, lowering the floor to $40\text{ Hz}$ (covering deep bass vocals down to $E_1 \approx 41.2\text{ Hz}$).
* Alternatively, for frequencies between $50\text{ Hz}$ and $71\text{ Hz}$, clamp $f_0$ to $71.0\text{ Hz}$ internally for the CheapTrick envelope analysis while preserving the true $f_0$ for synthesis pitch generation.

---

### Issue 3: Incomplete Clean-up of External F0 Disconnect in `VocalTransformer`

**Location:** `plugins/vocal_transformer.py`, lines 403–420

**Analysis:**
When `f0_in` is disconnected, `VocalTransformer` resets mode, frequency, and voicing:
```python
elif self._was_f0_connected:
    self.lib.set_param(self.dsp_handle, self.PARAM_EXT_F0_MODE_ID, self.EXT_F0_MODE_OFF)
    self.lib.set_param(self.dsp_handle, self.PARAM_EXT_F0_HZ_ID, 0.0)
    self.lib.set_param(self.dsp_handle, self.PARAM_EXT_VOICED_ID, 0.0)
    self._last_ext_mode = self.EXT_F0_MODE_OFF
    self._was_f0_connected = False
```
However, in `cpp/vocal_transformer.cpp`, switching `ext_f0_mode_` from `1` (External) back to `0` (Internal) does not reset the internal `PitchTracker` instance (`pitch_tracker_.reset()`). The internal tracker's history buffers and hangover counters still hold state from when external mode was first engaged. As a result, the first few frames after disconnecting `f0_in` can jump to an old pitch value before settling.

**Recommended Fix:**
In `cpp/vocal_transformer.cpp`:
```cpp
case 15: {
    int m = static_cast<int>(v);
    if (m != ext_f0_mode_) {
        ext_f0_mode_ = static_cast<float>(m);
        reset_tracker(); // Reset both external and internal tracking state
    }
    break;
}
```

---

## 5. Summary & Verdict

ANode is a well-engineered project that avoids the common pitfalls of Python-based audio applications (such as real-time memory allocations, unbounded worker blocking, and UI-induced audio dropouts). 

Its vocal transformer subsystem is particularly strong:
* **`VocalTransformer`** is a production-ready vocal processor that combines True-Envelope estimation, anatomical VTLN, formant-preserving envelope replacement, and pitch-synchronous aspiration into a coherent, low-artifact phase vocoder.
* **`WorldVoiceTransformer`** successfully integrates the reference WORLD vocoder into a real-time, low-latency streaming context through downmixed mono analysis, half-rate aperiodicity estimation, and stereo synthesis.
* **`SignalsmithVocal`** provides a complementary sub-band approach that requires no pitch tracking, making it an ideal option for polyphonic, backing, or extreme-range transposition.

Addressing the minor items identified above (burst-limit headroom on high upward shifts, F0 floor handling for bass vocals, and internal tracker state resets on external CV disconnects) will further solidify the engine's reliability in demanding real-time vocal scenarios.






 User 6:55 PM

what better voice changers could be implemented? rvc or similar or  mixed ai and traditional ?


In an engine with ANode’s architectural constraints (real-time 512-sample CPU blocks, zero-allocation processing loops, and low-latency targets), selecting the right voice conversion (VC) architecture is a trade-off between **identity transformation fidelity**, **latency**, **compute cost**, and **vocal expressiveness**.

Traditional vocoders (like ANode’s `VocalTransformer` and `WorldVoiceTransformer`) excel at **instantaneous response, deterministic control, and preserving natural dynamics**, but they cannot transform the speaker’s fundamental timbre into someone else's. Pure AI models (like RVC) excel at **complete voice identity replacement**, but introduce significant latency, GPU compute burdens, and phoneme dropouts.

A technical evaluation of the best voice changer architectures to implement next covers pure neural, hybrid, and modern DSP approaches, along with how they integrate into ANode.

---

### Comparative Architecture Matrix

| Architecture | Latency | Compute Target | Identity Swap Quality | Expression / Dynamics | Risk of Artifacts / Hallucinations | Best Suited For |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard RVC v2** (HuBERT + VITS/HiFi-GAN) | 150 – 350 ms | High (CUDA / DirectML GPU required) | **Flawless** (exact person cloning) | Moderate (squashes whispers, breathing) | High (dropping consonants, pitch warbles) | Live stream avatars, non-critical monitoring |
| **Streaming Causal VC** (e.g., Causal ContentVec + Vocos) | 30 – 60 ms | Medium (CPU ONNX / DirectML) | **Very High** | High | Low–Medium | Fast voice transformation |
| **DDSP Hybrid** (Differentiable DSP Harmonic + Noise) | **10 – 20 ms** | **Low (CPU-friendly, C++/ONNX)** | **High** (instrument/voice style) | **Exceptional** (100% dynamic retention) | **Zero** (no vocoder hallucinations) | Live monitoring, singing, instrument cross-synthesis |
| **Excitation-Replacement Hybrid** (DSP VTLN + Neural Glottal Pulse) | **10 – 25 ms** | **Very Low (CPU C++)** | **Medium–High** | **High** | **Zero** (deterministic consonants) | Vocal cleanup, gender morphing without buzz |
| **Signalsmith / Phase Vocoder + Neural Timbre Restoration** | 40 – 120 ms | Low–Medium (CPU) | **Medium** (restyled target) | **Exceptional** | Very Low | Studio vocal production |

---

### 1. Pure AI: Streaming Causal RVC (Causal ContentVec + Vocos)

#### Why standard RVC is flawed in real-time DAWs:
Standard RVC (Retrieval-based Voice Conversion) is built around **HuBERT** or **ContentVec**, which are non-causal vision/audio transformers with large multi-head attention context windows. They expect chunks of 200–500 ms with bidirectional lookahead. If forced into a DAW buffer:
1. Buffering introduces **200+ ms latency**, making real-time live performance or in-ear monitoring impossible.
2. The HiFi-GAN vocoder can glitch on per-block chunk boundaries unless complex overlapping OLA state machines are maintained.

#### The Modern Solution: "Streaming Fast-VC"
To build a high-performance RVC-class node in ANode:
1. **Causal Content Encoder:** Use a causal, streaming-friendly speech encoder trained without future lookahead (e.g., *Causal ContentVec* or *WavLM Streaming* quantized to INT8/FP16).
2. **Vocos instead of HiFi-GAN:** Replace HiFi-GAN with **Vocos**. Vocos uses ConvNeXt backbones to predict Fourier coefficients (real/imaginary STFT components) rather than generating audio sample-by-sample with dilated convolutions. It runs **5–10× faster than HiFi-GAN** on CPU and easily executes in real time via ONNX Runtime.
3. **Integration into ANode:**
   * Build a dedicated `StreamingRvcNode` structured similarly to `SwiftF0Node`:
   * Audio thread pushes 512-sample blocks into an SPSC ring buffer.
   * A dedicated streaming background thread runs ONNX Runtime (`onnxruntime-directml` or CPU OpenVINO/TensorRT) over rolling 40 ms chunks.
   * Latency drops from ~300 ms down to **40–60 ms**.

---

### 2. The Ideal Hybrid: DDSP (Differentiable Digital Signal Processing)

If the goal is **musical responsiveness, zero latency, singing expressiveness, and CPU-only execution**, DDSP is the strongest architecture available.

```
Incoming Audio (Vocal)
       │
       ├───────────────────────────────────────┐
       ▼                                       ▼
Pitch Tracking (SwiftF0 or NSDF)       Loudness / RMS Extractor
       │ (F0 in Hz)                            │ (dB envelope)
       └───────────────────┬───────────────────┘
                           ▼
              Lightweight Control Mapping
        (Tiny MLP or GRU: ~50k parameters in ONNX)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Harmonic Amplitudes          Noise Band Filter Gains
  (Amplitudes of 60 partials)     (Filtered pink/white noise)
             │                           │
             ▼                           ▼
     Additive Synthesis          Noise Convolution
     (Bank of Sinusoids)         (Aspiration / Breath)
             │                           │
             └─────────────┬─────────────┘
                           ▼
                Synthesized Voice Output
```

#### Why DDSP is exceptionally suited for ANode:
1. **Deterministic & Glitch-Free:**
   Standard neural vocoders guess audio waveforms, which causes phoneme hallucinations, robotic buzzing when tracking drops, or dropped consonants. DDSP does **not** guess raw audio; it predicts the physical parameters of an additive synthesizer:
   * Harmonic frequencies: $f_k = k \cdot F_0$
   * Harmonic amplitudes: $A_k(t)$
   * Noise transfer function: $H_{\text{noise}}(\omega, t)$
2. **True Low Latency (10.67 ms / 1 Block):**
   The neural network is tiny (a small 2-layer GRU or MLP, <100k parameters). It takes scalar $F_0$ and loudness as inputs and outputs synthesis parameters. The actual audio synthesis is **pure C++ additive synthesis**, running inside ANode's 512-sample processing block with **zero algorithmic lookahead latency**.
3. **Flawless Dynamic Range:**
   Because volume is explicitly controlled by the physical loudness curve, whispering, shouting, vibrato, and dynamic expression translate directly to the target voice without training-data normalization compression.

---

### 3. Source-Filter Hybrid: Neural Glottal Pulse + 3-Zone VTLN

Currently, `VocalTransformer` uses an *excitation shaper* (spectral tilt + $H_1/H_2$ boost) combined with an analytical True-Envelope. While clean, phase vocoders can struggle to replicate the acoustic richness of a real human glottal flow wave (Liljencrants-Fant model).

A powerful hybrid approach:
1. **Classical Decomposition:**
   * Extract the vocal tract spectral envelope $V(\omega)$ using ANode’s existing True-Envelope (Roebel-Rodet with PCHIP).
   * Inverse filter the input vocal to extract the raw glottal excitation $E(t) = \text{IFFT}(X(\omega) / V(\omega))$.
2. **Neural Excitation Synthesis:**
   * Instead of generating artificial noise or relying on stretched harmonic phase locking, run a microscopic 1D WaveNet/ConvNet (conditioned on target pitch $F_0$) to generate the **target speaker's glottal pulse train**.
3. **Classical Filtering:**
   * Filter the neural excitation back through ANode’s C++ 3-zone anatomical VTLN filter.
4. **Why this works:**
   * Consonants and sibilants are already separated by ANode's $V/UV$ detector, so the neural model only handles the periodic vocal cord vibration.
   * Eliminates the "buzz" and "metallic" artifacts of phase vocoders while maintaining 100% clean consonant articulation.

---

### 4. DSP + Neural Post-Restyler / De-Vocoder

Rather than generating voice from scratch with AI, use traditional DSP to do the heavy mathematical shifting, then use a neural network strictly for **spectral restoration and artifact removal**:

```
Microphone In 
     │
     ▼
ANode C++ Engine (SignalsmithVocal / VocalTransformer)
  - Shifts pitch by +12 st
  - Warps formants by +3.5 st
  - Zero/fixed low latency
     │
     ▼ (Contains slight vocoder artifacts, synthetic phase)
Neural Post-Restyler (Lightweight Streaming Diffusion / UNet)
  - Trained on paired: [Vocoder-Processed Voice ──► Natural Human Voice]
  - Restores natural harmonic phase alignment
  - Injects realistic vocal fry, micro-tremor, and natural breathiness
     │
     ▼
Clean, Natural Voice Out
```

* **Advantage:** If the neural network drops out or experiences buffer underruns, the system cleanly degrades to standard DSP audio rather than producing digital clicks or silence.
* **Training Simplicity:** This model is easy to train because it only learns an image-to-image or STFT-to-STFT residual enhancement mapping.

---

### Recommended Implementation Roadmap for ANode

To implement advanced voice transformation while preserving real-time stability:

#### Phase 1: Implement a `DDSPVoiceNode` (Highest ROI for Low Latency)
* **Architecture:** $F_0$ (from `SwiftF0` or `PitchTracker`) + RMS Loudness $\to$ Tiny ONNX Runtime GRU $\to$ C++ Additive Sinusoidal Oscillator Bank + Filtered Noise generator.
* **Why first:** It provides custom voice identity cloning (e.g., training a model on a specific singer's voice) with **<15 ms latency** and zero GPU requirement.

#### Phase 2: Implement a Causal `StreamingRvcNode` (For Complete Identity Swapping)
* **Architecture:** Causal ContentVec encoder + streaming Vocos vocoder wrapped in a background streaming worker (`spawn_stream`), communicating via `SPSCRingBuffer`.
* **Hardware:** DirectML execution provider on Windows, allowing both NVIDIA and AMD/Intel GPUs to accelerate the neural vocoder without breaking the CPU audio thread.

#### Phase 3: Enhance the Existing Vocal Transformers
* Add a **Liljencrants-Fant (LF) glottal model generator** to `cpp/vocal_transformer.cpp` as an alternative excitation mode to replace the raw linear prediction excitation on synthesized voiced frames.





what better voice changers could be implemented? rvc or similar or  mixed ai and traditional ?

Yes, there are **vast ecosystems of ready-made models** you can download and use immediately, as well as **established open-source training pipelines** that allow you to train a custom voice with only 5–15 minutes of clean audio.

Depending on which path you take for ANode, here is where ready models exist and how custom training works:

---

## 1. RVC (Retrieval-based Voice Conversion)

RVC is currently the most popular voice identity conversion system.

### Where to Find Ready Models
* **Hugging Face (`huggingface.co`)**: Search for `RVC` or specific character/singer models. Thousands of trained `.pth` + `.index` bundles are hosted as public repositories.
* **Weights.gg (formerly Voice-Models)**: A community hub dedicated exclusively to RVC v2 voice models.
* **Base Pre-trained Backbones** (needed for training/inference):
  * **Content Encoders**: `ContentVec` (`checkpoint_best_legacy_500.pt`) or `WavLM`.
  * **Pitch Estimators**: `RMVPE` (`rmvpe.pt`) or `Crepe`.
  * **Vocoders**: Pre-trained HiFi-GAN generator weights.

### How to Train a Custom RVC Model
Training an RVC v2 model takes about **10–30 minutes on a modern GPU** (NVIDIA RTX 3060 or better) and requires no coding.

#### The Modern Tool: **Applio**
Instead of the original RVC repository (which has legacy dependencies), use **Applio**—a modernized, actively maintained RVC fork with built-in export to ONNX.

1. **Prepare Audio Data:**
   * Collect **5 to 15 minutes** of the target speaker talking or singing.
   * Audio must be **completely dry** (no background music, no heavy reverb, no echo).
   * If working with existing songs, use **Ultimate Vocal Remover (UVR5)** with the `MDX-Net` or `VR Architecture` models to isolate clean vocals.
2. **Preprocessing (in Applio / RVC WebUI):**
   * Feed your clean audio folder into the WebUI.
   * It automatically slices long files into short 2–4 second chunks and resamples them to 40 kHz or 48 kHz.
3. **Feature Extraction:**
   * The pipeline runs `ContentVec` (to extract phonetic speech representations) and `RMVPE` (to extract continuous pitch curves).
4. **Model Training:**
   * Architecture: VITS-style generator + Multi-Period Discriminators (MPD/MRD).
   * Recommended epochs: **150 to 250 epochs** (stops before overfitting into robotic rasps).
5. **Index Building:**
   * The UI builds a **FAISS index** (`.index` file). This stores vector embeddings of the target speaker's phonemes so the model can pull the exact timbre during inference.
6. **Export for ANode:**
   * Export the trained model to `.onnx` directly from Applio's export tab.

---

## 2. DDSP-SVC (Differentiable DSP Singing Voice Conversion)

DDSP-SVC is the best option if your goal is **musical live performance, zero vocoder artifacts, and low CPU usage**.

### Where to Find Ready Models
* **GitHub (`yxlllc/DDSP-SVC`)**: The primary open-source repo maintains pre-trained singer checkpoints and multi-speaker models on Hugging Face.
* Community vocal packs include trained Japanese, English, and Mandarin singing voice profiles.

### How to Train a DDSP-SVC Model
DDSP does not try to hallucinate raw waveforms; it learns to predict **harmonic partial amplitudes** and **filtered noise bands** conditioned on pitch and loudness. Because it optimizes physical synthesizer parameters, training converges much faster than GANs or Diffusion models.

```
Clean Audio (10–20 min) 
   │
   ├──► Extract F0 (RMVPE) + Loudness (RMS) + Content (ContentVec)
   ▼
Train Tiny Neural Controller (GRU / MLP, ~50k–200k params)
   │  Loss: Multi-Scale Spectral Loss (STFT L1 + Log-magnitude)
   ▼
Export to .onnx for ANode C++ Additive Synthesizer
```

#### Training Steps:
1. **Dataset**: 10 to 30 minutes of clean, isolated singing or expressive speech (WAV, 44.1k or 48k).
2. **Preprocessing**:
   ```bash
   python preprocess.py -c configs/combsub.yaml
   ```
   This extracts $F_0$ (via RMVPE), volume curves, and content features.
3. **Training**:
   ```bash
   python train.py -c configs/combsub.yaml
   ```
   * Can be trained on a consumer GPU in **1 to 2 hours**, or even on a modern multi-core CPU overnight.
4. **ONNX Export**:
   * DDSP-SVC includes an official script to trace and export the controller network to ONNX:
   ```bash
   python onnx_export.py -c configs/combsub.yaml --output model.onnx
   ```
5. **Inference in ANode**:
   * ANode’s C++ engine runs the additive oscillator bank (`\sum A_k \sin(k \omega_0 t)`) while ONNX Runtime updates $A_k$ every block.

---

## 3. Universal Neural Vocoders (Zero Training Required)

If you only want to eliminate the phase-vocoder metallic artifacts from ANode's current `SignalsmithVocal` or `VocalTransformer`, **you do not need to train a voice model at all**. You can use a **universal pre-trained neural vocoder**:

### 1. **Vocos** (`charactr/vocos-mel-24khz` / `gemelo-ai/vocos`)
* **What it is:** A Fourier-based neural vocoder using ConvNeXt backbones.
* **Pre-trained:** Available on Hugging Face. Pre-trained on thousands of hours of speech.
* **Why it matters:** It takes a mel-spectrogram or pitch/envelope features and produces studio-grade audio **without being trained on a specific person**.
* **Speed:** Runs in pure real-time on CPU via ONNX Runtime (~2–3 ms per hop).

### 2. **BigVGAN v2** (NVIDIA)
* **What it is:** The current state of the art in universal generative vocoders (supports 24 kHz and 44.1 kHz).
* **Pre-trained:** Pre-trained weights are public on Hugging Face. Handles singing, whispering, extreme screaming, and background instruments without glitching.
* **Usage:** Best run on a GPU (via DirectML/CUDA in a background worker).

---

## 4. Zero-Shot / In-Context Voice Cloners (Seed-VC & CosyVoice)

A newer paradigm is **Zero-Shot Voice Conversion** (like **Seed-VC** or **CosyVoice**).

* **How they work:** You don't train them at all. You give the model:
  1. Your live microphone input.
  2. A **3-second reference audio clip** of any target voice.
* The model analyzes the reference clip's timbre on the fly and converts the source voice to match it.
* **Current limitation for ANode:** These models rely on Diffusion or Flow Matching transformers. They typically require an NVIDIA RTX 3080/4080 or better to achieve real-time streaming, and their latency sits around **150–250 ms**. They are well-suited for streaming or voice chat avatars, but too slow for live in-ear monitoring.

---

## Recommended Strategy for ANode

| Priority | Approach | Action Plan |
| :--- | :--- | :--- |
| **Immediate (Lowest Friction)** | **RVC via ONNX** | 1. Download ready-to-use `.pth` models from Hugging Face or Weights.gg.<br>2. Convert them to `.onnx` using Applio.<br>3. Load them into an ANode background worker using `onnxruntime` (already in `environment.yml`). |
| **Best for Real-Time & Singing** | **DDSP-SVC** | 1. Clone `yxlllc/DDSP-SVC`.<br>2. Record 15 minutes of your voice (or download public singing acapellas).<br>3. Train for 1 hour on GPU.<br>4. Export to ONNX and feed it from ANode's `SwiftF0` node. |
| **Universal Enhancement** | **Vocos ONNX** | Download pre-trained Vocos ONNX weights from Hugging Face to serve as a real-time neural resynthesizer for ANode's traditional C++ DSP nodes. |