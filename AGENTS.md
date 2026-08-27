# AGENTS.md --- ANode Developer Guide & Architectural Invariants

## 1. Project Overview

ANode is a modular, low-latency, node-based audio workstation combining Python/PyTorch audio processing, native C++ DSP through a C ABI/ctypes FFI, and a PySide6/Qt UI.

This file is the authoritative architectural contract for AI coding agents and developers.

**When implementation, tests, and this document disagree, do not silently choose one. Investigate the discrepancy. Preserve explicit architectural invariants unless the task intentionally and explicitly changes the architecture.**

---

## 2. Immutable Audio Format

Project-wide constants defined in `base.py`:

- `BLOCK_SIZE = 512`
- `SAMPLE_RATE = 48000`
- `CHANNELS = 2` by default
- `DTYPE = torch.float32`
- Audio tensors are CPU-resident and planar: `(channels, BLOCK_SIZE)`

Do not introduce per-node block sizes or silently change these constants.

A node's output channel count is an immutable part of its port contract and must not dynamically change during `process()`.

---

## 3. Thread Ownership Model

The engine thread exclusively owns live DSP execution, graph topology, parameter synchronization, and native DSP state during playback.

```text
UI / CONTROL THREAD
        |
        | commands / staged parameters
        v
  ENGINE THREAD
  authoritative graph + live DSP
  parameter sync + process()
        |
        | immutable results / snapshots / telemetry
        v
 NRT workers <----> UI
 disk / loading     telemetry / rendering
 preparation
```

**No thread other than the engine thread may directly mutate live graph topology, node instances, engine parameter values, or native C++ DSP handles.**

UI and NRT threads may request changes, stage parameter values, or consume immutable results/snapshots.

NRT workers must return results through the NRT completion mechanism rather than mutating live nodes directly.

---

## 4. Development Environment & Build

- Conda environment: `anode-dev`
- Python version: 3.11
- Preferred shell: Git Bash on Windows, Bash/Zsh on Linux/macOS.
- Use `/` (forward slashes) in file paths.

Run tests with:

```bash
conda run -n anode-dev python -m pytest tests/ -v
```

For C++ extension builds, fully activate the environment (do not use bare `conda run` on Windows to avoid compiler toolchain stripping):

```bash
conda activate anode-dev
pip install -e . -v
```

Do not manually modify or commit generated `.dll`, `.so`, or `.dylib` binaries.

The canonical project build uses CMake / scikit-build-core. Do not bypass it with ad-hoc compilation when building project plugins. Keep `cpp/CMakeLists.txt` synchronized with all native targets.

---

## 5. Real-Time Audio Rules

`Engine._worker()` is the real-time (RT) audio execution loop. `process()` runs under `torch.no_grad()`.

`process()` must **NOT**:

- Perform disk, network, or pipe I/O.
- Acquire blocking mutexes or locks.
- Wait on futures, promises, semaphores, or condition variables.
- Start, join, or terminate OS threads.
- Perform model loading, IR file decoding, or heavy buffer initialization.
- Intentionally raise exceptions as normal control flow.
- Log, print, format strings, or construct diagnostic payloads.
- Allocate, resize, or replace fixed audio/scratch buffers.
- Create avoidable per-block Python objects (dictionaries, lists, string formatting).
- Call `.tobytes()`, `.item()`, or perform per-sample Python loops.
- Call arbitrary user or un-bounded library code.

### Memory Allocation
The architectural requirement is **zero avoidable steady-state heap allocation** on the audio thread. Pre-allocate all scratch buffers, index arrays, boolean masks, and analysis buffers during node `__init__`.

Library operations (such as PyTorch FFT or C++ convolution) may internally utilize pre-allocated workspace memory.

`tracemalloc` tests are regression detectors for retained/unexpected allocations; passing a net-growth test does not prove that no transient allocations occurred.

Automatic Python garbage collection is disabled during active playback (`gc.disable()`). Write code that avoids per-block temporary objects entirely rather than relying on GC.

### Output Buffer Contract
Every `process()` implementation must completely write every output channel for every frame:
- Never leave stale samples in an output buffer.
- When processing mono input to stereo output, explicitly duplicate or zero unused channels to prevent audio ghosting.
- Do not use `out=` tensor operations in a way that shrinks or resizes the destination tensor.
- Do not use `torch.nan_to_num`, `torch.clamp`, normalization, or other sanitization on the main audio path unless that transformation is explicitly the node's DSP function (e.g., a clipper or saturator).

---

## 6. Visual / Monitoring Nodes

Visualization nodes (`WaveformDisplay`, `DataDisplayNode`, `SpectrogramDisplay`, `SpectrumDisplay`) are **strict bit-exact pass-through nodes**.

```text
input -----------------> exact output copy (untouched)
  |
  +--------------------> private analysis buffer -> UI telemetry
```

A visual node must never clip, sanitize, normalize, denoise, or otherwise modify the audio continuing through the graph.

Analysis routines may sanitize, downsample, or transform their private copy.

### Telemetry Transport
Do not use synchronized `queue.Queue` as an audio-thread queue. `queue.Queue` relies on mutex locks and can block the real-time thread during UI polling.

Visual nodes must use a bounded, non-blocking Single-Producer Single-Consumer (SPSC) ring buffer or lock-free array queue:
- If the telemetry queue is full, **drop the visual frame immediately**. Never block the audio thread.
- Pre-allocate all analysis, downsampling, and telemetry storage during initialization.
- Do not construct heap dictionaries or format strings per block to package visual frames.

*(For streaming audio I/O nodes such as media players, refer to §12).*

---

## 7. Parameters

Parameters use a decoupled staging model:

```text
UI / Script / Undo -> Parameter._staging -> Parameter.sync()
                   -> engine-owned Parameter.value -> native DSP state
```

- UI changes and external scripts write exclusively to `Parameter._staging`.
- `Parameter.sync()` crosses the thread boundary on the engine thread.
- UI code must never directly mutate engine-owned `Parameter.value`.
- Native mutable DSP state is updated strictly by the engine thread.
- `Parameter.set()` must never perform blocking work or file I/O.

Use `Parameter.get_staging_safe()` for UI rendering and `AppController.set_parameter()` for UI dispatch.

---

## 8. NRT (Non-Real-Time) Work

Heavy tasks must use `node.submit_nrt(...)`, including file loading, audio file decoding, resampling, neural network model loading, FIR/IR convolution partitioning, and device probing.

NRT workers must **never directly mutate live nodes, graphs, parameters, or active native DSP handles**.

They return immutable/prepared results through `on_nrt_complete(tag, ok, result)`.

The engine thread installs prepared results during `sync()` or between process cycles, but must not perform expensive preparation itself.

### Prepared State Pattern
For complex DSP state (e.g., convolution reverb):

```text
NRT worker
  -> Load file / data
  -> Partition / compute FFTs / pre-allocate runtime buffers
  -> Validate complete PreparedState object
  -> Return to engine thread
  -> Engine thread atomically swaps/installs PreparedState
  -> process() executes with zero allocation
```

`process()` must never lazily construct missing DSP state. If prepared state is absent, the node must pass through dry audio or output silence according to its documented contract.

### Epoch Invalidation
NRT tasks must be invalidated via the node's epoch mechanism:
- When a node is deleted, call `nrt.discard(node)`.
- When global graph resets occur (`clear`, `load`, `reload`), `Engine` must invoke `nrt.discard(node)` for all displaced nodes before destroying or replacing them.

---

## 9. Native C++ FFI

Required C ABI exports:

```text
void* create()
void  destroy(void* handle)
void  process(void* handle, float* in, float* out, int channels, int frames)
void  set_param(void* handle, int param_id, float value)
```

Optional standard exports:

```text
void set_samplerate(void* handle, float samplerate)
void reset(void* handle)
```

### ctypes Signatures
**Every bound native function must explicitly declare both `restype` and `argtypes`.**

In particular:
- `create.restype` must be `ctypes.c_void_p`.
- Handle arguments must be `ctypes.c_void_p`.
- Pointer arguments must use explicit pointer types (e.g., `ctypes.POINTER(ctypes.c_float)`).

Never rely on ctypes defaults (which default to 32-bit `c_int` and silently truncate 64-bit pointers on 64-bit platforms).

### Parameter Synchronization
Parameter synchronization follows a single canonical path owned by `FFINode`:

```text
Parameter staging
 -> engine-thread Parameter.sync()
 -> FFINode._sync_params_to_cpp() (cached tuple comparison)
 -> native set_param()
 -> native process()
```

- `FFINode.on_ui_param_change()` must **never** call native `set_param()` directly.
- `FFINode` synchronizes staged parameters exactly once **before** native `process()` executes.
- `start()` and `load_state()` invalidate the cached parameter state (`_cpp_param_state = None`) to guarantee native parameter synchronization prior to audio processing.
- If a native library exports `reset()`, bind it and invoke it during `FFINode.start()` on the engine thread.

#### Audio-Rate Modulation Exception
When an input slot is explicitly designed to modulate a parameter per block (e.g., `BiquadFilter.in_mod` modulating cutoff frequency), `process()` on the engine thread may calculate the modulation value and push it directly via `lib.set_param()` after staged parameters have synced.

#### Model & Resource Deallocation
When native C++ models or buffers are updated (e.g., in Neural Amp Modeler), old DSP models must not be deallocated inside the audio thread's `process()` call. Stage outgoing models for deferred deallocation on an NRT/worker thread.

### Buffer Contract
FFI audio buffers are CPU-resident, contiguous, planar float buffers:

```text
[Ch0_0 ... Ch0_N, Ch1_0 ... Ch1_N]
```

Before casting pointers:
- Verify CPU residency (`device.type == "cpu"`).
- Verify contiguity (`is_contiguous()`); if non-contiguous, copy into a pre-allocated contiguous scratch buffer.
- Pass actual channel and frame counts.
- For multiple buffers (e.g. sidechains), every buffer must receive its own explicit channel count. Never index a secondary buffer using the main signal's channel count.

---

## 10. Graph Topology & DAG Invariants

ANode's graph is a Directed Acyclic Graph (DAG).

All UI-originated topology mutations must be dispatched as `AppController` commands:
- `add`
- `del`
- `move`
- `conn`
- `disconn`
- `restore`

Purely visual operations (selection, zooming, panning) are scene-level operations and not graph commands.

### Connection-Time Cycle Rejection
**A connection that would introduce a feedback cycle or self-loop must be rejected at connection time.**

- `Graph.connect()` must perform a reachability / DFS check: if `dst_id` can reach `src_id` (or if `src_id == dst_id`), reject the connection immediately and return `False`.
- Do not permit an invalid cyclic graph and merely discover it later during topological sorting.

### Commands & Undo/Redo State Ownership
Commands own the authoritative state required to undo and redo themselves:
- Destructive commands (such as `DeleteNodeCommand`) must capture the exact serialized node state and attached connection data from the authoritative graph model at command creation or execution time.
- **Do not use `_latest_snapshot` as an authoritative undo source.** `_latest_snapshot` is an asynchronous UI rendering cache and may be stale when rapid actions occur.
- `CompoundCommand` executes children forward and undoes them in reverse order.

---

## 11. Graph Serialization / Save

Graph serialization must be completely thread-safe:

1. When a `save` command is processed by the engine on the engine thread, ensure all preceding queued commands have been applied.
2. Serialize the authoritative graph state there (`self.graph.to_json()`).
3. Produce an immutable snapshot string/bytes.
4. Pass only that immutable snapshot to the background writer thread.
5. Perform file I/O outside the engine critical path.

**Never permit a background thread to traverse the live mutable graph.**

---

## 12. Audio I/O & File Streaming Nodes

### FileRecorder
Recording audio to disk must not introduce audio-thread latency:
- File creation, directory checks, and `wave.open()` / `wave.close()` must run on a persistent background writer thread.
- Never spawn, join, or control OS threads from audio-thread parameter callbacks (`on_ui_param_change`).
- Use a recorder-owned bounded pool/ring of pre-allocated audio blocks.
- In `process()`, copy engine audio into an available pool slot and pass slot indices over a lock-free queue.
- Never call `.tobytes()` on the audio thread.
- If the recorder pool is exhausted, drop or defer frames according to an explicit policy rather than blocking the audio loop.

### Streaming Audio Producers (e.g., MediaPlayerNode)
For long-running decoders streaming audio into the graph:
- Decoding, demuxing, URL resolution, and resampling run strictly on dedicated worker threads.
- The worker pushes pre-allocated blocks into a bounded queue.
- The engine thread's `process()` consumes blocks strictly via non-blocking `get_nowait()`.
- On queue underrun, `process()` writes exact zeros and sets status telemetry without blocking.

---

## 13. ScriptNode Contract

Arbitrary Python `exec()` is **not real-time safe**. Python bytecode execution may allocate heap memory, invoke arbitrary libraries, trigger GIL contention, perform I/O, or raise unhandled exceptions.

`ScriptNode` is explicitly contracted as a **Non-Real-Time (Non-RT) / Prototyping Node**:
- Document clearly that arbitrary scripts may cause latency jitter and dropouts during playback.
- If script compilation fails or runtime exceptions occur in `process()`, zero output buffers safely and report error telemetry without printing or logging inside `process()`.

---

## 14. DSP Node Contracts

All nodes must:
- Respect the project constants (`BLOCK_SIZE = 512`, `SAMPLE_RATE = 48000`, `CHANNELS = 2`, `torch.float32`).
- Fully write all output channel buffers on every block.
- Preserve pass-through bit-exactness where specified.
- Keep persistent DSP state in node-owned instance variables.
- Reset internal state (phases, delay lines, filter history, envelopes) on `start()`.

### InputSlot Contract
- For a connected input, `InputSlot.get_tensor()` copies connected buffers into its pre-allocated scratch buffer.
- For an unconnected input bound to a parameter (`param_name`), `InputSlot.get_tensor()` returns the parameter's cached constant tensor.
- For an unbound, unconnected input, `InputSlot.get_tensor()` clears `_scratch` to zero and returns it.
- Never treat `_scratch` as persistent audio state across blocks.
- Tests should mock `slot.get_tensor` when injecting explicit test signals rather than writing test data into `_scratch`.

---

## 15. Plugin Imports & Dependency Management

Plugin discovery must degrade gracefully when optional or heavy dependencies (e.g. `sounddevice`, `soundfile`, `resampy`, `av`, `yt-dlp`) are unavailable.

- Guard optional imports inside plugin modules or execution paths.
- Keep the core engine environment minimal.
- Do not add external dependencies for simple DSP algorithms that can be implemented cleanly with existing PyTorch/C++ facilities.

---

## 16. Testing Requirements

Testing workflow:
1. Run targeted node/unit tests first.
2. Run subsystem regression suites.
3. Run the complete test suite.

Architectural regression tests must cover:
- Visual bit-exact pass-through with out-of-bounds input amplitudes.
- Isolation of visual sanitization to private analysis buffers.
- Lock-free, non-blocking telemetry queues in visual nodes.
- Zero heap allocations in steady-state real-time processing (`tracemalloc`).
- Thread safety and pre-allocated buffer ownership in `FileRecorder` and `AudioDeviceInput`.
- Canonical FFI parameter synchronization order (before native `process()`).
- Native state reset on `start()` lifecycle calls.
- Connection-time cycle rejection in `Graph.connect()`.
- Immediate add/delete/undo topology integrity.
- Thread-safe serialization snapshots during active playback.
- NRT worker epoch invalidation on node deletion and global graph resets.

Node tests must verify `category`, `label`, and registration metadata.

FFT and spectral assertions should be used where frequency-domain behavior is part of the DSP contract.

### Visual Node Testing Invariant
Unit tests for visual nodes must exercise the multi-threaded GUI pipeline, not just DSP in isolation:

1. **Concurrent Producer-Consumer Fuzzing:** Run an audio-thread-simulator parallel to UI-thread polling to catch memory tearing, GIL races, and shared mutable array views.
2. **Headless Widget & paintEvent Execution:** Use `QT_QPA_PLATFORM=offscreen` to render `QWidget` to `QPixmap` and verify coordinate transforms, layout strings, and `QPainter` execute without exceptions.
3. **Pixel Energy Assertions:** Assert that silent signal renders the background color and active signal produces measurable pixel energy - proving the display actually shows sound.
4. **Temporal Frame Accounting:** Verify waterfall frames are never dropped between audio blocks and UI polling (no 66% frame starvation).

Example guard test:
```python
def test_waveform_display_thread_safety():
    # Dual-thread producer (audio sim) + consumer (UI poll)
    # Fail if torn frames detected via non-uniform channel values
```

---

## 17. AI Coding Agent Working Rules

1. **Inspect before modifying.** Read relevant code and understand data flows before editing.
2. **Search accurately.** Locate exact symbol definitions rather than guessing filenames.
3. **Preserve public behavior.** Do not alter port names, parameter keys, or node labels unless explicitly requested.
4. **Prefer small, precise patches.** Avoid mass refactoring when targeted changes suffice.
5. **Never weaken an architectural invariant.** If current code violates an invariant, fix the implementation to satisfy the invariant; never weaken the rule to accommodate a shortcut.
6. **Fix contradictory tests.** If an existing test asserts an anti-pattern or broken behavior (e.g. asserting that cycles are permitted), correct the test to assert the architectural rule.
7. **Verify native builds.** If modifying C++ source files (`cpp/*.cpp`) or `CMakeLists.txt`, ensure the native extension compiles cleanly.
8. **Run tests after every logical step.** Execute the relevant pytest suite and confirm passing status before marking work complete.
9. **Never claim edits succeeded or tests passed unless verified.**
10. **Do not commit or push unless explicitly requested.**

---

## 18. Project Layout

- `core.py`, `base.py`, `commands.py`, `controller.py`: Core engine, graph DAG, thread dispatch, parameter staging.
- `ffi_base.py`: Base ctypes / C-ABI bridging layer for native DSP nodes.
- `cpp/`: Native C++ DSP source code and `CMakeLists.txt`.
- `plugins/`: Node implementations (I/O, sources, effects, spatial, utilities, visual).
- `tests/`: Automated pytest regression suite.
- `docs/`: Technical specifications and DSP design notes.

*Use filesystem search/tools rather than assuming a static list of node or test files.*