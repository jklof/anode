Here is a complete, production-ready `AGENTS.md` file tailored specifically for the **ANode** 
# AGENTS.md — ANode Developer Guide for AI Coding Agents

## Project Overview
ANode is a modular, real-time node-based audio processing engine and GUI workstation. It combines a Python/PyTorch audio graph with high-performance native C++ (C++20, CMake, ctypes FFI) DSP plugins and a PySide6 (Qt) graphical canvas.

- **Audio Engine**: Python real-time loop + C++ FFI plugins.
- **Audio Format**: Fixed 512 samples/block (`BLOCK_SIZE = 512`), 48,000 Hz (`SAMPLE_RATE`), 2 channels (`CHANNELS = 2`), `torch.float32`.
- **UI Framework**: PySide6 (`QGraphicsScene`, `QGraphicsView`, custom widgets).
- **Core Dependencies**: PyTorch, NumPy, SoundDevice, SoundFile, Resampy, PyAV, yt-dlp, NeuralAmpModelerCore.

---

## Development Environment & Shell Rules

### Environment & Tooling
- **Conda Environment**: `anode-dev` (Python 3.11).
- **Default Shell**: git bash on windows, bash on linux and mac

### Command Execution Best Practices
- Run Python scripts and pytest using:
  ```bash
  conda run -n anode-dev <command>
  ```
- **Do not** use bare `python` or `py` commands.
- **Do not** use Windows backslash paths in shell commands (they get stripped). Use forward slashes (`/`).
- **Do not** use complex shell operators (`&&` or `||`) if chaining commands across heterogeneous shells; run discrete commands sequentially.

---

## Build & Installation Instructions

### 1. Environment Setup
```bash
conda env create -f environment.yml
```

### 2. Compiling C++ Extensions (CRITICAL)
> **WARNING**: **Never** use `conda run` to build C++ extensions (`pip install -e .`). `conda run` fails to inject the MinGW compiler toolchain environment variables on Windows, causing MSVC fallback errors.

Always activate the environment in the active shell first before building:


**Git Bash / POSIX Shell:**
```bash
conda activate anode-dev
pip install -e . -v
```

### 3. Running the Application
```bash
conda run -n anode-dev python main.py
```

---

## Testing Instructions

Run the complete test suite with pytest:
```bash
conda run -n anode-dev python -m pytest tests/ -v
```

To run a specific test file or test case:
```bash
conda run -n anode-dev python -m pytest tests/test_core.py -v
conda run -n anode-dev python -m pytest tests/test_nodes.py -k "test_gain" -v
```

---

## Architecture & Code Rules for Agents

### 1. Real-Time Audio Constraints (Audio Thread Safety)
- **Zero Allocations in `process()`**: Never allocate tensors, resize buffers, or create new Python objects inside a node's `process()` method. Always use pre-allocated buffers with in-place PyTorch operators (`copy_`, `mul_`, `add_`, `zero_`, `fill_`).
  - *Unavoidable exception*: some APIs have no `out=` variant (`torch.fft.rfft/irfft`, `F.conv1d`). Small per-block transients from these are acceptable, but must be documented at the call site (see `plugins/filters.py`, `plugins/convolution_reverb.py`). Acceptance tests assert near-zero *net* growth (tracemalloc), not literally zero allocations.
- **No Blocking Calls on Audio Thread**: Never perform disk I/O, network requests, device queries, or model weight loading in `process()`, `start()`, or parameter setters.
- **Background Tasks via `NRTExecutor`**: Use `self.submit_nrt(fn, *args, tag=...)` and handle results in `on_nrt_complete(tag, ok, result)`. The engine handles task invalidation (`_nrt_epoch`) automatically.
- **Thread Concurrency**: Keep `torch.set_num_threads(1)` active to prevent OpenMP deadlocks between background loader threads and audio processing.
- **Per-sample budgets**: one block = ~10.7 ms; aim for <5% (~0.5 ms) per node. Do NOT use per-sample loops in Python or TorchScript for IIR-style kernels — every `.item()`/setitem goes through the dispatcher and measured ~12 ms/block; a plain-Python float loop gets to ~0.5 ms/block but is only a stopgap. Real per-sample DSP belongs behind the FFI in C++: the same biquad runs at ~25 us/block natively (see `plugins/filters.py` + `cpp/biquad.cpp` / `cpp/fir_eq.cpp` for the wrapper + native pattern).

### 2. Node & Slot Development Rules
- Every node must subclass `base.Node` (or `ffi_base.FFINode` for C++ plugins).
- Every node must declare class-level metadata:
  ```python
  category = "Effects"  # One of: Sources, Utilities, Effects, I/O, Visual, Uncategorized
  label = "Human Friendly Name"
  ```
- **Anti-Ghosting**: When processing or routing between mismatched channel counts (e.g. Mono $\to$ Stereo), always explicit zero-out unused channels in output buffers.
- **Channel-count writes**: `Tensor.copy_()` broadcasts the source to the destination's shape — copying a wide signal into a narrowed slice (e.g. `seg[:, :1].copy_(mono)`) fails at runtime. Fill rows explicitly (`seg[c].copy_(...)`) and exercise channel paths in tests with *genuine* `(1, BLOCK)` mono tensors, not tiled stereo stand-ins.
- **`out=` resize trap**: a torch op with `out=` whose operands broadcast *narrower* than the destination (mono `(1, BLOCK)` source into a stereo `(CHANNELS, BLOCK)` buffer) silently **resizes the destination down** to the broadcast shape — only a deprecation warning, but it permanently shrinks pre-allocated buffers and corrupts every later block. Never pass `out=` a buffer wider than the computed result. Instead: `dest.copy_(src)` first (broadcasts without resizing), then in-place ops (`dest.mul_(g)`, `dest.add_(other, alpha=mix)`). This bit `torch.pow(out=)` (Data Display), `torch.mul(out=)` (WaveShaper drive), and every MathOp binary before it was codified.
- **Modulation inputs must be param-bound**: register optional audio-rate modulators as `add_input("x_in", "x")`. An unconnected slot then returns the parameter's constant tensor cache automatically; an unbound unconnected slot returns zeros — e.g. an oscillator reading raw `freq_in` goes silent (~0 Hz after dt clamping) with nothing patched. Never hand-check `"x_in" in self.inputs` (always true after `add_input`); check `slot.connected_outputs` only when connected-vs-param behavior must genuinely differ (see `WaveformOscillator`, `StereoPanner`, vs. `NoiseGate` sidechain fallback).
- **Test anti-aliasing spectrally, not locally**: BLEP/BLAMP-style correctors have a correction region of exactly `dt · (samples per cycle)` = 1 sample by construction, so inter-sample delta bounds near discontinuities stay O(1) at any frequency and prove nothing. Assert instead on FFT harmonic content: collect an integer number of periods, measure the largest non-harmonic bin relative to the fundamental (convention: `tests/test_oscillators.py::test_sawtooth_polyblep_suppresses_aliasing`, < -35 dB).
- **Visual nodes & `monitor_queue`**: dispatch frames via `self.monitor_queue = queue.Queue(maxsize=2)`; guard `put_nowait` with `full()` checked *before* copying the payload so overflow costs nothing, and let the widget drain it on its own QTimer. Visualization must be strictly pass-through — never sanitize or otherwise modify the audio path (see `plugins/spectrogram.py`, `plugins/spectrum.py`; frame-draining test convention in `tests/test_spectrogram.py`).
- **`InputSlot.get_tensor()` semantics**: an unconnected input zeroes its scratch buffer on every call and returns it. Never stash data in `_scratch` across blocks, and never feed test signals by writing to `_scratch` — mock `slot.get_tensor = lambda: block` instead (convention used in `tests/test_nodes.py` and `tests/test_filters.py`).
- **Plugin import granularity**: `plugin_system.load_plugins()` skips the entire module if any module-level import fails, killing every node defined in that file. Guard optional/heavy dependencies with try/except at module top and degrade gracefully (see `plugins/media_player.py`). The conda env is intentionally minimal — do not add dependencies without need (e.g. there is no scipy; do coefficient design with numpy and per-sample DSP in C++ as in `plugins/filters.py` / `cpp/fir_eq.cpp`).

### 3. Parameter Lifecycle & Synchronization
- Parameters use a staging mechanism (`_staging` $\to$ `sync()`) to cross the UI/Engine thread boundary safely.
- Do not access `Parameter.value` directly from the UI thread; use `Parameter.get_staging_safe()` or send updates through `AppController.set_parameter()`.

### 4. Graph Topology & Undo/Redo (Command Pattern)
- All structural alterations (adding nodes, deleting nodes, moving nodes, creating/breaking connections) **must** go through `AppController` commands (`commands.py`).
- Always snapshot the node state **at command instantiation time** (not during delayed execution) to avoid state race conditions.
- Batch operations (e.g. deleting selections or pasting subgraphs) must be grouped into `CompoundCommand` macros.

### 5. C++ FFI (`ffi_base.py` & `cpp/`)
- All native shared libraries must export standard C-ABI functions:
  `create()`, `destroy()`, `process()`, `set_param()`, `set_samplerate()`. `ffi_base` auto-binds and calls `set_samplerate` on creation when the library exports it.
- Audio data passed through FFI is in planar/flat `float*` buffers (`[Ch0_0..Ch0_N, Ch1_0..Ch1_N]`). Ensure tensors are contiguous (`.is_contiguous()`) and on the CPU before casting to C pointers.
- **Native param pushes**: replicate the `_CppParamMixin` pattern from `plugins/filters.py` — lazily compare a value tuple against the last-pushed state inside `process()` so every path (UI edits, `load_state`, direct `Parameter.set()+sync()` from tests) is covered uniformly, and re-push after `start()`/`load_state`. Extended exports beyond `create` need explicit `restype`/`argtypes` bindings plus a `reset()` export cleared from `start()`.
- For iteration, individual plugins can be compiled directly (`g++ -O2 -std=c++17 -shared -fPIC -o plugins/libfoo.so cpp/foo.cpp`) without a full scikit-build cycle; keep `cpp/CMakeLists.txt` updated regardless.
- When an extended API mixes buffers with different channel counts (e.g. mono sidechain into a stereo compressor), the C side must receive explicit channel counts per buffer — never index a smaller buffer with the main channel count (see `process_with_sidechain` in `cpp/compressor.cpp`).
- When binding additional exports (`reset`, `load_model_sync`, extended process calls, ...), always declare `restype`/`argtypes`. An unannotated handle argument defaults to 32-bit `c_int`, silently truncating 64-bit pointers and producing delayed, unrelated-looking segfaults.

---

## Directory Structure

```
.
├── base.py                 # Core data types (Node, Slots, Parameter, Clock)
├── commands.py             # Undo/Redo command pattern implementations
├── controller.py           # AppController mediator between UI and Engine
├── core.py                 # Audio Engine, Graph execution order, NRTExecutor
├── ffi_base.py             # Ctypes base class for C++ shared library nodes
├── main.py                 # Application entry point & Qt window setup
├── plugin_system.py        # Dynamic plugin discovery & hot-reloading
├── theme.py                # Visual styling constants & palettes
├── ui_icons.py             # Vector icons & logo rendering
├── ui_system.py            # QGraphicsScene canvas, wire drawing, NodeItems
├── cpp/                    # Native C++ source code & CMakeLists.txt
├── plugins/                # Python & C++ node implementations
├── tests/                  # Pytest unit & integration test suite
├── environment.yml         # Conda environment definition
└── pyproject.toml          # scikit-build-core configuration
```

---

## PR & Commit Guidelines
1. **Verify Builds**: If C++ files in `cpp/` are touched, ensure the extension builds cleanly via activated conda `pip install -e . -v`.
2. **Run Tests**: Ensure all tests in `tests/` pass before finishing tasks (`pytest tests/ -v`).
3. **Preserve Real-Time Safety**: Do not introduce dynamic memory allocation, disk I/O, or locking into any `process()` method.
