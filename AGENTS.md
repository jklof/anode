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
- **No Blocking Calls on Audio Thread**: Never perform disk I/O, network requests, device queries, or model weight loading in `process()`, `start()`, or parameter setters.
- **Background Tasks via `NRTExecutor`**: Use `self.submit_nrt(fn, *args, tag=...)` and handle results in `on_nrt_complete(tag, ok, result)`. The engine handles task invalidation (`_nrt_epoch`) automatically.
- **Thread Concurrency**: Keep `torch.set_num_threads(1)` active to prevent OpenMP deadlocks between background loader threads and audio processing.

### 2. Node & Slot Development Rules
- Every node must subclass `base.Node` (or `ffi_base.FFINode` for C++ plugins).
- Every node must declare class-level metadata:
  ```python
  category = "Effects"  # One of: Sources, Utilities, Effects, I/O, Visual, Uncategorized
  label = "Human Friendly Name"
  ```
- **Anti-Ghosting**: When processing or routing between mismatched channel counts (e.g. Mono $\to$ Stereo), always explicit zero-out unused channels in output buffers.

### 3. Parameter Lifecycle & Synchronization
- Parameters use a staging mechanism (`_staging` $\to$ `sync()`) to cross the UI/Engine thread boundary safely.
- Do not access `Parameter.value` directly from the UI thread; use `Parameter.get_staging_safe()` or send updates through `AppController.set_parameter()`.

### 4. Graph Topology & Undo/Redo (Command Pattern)
- All structural alterations (adding nodes, deleting nodes, moving nodes, creating/breaking connections) **must** go through `AppController` commands (`commands.py`).
- Always snapshot the node state **at command instantiation time** (not during delayed execution) to avoid state race conditions.
- Batch operations (e.g. deleting selections or pasting subgraphs) must be grouped into `CompoundCommand` macros.

### 5. C++ FFI (`ffi_base.py` & `cpp/`)
- All native shared libraries must export standard C-ABI functions:
  `create()`, `destroy()`, `process()`, `set_param()`, `set_samplerate()`.
- Audio data passed through FFI is in planar/flat `float*` buffers (`[Ch0_0..Ch0_N, Ch1_0..Ch1_N]`). Ensure tensors are contiguous (`.is_contiguous()`) and on the CPU before casting to C pointers.

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
