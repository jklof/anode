# AGENTS.md — ANode Development Guide

## 1. Project Model

ANode is a Python/PyTorch node-based audio application with:

* a graph/model layer,
* a Qt UI,
* background workers for loading and other slow operations,
* optional native C++ DSP through ctypes,
* a block-based audio processing loop.

The priority order is:

1. Correct audio and application behavior.
2. No unsafe cross-thread access to shared mutable state.
3. No blocking disk/network/UI work in audio processing.
4. Small, understandable changes.
5. Performance optimizations only where measurements justify them.

This document describes important invariants. It is not a requirement to turn ANode into a hard-real-time audio framework.

When code, tests, and documentation disagree, investigate the discrepancy rather than blindly following any one source.

---

## 2. Audio Format

The default engine format is:

* `BLOCK_SIZE = 512`
* `SAMPLE_RATE = 48000`
* `CHANNELS = 2`
* `DTYPE = torch.float32`
* CPU tensors, normally shaped `(channels, frames)`

Do not introduce per-node block sizes or silently change the global format.

Node port channel counts are part of the port contract. Channel adaptation must be explicit and deterministic.

Ports are additionally typed: `slot_type` is `audio` or `midi`. Audio ports carry a `(channels, BLOCK_SIZE)` tensor buffer; MIDI ports carry a `MIDIPacket` (a list of `(sample_offset, mido.Message)` sorted ascending by sample offset) instead of a numeric buffer. Never wire an audio output into a MIDI input (or vice versa); `Graph.connect()` rejects mismatched types.

Channel adaptation policy (mono source into a wider input):

* A mono (`channels=1`) source is broadcast/duplicated to fill a stereo input.
* Pure-Python nodes do this with `buffer.copy_(sig)` followed by in-place ops (`mul_`, `add_`).
* `FFINode` duplicates 1-channel inputs into both channels of its scratch buffer before calling native C++ (`ffi_base.FFINode.process()`, mirrored by `BiquadFilter.process()`). Native code is called with the full output channel count so both channels are written.

PyTorch `out=` buffer-shrinkage hazard: functional ops with `out=buf` (e.g. `torch.mul(a, b, out=buf)`) silently RESIZE `buf` to the broadcast shape. Feeding a potentially mono input into a `(CHANNELS, BLOCK_SIZE)` output buffer this way shrinks it to `(1, BLOCK_SIZE)` and downgrades the port. Never use functional `out=` with inputs that may be mono; use `buf.copy_(sig)` followed by in-place ops instead (`Tensor.copy_` broadcasts without resizing).

Every audio output must be fully written for every processed block. Never leave stale samples in an output buffer.

The same anti-ghosting rule applies to MIDI: every MIDI output packet must be cleared at the top of `process()` and refilled for the current block. A stale message carried across a block boundary duplicates events just like a stale audio sample. `InputSlot.get_packet()` aggregates packets from all connected MIDI outputs.

---

## 3. Threading and Ownership

ANode has several execution contexts:

* UI/control thread
* engine audio-processing thread
* NRT worker threads
* audio-device callback threads
* optional native/library threads

The important rule is **single ownership of shared mutable state**.

A worker or callback may freely mutate state that it exclusively owns. It must not concurrently modify graph topology or DSP state that is also being processed by another thread.

Use message passing, staging, queues, or prepared-state handoff when ownership crosses a thread boundary.

Node construction must occur off the real-time audio thread. When queuing structural commands to a running engine (`("add", ...)`), pass a pre-instantiated node object, not a type name string: `cls()` may load native libraries (`ctypes.CDLL`), design FIR filters, or allocate large structures. Graph insertion and execution-plan recompilation are the only engine-thread work for an add.

Restore/undo flows hand a bare node instance to the engine so that `node.graph` is attached BEFORE `load_state()` runs — nodes that submit background tasks from `load_state()` (ConvolutionReverb, SamplePlayer, NamNode, ...) need a valid graph reference.

Graph serialization (`to_json()`) and file I/O must run on control or NRT threads, never in the steady-state processing loop. Save serialization happens at the coherent command boundary (all queued commands already applied); the file write is delegated to a background stream.

Exception logging on the audio path must be minimal: store the error string on `node.error_msg` and defer detailed logging/reporting to control/UI contexts.

It is acceptable for some control operations to run synchronously while the engine is stopped.

---

## 4. Audio Processing Rules

Audio processing must not perform:

* disk I/O
* network I/O
* UI operations
* blocking waits
* arbitrary long-running user code
* model/file loading

Avoid unnecessary allocations in steady-state processing.

Pre-allocate buffers when this is simple and materially useful, but do not treat every PyTorch/backend allocation as a correctness violation.

Do not use logging or expensive diagnostic formatting as normal audio-path behavior.

If a node fails during processing, fail it safely and defer detailed reporting/logging to the control/UI side.

---

## 5. Parameters

Parameters use staging:

```
UI/script
   -> staging value
   -> engine/control synchronization
   -> active DSP value
```

UI code should not directly modify engine-owned DSP state.

Parameter callbacks should request lifecycle changes rather than directly manipulating hardware streams, native DSP objects, or other shared resources.

Lifecycle contract: `on_ui_param_change(name)` observes a synchronized parameter. The engine's `param`/`add` command handlers call `param.set(val)` followed by `param.sync()` BEFORE invoking `on_ui_param_change(name)`, so `param.value` is committed and caches are updated. Node callbacks may read the staged value via `param.get_staging_safe()` (equal to `value` after the sync) — but active DSP-side consumption always reads `param.value`, which is committed at the block boundary via `sync()` in the processing loop.

Because the engine has already committed the value, `on_ui_param_change()` implementations must not defensively call `param.sync()` on the changed parameter — it is redundant and only masks contract violations. A deliberate re-stage followed by `sync()` (e.g. resetting a transient parameter like `seek_ratio` back to its neutral value) is fine.

Audio-rate modulation is allowed where a node explicitly defines such an input.

---

## 6. NRT Work

Use NRT workers for operations that may block or take substantial time:

* file loading
* decoding
* resampling
* model loading
* IR preparation
* device enumeration
* stream setup/teardown

NRT work should preferably produce a result that can be installed by the owning control/engine side.

Workers may own their own resources and worker-local state. They must not concurrently manipulate shared DSP state that the audio thread is processing.

For replaceable DSP resources, prefer:

```
prepare new resource
    -> install new resource
    -> retire old resource safely
```

Avoid loading or destroying large native DSP objects inside the audio processing function. This includes retirement of replaced resources: large native object destruction (e.g. NAM neural models) must be submitted back to NRT for background deallocation rather than run on the engine/audio thread.

`on_nrt_complete()` is invoked by `Engine._drain_nrt_all()`, which runs at periodic telemetry intervals (~100 ms) and at command execution boundaries (and from the UI poll timer when the engine is stopped) — not inside `Node.sync()` per block. Treat it as an engine/control-thread callback between blocks.

Epoch/version checks may be used to discard stale asynchronous results.

---

## 7. Native C++ DSP

All ctypes bindings must explicitly define `restype` and `argtypes`.

Native DSP handles must have clear ownership. Do not simultaneously reconfigure a native handle from a worker while the audio thread is using it.

Prefer constructing replacement native state off the audio thread and swapping ownership at a safe boundary.

Native `process()` functions must receive correctly sized CPU-resident contiguous buffers.

Every native DSP translation unit must define the standard export macro and prefix all C-ABI functions with it:

```cpp
#if defined(_WIN32)
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C"
#endif
```

All native DSP classes must implement and export `reset(void* handle)` (clearing delay lines, envelopes, and internal state) so transport restarts cannot leak stale audio. `FFINode.start()` calls it via `lib.reset()` when available.

---

## 8. Graph

The graph is a DAG.

`Graph.connect()` must reject:

* self-loops
* connections that would introduce a cycle
* invalid node/port references
* connections between mismatched port types (audio vs midi)

Invalid topology should be rejected at connection time rather than merely detected during execution-order generation.

When code iterates all ports generically (e.g. the engine's startup buffer reset), it must respect `slot_type`: only audio slots have `.buffer` / `._scratch`; MIDI slots have `.packet` / `._scratch_packet`. Blindly touching `.buffer` on every output slot raises `AttributeError` on MIDI nodes.

Graph mutations should go through the command/control mechanism.

Undo state must come from authoritative command state, not an asynchronous UI rendering cache.

Commands that execute asynchronously should have an identifiable request/result so history does not depend on guessing which command was accepted.

---

## 9. Save/Load

Saving must serialize one coherent authoritative graph state.

Before serialization, pending UI changes that are part of the saved state must be committed to the authoritative state.

Serialization runs at the engine's coherent command boundary (so queued commands are already applied); the file write itself is delegated to a background stream thread and never runs on the steady-state processing loop.

Loading and reload operations must invalidate stale asynchronous work associated with replaced nodes.

Malformed or incompatible connections should not be silently converted into a different graph without an explicit policy.

---

## 10. Visual / Monitoring Nodes

Visual nodes must not alter audio passing through the graph except for the explicitly documented channel-adaptation behavior.

Analysis may use a private copy and may sanitize/downsample/transform that copy.

Telemetry queues must be bounded and non-blocking from the audio side.

Dropping visual telemetry is preferable to delaying audio processing.

---

## 11. File and Streaming I/O

File creation, opening, decoding, and writing belong outside the audio processing function.

Streaming producers may own their own worker thread and queue.

The audio side must consume streaming data without blocking. On underrun, use the node's documented fallback behavior, normally silence or a safe fallback.

---

## 12. Script / Experimental Nodes

Nodes executing arbitrary Python code are not guaranteed to be glitch-free under load.

Such nodes may be used for prototyping, but this limitation must remain explicit in their documentation.

`ScriptNode` dynamically alters its own port topology (inputs/outputs) in response to script changes. These topology changes bypass the command/undo history system: they are applied directly to the node and surfaced via graph-structure dirty marking, and are not individually undoable.

`ScriptNode` applies the standard channel-adaptation rule to its script outputs: a mono `(1, B)` value assigned to a wider output is broadcast across all output channels (via `copy_`, which never resizes the destination); values that are too wide are copied into the leading channels; and any output that is unassigned, non-tensor, or has leftover channels is zero-filled every block so stale audio never survives.

---

## 13. Testing

Tests should prioritize:

* graph cycle rejection
* save/load round trips
* undo/redo correctness
* parameter synchronization
* node lifecycle/reset behavior
* stale NRT result rejection
* audio output buffer correctness
* important DSP numerical behavior
* non-blocking I/O behavior
* mono-input channel adaptation: output buffers must keep their `(CHANNELS, BLOCK_SIZE)` shape (PyTorch `out=` shrinkage regression guard)
* allocation-free steady-state RT paths (e.g. FileRecorder conversion into the pre-allocated pool)
* undo/restore graph-attachment ordering for nodes that spawn NRT work in `load_state()`

Real-time allocation tests are regression indicators, not proofs of zero allocation.

For visual nodes, test pass-through correctness and telemetry behavior. Add GUI rendering/concurrency tests where those behaviors are important enough to justify the maintenance cost.

---

## 14. Coding Style for Agents

Inspect before modifying.

Prefer small, local changes.

Preserve existing public node names, port names, parameter keys, and saved-patch compatibility unless the task explicitly changes them.

Do not refactor an architecture merely to satisfy this document.

Fix real correctness, ownership, lifecycle, and regression problems first.

Run the smallest relevant tests after each logical change, then the full suite when practical.

Do not claim tests passed or builds succeeded unless they were actually run.

Do not commit or push unless explicitly requested.
