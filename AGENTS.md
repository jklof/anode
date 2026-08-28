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

Every audio output must be fully written for every processed block. Never leave stale samples in an output buffer.

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

Avoid loading or destroying large native DSP objects inside the audio processing function.

Epoch/version checks may be used to discard stale asynchronous results.

---

## 7. Native C++ DSP

All ctypes bindings must explicitly define `restype` and `argtypes`.

Native DSP handles must have clear ownership. Do not simultaneously reconfigure a native handle from a worker while the audio thread is using it.

Prefer constructing replacement native state off the audio thread and swapping ownership at a safe boundary.

Native `process()` functions must receive correctly sized CPU-resident contiguous buffers.

---

## 8. Graph

The graph is a DAG.

`Graph.connect()` must reject:

* self-loops
* connections that would introduce a cycle
* invalid node/port references

Invalid topology should be rejected at connection time rather than merely detected during execution-order generation.

Graph mutations should go through the command/control mechanism.

Undo state must come from authoritative command state, not an asynchronous UI rendering cache.

Commands that execute asynchronously should have an identifiable request/result so history does not depend on guessing which command was accepted.

---

## 9. Save/Load

Saving must serialize one coherent authoritative graph state.

Before serialization, pending UI changes that are part of the saved state must be committed to the authoritative state.

File I/O may run outside the audio thread.

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
