# ANode Bug Backlog

Prioritized, session-sized batches from full review (2026-09-19).
Source invariants: `AGENTS.md`.

## Workflow (every session)

1. Pick one batch below (3–5 items max, in order).
2. Track with `TodoWrite` (one `in_progress` at a time).
3. One logical fix at a time, small/local, preserve public node/port/param names + patch compat.
4. Add/extend regression test per bug where noted.
5. Run smallest relevant tests (`conda run -n anode-dev pytest tests/<file> -q`), then full suite when practical.
6. Do not commit/push unless explicitly requested.
7. Mark item `[x]` here when done + verified.

Legend: `[ ]` pending, `[~]` in progress, `[x]` done. Severity: H/M/L.

## Session 1 — Engine thread-safety (do first)

- [x] **B-001 (H)** Unguarded `on_nrt_complete` kills engine thread — `core.py:445`
  Fixed: try/except + `error_msg`. Verified 2026-09-19 via subagent, 50 passed.
- [x] **B-002 (H)** Blocking `output_queue.put()` on engine thread — `core.py:623`
  Fixed: `put_nowait`, drop on `Full`. Verified 2026-09-19, 50 passed.
- [x] **B-003 (H)** `logging.exception` per failing block + per telemetry tick — `core.py:1069-1072,1092-1093`
  Fixed: transition-only logging (+ `_telemetry_error_ids`). Verified 2026-09-19, 50 passed.
- [x] **B-004 (H)** `load` while running constructs nodes on audio thread — `core.py:879-891`, `controller.py:431-441`
  Fixed: `stop_audio()` before load (same as reload). Verified 2026-09-19, 50 passed.
- [x] **B-005 (M)** `add` string-type fallback runs `cls()` on engine thread — `core.py:632-639`
  Fixed: reject string adds with warning. Verified 2026-09-19, 50 passed.

## Session 2 — Silent audio / stale state

- [ ] **B-006 (H)** `FFINode.process` silent return leaves stale audio — `ffi_base.py:157-158`, also `plugins/envelope.py:70-71`, `plugins/dynamics.py:78-79`, `plugins/filters.py:62-64`
  Fix: `zero_()` all audio outputs before return. Test: missing lib/handle → silent block.
- [ ] **B-007 (H)** `SamplePlayer`/`ABCPlayer` lose file on reload (no `load_state`) — `plugins/sample_player.py:88-98`, `plugins/abc_player.py`
  Fix: add `load_state()` mirroring `convolution_reverb.py:156-161`. Test: save/load round-trip restores audio.
- [x] **B-008 (M)** Engine-replacement ops leak stale stats/error — `core.py:842-851,866-914,916-962`
  Fixed 2026-09-19: full reset of `_stats_buffer/_process_error_ids/_telemetry_error_ids/_last_error_sent` on clear/load/reload + `del` prune discards telemetry id. Tests: core/save/arch/error-visibility 56 passed.
- [ ] **B-009 (M)** Load/restore silently drops bad wires + unknown types — `core.py:892-898,826-827,935-945,884-885`
  Fix: collect rejected wires, one warning + snapshot event. Test: bad wire reported, not silent.
- [x] **B-010 (L)** MIDI merge unsorted — `base.py:321-328`
  Fixed 2026-09-19: in-place sort. Tests: `test_midi.py` + `test_core.py` 63 passed.
  Fix: in-place sort after loop. Test: two MIDI sources → ascending offsets.
- [x] **B-011 (L)** Menu params unclamped — `base.py:374-375`
  Fixed 2026-09-19: clamp to `[0,len(items))`. Verified `set(99)` → `1`.
  Fix: clamp to `[0,len(items))`. Test: crafted patch does not `IndexError`.
- [ ] **B-012 (H)** NAM type-pun `(NAM_SAMPLE*)inputs` — `cpp/neural_amp.cpp:87-90`
  Fix: `static_assert(sizeof==4)` or float scratch. Test: build-time assert + audio sanity.

## Session 3 — NRT ownership / params

- [ ] **B-013 (H)** `MediaPlayer.on_nrt_discarded` joins 2s on engine thread — `plugins/media_player.py:554-562`
  Fix: detach + `nrt.stop_stream` like `:659-666`. Test: discard does not block.
- [ ] **B-014 (H)** `NamNode` sync destroy on engine thread (stale-load + `remove`) — `plugins/neural_amp.py:169-177`, `ffi_base.py:237-241`
  Fix: background destroy via `submit_nrt`; override `remove()`. Test: code path review + no audio-thread destroy.
- [ ] **B-015 (M)** Teardown submits pollute shared NRT epoch — `plugins/neural_amp.py:188` + `core.py:480-498,406`
  Fix: pool-only submit skipping epoch bump for teardown. Test: load still installs after cleanup submit.
- [ ] **B-016 (M)** Reverb IR load has no epoch/tag — `plugins/convolution_reverb.py:163-168,254-267`
  Fix: add `_load_epoch` + stale-reject. Test: rapid IR switch → newest wins.
- [ ] **B-017 (M)** Sync device close in `on_nrt_complete` — `plugins/midi_devices.py:101-105,233-237`
  Fix: route via `nrt.submit/stop_stream`. Test: review, no close on engine thread.
- [ ] **B-018 (M)** Stopped-engine poll misses discarded nodes — `controller.py:98-113`
  Fix: wake drain when `nrt._discarded_nodes` non-empty. Test: delete-while-stopped drains.
- [ ] **B-019 (M)** FileRecorder writer mutates `params["record"]` cross-thread — `plugins/extended_nodes.py:161-165`
  Fix: `push_command(("param",...))` like `deesser.py:299-301`. Test: no direct set from writer.
- [ ] **B-020 (M)** PyAV container leak — `audio_io.py:63-84`
  Fix: try/finally `container.close()`. Test: handle count stable.

## Session 4 — DSP / I/O / UI correctness (small)

- [ ] **B-021 (M)** Reverb duplicates loader, MP3 fails — `plugins/convolution_reverb.py:170-186` vs `audio_io.py:125-145`
  Fix: call `audio_io.to_engine_audio`. Test: MP3 IR decodes.
- [ ] **B-022 (M)** Rubberband per-block alloc on audio thread — `plugins/rubberband_pitch_shifter.py:283,221,224`
  Fix: pool or document exception. Test: quarantine / doc.
- [x] **B-023 (M)** Unknown icon crashes — `ui_icons.py:165`
  Fixed 2026-09-19: `or ""` guard → empty icon via failed load + warning.
  Fix: `if svg is None: return QIcon()`. Test: unknown name returns empty icon.
- [ ] **B-024 (M)** `audiocpp_backend.py:717-720` hides valid repairs
  Fix: rely on `missing_specs()` alone. Test: corrupt cudart re-offered.
- [ ] **B-025 (L)** `FFINode` hardcoded `BLOCK_SIZE` + missing float32 check — `ffi_base.py:220,189-197`, `plugins/envelope.py:92`
  Fix: pass `shape[1]`, assert dtype. Test: non-standard frames + dtype guard.
- [ ] **B-026 (L)** Misc low-risk: stale-plan one tick (`core.py:1075-1095`), spurious `connect_rejected` (`:707-734`), unguarded stopped telemetry (`controller.py:108-113`), `disconnect` unconditional dirty (`core.py:248-256`), param-array always-changed (`base.py:379-393`), gain-race (`plugins/dynamics.py:150-156`), ScriptNode `ndim>2` (`plugins/scripting.py:249-265`), visual step assume (`plugins/visualization.py:50`), sentinel drop (`extended_nodes.py:257`), delete-vs-snapshot race (`controller.py:289-299`), `gc.disable` global (`core.py:1000,605-608`)
  Fix: per review notes. Test: as touched.

## Session 5 — Gemini batch (verified 2026-09-19)

- [x] **B-027 (H)** `MathOp` Divide crashes on mono `in_b` (`out=` shrinkage) — `plugins/math_op.py:74-78`
  Fixed 2026-09-19: copy_-first + `sign_()`. Tests: `test_math_op.py` 36 passed (existing mono Min/Max guards cover pattern).
  `torch.sign(sig_b, out=_tmp)` resizes `(2,512)->(1,512)`, then `_tmp.add_(_maskf)` raises broadcast `RuntimeError` on audio thread. Rest of file already uses copy_-first.
  Fix: `_tmp.copy_(sig_b); _tmp.sign_()` then `eq` as now. Test: divide with mono B over several blocks, assert no throw + `_tmp/_maskb` stay `(2,512)`.
- [x] **B-028 (M)** `get_node_documentation()` leaks native handles — `plugin_system.py:127`, `ffi_base.py:47`
  Fixed 2026-09-19: try/finally `remove()`.
  `cls()` creates `lib.create()` handle, never `remove()`d (no `__del__`). Once per type (cached) but leaks every native processor.
  Fix: try/finally `instance.remove()` (guarded). Test: doc call for FFINode type leaves no live handle.
- [x] **B-029 (M)** `ABCPlayer._all_off()` appends offset-0 note-offs unsorted — `plugins/abc_player.py:209-219,321-332`
  Fixed 2026-09-19: offset `BLOCK_SIZE-1` + sort. Tests: `test_abc_player.py` passed.
  Breaks ascending-offset packet contract; end-of-song offs appended after same-block note-ons at higher offsets.
  Fix: offset `BLOCK_SIZE-1` + sort after append. Test: end-of-score block packet sorted, offs at end of block.
- [x] **B-030 (M)** `abc_score._read_length()` `ZeroDivisionError` on `/0` — `abc_score.py:132-139` vs never-raises doc `:204-205`
  Fixed 2026-09-19: `denom != "0"` guard. Tests: `test_abc_score.py` passed + manual `C/0` parse ok.
  `C/0` → `Fraction(num,0)` escapes (siblings guard via `except ZeroDivisionError` at `:103,116`).
  Fix: `d = int(denom) if (denom and denom != "0") else 2`. Test: `parse_score("C/0")` does not raise.
- [ ] **B-031 (L)** VocalTransformer breath sideband reads uninitialized/OOB at non-48k rates — `cpp/vocal_transformer.cpp:1334-1335,1351-1357,1377-1388`
  Latent at shipped 48k/2048 (comment `:1382` claims `kd+k0<=nb_hi` "by construction" — false once `nb_hi` clamps). At low SR `k_high+k0` exceeds `num_bins_-1`.
  Fix: clamp `k_high` to `num_bins_-1-k0` when pitch_sync + guard `kd±k0`. Test: code review + low-SR unit case. Not a current-rate crash.

## Backlog — Optimizations (easy wins, do between sessions)

- [ ] **O-01** MediaPlayer queue 500 → 64–128 — `plugins/media_player.py:382,519,169`
- [ ] **O-02** Gate per-block double `perf_counter` (~3750/s @20 nodes) — `core.py:1062/1067`
  Fix: total block time once per block; per-node timers only on telemetry tick or behind flag.
- [ ] **O-03** Throttle fetch progress (every 5%) + fix `_in_flight` — `audiocpp_backend.py:729-740`, `core.py:431-434`
- [ ] **O-04** Raw thread per save/GC → pool — `core.py:600-601,608`
- [ ] **O-05** `_get_downstream_nodes` O(V·E) only if slow — `core.py:69-79`
- [ ] **O-06** Analyzer `.item()` syncs already rate-limited, keep/document — `plugins/data_display.py:124`, `signal_analyzer.py:91-98`
- [ ] **O-07** Cache `QPainterPathStroker` shape — `ui_system.py:333-336`
  `shape()` runs per mouse-move hit-test. Fix: build stroked path once in `update_path()`, return cached.
- [ ] **O-08** Reuse `energy` in NSDF loop (~115k redundant squarings/hop) — `cpp/pitch_tracker.h:77-103`
  Fix: seed `den` with `energy`, accumulate only `y*y`. Numerically identical.
- [x] **O-09** Single-pass RMS in `AutoGain` — `plugins/dynamics.py:483-485`
  Fixed 2026-09-19. Tests: `test_dynamics.py` passed.
  `mean(dim=0)` then `mean()` == global `mean()`. Fix: `rms = sqrt(mean(_pow_scratch))`. Identical output.
- [ ] **O-10** Cache colored icons — `ui_icons.py:138-170`
  `create_icon()` re-parses SVG + Qt render per call. Fix: dict cache on `(name, hex_color)`. Complements B-023 (which still needs the `None` guard).
- [ ] **O-11** Skip hidden visual timers — `plugins/spectrum.py:176-192`, `spectrogram.py:236-272`, `visualization.py:80`
  30–40 FPS `QTimer` runs even when hidden. Fix: `if not self.isVisible(): return` at top of `poll_queue`.

## Backlog — KISS / Maintainability (refactor only when touching)

- [ ] **K-01** Shared sliding-FFT helper — `plugins/spectrum.py:69-87` vs `spectrogram.py:116-133`
- [ ] **K-02** Single `normalize_param_value()` — `core.py:655-667` + `base.py:571-579`
- [ ] **K-03** `BiquadFilter` reuse `FFINode` mono-dup — `plugins/filters.py:106-115`
- [ ] **K-04** Split `_apply_command` (~350 lines) to `_op_<name>` — `core.py:625-973`
- [ ] **K-05** One epoch idiom doc (executor = staleness, node counters = intra-tag) — `core.py:406`
- [ ] **K-06** Comment `remove_node`/`del` pairing, `stop_stream` epoch bump, `_tensor_cache` read-only — `core.py:89-110,480-498`, `base.py:315-316`
- [ ] **K-07** Fix CoW breaks: ghost wires + in-place `n["pos"]` — `controller.py:297-299,313-317,345-348`
- [ ] **K-08** Dead code: reverb display bins, theme container, `self.graph=None`, editor size, `sf is None`, `queue` shadow, redundant save branch — see review
- [ ] **K-09** Standardize on `.value` (not `get_staging_safe`) + push UI state from completions via `_refresh_ui/_push_telemetry` — `audiocpp_backend.py:791-837`
- [ ] **K-10** `ui_system.py:1983` split only when touching; test gaps: SamplePlayer restore, FFINode zero-fill, unknown icon, rubberband quarantine.
- [ ] **K-11** Deduplicate `load`/`reload` deserialization — `core.py:866-962`
  ~30-line instantiate→attach→`load_state`→connect→clock loops duplicated. Fix: `_deserialize_graph(data, target_graph)` called from both.
- [x] **K-12** Simplify compressor read-head math (no behavior change) — `cpp/compressor.cpp:143`
  Fixed 2026-09-19.
  `(_write_head - (_delay_samples-1) + _delay_samples) % _delay_samples` == `(_write_head+1) % _delay_samples`.
- [x] **K-13** Add missing modules to `_PROJECT_MODULES` shadow guard — `plugin_system.py:20-22`
  Fixed 2026-09-19.
  Missing `abc_score, audio_io, audiocpp_backend, download_util, offline_job_widget` (all exist at root).
- [x] **K-14** Fix `delay.cpp` "soft clipping" comment (do NOT swap to tanh without audition) — `cpp/delay.cpp:99-103`
  Fixed 2026-09-19: comment corrected to hard clamp.
  Code is hard clamp ±2.0, comment says soft. Fix comment only; tanh changes DSP sound.
