# ANode Bug Backlog

Prioritized, session-sized batches. Source invariants: `AGENTS.md`.

## Workflow (every session)

1. Pick one batch below (3–5 items max, in order).
2. Track with `TodoWrite` (one `in_progress` at a time).
3. One logical fix at a time, small/local, preserve public node/port/param names + patch compat.
4. Add/extend regression test per bug where noted.
5. Run smallest relevant tests (`conda run -n anode-dev pytest tests/<file> -q`), then full suite when practical.
6. Do not commit/push unless explicitly requested.
7. Mark item `[x]` here when done + verified.

Legend: `[ ]` pending, `[~]` in progress, `[x]` done. Severity: H/M/L.

## Archive — fixed sessions (summary, details removed 2026-09-19)

All Session 1–5 items (B-001…B-023, B-025…B-031, O-01…O-04, O-07…O-11,
K-02, K-05…K-08, K-11…K-14) were fixed + test-verified 2026-09-19.
Full notes removed to keep this file actionable. Still open: B-024,
O-05, O-06, K-01, K-03, K-04, K-09, K-10 (carried forward below).

## Carried-forward (still open)
- [ ] **B-024 (M)** installed-runtime health check missing, gate hides valid
  repairs — `audiocpp_backend.py`, REVERTED 2026-09-19, needs redesign.
  Staging zips are deleted after install so `missing_specs()` alone always
  re-offers runtime even when installed (broke 3 fetch tests + "Everything
  already downloaded" UX). Fix needs CLI+libs health check in `bin_dir`,
  not gate removal. (Superseded in part by F-03 for non-Windows; keep
  B-024 for the Windows health-check redesign.)
- [ ] **O-05** `_get_downstream_nodes` O(V·E) only if slow — `core.py`
- [ ] **O-06** Analyzer `.item()` syncs already rate-limited, keep/document
- [ ] **K-01** Shared sliding-FFT helper — spectrum vs spectrogram plugins
- [ ] **K-03** `BiquadFilter` reuse `FFINode` mono-dup — `plugins/filters.py`
- [ ] **K-04** Split `_apply_command` (~350 lines) to `_op_<name>` — `core.py`
- [ ] **K-09** Standardize on `.value` (not `get_staging_safe`) — `audiocpp_backend.py`
  (`.value` cleanup half; UI-push half superseded by F-04 pattern)
- [ ] **K-10** `ui_system` split only when touching + test gaps (SamplePlayer
  restore, FFINode zero-fill, unknown icon, rubberband quarantine)

## Session 6 — Offline review batch (new 2026-09-19, do first, in order)

- [x] **F-01 (H)** `_build_bs_roformer_argv` emits broken CLI order for
  non-default `num_overlap` — fixed 2026-09-20: slice-insert flag+value as
  one pair before `--audio`; extended `test_argv_nondefault_overlap_emitted`
  with order assertions. Full suite green (890 passed).
  `argv.insert(argv.index("--audio"), value)` then
  `argv.insert(argv.index("--audio"), "--session-option")` leaves
  `[..., "bs_roformer.num_overlap=N", "--session-option", "--audio", ...]`:
  the value dangles as a positional and `--session-option` consumes
  `--audio` as its value. Every non-default overlap run fails.
  Fix: insert flag+value as one pair before `--audio`
  (`i = argv.index("--audio"); argv[i:i] = ["--session-option", ...]`).
  Test: extend `test_argv_nondefault_overlap_emitted` — flag immediately
  followed by value, both before `--audio`.
- [x] **F-02 (H)** `proc.terminate()` without `wait()` leaks sidecars in all
  five `run_*` workers — fixed 2026-09-20: shared `_terminate(proc)` helper
  (terminate, wait with timeout, kill+wait fallback, OSError guards) called
  from all five `finally` blocks; fake-Popen reap + escalate tests.
  Full suite green (892 passed).
  Cancel/timeout path never reaps the child (zombie on POSIX, lingering
  CUDA holder on Windows); a relaunched job races the dying process for
  VRAM. Fix: shared `_terminate(proc)` helper — `terminate()`,
  `wait(timeout)`, `kill()` + `wait()` fallback, best-effort OSError
  guards. Test: fake-Popen cancel path asserts `wait` called, no
  `poll() is None` survivor.
- [x] **F-03 (M)** `_build_fetch_payload` blocks model-only/manual-runtime
  downloads on Linux/macOS — fixed 2026-09-20: catch the platform error,
  empty runtime spec list + same staging/bin_dir layout, `"runtime_note"`
  hint in the download detail line (Windows strings byte-identical).
  Non-win32 + manual CLI test; full suite green modulo known-flaky
  vocal-latency timing test (passes standalone).
- [x] **F-04 (M)** Note/source `_fail`/status paths invisible when engine is
  stopped — fixed 2026-09-20: `_NoteNodeBase._fail` and
  `_FileSourceBase._publish` call `_refresh_ui()` (new file-source helper,
  LyricFitter pattern); `"busy"` appended to both telemetries (notes busy
  while Publishing/Loading, sources always idle). 3 tests. Full suite
  green except known-flaky vocal-latency timing test (flakes identically
  with and without this change — environmental).
- [x] **F-05 (M)** Score section click misses `%name` (no space) and empty
  sections — fixed 2026-09-20: tolerant finder (`"% name"` then `"%"name"`)
  in both `OfflineJobWidget` and note widgets; `"—"`/empty is a safe
  no-op. Click tests for no-space headers + safe no-op. Full suite green
  (900 passed).
- [x] **F-06 (L)** Docstring advertises `ABCFileSource`, class missing —
  fixed 2026-09-20: docstring corrected to `TextFileSource`-only (removal
  was deliberate); registry test locks `ABCFileSource` absence.
- [x] **F-07 (L)** `LyricFitterWidget` never shows detail errors —
  reviewed 2026-09-20, already fixed: `on_telemetry`
  (`plugins/lyric_fitter.py:86-97`) renders the `audio` key into
  `lbl_detail` with red-on-Error styling. No change needed.

## Session 7 — next (only after Session 6)

- B-024 (Windows health-check redesign), O-05, O-06, K-01, K-03, K-04,
  K-09 (`.value` half), K-10.

## Deliberately NOT filed (review false positives, verified)

- Pulse contracts (`ready`/`done` naming, pulse reset in `start`,
  `buf.zero_()` + `fill_(1.0)`): all `AudioCppJob` subclasses,
  `LyricFitter`, note/source nodes implement the AGENTS.md section 6
  one-block pulse + anti-ghosting correctly.
- Audio-thread `submit_nrt` / rising-edge `push_command(("param", …))`:
  sanctioned patterns (per-node epoch/inbox is engine-side; URI snapshot
  is the allowed audio-thread read).
- `on_nrt_complete` direct `param.set()+sync()`: engine/control thread
  between blocks, not audio thread — permitted.
- Empty-transcript RuntimeError on silence (Qwen3): intended fail-loudly,
  surfaced via `_fail` + telemetry.
- Qwen3 >30 s pre-check, BSRoFormer 44.1 kHz ingest + instrumental=None
  tolerance, YuE2 argv validation, SheetSage2 model-dir check: present.
- `outputs["text"]` alias on file sources: intentional saved-patch compat.
- `communicate(timeout)` step-loop + GenerationCancelled flow: correct;
  F-02 covers only the missing reap.

