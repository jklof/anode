"""Shared client for the in-tree audio.cpp native runtime (tools/audiocpp/).

audio.cpp is a C++ inference runtime for local audio models. ANode drives
it as a sidecar *subprocess* for offline jobs (song generation, score
transcription, …) that must never run on the real-time audio thread
(AGENTS.md sections 4 and 6).

This module has no Qt dependency and performs no audio processing itself:

* :class:`AudioCppRuntime` — locates the `audiocpp_cli` binary and the
  model directories, plus a cheap health check.
* :func:`run_yue2_gen` — blocking YuE2 generation call, designed to run
  as an NRT worker function (``Node.submit_nrt``). Supports cooperative
  cancellation so a superseded generation does not burn GPU time.
* :class:`AudioCppJob` — :class:`base.Node` base with generate/cancel
  plumbing shared by all audio.cpp-backed nodes. Lives here (not under
  ``plugins/``) so it is not auto-registered in the node palette.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from base import Node

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
AUDIOCPP_PIN_VERSION = "v0.8.0"


class GenerationCancelled(RuntimeError):
    """Raised when a generation job is cancelled before producing output."""


class AudioCppRuntime:
    """Locates the in-tree audio.cpp runtime. Cheap filesystem checks only."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else REPO_ROOT / "tools" / "audiocpp"
        self.bin_dir = self.root / "bin"
        self.cli = self.bin_dir / ("audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli")

    def check_ready(self):
        """Return (ok, hint). Never spawns a process."""
        if not self.cli.exists():
            return False, (
                "audio.cpp runtime not found. Run from the repo root: "
                "python tools/audiocpp/fetch_audiocpp.py"
            )
        return True, ""

    def health(self, timeout_s=60):
        """Run `--list-devices`. Returns (ok, output). Only called off the
        audio thread (NRT worker or control thread)."""
        ok, hint = self.check_ready()
        if not ok:
            return False, hint
        try:
            proc = subprocess.run(
                [str(self.cli), "--list-devices"],
                capture_output=True, text=True, timeout=timeout_s,
                **_no_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"audio.cpp health check failed: {e}"
        if proc.returncode != 0:
            return False, f"audio.cpp health check failed: {proc.stderr[-500:]}"
        return True, proc.stdout


def _no_window_kwargs():
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _build_yue2_argv(cli, model_dir, *, lyrics, style, cot, seed,
                     main_gguf, vae_gguf, abc_file, out_wav):
    """Pure argv builder (no process). Tested without a GPU."""
    if cot not in ("off", "melody", "full"):
        raise ValueError(f"cot must be off/melody/full, got {cot!r}")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 63:
        raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
    if not lyrics or not lyrics.strip():
        raise ValueError("lyrics must be nonempty text")
    if not style or not style.strip():
        raise ValueError("style must be nonempty text")
    argv = [
        str(cli), "--task", "gen", "--family", "yue2",
        "--model", str(model_dir), "--backend", "cuda", "--threads", "8",
        "--lyrics", lyrics,
        "--request-option", f"style={style}",
        "--request-option", f"cot={cot}",
        "--session-option", f"yue2.model_gguf={main_gguf}",
        "--session-option", f"yue2.vae_gguf={vae_gguf}",
        "--seed", str(seed),
        "--out", str(out_wav), "--metrics",
    ]
    if abc_file:
        if cot == "off":
            raise ValueError("an ABC score requires cot=melody or cot=full")
        argv += ["--request-option", f"abc_file={abc_file}"]
    return argv


_METRIC_RE = re.compile(r"^metrics\.(\w+)=([0-9.]+)\s*$")


def _parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line.strip())
        if m:
            try:
                metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return metrics


def run_yue2_gen(cli, model_dir, *, lyrics, style, cot="full", seed=831001,
                 main_gguf="yue2-3b-q4_k_m.gguf", vae_gguf="yue2-vae-f16.gguf",
                 abc_file=None, out_wav, cancel_event=None, timeout_s=1800):
    """Run one YuE2 generation. Blocking; call only from an NRT worker.

    Returns ``{"wav": str, "metrics": dict}``. Raises
    :class:`GenerationCancelled` on cancellation, ``RuntimeError`` /
    ``ValueError`` / ``FileNotFoundError`` on failure.
    """
    cli, model_dir, out_wav = Path(cli), Path(model_dir), Path(out_wav)
    if not cli.exists():
        raise FileNotFoundError(f"audiocpp_cli not found: {cli}")
    main_path, vae_path = model_dir / main_gguf, model_dir / vae_gguf
    if not main_path.exists():
        raise FileNotFoundError(
            f"GGUF profile not downloaded: {main_path} "
            "(run tools/audiocpp/fetch_yue2_gguf.py)"
        )
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE GGUF not found: {vae_path}")
    if abc_file and not Path(abc_file).exists():
        raise FileNotFoundError(f"ABC score file not found: {abc_file}")
    argv = _build_yue2_argv(cli, model_dir, lyrics=lyrics, style=style,
                            cot=cot, seed=seed, main_gguf=main_gguf,
                            vae_gguf=vae_gguf, abc_file=abc_file, out_wav=out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, **_no_window_kwargs(),
        )
    except OSError as e:
        raise RuntimeError(f"failed to launch audio.cpp: {e}")
    try:
        elapsed = 0.0
        step = 0.5
        output_tail = ""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("generation cancelled")
            try:
                # May be retried after TimeoutExpired; returns full output.
                output_tail, _ = proc.communicate(timeout=step)
                break
            except subprocess.TimeoutExpired:
                elapsed += step
                if elapsed >= timeout_s:
                    raise TimeoutError(
                        f"YuE2 generation exceeded {timeout_s}s; terminating"
                    )
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"audio.cpp exited with code {proc.returncode}: "
            f"{output_tail[-2000:].strip()}"
        )
    if not out_wav.exists():
        raise RuntimeError("audio.cpp reported success but wrote no WAV file")
    return {"wav": str(out_wav), "metrics": _parse_metrics(output_tail)}


class AudioCppJob(Node):
    """Node base with generate/cancel plumbing for audio.cpp sidecar jobs.

    A new job cancels the previous one first: a superseded multi-minute
    GPU job must not keep burning VRAM behind the replacement. The cancel
    event is set from the engine/control thread; the pool-thread worker
    terminates its own subprocess — ``Popen.terminate()`` is never called
    from the audio path.
    """

    def __init__(self, name=""):
        super().__init__(name)
        self._job_lock = threading.Lock()
        self._job_cancel = None

    # ------------------------------------------------------------------
    # job control (engine/control thread)
    # ------------------------------------------------------------------
    def submit_job(self, tag, fn, *args):
        """Cancel any in-flight job, then submit ``fn(cancel_event, *args)``."""
        self._cancel_job()
        cancel = threading.Event()
        with self._job_lock:
            self._job_cancel = cancel
        self.submit_nrt(fn, cancel, *args, tag=tag)

    def _cancel_job(self):
        with self._job_lock:
            cancel = self._job_cancel
            self._job_cancel = None
        if cancel is not None:
            cancel.set()

    def remove(self):
        self._cancel_job()

    def on_nrt_discarded(self, tag, ok, result):
        self._cleanup_discarded(tag, ok, result)

    def _cleanup_discarded(self, tag, ok, result):
        """Subclass hook: release resources carried by a superseded result."""

    # ------------------------------------------------------------------
    # artifact helpers (NRT worker only)
    # ------------------------------------------------------------------
    @staticmethod
    def new_run_dir(model_dir, prefix):
        """Fresh per-generation directory for CLI output. Caller owns it."""
        outputs = Path(model_dir) / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(outputs)))

    @staticmethod
    def keep_artifacts(run_dir, keep_dir, names):
        """Move named artifacts into a timestamped keep directory. Returns it."""
        keep_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = Path(run_dir) / name
            if src.exists():
                shutil.move(str(src), str(keep_dir / name))
        shutil.rmtree(run_dir, ignore_errors=True)
        return keep_dir

    @staticmethod
    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
