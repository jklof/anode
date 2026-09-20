"""Direct tests for audiocpp_backend.run_audiocpp_subprocess.

The five run_* wrappers already cover the helper indirectly via stubbed
Popen; these pin the shared lifecycle itself: success tail, cancel,
timeout, and nonzero-exit reporting.
"""
import subprocess
import threading

import pytest

import audiocpp_backend as backend
from audiocpp_backend import GenerationCancelled, run_audiocpp_subprocess


class _StubProc:
    def __init__(self, text="", code=0):
        self._text = text
        self._code = code
        self.terminated = False

    def communicate(self, timeout=None):
        return self._text, ""

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self._code

    def kill(self):
        pass

    def poll(self):
        return None

    @property
    def returncode(self):
        return self._code


def _patch(monkeypatch, proc):
    monkeypatch.setattr(backend.subprocess, "Popen", lambda argv, **kw: proc)


def test_success_returns_tail(monkeypatch):
    proc = _StubProc(text="metrics.rtf=2.0\n", code=0)
    _patch(monkeypatch, proc)
    tail = run_audiocpp_subprocess(
        ["cli"], cancel_event=threading.Event(), timeout_s=10,
        task_label="YuE2 generation", cancel_message="generation cancelled",
    )
    assert "metrics.rtf=2.0" in tail


def test_cancel_raises(monkeypatch):
    proc = _StubProc(code=0)
    _patch(monkeypatch, proc)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GenerationCancelled, match="generation cancelled"):
        run_audiocpp_subprocess(
            ["cli"], cancel_event=cancel, timeout_s=10,
            task_label="YuE2 generation", cancel_message="generation cancelled",
        )


def test_timeout_message_uses_label(monkeypatch):
    class Hanging(_StubProc):
        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="cli", timeout=timeout)

    _patch(monkeypatch, Hanging())
    with pytest.raises(TimeoutError, match="BS-RoFormer separation exceeded"):
        run_audiocpp_subprocess(
            ["cli"], cancel_event=threading.Event(), timeout_s=1,
            task_label="BS-RoFormer separation",
            cancel_message="separation cancelled",
        )


def test_nonzero_exit_reports_tail(monkeypatch):
    proc = _StubProc(text="x" * 5000 + "boom", code=1)
    _patch(monkeypatch, proc)
    with pytest.raises(RuntimeError, match="exited with code 1.*boom"):
        run_audiocpp_subprocess(
            ["cli"], cancel_event=threading.Event(), timeout_s=10,
            task_label="job", cancel_message="cancelled",
        )


def test_launch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backend.subprocess, "Popen",
        lambda argv, **kw: (_ for _ in ()).throw(OSError("nope")),
    )
    with pytest.raises(RuntimeError, match="failed to launch"):
        run_audiocpp_subprocess(["cli"], timeout_s=10)
