"""Tests for download_util: resume, cancellation, verification.

A tiny in-process HTTP server serves deterministic bytes with explicit
Range support (plus modes that ignore Range or fail transiently), so no
network or GPU is needed.
"""
import hashlib
import http.server
import threading

import pytest

from download_util import (
    DownloadCancelled,
    DownloadError,
    DownloadSpec,
    fetch_all,
    format_bytes,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "TestDL/1"

    def _send_body(self, head_only):
        srv = self.server
        srv.requests.append((self.command, self.path,
                             self.headers.get("Range")))
        if srv.fail_first > 0 and self.command == "GET":
            srv.fail_first -= 1
            self.send_response(500)
            self.end_headers()
            return
        body = srv.content
        if srv.support_range:
            rng = self.headers.get("Range")
            if rng:
                try:
                    start = int(rng.split("=")[1].split("-")[0])
                except ValueError:
                    start = 0
                if start >= len(body):
                    self.send_response(416)
                    self.end_headers()
                    return
                body = body[start:]
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{start + len(body) - 1}/{len(srv.content)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(body)
                return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def do_GET(self):
        self._send_body(head_only=False)

    def do_HEAD(self):
        self._send_body(head_only=True)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server(tmp_path):
    content = bytes(range(256)) * 256  # 64 KiB deterministic
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.content = content
    httpd.support_range = True
    httpd.fail_first = 0
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def _spec(server, tmp_path, name="f.bin", size=None, sha=None):
    port = server.server_address[1]
    content = server.content
    return DownloadSpec(f"http://127.0.0.1:{port}/{name}", tmp_path / name,
                        len(content) if size is None else size,
                        _sha(content) if sha is None else sha,
                        label=name)


def test_full_download_verifies(server, tmp_path):
    spec = _spec(server, tmp_path)
    (dest,) = fetch_all([spec])
    assert dest.read_bytes() == server.content
    assert any(r[2] is None for r in server.requests)  # no Range on fresh fetch


def test_present_file_skips_network(server, tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(server.content)
    (out,) = fetch_all([_spec(server, tmp_path)])
    assert out == dest
    assert server.requests == []


def test_resume_from_partial(server, tmp_path):
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(server.content[:8192])
    (out,) = fetch_all([_spec(server, tmp_path)])
    assert out.read_bytes() == server.content
    ranges = [r[2] for r in server.requests if r[2]]
    assert ranges and ranges[0].startswith("bytes=8192-")


def test_cancel_keeps_part_then_resumes(server, tmp_path):
    import threading as th
    server.content = bytes(range(256)) * 256 * 64  # 4 MiB
    spec = _spec(server, tmp_path)
    cancel = th.Event()

    def progress(label, done, total, state):
        if state == "downloading" and done > 0:
            cancel.set()

    with pytest.raises(DownloadCancelled):
        fetch_all([spec], progress_cb=progress, cancel_event=cancel)
    part = spec.dest.with_name(spec.dest.name + ".part")
    assert part.exists() and not spec.dest.exists()
    (out,) = fetch_all([spec])  # resume to completion
    assert out.read_bytes() == server.content


def test_hash_mismatch_rejects(server, tmp_path):
    spec = _spec(server, tmp_path, sha="0" * 64)
    with pytest.raises(DownloadError, match="[Ss][Hh][Aa]256"):
        fetch_all([spec])
    assert not spec.dest.exists()
    assert not spec.dest.with_name(spec.dest.name + ".part").exists()


def test_size_mismatch_rejects(server, tmp_path):
    spec = _spec(server, tmp_path, size=len(server.content) + 1)
    with pytest.raises(DownloadError, match="[Ss]ize"):
        fetch_all([spec])
    assert not spec.dest.exists()


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    server.support_range = False
    dest = tmp_path / "f.bin"
    dest.with_name("f.bin.part").write_bytes(b"stale-partial")
    (out,) = fetch_all([_spec(server, tmp_path)])
    assert out.read_bytes() == server.content


def test_transient_500_retries(server, tmp_path):
    server.fail_first = 2
    (out,) = fetch_all([_spec(server, tmp_path)])
    assert out.read_bytes() == server.content
    assert len(server.requests) == 3


def test_progress_reports_done_and_total(server, tmp_path):
    events = []
    fetch_all([_spec(server, tmp_path)],
              progress_cb=lambda *a: events.append(a))
    states = [e[3] for e in events]
    assert "downloading" in states and states[-1] == "done"
    label, done, total, _ = events[-1]
    assert done == total == len(server.content)


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(None) == "?"
