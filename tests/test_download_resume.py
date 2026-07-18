"""
Tests for data/download_gravityspy.py's resumable download logic.

Uses a local HTTP server (tests/_local_http_server.py) rather than mocking
requests internals -- this exercises the real Range-request / retry code
path exactly as it runs against the real Zenodo server, just pointed at
localhost instead. This is what originally caught and verified the fix
for a real download failure: a user's 5.5GB download dropped at 85% with
an IncompleteRead, and the old code had no resume logic at all, forcing
a full restart.
"""
import hashlib

import pytest

from data.download_gravityspy import download_file
from tests._local_http_server import make_server


@pytest.fixture
def payload():
    # Deterministic, verifiable content (not just size) -- catches
    # corruption that a size-only check would miss.
    return bytes((i % 251) for i in range(2 * 1024 * 1024))


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_clean_download_no_interruption(tmp_path, payload):
    httpd, port = make_server(payload)
    dest = tmp_path / "out.bin"
    try:
        download_file(f"http://127.0.0.1:{port}/file", dest,
                       expected_size=len(payload), chunk_size=64 * 1024,
                       _sleep=lambda s: None)
        assert dest.stat().st_size == len(payload)
        assert _hash(dest.read_bytes()) == _hash(payload)
    finally:
        httpd.shutdown()


def test_resumes_after_connection_drop_within_one_call(tmp_path, payload):
    # download_file's own internal retry loop should recover from a single
    # interruption without the caller needing to do anything.
    drop_point = int(len(payload) * 0.85)
    httpd, port = make_server(payload, drop_after_bytes=drop_point)
    dest = tmp_path / "out.bin"
    try:
        download_file(f"http://127.0.0.1:{port}/file", dest,
                       expected_size=len(payload), chunk_size=64 * 1024,
                       max_retries=3, _sleep=lambda s: None)
        assert dest.stat().st_size == len(payload)
        assert _hash(dest.read_bytes()) == _hash(payload)
    finally:
        httpd.shutdown()


def test_resumes_across_separate_invocations(tmp_path, payload):
    # The realistic CLI scenario: one process invocation exhausts its
    # retries and exits with an error (as the real download did), then a
    # completely separate re-run of the script picks up where it left off.
    drop_point = int(len(payload) * 0.85)
    httpd, port = make_server(payload, drop_after_bytes=drop_point)
    dest = tmp_path / "out.bin"
    url = f"http://127.0.0.1:{port}/file"
    try:
        with pytest.raises(RuntimeError, match="re-run this script"):
            download_file(url, dest, expected_size=len(payload),
                           chunk_size=64 * 1024, max_retries=1, _sleep=lambda s: None)
        assert dest.exists()
        assert 0 < dest.stat().st_size < len(payload)

        # Fresh call, simulating a new process invocation
        download_file(url, dest, expected_size=len(payload),
                       chunk_size=64 * 1024, max_retries=3, _sleep=lambda s: None)
        assert dest.stat().st_size == len(payload)
        assert _hash(dest.read_bytes()) == _hash(payload)
    finally:
        httpd.shutdown()


def test_falls_back_to_clean_restart_when_server_ignores_range(tmp_path, payload):
    # Some servers/CDNs don't support Range requests. If ours ever hits
    # one, it must restart cleanly rather than append the full content
    # onto an existing partial file (which would corrupt it).
    httpd, port = make_server(payload, support_range=False)
    dest = tmp_path / "out.bin"
    # Pre-seed a "partial" file as if from an earlier interrupted attempt
    dest.write_bytes(payload[: len(payload) // 2])
    try:
        download_file(f"http://127.0.0.1:{port}/file", dest,
                       expected_size=len(payload), chunk_size=64 * 1024,
                       _sleep=lambda s: None)
        assert dest.stat().st_size == len(payload)
        assert _hash(dest.read_bytes()) == _hash(payload)
    finally:
        httpd.shutdown()


def test_already_complete_file_is_not_redownloaded(tmp_path, payload):
    dest = tmp_path / "out.bin"
    dest.write_bytes(payload)
    # Point at a port nothing is listening on -- if this tries to make a
    # request at all, it'll fail with a connection error, so success here
    # proves it correctly skipped the network call entirely.
    download_file("http://127.0.0.1:1/unreachable", dest,
                   expected_size=len(payload), _sleep=lambda s: None)
    assert dest.read_bytes() == payload
