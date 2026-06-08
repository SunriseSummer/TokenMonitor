from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.proxy import ProxyHandler


class _DummyWriter:
    def __init__(self, fail_on_write: bool = False, fail_on_flush: bool = False):
        self.fail_on_write = fail_on_write
        self.fail_on_flush = fail_on_flush
        self.writes: list[bytes] = []
        self.flushed = 0

    def write(self, data: bytes) -> None:
        if self.fail_on_write:
            raise ConnectionResetError(10054, "client disconnected")
        self.writes.append(data)

    def flush(self) -> None:
        if self.fail_on_flush:
            raise BrokenPipeError("pipe closed")
        self.flushed += 1


class _FakeHandler:
    _write_response = ProxyHandler._write_response
    _write_response_headers = ProxyHandler._write_response_headers
    _send_error = ProxyHandler._send_error

    def __init__(
        self,
        *,
        fail_on_headers: bool = False,
        fail_on_write: bool = False,
        fail_on_flush: bool = False,
    ):
        self.fail_on_headers = fail_on_headers
        self.responses: list[int] = []
        self.headers: list[tuple[str, str]] = []
        self.headers_ended = 0
        self.wfile = _DummyWriter(
            fail_on_write=fail_on_write,
            fail_on_flush=fail_on_flush,
        )

    def send_response(self, status: int) -> None:
        self.responses.append(status)

    def send_header(self, key: str, value: str) -> None:
        self.headers.append((key, value))

    def end_headers(self) -> None:
        self.headers_ended += 1
        if self.fail_on_headers:
            raise ConnectionResetError(10054, "client disconnected")


class ProxyResponseWriteTests(unittest.TestCase):
    def test_write_response_headers_returns_false_on_disconnect(self) -> None:
        handler = _FakeHandler(fail_on_headers=True)
        ok = handler._write_response_headers(200, {"Content-Type": "application/json"})
        self.assertFalse(ok)

    def test_write_response_returns_false_when_body_write_fails(self) -> None:
        handler = _FakeHandler(fail_on_write=True)
        ok = handler._write_response(200, {"Content-Length": "4"}, b"test")
        self.assertFalse(ok)

    def test_write_response_returns_false_when_flush_fails(self) -> None:
        handler = _FakeHandler(fail_on_flush=True)
        ok = handler._write_response(200, {"Content-Length": "4"}, b"test")
        self.assertFalse(ok)

    def test_send_error_swallows_client_disconnect(self) -> None:
        handler = _FakeHandler(fail_on_headers=True)
        handler._send_error(400, "Invalid JSON body")
        self.assertEqual(handler.responses, [400])
