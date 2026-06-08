"""本地 OpenAI 兼容代理服务"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import MonitorConfig
from .sse_handler import SSEResult, iter_sse_chunks
from .token_stats import RequestUsage, TokenStatsCollector
from .upstream import forward_non_stream, forward_stream


class ProxyHandler(BaseHTTPRequestHandler):
    """OpenAI Chat Completions 请求处理器"""

    config: MonitorConfig
    stats: TokenStatsCollector

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        start_time = time.time()
        body = self._read_json_body()
        if body is None:
            return

        model = self._apply_model_override(body)
        headers = self._forward_headers()
        if body.get("stream", False):
            self._handle_stream(body, headers, model, start_time)
        else:
            self._handle_non_stream(body, headers, model, start_time)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        try:
            return json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON body")
            return None

    def _apply_model_override(self, body: dict) -> str:
        model = body.get("model", "")
        if self.config.upstream_model:
            body["model"] = self.config.upstream_model
            model = self.config.upstream_model
        return model

    def _forward_headers(self) -> dict[str, str]:
        skip = {"host", "content-length"}
        return {k: v for k, v in self.headers.items() if k.lower() not in skip}

    def _handle_non_stream(
        self,
        body: dict,
        headers: dict[str, str],
        model: str,
        start_time: float,
    ) -> None:
        status, resp_headers, resp_body, result = forward_non_stream(
            self.config,
            body,
            headers,
        )
        self._record_usage(result, model, start_time, stream=False)
        content_type = self._content_type(resp_headers, default="application/json")
        self._write_response(
            status,
            {
                "Content-Type": content_type,
                "Content-Length": str(len(resp_body)),
            },
            resp_body,
        )

    def _handle_stream(
        self,
        body: dict,
        headers: dict[str, str],
        model: str,
        start_time: float,
    ) -> None:
        status, resp_headers, line_iter, finalizer = forward_stream(
            self.config,
            body,
            headers,
        )
        if status >= 400:
            self._write_stream_error(status, line_iter, finalizer)
            return

        if not self._write_response_headers(
            200,
            {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        ):
            finalizer()
            return

        result = self._relay_sse(line_iter, finalizer)
        self._record_usage(result, model, start_time, stream=True)

    def _write_stream_error(self, status: int, line_iter, finalizer) -> None:
        chunks = list(line_iter)
        finalizer()
        body = b"".join(chunks)
        self._write_response(
            status,
            {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body,
        )

    def _relay_sse(self, line_iter, finalizer) -> SSEResult:
        sse_gen = iter_sse_chunks(line_iter)
        result = SSEResult()
        try:
            while True:
                try:
                    raw_bytes, _ = next(sse_gen)
                    self._write_chunk(raw_bytes)
                except StopIteration as stop:
                    result = stop.value or SSEResult()
                    break
        finally:
            finalizer()
            self._write_chunk(b"")
        return result

    def _record_usage(
        self,
        result: SSEResult,
        fallback_model: str,
        start_time: float,
        *,
        stream: bool,
    ) -> None:
        usage = RequestUsage(
            request_id=result.request_id,
            timestamp=start_time,
            model=result.model or fallback_model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            duration=time.time() - start_time,
            stream=stream,
        )
        self.stats.record(usage)

    def _content_type(
        self,
        headers: dict[str, str],
        *,
        default: str,
    ) -> str:
        return next(
            (v for k, v in headers.items() if k.lower() == "content-type"),
            default,
        )

    def _write_chunk(self, data: bytes) -> None:
        try:
            chunk = f"{len(data):x}\r\n".encode() + data + b"\r\n"
            self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _write_response(
        self,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        if not self._write_response_headers(status, headers):
            return False
        try:
            self.wfile.write(body)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _write_response_headers(
        self,
        status: int,
        headers: dict[str, str],
    ) -> bool:
        try:
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self._write_response(
            code,
            {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body,
        )


class MonitorHTTPServer(ThreadingHTTPServer):
    """Monitor 专用 HTTP 服务"""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def create_proxy_server(
    config: MonitorConfig,
    stats: TokenStatsCollector,
) -> MonitorHTTPServer:
    """创建代理服务实例"""
    handler = type(
        "ConfiguredHandler",
        (ProxyHandler,),
        {"config": config, "stats": stats},
    )
    return MonitorHTTPServer((config.host, config.port), handler)


def start_proxy(
    config: MonitorConfig,
    stats: TokenStatsCollector,
) -> tuple[MonitorHTTPServer, threading.Thread]:
    """启动后台代理服务"""
    server = create_proxy_server(config, stats)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
