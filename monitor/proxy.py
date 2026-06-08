"""Token Monitor HTTP 代理服务器——透传请求至上游并统计 token 用量。"""

from __future__ import annotations

import json
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import MonitorConfig
from .token_stats import RequestUsage, TokenStatsCollector
from .sse_handler import iter_sse_chunks
from .upstream import forward_non_stream, forward_stream


class ProxyHandler(BaseHTTPRequestHandler):
    """OpenAI Chat Completions 代理请求处理器。"""

    config: MonitorConfig
    stats: TokenStatsCollector

    def log_message(self, format: str, *args: Any) -> None:
        pass  # 静默

    def do_POST(self) -> None:
        start_time = time.time()
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        try:
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON body")
            return

        is_stream = body.get("stream", False)
        model = body.get("model", "")
        # 如果配置了 upstream_model，替换请求中的模型名称
        if self.config.upstream_model:
            body["model"] = self.config.upstream_model
            model = self.config.upstream_model
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length")}
        if is_stream:
            self._handle_stream(body, hdrs, model, start_time)
        else:
            self._handle_non_stream(body, hdrs, model, start_time)

    def _handle_non_stream(self, body, headers, model, start_time):
        """处理非流式请求。"""
        status, resp_headers, resp_body, result = forward_non_stream(
            self.config, body, headers)
        duration = time.time() - start_time
        usage = RequestUsage(
            request_id=result.request_id, timestamp=start_time,
            model=result.model or model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            duration=duration, stream=False)
        self.stats.record(usage)
        ct = next((v for k, v in resp_headers.items()
                    if k.lower() == "content-type"), "application/json")
        self._write_response(status, {
            "Content-Type": ct,
            "Content-Length": str(len(resp_body)),
        }, resp_body)

    def _handle_stream(self, body, headers, model, start_time):
        """处理流式 SSE 请求。"""
        status, resp_headers, line_iter, finalizer = forward_stream(
            self.config, body, headers)
        if status >= 400:
            chunks = list(line_iter)
            finalizer()
            error_body = b"".join(chunks)
            self._write_response(status, {
                "Content-Type": "application/json",
                "Content-Length": str(len(error_body)),
            }, error_body)
            return

        if not self._write_response_headers(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
        }):
            finalizer()
            return

        sse_gen = iter_sse_chunks(line_iter)
        pt = ct = tt = cach = rt = 0
        resp_model = request_id = ""
        try:
            while True:
                try:
                    raw_bytes, _ = next(sse_gen)
                    self._write_chunk(raw_bytes)
                except StopIteration as e:
                    if e.value:
                        r = e.value
                        pt, ct, tt = r.prompt_tokens, r.completion_tokens, r.total_tokens
                        cach, rt = r.cached_tokens, r.reasoning_tokens
                        resp_model, request_id = r.model, r.request_id
                    break
        finally:
            finalizer()
            self._write_chunk(b"")

        duration = time.time() - start_time
        usage = RequestUsage(
            request_id=request_id, timestamp=start_time,
            model=resp_model or model,
            prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
            cached_tokens=cach, reasoning_tokens=rt,
            duration=duration, stream=True)
        self.stats.record(usage)

    def _write_chunk(self, data: bytes) -> None:
        """写入 HTTP chunked 编码的数据块。"""
        try:
            chunk = f"{len(data):x}\r\n".encode() + data + b"\r\n"
            self.wfile.write(chunk)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _write_response(self, status: int, headers: dict[str, str], body: bytes) -> bool:
        """安全写回完整 HTTP 响应，客户端提前断开时静默返回。"""
        if not self._write_response_headers(status, headers):
            return False
        try:
            self.wfile.write(body)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _write_response_headers(self, status: int, headers: dict[str, str]) -> bool:
        """安全写回响应头，客户端提前断开时返回 False。"""
        try:
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _send_error(self, code: int, message: str) -> None:
        """发送 JSON 格式的错误响应。"""
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self._write_response(code, {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }, body)


class MonitorHTTPServer(ThreadingHTTPServer):
    """Monitor 专用 HTTP Server，关闭时不等待请求线程阻塞主进程"""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        """客户端提前断开属于正常网络事件，不打印整段异常栈。"""
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def create_proxy_server(
    config: MonitorConfig, stats: TokenStatsCollector
) -> MonitorHTTPServer:
    """创建代理服务器实例。"""
    handler = type(
        "ConfiguredHandler",
        (ProxyHandler,),
        {"config": config, "stats": stats},
    )
    server = MonitorHTTPServer((config.host, config.port), handler)
    return server


def start_proxy(
    config: MonitorConfig, stats: TokenStatsCollector
) -> tuple[MonitorHTTPServer, threading.Thread]:
    """启动代理服务器（后台线程）并返回 (server, thread)。"""
    server = create_proxy_server(config, stats)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
