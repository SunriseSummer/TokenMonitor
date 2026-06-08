"""
上游服务商路由模块。

负责将下游请求转发到上游 OpenAI 兼容接口，
支持流式和非流式两种模式，确保 usage 信息完整返回。
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
import ssl
from typing import Any

from .config import MonitorConfig
from .provider_adapters import (
    iter_normalized_minimax_sse,
    normalize_minimax_non_stream_response,
)
from .sse_handler import SSEResult, extract_usage_from_chunk


def _build_ssl_context() -> ssl.SSLContext:
    """构建不验证证书的 SSL 上下文（与 Coder 的 INSECURE_TLS 对齐）。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def prepare_request_body(body: dict) -> dict:
    """修改请求体以确保上游返回 usage 信息。

    对于流式请求，注入 stream_options.include_usage = true，
    让上游在最后一个 SSE chunk 中附带完整的 usage 数据。
    """
    result = dict(body)
    if result.get("stream", False):
        opts = result.get("stream_options") or {}
        opts["include_usage"] = True
        result["stream_options"] = opts
    return result


def forward_non_stream(
    config: MonitorConfig, body: dict, headers: dict[str, str]
) -> tuple[int, dict[str, str], bytes, SSEResult]:
    """非流式转发：发送请求并解析完整响应。"""
    endpoint = config.resolve_endpoint()
    payload = json.dumps(prepare_request_body(body)).encode("utf-8")

    req = urllib.request.Request(
        endpoint, data=payload, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {config.upstream_api_key}")

    ssl_ctx = _build_ssl_context()
    result = SSEResult()

    try:
        with urllib.request.urlopen(req, context=ssl_ctx,
                                    timeout=300) as resp:
            resp_body = resp.read()
            resp_headers = dict(resp.headers)
            status = resp.status
    except urllib.error.HTTPError as e:
        resp_body = e.read()
        resp_headers = dict(e.headers)
        status = e.code

    # 解析 usage
    if status < 400:
        try:
            data = json.loads(resp_body)
            if config.upstream_provider == "minimax":
                data = normalize_minimax_non_stream_response(data)
                resp_body = json.dumps(
                    data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            usage_info = extract_usage_from_chunk(data)
            if usage_info:
                result.prompt_tokens = usage_info["prompt_tokens"]
                result.completion_tokens = usage_info["completion_tokens"]
                result.total_tokens = usage_info["total_tokens"]
                result.cached_tokens = usage_info.get("cached_tokens", 0)
                result.reasoning_tokens = usage_info.get(
                    "reasoning_tokens", 0
                )
            result.model = data.get("model", "")
            result.request_id = data.get("id", "")
        except (json.JSONDecodeError, KeyError):
            pass

    return status, resp_headers, resp_body, result


def forward_stream(
    config: MonitorConfig, body: dict, headers: dict[str, str]
):
    """流式转发：返回 (status, resp_headers, line_iterator, finalizer)。

    line_iterator 产生原始字节行，供 proxy 层 SSE 解析和转发。
    finalizer 是一个无参函数，调用后关闭上游连接。
    """
    endpoint = config.resolve_endpoint()
    payload = json.dumps(prepare_request_body(body)).encode("utf-8")

    req = urllib.request.Request(
        endpoint, data=payload, method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    req.add_header("Authorization", f"Bearer {config.upstream_api_key}")

    ssl_ctx = _build_ssl_context()

    try:
        resp = urllib.request.urlopen(req, context=ssl_ctx, timeout=300)
    except urllib.error.HTTPError as e:
        error_body = e.read()
        resp_headers = dict(e.headers)

        def empty_iter():
            yield error_body
        return e.code, resp_headers, empty_iter(), lambda: None

    resp_headers = dict(resp.headers)

    def line_iter():
        """逐行读取上游 SSE 流。"""
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                if buf:
                    yield buf
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line + b"\n"

    lines = line_iter()
    if config.upstream_provider == "minimax":
        lines = iter_normalized_minimax_sse(lines)

    return resp.status, resp_headers, lines, lambda: resp.close()
