"""上游请求转发"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

from .config import MonitorConfig
from .provider_adapters import adapter_for
from .sse_handler import SSEResult, extract_usage_from_chunk


TIMEOUT_SECONDS = 300


def _build_ssl_context() -> ssl.SSLContext:
    """构造宽松 TLS 上下文"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def prepare_request_body(body: dict) -> dict:
    """准备上游请求体"""
    result = dict(body)
    if result.get("stream", False):
        options = result.get("stream_options") or {}
        options["include_usage"] = True
        result["stream_options"] = options
    return result


def _build_request(
    config: MonitorConfig,
    body: dict,
    *,
    accept: str,
) -> urllib.request.Request:
    payload = json.dumps(prepare_request_body(body), ensure_ascii=False).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        config.resolve_endpoint(),
        data=payload,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", accept)
    req.add_header("Authorization", f"Bearer {config.upstream_api_key}")
    return req


def _extract_result(data: dict) -> SSEResult:
    result = SSEResult()
    usage = extract_usage_from_chunk(data)
    if usage:
        result.prompt_tokens = usage["prompt_tokens"]
        result.completion_tokens = usage["completion_tokens"]
        result.total_tokens = usage["total_tokens"]
        result.cached_tokens = usage.get("cached_tokens", 0)
        result.reasoning_tokens = usage.get("reasoning_tokens", 0)
    result.model = data.get("model", "")
    result.request_id = data.get("id", "")
    return result


def _line_iter(resp):
    """按行读取上游 SSE 响应"""
    buffer = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            if buffer:
                yield buffer
            return
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line + b"\n"


def forward_non_stream(
    config: MonitorConfig,
    body: dict,
    headers: dict[str, str],
) -> tuple[int, dict[str, str], bytes, SSEResult]:
    """转发非流式请求"""
    req = _build_request(config, body, accept="application/json")
    adapter = adapter_for(config.upstream_provider)
    result = SSEResult()

    try:
        with urllib.request.urlopen(
            req,
            context=_build_ssl_context(),
            timeout=TIMEOUT_SECONDS,
        ) as resp:
            status = resp.status
            resp_headers = dict(resp.headers)
            resp_body = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        resp_headers = dict(exc.headers)
        resp_body = exc.read()

    if status < 400:
        try:
            data = adapter.normalize_response(json.loads(resp_body))
            resp_body = json.dumps(
                data,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            result = _extract_result(data)
        except (json.JSONDecodeError, KeyError):
            pass

    return status, resp_headers, resp_body, result


def forward_stream(
    config: MonitorConfig,
    body: dict,
    headers: dict[str, str],
):
    """转发流式请求"""
    req = _build_request(config, body, accept="text/event-stream")
    adapter = adapter_for(config.upstream_provider)

    try:
        resp = urllib.request.urlopen(
            req,
            context=_build_ssl_context(),
            timeout=TIMEOUT_SECONDS,
        )
    except urllib.error.HTTPError as exc:
        error_body = exc.read()

        def error_iter():
            yield error_body

        return exc.code, dict(exc.headers), error_iter(), lambda: None

    lines = adapter.normalize_stream(_line_iter(resp))
    return resp.status, dict(resp.headers), lines, lambda: resp.close()
