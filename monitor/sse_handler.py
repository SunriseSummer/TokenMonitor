"""
SSE（Server-Sent Events）流式响应处理模块。

负责：
  1. 逐行解析上游 SSE 数据流
  2. 从最后一个 chunk 提取 usage 信息
  3. 将数据透传给下游客户端
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class SSEResult:
    """SSE 流处理结果，包含完整的 usage 数据。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    model: str = ""
    request_id: str = ""


def parse_sse_line(line: str) -> str | None:
    """解析单行 SSE 数据，返回 data 字段内容或 None。"""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if stripped.startswith("data:"):
        return stripped[5:].strip()
    return None


def _int_usage_value(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_lines(raw_item):
    if isinstance(raw_item, bytes):
        lines = raw_item.splitlines(keepends=True)
    else:
        lines = str(raw_item).splitlines(keepends=True)
    return lines or [raw_item]


def extract_usage_from_chunk(data: dict) -> dict:
    """从 SSE chunk 中提取 usage 信息。

    OpenAI 兼容接口在 stream_options.include_usage=true 时，
    最后一个 chunk 会包含完整的 usage 字段。
    """
    usage = data.get("usage")
    if not usage:
        return {}
    result = {
        "prompt_tokens": _int_usage_value(usage.get("prompt_tokens")),
        "completion_tokens": _int_usage_value(
            usage.get("completion_tokens")
        ),
        "total_tokens": _int_usage_value(usage.get("total_tokens")),
    }
    # 提取 prompt_tokens_details 中的缓存 token
    details = usage.get("prompt_tokens_details") or {}
    result["cached_tokens"] = _int_usage_value(
        details.get(
            "cached_tokens",
            usage.get(
                "cached_tokens",
                usage.get("prompt_cache_hit_tokens", 0),
            ),
        )
    )
    # 提取 completion_tokens_details 中的推理 token
    comp_details = usage.get("completion_tokens_details") or {}
    result["reasoning_tokens"] = _int_usage_value(
        comp_details.get(
            "reasoning_tokens",
            usage.get("reasoning_tokens", 0),
        )
    )
    return result


def iter_sse_chunks(
    raw_iter,
) -> Generator[tuple[bytes, dict | None], None, SSEResult]:
    """迭代 SSE 数据流，透传原始字节并收集 usage。

    Args:
        raw_iter: 产生原始字节行的可迭代对象。

    Yields:
        (raw_bytes, parsed_data_or_None) 二元组。

    Returns:
        SSEResult 汇总。
    """
    result = SSEResult()
    pending_event_lines: list[bytes] = []
    pending_data_lines: list[str] = []

    def finalize_event() -> Generator[tuple[bytes, dict | None], None, None]:
        if not pending_event_lines and not pending_data_lines:
            return

        data_text = "\n".join(pending_data_lines)
        event_bytes = b"".join(pending_event_lines) + b"\n"
        pending_event_lines.clear()
        pending_data_lines.clear()

        if data_text.strip() == "[DONE]":
            yield (b"data: [DONE]\n\n", None)
            return

        if data_text:
            try:
                data = json.loads(data_text)
                usage_info = extract_usage_from_chunk(data)
                if usage_info:
                    result.prompt_tokens = usage_info["prompt_tokens"]
                    result.completion_tokens = usage_info[
                        "completion_tokens"
                    ]
                    result.total_tokens = usage_info["total_tokens"]
                    result.cached_tokens = usage_info.get(
                        "cached_tokens", 0
                    )
                    result.reasoning_tokens = usage_info.get(
                        "reasoning_tokens", 0
                    )
                if not result.model:
                    result.model = data.get("model", "")
                if not result.request_id:
                    result.request_id = data.get("id", "")
            except (json.JSONDecodeError, KeyError):
                pass

        yield (event_bytes, None)

    for raw_item in raw_iter:
        for raw_line in _iter_lines(raw_item):
            if isinstance(raw_line, bytes):
                line_bytes = raw_line
                line_str = raw_line.decode("utf-8", errors="replace")
            else:
                line_str = raw_line
                line_bytes = raw_line.encode("utf-8")

            stripped = line_str.rstrip("\r\n")

            if not stripped:
                yield from finalize_event()
                continue

            if stripped.startswith(":"):
                yield from finalize_event()
                yield (
                    line_bytes
                    + (b"" if line_bytes.endswith(b"\n") else b"\n"),
                    None,
                )
                continue

            pending_event_lines.append(line_bytes)
            data = parse_sse_line(stripped)
            if data is not None:
                pending_data_lines.append(data)

    yield from finalize_event()

    return result
