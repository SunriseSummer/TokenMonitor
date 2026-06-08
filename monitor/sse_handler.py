"""SSE 透传与 usage 收集"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator

from .provider_adapters import estimate_reasoning_tokens
from .sse import encode_done, iter_events


@dataclass
class SSEResult:
    """SSE 处理结果"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    model: str = ""
    request_id: str = ""


def _int_usage_value(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text_len(value) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        total = 0
        for item in value:
            if isinstance(item, dict):
                total += _text_len(item.get("text", ""))
            else:
                total += _text_len(item)
        return total
    return 0


def extract_usage_from_chunk(data: dict) -> dict:
    """从响应 chunk 中提取 usage"""
    usage = data.get("usage")
    if not usage:
        return {}

    result = {
        "prompt_tokens": _int_usage_value(usage.get("prompt_tokens")),
        "completion_tokens": _int_usage_value(usage.get("completion_tokens")),
        "total_tokens": _int_usage_value(usage.get("total_tokens")),
    }

    prompt_details = usage.get("prompt_tokens_details") or {}
    result["cached_tokens"] = _int_usage_value(
        prompt_details.get(
            "cached_tokens",
            usage.get("cached_tokens", usage.get("prompt_cache_hit_tokens", 0)),
        )
    )

    completion_details = usage.get("completion_tokens_details") or {}
    result["reasoning_tokens"] = _int_usage_value(
        completion_details.get(
            "reasoning_tokens",
            usage.get("reasoning_tokens", 0),
        )
    )
    return result


class UsageCollector:
    """从 SSE chunk 中累计 usage 和文本长度"""

    def __init__(self) -> None:
        self.result = SSEResult()
        self.content_text_len = 0
        self.reasoning_text_len = 0

    def consume(self, data: dict) -> None:
        self._collect_delta_text(data)
        usage_info = extract_usage_from_chunk(data)
        if usage_info:
            self._apply_usage(data, usage_info)
        if not self.result.model:
            self.result.model = data.get("model", "")
        if not self.result.request_id:
            self.result.request_id = data.get("id", "")

    def _collect_delta_text(self, data: dict) -> None:
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            self.content_text_len += _text_len(delta.get("content"))
            self.reasoning_text_len += _text_len(delta.get("reasoning_content"))

    def _apply_usage(self, data: dict, usage_info: dict) -> None:
        if not usage_info.get("reasoning_tokens") and self.reasoning_text_len > 0:
            usage_info["reasoning_tokens"] = estimate_reasoning_tokens(
                usage_info["completion_tokens"],
                self.reasoning_text_len,
                self.content_text_len,
            )
        self.result.prompt_tokens = usage_info["prompt_tokens"]
        self.result.completion_tokens = usage_info["completion_tokens"]
        self.result.total_tokens = usage_info["total_tokens"]
        self.result.cached_tokens = usage_info.get("cached_tokens", 0)
        self.result.reasoning_tokens = usage_info.get("reasoning_tokens", 0)
        self.result.model = data.get("model", self.result.model)
        self.result.request_id = data.get("id", self.result.request_id)


def iter_sse_chunks(
    raw_iter,
) -> Generator[tuple[bytes, dict | None], None, SSEResult]:
    """透传 SSE 事件并在结束时返回 usage"""
    collector = UsageCollector()
    for event in iter_events(raw_iter):
        if event.is_comment:
            yield event.raw, None
            continue
        if event.is_done:
            yield encode_done(), None
            continue
        if event.data:
            try:
                collector.consume(json.loads(event.data))
            except (json.JSONDecodeError, KeyError):
                pass
        yield event.raw, None
    return collector.result
