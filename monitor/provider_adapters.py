"""上游服务商响应适配"""

from __future__ import annotations

import json
from typing import Iterable

from .sse import encode_data, encode_done, iter_events


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def estimate_reasoning_tokens(
    completion_tokens: int,
    reasoning_text_len: int,
    content_text_len: int,
) -> int:
    """根据文本占比估算 reasoning tokens"""
    if completion_tokens <= 0 or reasoning_text_len <= 0:
        return 0
    total_len = reasoning_text_len + max(0, content_text_len)
    if total_len <= 0:
        return 0
    estimated = int(completion_tokens * reasoning_text_len / total_len + 0.5)
    return max(1, min(completion_tokens, estimated))


def split_think_tags(text: str) -> tuple[str, str]:
    """拆分正文和 think 标签中的思考内容"""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    pos = 0
    while pos < len(text):
        start = text.find(THINK_OPEN, pos)
        if start < 0:
            content_parts.append(text[pos:])
            break
        content_parts.append(text[pos:start])
        reasoning_start = start + len(THINK_OPEN)
        end = text.find(THINK_CLOSE, reasoning_start)
        if end < 0:
            reasoning_parts.append(text[reasoning_start:])
            break
        reasoning_parts.append(text[reasoning_start:end])
        pos = end + len(THINK_CLOSE)
    return "".join(content_parts), "".join(reasoning_parts)


def _text_len(value) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(
            _text_len(item.get("text", "") if isinstance(item, dict) else item)
            for item in value
        )
    return 0


def _set_reasoning_tokens(
    data: dict,
    reasoning_text_len: int,
    content_text_len: int,
) -> None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return

    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = {}
        usage["completion_tokens_details"] = details

    if details.get("reasoning_tokens") is not None:
        return
    if usage.get("reasoning_tokens") is not None:
        details["reasoning_tokens"] = usage["reasoning_tokens"]
        return

    try:
        completion_tokens = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        completion_tokens = 0

    details["reasoning_tokens"] = estimate_reasoning_tokens(
        completion_tokens,
        reasoning_text_len,
        content_text_len,
    )


class ProviderAdapter:
    """服务商响应适配接口"""

    def normalize_response(self, data: dict) -> dict:
        return data

    def normalize_stream(self, raw_iter: Iterable[bytes | str]):
        yield from raw_iter


class ThinkTagStreamNormalizer:
    """增量转换 MiniMax think 标签"""

    def __init__(self) -> None:
        self.in_reasoning = False
        self.pending = ""

    def consume(self, text: str) -> tuple[str, str]:
        buf = self.pending + text
        self.pending = ""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        while buf:
            tag = THINK_CLOSE if self.in_reasoning else THINK_OPEN
            idx = buf.find(tag)
            if idx >= 0:
                self._append(buf[:idx], content_parts, reasoning_parts)
                buf = buf[idx + len(tag):]
                self.in_reasoning = not self.in_reasoning
                continue

            safe, pending = self._split_tag_prefix(buf, tag)
            self._append(safe, content_parts, reasoning_parts)
            self.pending = pending
            break

        return "".join(content_parts), "".join(reasoning_parts)

    def flush(self) -> tuple[str, str]:
        pending = self.pending
        self.pending = ""
        if not pending:
            return "", ""
        if self.in_reasoning:
            return "", pending
        return pending, ""

    def _append(
        self,
        text: str,
        content_parts: list[str],
        reasoning_parts: list[str],
    ) -> None:
        if not text:
            return
        if self.in_reasoning:
            reasoning_parts.append(text)
        else:
            content_parts.append(text)

    def _split_tag_prefix(self, text: str, tag: str) -> tuple[str, str]:
        max_len = min(len(tag) - 1, len(text))
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if tag.startswith(suffix):
                return text[:-size], suffix
        return text, ""


class MiniMaxAdapter(ProviderAdapter):
    """MiniMax 响应适配器"""

    def normalize_response(self, data: dict) -> dict:
        reasoning_len = 0
        content_len = 0
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or THINK_OPEN not in content:
                continue
            clean_content, reasoning = split_think_tags(content)
            message["content"] = clean_content
            if reasoning:
                existing = message.get("reasoning_content") or ""
                message["reasoning_content"] = f"{existing}{reasoning}"
            reasoning_len += _text_len(message.get("reasoning_content"))
            content_len += _text_len(message.get("content"))

        _set_reasoning_tokens(data, reasoning_len, content_len)
        return data

    def normalize_stream(self, raw_iter: Iterable[bytes | str]):
        normalizer = ThinkTagStreamNormalizer()
        for event in iter_events(raw_iter):
            if event.is_comment:
                yield event.raw
                continue
            if event.is_done:
                yield from self._flush_pending(normalizer)
                yield encode_done()
                continue
            if not event.data:
                yield event.raw
                continue
            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                yield event.raw
                continue
            yield encode_data(self._normalize_stream_chunk(data, normalizer))

        yield from self._flush_pending(normalizer)

    def _normalize_stream_chunk(
        self,
        data: dict,
        normalizer: ThinkTagStreamNormalizer,
    ) -> dict:
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if not isinstance(content, str):
                continue
            clean_content, reasoning = normalizer.consume(content)
            if clean_content:
                delta["content"] = clean_content
            else:
                delta.pop("content", None)
            if reasoning:
                existing = delta.get("reasoning_content") or ""
                delta["reasoning_content"] = f"{existing}{reasoning}"
        return data

    def _flush_pending(self, normalizer: ThinkTagStreamNormalizer):
        clean_content, reasoning = normalizer.flush()
        if not clean_content and not reasoning:
            return
        delta = {}
        if clean_content:
            delta["content"] = clean_content
        if reasoning:
            delta["reasoning_content"] = reasoning
        yield encode_data({"choices": [{"delta": delta}]})


def adapter_for(provider: str) -> ProviderAdapter:
    """按服务商返回响应适配器"""
    if provider == "minimax":
        return MiniMaxAdapter()
    return ProviderAdapter()
