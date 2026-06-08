from __future__ import annotations

import json


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _iter_lines(raw_item):
    if isinstance(raw_item, bytes):
        lines = raw_item.splitlines(keepends=True)
    else:
        lines = str(raw_item).splitlines(keepends=True)
    return lines or [raw_item]


def split_think_tags(text: str) -> tuple[str, str]:
    """Split MiniMax <think>...</think> text into content and reasoning."""
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


def normalize_minimax_non_stream_response(data: dict) -> dict:
    """Move MiniMax <think> blocks from message.content to reasoning_content."""
    choices = data.get("choices") or []
    for choice in choices:
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
    return data


class ThinkTagStreamNormalizer:
    """Incrementally convert MiniMax streamed <think> tags to reasoning_content."""

    def __init__(self) -> None:
        self._in_reasoning = False
        self._pending = ""

    def consume(self, text: str) -> tuple[str, str]:
        buf = self._pending + text
        self._pending = ""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        while buf:
            tag = THINK_CLOSE if self._in_reasoning else THINK_OPEN
            idx = buf.find(tag)
            if idx >= 0:
                self._append_current(
                    buf[:idx],
                    content_parts,
                    reasoning_parts,
                )
                buf = buf[idx + len(tag):]
                self._in_reasoning = not self._in_reasoning
                continue

            safe, pending = self._split_tag_prefix(buf, tag)
            self._append_current(safe, content_parts, reasoning_parts)
            self._pending = pending
            break

        return "".join(content_parts), "".join(reasoning_parts)

    def flush(self) -> tuple[str, str]:
        pending = self._pending
        self._pending = ""
        if not pending:
            return "", ""
        if self._in_reasoning:
            return "", pending
        return pending, ""

    def _append_current(
        self,
        text: str,
        content_parts: list[str],
        reasoning_parts: list[str],
    ) -> None:
        if not text:
            return
        if self._in_reasoning:
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


def normalize_minimax_stream_chunk(
    data: dict,
    normalizer: ThinkTagStreamNormalizer,
) -> dict:
    choices = data.get("choices") or []
    for choice in choices:
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


def iter_normalized_minimax_sse(raw_iter):
    """Yield SSE lines with MiniMax think tags converted to reasoning_content."""
    normalizer = ThinkTagStreamNormalizer()
    pending_event_lines: list[bytes] = []
    pending_data_lines: list[str] = []

    def finalize_event():
        if not pending_event_lines and not pending_data_lines:
            return
        event_bytes = b"".join(pending_event_lines) + b"\n"
        data_text = "\n".join(pending_data_lines)
        pending_event_lines.clear()
        pending_data_lines.clear()

        if data_text.strip() == "[DONE]":
            clean_content, reasoning = normalizer.flush()
            if clean_content or reasoning:
                delta = {}
                if clean_content:
                    delta["content"] = clean_content
                if reasoning:
                    delta["reasoning_content"] = reasoning
                body = {"choices": [{"delta": delta}]}
                yield f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(
                    "utf-8"
                )
            yield b"data: [DONE]\n\n"
            return

        if not data_text:
            yield event_bytes
            return

        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            yield event_bytes
            return

        data = normalize_minimax_stream_chunk(data, normalizer)
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        yield f"data: {body}\n\n".encode("utf-8")

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
                yield line_bytes + (b"" if line_bytes.endswith(b"\n") else b"\n")
                continue
            pending_event_lines.append(line_bytes)
            if stripped.startswith("data:"):
                pending_data_lines.append(stripped[5:].strip())

    yield from finalize_event()
    clean_content, reasoning = normalizer.flush()
    if clean_content or reasoning:
        delta = {}
        if clean_content:
            delta["content"] = clean_content
        if reasoning:
            delta["reasoning_content"] = reasoning
        body = {"choices": [{"delta": delta}]}
        yield f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(
            "utf-8"
        )
