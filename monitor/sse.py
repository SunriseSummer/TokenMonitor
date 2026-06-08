"""SSE 分帧与编码工具"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SSEEvent:
    """单个 SSE 事件"""

    raw: bytes
    data: str = ""
    is_comment: bool = False

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"


def iter_lines(raw_item) -> list[bytes | str]:
    """将一次网络读取拆成保留换行的行"""
    if isinstance(raw_item, bytes):
        lines = raw_item.splitlines(keepends=True)
    else:
        lines = str(raw_item).splitlines(keepends=True)
    return lines or [raw_item]


def parse_data_line(line: str) -> str | None:
    """解析 data 行"""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if stripped.startswith("data:"):
        return stripped[5:].strip()
    return None


def iter_events(raw_iter: Iterable[bytes | str]):
    """从字节流中迭代完整 SSE 事件"""
    event_lines: list[bytes] = []
    data_lines: list[str] = []

    def flush_event():
        if not event_lines and not data_lines:
            return None
        raw = b"".join(event_lines) + b"\n"
        data = "\n".join(data_lines)
        event_lines.clear()
        data_lines.clear()
        return SSEEvent(raw=raw, data=data)

    for raw_item in raw_iter:
        for raw_line in iter_lines(raw_item):
            if isinstance(raw_line, bytes):
                line_bytes = raw_line
                line_text = raw_line.decode("utf-8", errors="replace")
            else:
                line_text = raw_line
                line_bytes = raw_line.encode("utf-8")

            stripped = line_text.rstrip("\r\n")
            if not stripped:
                event = flush_event()
                if event is not None:
                    yield event
                continue

            if stripped.startswith(":"):
                event = flush_event()
                if event is not None:
                    yield event
                raw = line_bytes + (b"" if line_bytes.endswith(b"\n") else b"\n")
                yield SSEEvent(raw=raw, is_comment=True)
                continue

            event_lines.append(line_bytes)
            data = parse_data_line(stripped)
            if data is not None:
                data_lines.append(data)

    event = flush_event()
    if event is not None:
        yield event


def encode_data(data: dict) -> bytes:
    """编码 JSON data 事件"""
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode("utf-8")


def encode_done() -> bytes:
    """编码结束事件"""
    return b"data: [DONE]\n\n"
