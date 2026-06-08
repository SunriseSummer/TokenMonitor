"""Token 用量记录与汇总"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RequestUsage:
    """单次请求 token 用量"""

    request_id: str = ""
    timestamp: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    effective_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    duration: float = 0.0
    stream: bool = False

    def finalize(self) -> None:
        """补齐派生字段"""
        self.effective_prompt_tokens = max(
            0,
            self.prompt_tokens - self.cached_tokens,
        )
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class AggregatedStats:
    """汇总统计"""

    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_effective_prompt_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cost_equivalent_tokens: float = 0.0
    total_duration: float = 0.0
    model: str = ""


class TokenStatsCollector:
    """线程安全的 token 收集器"""

    def __init__(self, log_file: str = "token_usage.jsonl") -> None:
        self._records: list[RequestUsage] = []
        self._lock = threading.Lock()
        self._log_path = Path(log_file)
        self._on_record: Callable[[], None] | None = None

    def set_record_callback(self, callback: Callable[[], None] | None) -> None:
        self._on_record = callback

    def record(self, usage: RequestUsage) -> None:
        usage.finalize()
        with self._lock:
            self._records.append(usage)
        self._append_log(usage)
        self._notify_recorded()

    def load_from_jsonl(self, path: str | Path | None = None) -> int:
        log_path = Path(path) if path is not None else self._log_path
        records: list[RequestUsage] = []
        with open(log_path, "r", encoding="utf-8") as file:
            for line_no, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    body = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {log_path}: {exc.msg}"
                    ) from exc
                usage = RequestUsage(**body)
                usage.finalize()
                records.append(usage)
        with self._lock:
            self._records = records
        return len(records)

    def aggregate(self) -> AggregatedStats:
        with self._lock:
            records = list(self._records)
        stats = AggregatedStats(total_requests=len(records))
        for record in records:
            self._add_record(stats, record)
        return stats

    def report(self) -> str:
        stats = self.aggregate()
        lines = [
            "=" * 56,
            "  Token 用量统计报告",
            "=" * 56,
            f"  模型:             {stats.model}",
            f"  总请求数:         {stats.total_requests}",
            f"  总耗时:           {stats.total_duration:.1f}s",
            "-" * 56,
            f"  总 Prompt Tokens:      {stats.total_prompt_tokens:>10,}",
            f"    其中缓存命中:        {stats.total_cached_tokens:>10,}",
            f"    有效输入 Tokens:      {stats.total_effective_prompt_tokens:>10,}",
            f"  总 Completion Tokens:  {stats.total_completion_tokens:>10,}",
            f"    其中 Reasoning:      {stats.total_reasoning_tokens:>10,}",
            f"  总 Tokens:             {stats.total_tokens:>10,}",
            f"  开销当量 Tokens:       {stats.total_cost_equivalent_tokens:>10,.1f}",
            "=" * 56,
        ]
        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.report() + "\n\n")
            file.write("JSON Summary:\n")
            file.write(
                json.dumps(asdict(self.aggregate()), ensure_ascii=False, indent=2)
                + "\n"
            )

    def _append_log(self, usage: RequestUsage) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(usage), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _notify_recorded(self) -> None:
        if self._on_record is None:
            return
        try:
            self._on_record()
        except Exception:
            pass

    def _add_record(
        self,
        stats: AggregatedStats,
        record: RequestUsage,
    ) -> None:
        stats.total_prompt_tokens += record.prompt_tokens
        stats.total_completion_tokens += record.completion_tokens
        stats.total_tokens += record.total_tokens
        stats.total_cached_tokens += record.cached_tokens
        stats.total_effective_prompt_tokens += record.effective_prompt_tokens
        stats.total_reasoning_tokens += record.reasoning_tokens
        stats.total_cost_equivalent_tokens += (
            record.effective_prompt_tokens
            + 0.1 * record.cached_tokens
            + 5 * record.completion_tokens
        )
        stats.total_duration += record.duration
        if record.model and not stats.model:
            stats.model = record.model
