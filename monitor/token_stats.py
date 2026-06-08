"""
Token 用量统计模块。

精确记录每次请求的 prompt_tokens、completion_tokens、
cached_tokens 等详细数据，并支持汇总报告。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class RequestUsage:
    """单次请求的 token 用量记录。"""

    request_id: str = ""
    timestamp: float = 0.0
    model: str = ""
    # 输入 token
    prompt_tokens: int = 0
    # 输出 token
    completion_tokens: int = 0
    # 总 token
    total_tokens: int = 0
    # 缓存命中的输入 token（prompt_tokens_details.cached_tokens）
    cached_tokens: int = 0
    # 实际有效输入 token = prompt_tokens - cached_tokens
    effective_prompt_tokens: int = 0
    # 推理 token（reasoning_tokens，思考模型专用）
    reasoning_tokens: int = 0
    # 请求耗时（秒）
    duration: float = 0.0
    # 是否为流式请求
    stream: bool = False

    def finalize(self) -> None:
        """根据已有字段计算派生字段。"""
        self.effective_prompt_tokens = max(
            0, self.prompt_tokens - self.cached_tokens
        )
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class AggregatedStats:
    """汇总统计数据。"""

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
    """线程安全的 token 用量收集器。"""

    def __init__(self, log_file: str = "token_usage.jsonl") -> None:
        self._records: list[RequestUsage] = []
        self._lock = threading.Lock()
        self._log_path = Path(log_file)
        self._on_record: Callable[[], None] | None = None

    def set_record_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a callback that runs after each completed request is recorded."""
        self._on_record = callback

    def record(self, usage: RequestUsage) -> None:
        """记录一次请求的 token 用量。"""
        usage.finalize()
        with self._lock:
            self._records.append(usage)
        self._append_log(usage)
        callback = self._on_record
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _append_log(self, usage: RequestUsage) -> None:
        """追加写入 JSONL 日志文件。"""
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(usage), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def load_from_jsonl(self, path: str | Path | None = None) -> int:
        """从 JSONL 文件加载历史记录并返回成功读取的条数。"""
        log_path = Path(path) if path is not None else self._log_path
        records: list[RequestUsage] = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
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
        """生成汇总统计。"""
        with self._lock:
            records = list(self._records)
        stats = AggregatedStats(total_requests=len(records))
        for r in records:
            stats.total_prompt_tokens += r.prompt_tokens
            stats.total_completion_tokens += r.completion_tokens
            stats.total_tokens += r.total_tokens
            stats.total_cached_tokens += r.cached_tokens
            stats.total_effective_prompt_tokens += r.effective_prompt_tokens
            stats.total_reasoning_tokens += r.reasoning_tokens
            stats.total_cost_equivalent_tokens += (
                r.effective_prompt_tokens
                + 0.1 * r.cached_tokens
                + 5 * r.completion_tokens
            )
            stats.total_duration += r.duration
            if r.model and not stats.model:
                stats.model = r.model
        return stats

    def report(self) -> str:
        """生成人类可读的统计报告。"""
        s = self.aggregate()
        lines = [
            "=" * 56,
            "  Token 用量统计报告",
            "=" * 56,
            f"  模型:             {s.model}",
            f"  总请求数:         {s.total_requests}",
            f"  总耗时:           {s.total_duration:.1f}s",
            "-" * 56,
            f"  总 Prompt Tokens:      {s.total_prompt_tokens:>10,}",
            f"    其中缓存命中:        {s.total_cached_tokens:>10,}",
            f"    有效输入 Tokens:      {s.total_effective_prompt_tokens:>10,}",
            f"  总 Completion Tokens:  {s.total_completion_tokens:>10,}",
            f"    其中 Reasoning:      {s.total_reasoning_tokens:>10,}",
            f"  总 Tokens:             {s.total_tokens:>10,}",
            f"  开销当量 Tokens:       {s.total_cost_equivalent_tokens:>10,.1f}",
            "=" * 56,
        ]
        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        """将报告保存到文件。"""
        report = self.report()
        summary = asdict(self.aggregate())
        with open(path, "w", encoding="utf-8") as f:
            f.write(report + "\n\n")
            f.write("JSON Summary:\n")
            f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
