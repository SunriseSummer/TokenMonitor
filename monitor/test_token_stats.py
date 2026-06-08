from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.token_stats import RequestUsage, TokenStatsCollector


class TokenStatsCollectorTests(unittest.TestCase):
    def test_cost_equivalent_tokens_uses_effective_cached_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = TokenStatsCollector(str(Path(tmp) / "usage.jsonl"))
            stats.record(
                RequestUsage(
                    model="m",
                    prompt_tokens=100,
                    cached_tokens=40,
                    completion_tokens=20,
                    total_tokens=120,
                )
            )
            stats.record(
                RequestUsage(
                    model="m",
                    prompt_tokens=10,
                    cached_tokens=0,
                    completion_tokens=3,
                    total_tokens=13,
                )
            )

            aggregate = stats.aggregate()

        self.assertEqual(aggregate.total_effective_prompt_tokens, 70)
        self.assertEqual(aggregate.total_cached_tokens, 40)
        self.assertEqual(aggregate.total_completion_tokens, 23)
        self.assertAlmostEqual(
            aggregate.total_cost_equivalent_tokens,
            70 + 0.1 * 40 + 5 * 23,
        )

    def test_report_puts_cost_equivalent_tokens_as_last_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = TokenStatsCollector(str(Path(tmp) / "usage.jsonl"))
            stats.record(
                RequestUsage(
                    prompt_tokens=10,
                    cached_tokens=5,
                    completion_tokens=2,
                )
            )

            lines = stats.report().splitlines()

        self.assertIn("开销当量 Tokens:", lines[-2])
        self.assertIn("15.5", lines[-2])

    def test_record_callback_runs_after_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = TokenStatsCollector(str(Path(tmp) / "usage.jsonl"))
            observed_totals: list[int] = []

            stats.set_record_callback(
                lambda: observed_totals.append(stats.aggregate().total_requests)
            )
            stats.record(RequestUsage(prompt_tokens=1, completion_tokens=2))

        self.assertEqual(observed_totals, [1])

    def test_record_callback_errors_do_not_block_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stats = TokenStatsCollector(str(Path(tmp) / "usage.jsonl"))

            def fail() -> None:
                raise RuntimeError("render failed")

            stats.set_record_callback(fail)
            stats.record(RequestUsage(prompt_tokens=1, completion_tokens=2))
            aggregate = stats.aggregate()

        self.assertEqual(aggregate.total_requests, 1)


if __name__ == "__main__":
    unittest.main()
