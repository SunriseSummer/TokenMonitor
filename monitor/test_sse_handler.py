from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.sse_handler import extract_usage_from_chunk, iter_sse_chunks


def _consume(gen):
    chunks = []
    while True:
        try:
            chunks.append(next(gen))
        except StopIteration as stop:
            return chunks, stop.value


class SSEHandlerTests(unittest.TestCase):
    def test_done_event_forwarded_once(self) -> None:
        raw_iter = [
            b'data: {"id":"r1","choices":[{"delta":{"content":"hi"}}]}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]

        chunks, _ = _consume(iter_sse_chunks(raw_iter))

        self.assertEqual(
            [raw for raw, _ in chunks],
            [
                b'data: {"id":"r1","choices":[{"delta":{"content":"hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ],
        )

    def test_last_event_flushed_without_trailing_blank_line(self) -> None:
        raw_iter = [
            b'data: {"id":"r2","model":"deepseek-v4-pro","usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n',
        ]

        chunks, result = _consume(iter_sse_chunks(raw_iter))

        self.assertEqual(
            chunks,
            [
                (
                    b'data: {"id":"r2","model":"deepseek-v4-pro","usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n',
                    None,
                )
            ],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.completion_tokens, 5)
        self.assertEqual(result.total_tokens, 8)

    def test_comment_line_passthrough(self) -> None:
        raw_iter = [b": keep-alive\n", b"\n", b"data: [DONE]\n", b"\n"]
        chunks, _ = _consume(iter_sse_chunks(raw_iter))
        self.assertEqual([raw for raw, _ in chunks], [b": keep-alive\n", b"data: [DONE]\n\n"])

    def test_extract_usage_accepts_top_level_cache_fields(self) -> None:
        usage = extract_usage_from_chunk(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_cache_hit_tokens": 30,
                    "reasoning_tokens": 8,
                }
            }
        )

        self.assertEqual(usage["cached_tokens"], 30)
        self.assertEqual(usage["reasoning_tokens"], 8)

    def test_usage_is_collected_when_multiple_events_arrive_in_one_chunk(self) -> None:
        raw_iter = [
            (
                b'data: {"id":"r1","model":"MiniMax-M3","choices":[{"delta":{"content":"hi"}}]}\n\n'
                b'data: {"id":"r1","model":"MiniMax-M3","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'
            )
        ]

        chunks, result = _consume(iter_sse_chunks(raw_iter))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(result.prompt_tokens, 7)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.total_tokens, 10)
        self.assertEqual(result.model, "MiniMax-M3")
        self.assertEqual(result.request_id, "r1")

    def test_reasoning_tokens_are_estimated_from_reasoning_deltas(self) -> None:
        raw_iter = [
            (
                b'data: {"id":"r1","model":"MiniMax-M3","choices":[{"delta":{"reasoning_content":"plan"}}]}\n\n'
                b'data: {"id":"r1","model":"MiniMax-M3","choices":[{"delta":{"content":"answer"}}]}\n\n'
                b'data: {"id":"r1","model":"MiniMax-M3","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":10,"total_tokens":17}}\n\n'
            )
        ]

        _, result = _consume(iter_sse_chunks(raw_iter))

        self.assertEqual(result.completion_tokens, 10)
        self.assertEqual(result.reasoning_tokens, 4)
