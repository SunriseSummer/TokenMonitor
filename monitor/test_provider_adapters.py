from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.provider_adapters import MiniMaxAdapter


def _json_event(raw: bytes) -> dict:
    text = raw.decode("utf-8")
    assert text.startswith("data: ")
    return json.loads(text[6:].strip())


class MiniMaxAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = MiniMaxAdapter()

    def test_non_stream_moves_think_tags_to_reasoning_content(self) -> None:
        data = {
            "usage": {"completion_tokens": 10},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>plan first</think>\n\nanswer",
                    }
                }
            ]
        }

        normalized = self.adapter.normalize_response(data)
        message = normalized["choices"][0]["message"]

        self.assertEqual(message["reasoning_content"], "plan first")
        self.assertNotIn("<think>", message["content"])
        self.assertEqual(message["content"], "\n\nanswer")
        self.assertGreater(
            normalized["usage"]["completion_tokens_details"]["reasoning_tokens"],
            0,
        )

    def test_stream_moves_inline_think_tags_to_reasoning_delta(self) -> None:
        raw_iter = [
            b'data: {"choices":[{"delta":{"content":"<think>plan</think>answer"}}]}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]

        chunks = list(self.adapter.normalize_stream(raw_iter))
        data = _json_event(chunks[0])
        delta = data["choices"][0]["delta"]

        self.assertEqual(delta["reasoning_content"], "plan")
        self.assertEqual(delta["content"], "answer")
        self.assertEqual(chunks[1], b"data: [DONE]\n\n")

    def test_stream_handles_split_think_tags_across_chunks(self) -> None:
        raw_iter = [
            b'data: {"choices":[{"delta":{"content":"<thi"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":"nk>plan</th"}}]}\n',
            b"\n",
            b'data: {"choices":[{"delta":{"content":"ink>answer"}}]}\n',
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]

        chunks = [
            _json_event(chunk)
            for chunk in self.adapter.normalize_stream(raw_iter)
            if chunk != b"data: [DONE]\n\n"
        ]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks]

        self.assertNotIn("content", deltas[0])
        self.assertEqual(deltas[1]["reasoning_content"], "plan")
        self.assertEqual(deltas[2]["content"], "answer")

    def test_stream_handles_multiple_events_in_one_network_chunk(self) -> None:
        raw_iter = [
            (
                b'data: {"choices":[{"delta":{"content":"<think>plan"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"</think>answer"}}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            )
        ]

        chunks = list(self.adapter.normalize_stream(raw_iter))
        payloads = [_json_event(chunk) for chunk in chunks]

        self.assertEqual(
            payloads[0]["choices"][0]["delta"]["reasoning_content"],
            "plan",
        )
        self.assertEqual(payloads[1]["choices"][0]["delta"]["content"], "answer")
        self.assertEqual(payloads[2]["usage"]["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
