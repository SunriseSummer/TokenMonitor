#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


DEFAULT_BASE_URL = "http://127.0.0.1:9100/v1/chat/completions"


@dataclass
class ChatResult:
    content: str = ""
    reasoning_content: str = ""
    model: str = ""
    request_id: str = ""
    usage: dict | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple end-to-end chat client for the local monitor gateway."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"monitor chat completions endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default="monitor-test",
        help="downstream API key sent to monitor; monitor uses its own upstream key",
    )
    parser.add_argument(
        "--model",
        default="monitor-test-model",
        help="downstream model name; monitor may override it with --model",
    )
    parser.add_argument(
        "--system",
        default="You are a concise coding assistant.",
        help="system prompt for the test chat",
    )
    parser.add_argument(
        "--once",
        default="",
        help="send one prompt and exit; useful for smoke tests",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="disable streaming and use a non-streaming request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="max completion tokens for each request",
    )
    parser.add_argument(
        "--assert-no-think-tags",
        action="store_true",
        help="fail if assistant content contains raw <think> tags",
    )
    return parser.parse_args()


def post_json(
    url: str,
    api_key: str,
    payload: dict,
    *,
    stream: bool,
) -> urllib.response.addinfourl:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream" if stream else "application/json")
    return urllib.request.urlopen(req, timeout=300)


def iter_sse_events(resp) -> Iterable[str]:
    event_lines: list[str] = []
    while True:
        raw = resp.readline()
        if not raw:
            if event_lines:
                yield "\n".join(event_lines)
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if event_lines:
                yield "\n".join(event_lines)
                event_lines.clear()
            continue
        if line.startswith(":"):
            continue
        event_lines.append(line)


def parse_event_data(event: str) -> str:
    data_lines: list[str] = []
    for line in event.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    return "\n".join(data_lines)


def read_non_stream(resp) -> ChatResult:
    data = json.loads(resp.read().decode("utf-8", errors="replace"))
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return ChatResult(
        content=message.get("content") or "",
        reasoning_content=message.get("reasoning_content") or "",
        model=data.get("model") or "",
        request_id=data.get("id") or "",
        usage=data.get("usage"),
    )


def read_stream(resp) -> ChatResult:
    result = ChatResult()
    for event in iter_sse_events(resp):
        data_text = parse_event_data(event)
        if not data_text:
            continue
        if data_text == "[DONE]":
            break
        try:
            data = json.loads(data_text)
        except json.JSONDecodeError:
            print(f"\n[warn] skipped invalid SSE data: {data_text[:200]}", file=sys.stderr)
            continue

        if not result.model:
            result.model = data.get("model") or ""
        if not result.request_id:
            result.request_id = data.get("id") or ""
        if data.get("usage"):
            result.usage = data["usage"]

        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        if reasoning:
            result.reasoning_content += reasoning
        if content:
            result.content += content
            print(content, end="", flush=True)
    print()
    return result


def call_chat(args: argparse.Namespace, messages: list[dict]) -> ChatResult:
    stream = not args.no_stream
    payload = {
        "model": args.model,
        "messages": messages,
        "stream": stream,
        "max_tokens": args.max_tokens,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    try:
        with post_json(args.base_url, args.api_key, payload, stream=stream) as resp:
            if stream:
                return read_stream(resp)
            return read_non_stream(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from monitor:\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot connect to monitor at {args.base_url}: {exc.reason}"
        ) from exc


def print_result(result: ChatResult, *, show_content: bool) -> None:
    if show_content and result.content:
        print(result.content)
    if result.reasoning_content:
        print("\n[reasoning_content]")
        print(result.reasoning_content.strip())
    if result.usage:
        print("\n[usage]")
        print(json.dumps(result.usage, ensure_ascii=False, indent=2))
    if result.model or result.request_id:
        print(
            f"\n[meta] model={result.model or '-'} request_id={result.request_id or '-'}"
        )


def assert_no_think_tags(result: ChatResult) -> None:
    if "<think>" in result.content or "</think>" in result.content:
        raise SystemExit("FAILED: assistant content contains raw <think> tags")


def run_once(args: argparse.Namespace, messages: list[dict]) -> None:
    messages.append({"role": "user", "content": args.once})
    print("[assistant]")
    result = call_chat(args, messages)
    if args.assert_no_think_tags:
        assert_no_think_tags(result)
    print_result(result, show_content=args.no_stream)
    print("\nPASS")


def run_repl(args: argparse.Namespace, messages: list[dict]) -> None:
    print(f"Connected target: {args.base_url}")
    print("Type /exit to quit. Type /reset to clear conversation.")
    while True:
        try:
            prompt = input("\n[user] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt in {"/exit", "exit", "quit"}:
            return
        if prompt == "/reset":
            messages[:] = [{"role": "system", "content": args.system}]
            print("conversation reset")
            continue

        messages.append({"role": "user", "content": prompt})
        print("[assistant]")
        result = call_chat(args, messages)
        if args.assert_no_think_tags:
            assert_no_think_tags(result)
        print_result(result, show_content=args.no_stream)
        messages.append(
            {
                "role": "assistant",
                "content": result.content,
                "reasoning_content": result.reasoning_content,
            }
        )


def main() -> None:
    args = parse_args()
    messages = [{"role": "system", "content": args.system}]
    if args.once:
        run_once(args, messages)
    else:
        run_repl(args, messages)


if __name__ == "__main__":
    main()
