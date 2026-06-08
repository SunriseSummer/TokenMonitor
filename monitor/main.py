"""
Token Monitor — OpenAI 兼容接口的 token 用量监控网关。

用法：
    python monitor --provider kimi --api-key <KEY> --model kimi-k2.6
    python monitor --port 9100 --provider kimi --api-key <KEY>

网关启动后，将 Coder 的 /connect custom 配置为：
    BaseURL: http://127.0.0.1:9100/v1/chat/completions
    API Key: 任意值（网关自行使用上游 Key）

网关会将请求透传至上游服务商，同时精确统计 token 用量。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from .config import MonitorConfig, PROVIDERS
from .proxy import create_proxy_server
from .token_stats import TokenStatsCollector


class LiveSummaryRenderer:
    """Refresh the interactive monitor screen with the latest token summary."""

    def __init__(
        self,
        config: MonitorConfig,
        stats: TokenStatsCollector,
        provider_name: str,
        endpoint: str,
        *,
        enabled: bool,
    ) -> None:
        self.config = config
        self.stats = stats
        self.provider_name = provider_name
        self.endpoint = endpoint
        self.enabled = enabled
        self._lock = threading.Lock()

    def refresh(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._clear_screen()
            print("Token Monitor running")
            print(f"  Provider:  {self.provider_name} ({self.config.upstream_provider})")
            print(f"  Upstream:  {self.endpoint}")
            print(f"  Listening: http://{self.config.host}:{self.config.port}")
            print(f"  Log file:  {self.config.log_file}")
            print()
            print(self.stats.report())
            print()
            print("Press Ctrl+C to stop.")
            sys.stdout.flush()

    def _clear_screen(self) -> None:
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Token Monitor: OpenAI 兼容接口的 token 用量监控网关"
    )
    parser.add_argument(
        "--summary",
        default="",
        help="读取指定 JSONL 日志并输出汇总报告，然后退出",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="监听地址 (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=9100, help="监听端口 (default: 9100)"
    )
    parser.add_argument(
        "--provider",
        default="kimi",
        choices=list(PROVIDERS.keys()),
        help="上游服务商 (default: kimi)",
    )
    parser.add_argument(
        "--endpoint", default="", help="自定义上游 endpoint（覆盖 provider）"
    )
    parser.add_argument(
        "--api-key", default="", help="上游服务商 API Key"
    )
    parser.add_argument(
        "--model", default="", help="覆盖请求中的模型名称（可选）"
    )
    parser.add_argument(
        "--log-file",
        default="token_usage.jsonl",
        help="token 用量日志文件 (default: token_usage.jsonl)",
    )
    args = parser.parse_args()
    if not args.summary and not args.api_key:
        parser.error("--api-key is required unless --summary is used")
    return args


def main() -> None:
    """主入口。"""
    args = parse_args()
    if args.summary:
        stats = TokenStatsCollector(log_file=args.summary)
        summary_path = Path(args.summary)
        try:
            count = stats.load_from_jsonl()
        except FileNotFoundError:
            print(f"summary file not found: {summary_path}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {count} request records from {summary_path}")
        print()
        print(stats.report())
        return

    config = MonitorConfig(
        host=args.host,
        port=args.port,
        upstream_provider=args.provider,
        upstream_api_key=args.api_key,
        upstream_endpoint=args.endpoint,
        upstream_model=args.model,
        log_file=args.log_file,
    )
    stats = TokenStatsCollector(log_file=config.log_file)

    provider_name = PROVIDERS.get(
        config.upstream_provider, ("custom", "")
    )[0]
    endpoint = config.resolve_endpoint()

    print(f"Token Monitor 启动中...")
    print(f"  上游服务商: {provider_name} ({config.upstream_provider})")
    print(f"  上游地址:   {endpoint}")
    print(f"  监听地址:   http://{config.host}:{config.port}")
    print(f"  日志文件:   {config.log_file}")
    print()

    server = create_proxy_server(config, stats)
    live_summary = LiveSummaryRenderer(
        config,
        stats,
        provider_name,
        endpoint,
        enabled=sys.stdout.isatty(),
    )
    stats.set_record_callback(live_summary.refresh)

    def request_shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_shutdown)
    print("Token Monitor 已启动，按 Ctrl+C 停止。")
    if live_summary.enabled:
        live_summary.refresh()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在关闭...")
        server.server_close()
        if live_summary.enabled:
            live_summary.refresh()
            print()
            print("Token Monitor stopped.")
        else:
            print(stats.report())


if __name__ == "__main__":
    main()
