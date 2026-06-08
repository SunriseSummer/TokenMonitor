"""Token Monitor 命令行入口"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from .config import PROVIDERS, MonitorConfig
from .proxy import create_proxy_server
from .token_stats import TokenStatsCollector


class LiveSummaryRenderer:
    """交互式实时汇总渲染器"""

    def __init__(
        self,
        config: MonitorConfig,
        stats: TokenStatsCollector,
        endpoint: str,
        *,
        enabled: bool,
    ) -> None:
        self.config = config
        self.stats = stats
        self.endpoint = endpoint
        self.enabled = enabled
        self._lock = threading.Lock()

    def refresh(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._clear_screen()
            print("Token Monitor running")
            print(
                f"  Provider:  {self.config.provider_display_name()} "
                f"({self.config.upstream_provider})"
            )
            print(f"  Upstream:  {self.endpoint}")
            print(f"  Listening: http://{self.config.host}:{self.config.port}")
            print(f"  Log file:  {self.config.log_file}")
            print()
            print(self.stats.report())
            print()
            print("Press Ctrl+C to stop")
            sys.stdout.flush()

    def _clear_screen(self) -> None:
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Token Monitor: OpenAI 兼容 token 用量网关"
    )
    parser.add_argument(
        "--summary",
        default="",
        help="读取指定 JSONL 日志并输出汇总后退出",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9100,
        help="监听端口",
    )
    parser.add_argument(
        "--provider",
        default="kimi",
        choices=list(PROVIDERS.keys()),
        help="上游服务商",
    )
    parser.add_argument(
        "--endpoint",
        default="",
        help="自定义上游 endpoint",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="上游 API Key",
    )
    parser.add_argument(
        "--model",
        default="",
        help="覆盖下游请求中的模型名",
    )
    parser.add_argument(
        "--log-file",
        default="token_usage.jsonl",
        help="token 用量日志文件",
    )
    args = parser.parse_args()
    if not args.summary and not args.api_key:
        parser.error("--api-key is required unless --summary is used")
    return args


def print_summary(path: str) -> None:
    """打印离线汇总"""
    stats = TokenStatsCollector(log_file=path)
    summary_path = Path(path)
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


def build_config(args: argparse.Namespace) -> MonitorConfig:
    """根据命令行参数构造配置"""
    return MonitorConfig(
        host=args.host,
        port=args.port,
        upstream_provider=args.provider,
        upstream_api_key=args.api_key,
        upstream_endpoint=args.endpoint,
        upstream_model=args.model,
        log_file=args.log_file,
    )


def install_signal_handlers() -> None:
    """安装关闭信号处理"""

    def request_shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_shutdown)


def main() -> None:
    args = parse_args()
    if args.summary:
        print_summary(args.summary)
        return

    config = build_config(args)
    stats = TokenStatsCollector(log_file=config.log_file)
    endpoint = config.resolve_endpoint()

    print("Token Monitor 启动中")
    print(f"  上游服务商: {config.provider_display_name()} ({config.upstream_provider})")
    print(f"  上游地址:   {endpoint}")
    print(f"  监听地址:   http://{config.host}:{config.port}")
    print(f"  日志文件:   {config.log_file}")
    print()

    server = create_proxy_server(config, stats)
    live_summary = LiveSummaryRenderer(
        config,
        stats,
        endpoint,
        enabled=sys.stdout.isatty(),
    )
    stats.set_record_callback(live_summary.refresh)
    install_signal_handlers()

    print("Token Monitor 已启动，按 Ctrl+C 停止")
    if live_summary.enabled:
        live_summary.refresh()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在关闭")
        server.server_close()
        if live_summary.enabled:
            live_summary.refresh()
            print()
            print("Token Monitor stopped")
        else:
            print(stats.report())


if __name__ == "__main__":
    main()
