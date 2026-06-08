"""
Token Monitor 配置模块。

定义上游服务商路由表和监控网关的基本配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 内建服务商配置：id -> (display_name, endpoint)
PROVIDERS: dict[str, tuple[str, str]] = {
    "huawei": (
        "华为云 MaaS",
        "https://api.modelarts-maas.com/v2/chat/completions",
    ),
    "kimi": (
        "月之暗面 Moonshot",
        "https://api.moonshot.cn/v1/chat/completions",
    ),
    "deepseek": (
        "DeepSeek",
        "https://api.deepseek.com/chat/completions",
    ),
    "minimax": (
        "MiniMax",
        "https://api.minimaxi.com/v1/chat/completions",
    ),
    "zhipu": (
        "智谱 GLM",
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    ),
}


@dataclass
class MonitorConfig:
    """Token 监控网关配置。"""

    host: str = "127.0.0.1"
    port: int = 9100
    upstream_provider: str = "kimi"
    upstream_api_key: str = ""
    upstream_endpoint: str = ""
    upstream_model: str = ""
    log_file: str = "token_usage.jsonl"
    # 请求中附加的参数，用于强制上游返回 usage 信息
    extra_params: dict = field(default_factory=dict)

    def resolve_endpoint(self) -> str:
        """解析最终的上游 endpoint。"""
        if self.upstream_endpoint:
            return self.upstream_endpoint
        if self.upstream_provider in PROVIDERS:
            return PROVIDERS[self.upstream_provider][1]
        raise ValueError(f"Unknown provider: {self.upstream_provider}")
