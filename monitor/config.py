"""Monitor 配置与内置服务商"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    """上游服务商配置"""

    display_name: str
    endpoint: str


PROVIDERS: dict[str, Provider] = {
    "huawei": Provider(
        "华为云 MaaS",
        "https://api.modelarts-maas.com/v2/chat/completions",
    ),
    "kimi": Provider(
        "月之暗面 Moonshot",
        "https://api.moonshot.cn/v1/chat/completions",
    ),
    "deepseek": Provider(
        "DeepSeek",
        "https://api.deepseek.com/chat/completions",
    ),
    "minimax": Provider(
        "MiniMax",
        "https://api.minimaxi.com/v1/chat/completions",
    ),
    "zhipu": Provider(
        "智谱 GLM",
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    ),
}


@dataclass
class MonitorConfig:
    """Token 网关运行配置"""

    host: str = "127.0.0.1"
    port: int = 9100
    upstream_provider: str = "kimi"
    upstream_api_key: str = ""
    upstream_endpoint: str = ""
    upstream_model: str = ""
    log_file: str = "token_usage.jsonl"
    extra_params: dict = field(default_factory=dict)

    def resolve_endpoint(self) -> str:
        """解析最终上游 endpoint"""
        if self.upstream_endpoint:
            return self.upstream_endpoint
        provider = PROVIDERS.get(self.upstream_provider)
        if provider is None:
            raise ValueError(f"Unknown provider: {self.upstream_provider}")
        return provider.endpoint

    def provider_display_name(self) -> str:
        """返回服务商展示名"""
        provider = PROVIDERS.get(self.upstream_provider)
        return provider.display_name if provider else "custom"
