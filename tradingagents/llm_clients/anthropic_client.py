import logging
import os
from collections.abc import Iterable
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content, warn_if_truncated
from .cc_switch_store import resolve_claude_api_key
from .provider_catalog import CC_SWITCH_CLAUDE_PROVIDERS, CLAUDE_PROVIDER_KEYS
from .validators import validate_model

logger = logging.getLogger(__name__)

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
    "default_headers", "betas",
)

# 走 anthropic 通道跑第三方模型（Kimi 等）时的默认输出上限。
# langchain-anthropic 只认识真正的 Claude 模型名，对别的模型会落到一个很小的
# 兜底值（1.5.x 是 4096，更早的版本是 1024）——够不上一篇完整的分析报告，表现
# 就是"报告写到一半结束"（#91）。8192 是各家 Anthropic 兼容端点普遍支持的档位；
# 需要更长就在 config 里显式设 `max_tokens`。
_THIRD_PARTY_DEFAULT_MAX_TOKENS = 8192

_CC_SWITCH_CLAUDE_CONFIG = {
    provider.key: (
        provider.base_url,
        provider.auth_token_envs,
        provider.api_key_envs,
        provider.default_betas,
    )
    for provider in CC_SWITCH_CLAUDE_PROVIDERS
}


def _normalize_betas(value: Any) -> list[str]:
    """Return beta names from strings or iterables, preserving order."""
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, Iterable):
        candidates = value
    else:
        candidates = (str(value),)

    betas: list[str] = []
    for beta in candidates:
        normalized = str(beta).strip()
        if normalized and normalized not in betas:
            betas.append(normalized)
    return betas


def _merge_betas(*values: Any) -> list[str]:
    betas: list[str] = []
    for value in values:
        for beta in _normalize_betas(value):
            if beta not in betas:
                betas.append(beta)
    return betas


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config, **kwargs)
        warn_if_truncated(response, self.model)
        return normalize_content(response)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models and cc-switch Claude providers."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "anthropic",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.provider in _CC_SWITCH_CLAUDE_CONFIG:
            default_base, auth_token_envs, api_key_envs, default_betas = (
                _CC_SWITCH_CLAUDE_CONFIG[self.provider]
            )
            llm_kwargs["base_url"] = self.base_url or default_base
            api_key = resolve_claude_api_key(
                self.provider,
                auth_token_envs,
                api_key_envs,
            )
            if api_key:
                llm_kwargs["api_key"] = api_key
            elif "api_key" not in self.kwargs:
                env_hint = " / ".join(auth_token_envs + api_key_envs) or "ANTHROPIC_AUTH_TOKEN"
                raise RuntimeError(
                    f"未找到 {self.provider} 的 API Key。请在 .env 或 cc-switch 中配置 "
                    f"`{env_hint}` 后重启程序。"
                )
            if default_betas:
                llm_kwargs["betas"] = list(default_betas)
        else:
            base_url = self.base_url or os.environ.get("ANTHROPIC_BASE_URL")
            if base_url:
                llm_kwargs["base_url"] = base_url

            # 用别家模型名走 anthropic 通道却没给端点，请求会发到 api.anthropic.com，
            # 拿一个第三方 token 去认证，结果是一句看不懂的 401 invalid x-api-key（#89）。
            # 与其让用户对着 Anthropic 的报错猜，不如在启动时说清楚缺了什么。
            if not base_url and not self.model.lower().startswith("claude"):
                raise RuntimeError(
                    f"模型 `{self.model}` 不是 Claude 模型，但没有配置 API 端点，"
                    f"请求会被发到 Anthropic 官方（api.anthropic.com）并以 "
                    f"401 invalid x-api-key 失败。\n"
                    f"用 Kimi 等 Anthropic 兼容服务时，请同时给出端点和 key：\n"
                    f"  · config 里设 `backend_url`（Web 侧栏是「API Base URL」），"
                    f"或设环境变量 ANTHROPIC_BASE_URL\n"
                    f"    例：backend_url=\"https://api.kimi.com/coding/\"\n"
                    f"  · 把该服务的 token 设成 ANTHROPIC_API_KEY\n"
                    f"    （注意：ANTHROPIC_AUTH_TOKEN 是 Claude Code CLI 的写法，"
                    f"本项目走 langchain，只认 ANTHROPIC_API_KEY）"
                )

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                if key == "betas":
                    llm_kwargs[key] = _merge_betas(
                        llm_kwargs.get(key),
                        self.kwargs[key],
                    )
                else:
                    llm_kwargs[key] = self.kwargs[key]

        # 第三方模型必须显式给输出上限，否则会被 langchain 的小兜底值静默截断（#91）。
        # 判据用"模型名是不是 claude 开头"，不用 validate_model()——后者只认我们
        # 目录里那几个短别名，会把 claude-sonnet-4-5-20250929 这种带日期的正规
        # Claude ID 也误判成第三方，反而把它从 64000 砍到 8192。
        if (
            self.provider not in CLAUDE_PROVIDER_KEYS
            and llm_kwargs.get("max_tokens") is None
            and not self.model.lower().startswith("claude")
        ):
            llm_kwargs["max_tokens"] = _THIRD_PARTY_DEFAULT_MAX_TOKENS
            logger.info(
                "模型 %s 不在已知 Claude 模型列表内，max_tokens 默认设为 %d "
                "（否则会被截断在更小的兜底值上）。需要更长的报告就在 config 里"
                "显式设置 `max_tokens`。",
                self.model, _THIRD_PARTY_DEFAULT_MAX_TOKENS,
            )

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic / cc-switch Claude providers."""
        return validate_model(self.provider, self.model)
