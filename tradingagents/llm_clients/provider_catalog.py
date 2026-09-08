"""cc-switch provider catalog used by TradingAgents-Astock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

ModelOption = Tuple[str, str]


@dataclass(frozen=True)
class CodexProvider:
    key: str
    display_name: str
    base_url: str
    api_key_envs: tuple[str, ...] = ()
    user_agent_env: str | None = None
    cc_switch_names: tuple[str, ...] = ()
    models: tuple[ModelOption, ...] = ()
    default_api_key: str | None = None
    use_responses_api: bool = True


@dataclass(frozen=True)
class ClaudeProvider:
    key: str
    display_name: str
    base_url: str
    models: tuple[ModelOption, ...]
    auth_token_envs: tuple[str, ...] = ()
    api_key_envs: tuple[str, ...] = ()
    cc_switch_names: tuple[str, ...] = ()
    default_betas: tuple[str, ...] = ()


GPT_56_MODEL: ModelOption = ("GPT-5.6 Sol", "gpt-5.6-sol")
CODEX_DEFAULT_MODELS: tuple[ModelOption, ...] = (GPT_56_MODEL,)
# Keep alias for older call sites / tests.
GPT_55_MODEL = GPT_56_MODEL

CLAUDE_DEFAULT_MODELS: tuple[ModelOption, ...] = (
    ("Claude Sonnet 4.6", "claude-sonnet-4-6"),
    ("Claude Opus 4.6", "claude-opus-4-6"),
    ("Claude Opus 5 (1M)", "claude-opus-5[1M]"),
    ("Claude Opus 5", "claude-opus-5"),
    ("Claude Sonnet 5 (1M)", "claude-sonnet-5[1M]"),
    ("Claude Opus 4.8 (1M)", "claude-opus-4-8[1M]"),
    ("Claude Haiku 4.5", "claude-haiku-4-5"),
)

# Lanln: verified available 2026-09-03 (see /tmp/lanln_model_probe.json).
LANLN_CLAUDE_MODELS: tuple[ModelOption, ...] = (
    ("Claude Fable 5.1", "claude-fable-5-1"),
    ("Claude Opus 5", "claude-opus-5"),
    ("Claude Opus 4.8", "claude-opus-4-8"),
    ("Claude Sonnet 5", "claude-sonnet-5"),
)


# Synced from `cc-switch -a codex provider list` (2026-09-03).
CC_SWITCH_CODEX_PROVIDERS: tuple[CodexProvider, ...] = (
    CodexProvider(
        key="cc_codex_aisystem",
        display_name="Codex / AI系统",
        base_url="https://api.zzzcoding.org",
        api_key_envs=("CC_CODEX_AISYSTEM_API_KEY",),
        cc_switch_names=("AI系统",),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_fastai",
        display_name="Codex / FastAI",
        base_url="https://www.fastaitoken.com",
        api_key_envs=("CC_CODEX_FASTAI_API_KEY",),
        cc_switch_names=("FastAI",),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_hyb",
        display_name="Codex / 黑与白",
        base_url="https://ai.hybgzs.com",
        api_key_envs=("CC_CODEX_HYB_API_KEY",),
        cc_switch_names=("黑与白",),
        models=CODEX_DEFAULT_MODELS,
        # /v1/responses 在本机出口 IP 上经常超时；chat completions 可用
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_aiwanwu",
        display_name="Codex / aiwanwu",
        base_url="https://www.aiwanwu.cc",
        api_key_envs=("CC_CODEX_AIWANWU_API_KEY",),
        cc_switch_names=("aiwanwu.cc", "aiwanwu"),
        models=CODEX_DEFAULT_MODELS,
        # /v1/responses 返回 503/超时；实测 /v1/chat/completions 正常
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_junxingchen",
        display_name="Codex / 君星辰",
        base_url="https://ai.centos.hk/v1",
        api_key_envs=("CC_CODEX_JUNXINGCHEN_API_KEY",),
        cc_switch_names=("君星辰",),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_fengwind",
        display_name="Codex / Fengwind API",
        base_url="https://api.fengwind.com",
        api_key_envs=("CC_CODEX_FENGWIND_API_KEY",),
        cc_switch_names=("Fengwind API", "Fengwidn API"),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_fluxionai",
        display_name="Codex / FluxionAI",
        base_url="https://fluxionai.space",
        api_key_envs=("CC_CODEX_FLUXIONAI_API_KEY",),
        cc_switch_names=("FluxionAI",),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_shuaiapi",
        display_name="Codex / 帅API",
        base_url="https://api.shuaiapi.com/v1",
        api_key_envs=("CC_CODEX_SHUAIAPI_API_KEY",),
        cc_switch_names=("帅API",),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        key="cc_codex_aihub",
        display_name="Codex / AIHub",
        base_url="https://aihub.dog",
        api_key_envs=("CC_CODEX_AIHUB_API_KEY",),
        cc_switch_names=("AIHub",),
        models=CODEX_DEFAULT_MODELS,
        use_responses_api=False,
    ),
    CodexProvider(
        # Prefer host gateway so Docker can reach the host cc-switch Codex proxy.
        key="ccswitch",
        display_name="Codex / cc-switch local",
        base_url="http://host.docker.internal:15722",
        api_key_envs=("CC_SWITCH_API_KEY",),
        models=CODEX_DEFAULT_MODELS,
        default_api_key="not-needed",
        use_responses_api=False,
    ),
)


# Synced from `cc-switch -a claude provider list` (2026-09-03).
CC_SWITCH_CLAUDE_PROVIDERS: tuple[ClaudeProvider, ...] = (
    ClaudeProvider(
        key="cc_claude_aisystem",
        display_name="Claude / AI系统",
        base_url="https://api.zzzcoding.org",
        models=CLAUDE_DEFAULT_MODELS,
        auth_token_envs=("CC_CLAUDE_AISYSTEM_AUTH_TOKEN",),
        cc_switch_names=("AI系统",),
    ),
    ClaudeProvider(
        key="cc_claude_ai",
        display_name="Claude / Ai",
        base_url="https://cc.atai8.cc",
        models=CLAUDE_DEFAULT_MODELS,
        auth_token_envs=("CC_CLAUDE_AI_AUTH_TOKEN",),
        cc_switch_names=("Ai",),
    ),
    ClaudeProvider(
        key="cc_claude_lanln",
        display_name="Claude / Lanln",
        base_url="https://ai.venlacy.com",
        models=LANLN_CLAUDE_MODELS,
        auth_token_envs=("CC_CLAUDE_LANLN_AUTH_TOKEN",),
        cc_switch_names=("Lanln",),
    ),
    ClaudeProvider(
        key="cc_claude_anyrouter",
        display_name="Claude / AnyRouter",
        base_url="https://a-ocnfniawgw.cn-shanghai.fcapp.run",
        models=CLAUDE_DEFAULT_MODELS,
        auth_token_envs=("CC_CLAUDE_ANYROUTER_AUTH_TOKEN",),
        cc_switch_names=("AnyRouter",),
        default_betas=("context-1m-2025-08-07",),
    ),
    ClaudeProvider(
        key="cc_claude_ccgui",
        display_name="Claude / ccgui",
        base_url="https://fufei.mossx.ai",
        models=CLAUDE_DEFAULT_MODELS,
        auth_token_envs=("CC_CLAUDE_CCGUI_AUTH_TOKEN",),
        cc_switch_names=("ccgui",),
    ),
)


CODEX_PROVIDER_KEYS = tuple(provider.key for provider in CC_SWITCH_CODEX_PROVIDERS)
CLAUDE_PROVIDER_KEYS = tuple(provider.key for provider in CC_SWITCH_CLAUDE_PROVIDERS)

SELECTABLE_PROVIDERS: tuple[tuple[str, str], ...] = (
    *((provider.display_name, provider.key) for provider in CC_SWITCH_CODEX_PROVIDERS),
    *((provider.display_name, provider.key) for provider in CC_SWITCH_CLAUDE_PROVIDERS),
)
