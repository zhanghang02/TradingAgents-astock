"""Unit tests for cc-switch provider catalog and key resolution."""

from __future__ import annotations

import pytest

from tradingagents.llm_clients.cc_switch_store import (
    ensure_openai_v1_base,
    resolve_claude_api_key,
    resolve_codex_api_key,
)
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.provider_catalog import (
    CC_SWITCH_CLAUDE_PROVIDERS,
    CC_SWITCH_CODEX_PROVIDERS,
    CLAUDE_PROVIDER_KEYS,
    CODEX_PROVIDER_KEYS,
    SELECTABLE_PROVIDERS,
)


@pytest.mark.unit
def test_requested_codex_providers_are_present():
    names = {provider.cc_switch_names[0] for provider in CC_SWITCH_CODEX_PROVIDERS if provider.cc_switch_names}
    for expected in (
        "AI系统",
        "FastAI",
        "黑与白",
        "aiwanwu.cc",
        "君星辰",
        "Fengwind API",
        "FluxionAI",
        "帅API",
        "AIHub",
    ):
        assert expected in names


@pytest.mark.unit
def test_requested_claude_providers_are_present():
    names = {provider.cc_switch_names[0] for provider in CC_SWITCH_CLAUDE_PROVIDERS}
    for expected in ("AI系统", "Ai", "Lanln", "AnyRouter", "ccgui"):
        assert expected in names


@pytest.mark.unit
def test_local_ccswitch_remains_a_codex_provider():
    assert "ccswitch" in CODEX_PROVIDER_KEYS
    assert ("Codex / cc-switch local", "ccswitch") in SELECTABLE_PROVIDERS


@pytest.mark.unit
def test_providers_do_not_use_generic_api_key_fallbacks():
    forbidden = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    }
    for provider in CC_SWITCH_CODEX_PROVIDERS:
        assert forbidden.isdisjoint(provider.api_key_envs)
    for provider in CC_SWITCH_CLAUDE_PROVIDERS:
        assert forbidden.isdisjoint(provider.auth_token_envs)
        assert forbidden.isdisjoint(provider.api_key_envs)


@pytest.mark.unit
def test_codex_default_model_is_gpt_56_sol():
    assert get_model_options("cc_codex_fastai", "quick") == [("GPT-5.6 Sol", "gpt-5.6-sol")]
    assert get_model_options("cc_codex_fastai", "deep") == [("GPT-5.6 Sol", "gpt-5.6-sol")]


@pytest.mark.unit
def test_claude_models_include_opus5():
    models = [value for _, value in get_model_options("cc_claude_anyrouter", "deep")]
    assert "claude-opus-5[1M]" in models


@pytest.mark.unit
def test_lanln_claude_models_are_verified_subset():
    models = [value for _, value in get_model_options("cc_claude_lanln", "quick")]
    assert models == [
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
    ]


@pytest.mark.unit
def test_ensure_openai_v1_base_is_idempotent():
    assert ensure_openai_v1_base("https://aihub.dog") == "https://aihub.dog/v1"
    assert ensure_openai_v1_base("https://aihub.dog/v1") == "https://aihub.dog/v1"
    assert ensure_openai_v1_base("https://ai.centos.hk/v1/") == "https://ai.centos.hk/v1"


@pytest.mark.unit
def test_local_ccswitch_uses_placeholder_api_key(monkeypatch):
    monkeypatch.delenv("CC_SWITCH_API_KEY", raising=False)
    api_key = resolve_codex_api_key("ccswitch", ("CC_SWITCH_API_KEY",))
    assert api_key == "not-needed"


@pytest.mark.unit
def test_resolve_codex_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("CC_CODEX_FASTAI_API_KEY", "from-env")
    assert resolve_codex_api_key("cc_codex_fastai", ("CC_CODEX_FASTAI_API_KEY",)) == "from-env"


@pytest.mark.unit
def test_resolve_claude_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("CC_CLAUDE_ANYROUTER_AUTH_TOKEN", "from-env")
    assert (
        resolve_claude_api_key(
            "cc_claude_anyrouter",
            ("CC_CLAUDE_ANYROUTER_AUTH_TOKEN",),
            (),
        )
        == "from-env"
    )


@pytest.mark.unit
def test_claude_provider_keys_are_unique():
    assert len(CLAUDE_PROVIDER_KEYS) == len(set(CLAUDE_PROVIDER_KEYS))
    assert len(CODEX_PROVIDER_KEYS) == len(set(CODEX_PROVIDER_KEYS))
