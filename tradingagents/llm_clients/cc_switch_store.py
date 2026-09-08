"""Helpers for reading optional local cc-switch provider settings.

The app can run purely from environment variables, but on this machine we also
support reading the cc-switch SQLite store so the configured provider keys do
not need to be duplicated into the repo or .env files.
"""

from __future__ import annotations

import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .provider_catalog import (
    CC_SWITCH_CLAUDE_PROVIDERS,
    CC_SWITCH_CODEX_PROVIDERS,
    CLAUDE_PROVIDER_KEYS,
    CODEX_PROVIDER_KEYS,
)


def _cc_switch_db_path() -> Path:
    override = os.environ.get("TRADINGAGENTS_CC_SWITCH_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cc-switch" / "cc-switch.db"


def ensure_openai_v1_base(base_url: str) -> str:
    """Ensure an OpenAI-compatible base URL ends with /v1 exactly once."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        normalized_path = path
    elif path:
        normalized_path = f"{path}/v1"
    else:
        normalized_path = "/v1"
    return urlunsplit(parsed._replace(path=normalized_path))


@lru_cache(maxsize=None)
def _load_provider_settings(db_path: str, app_type: str) -> tuple[dict[str, Any], ...]:
    path = Path(db_path)
    if not path.exists():
        return ()

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, name, website_url, settings_config, meta, app_type, sort_index, created_at
            FROM providers
            WHERE app_type = ?
            ORDER BY COALESCE(sort_index, 999999), created_at
            """,
            (app_type,),
        ).fetchall()
        parsed: list[dict[str, Any]] = []
        for row in rows:
            try:
                item = json.loads(row["settings_config"])
            except Exception:
                item = {}
            if not isinstance(item, dict):
                item = {}
            item.setdefault("id", row["id"])
            item.setdefault("name", row["name"])
            item.setdefault("website_url", row["website_url"])
            item.setdefault("app_type", row["app_type"])
            item.setdefault("sort_index", row["sort_index"])
            item.setdefault("created_at", row["created_at"])
            item.setdefault("meta", row["meta"])
            parsed.append(item)
        return tuple(parsed)
    except Exception:
        return ()
    finally:
        if conn is not None:
            conn.close()


def _provider_for_key(provider_key: str, app_type: str) -> Any | None:
    if app_type == "codex":
        providers = CC_SWITCH_CODEX_PROVIDERS
    elif app_type == "claude":
        providers = CC_SWITCH_CLAUDE_PROVIDERS
    else:
        return None

    try:
        return next(item for item in providers if item.key == provider_key)
    except StopIteration:
        return None


def _settings_matches_provider_name(settings: dict[str, Any], provider: Any) -> bool:
    name = settings.get("name")
    if not isinstance(name, str) or not name.strip():
        return False

    lowered = name.strip().lower()
    cc_switch_names = getattr(provider, "cc_switch_names", ())
    return lowered == provider.display_name.lower() or any(
        lowered == candidate.lower() for candidate in cc_switch_names
    )


def _settings_matches_provider_url(settings: dict[str, Any], provider: Any) -> bool:
    if settings.get("name") and getattr(provider, "cc_switch_names", ()):
        return False

    website_url = settings.get("website_url")
    if isinstance(website_url, str) and website_url.strip():
        normalized = website_url.strip().rstrip("/")
        provider_base = provider.base_url.rstrip("/")
        if normalized == provider_base:
            return True

    config = settings.get("config")
    if isinstance(config, str):
        if provider.base_url.rstrip("/") in config:
            return True

    return False


def _provider_settings_by_match(app_type: str, provider_key: str) -> dict[str, Any] | None:
    provider = _provider_for_key(provider_key, app_type)
    if provider is None:
        return None

    settings_rows = _load_provider_settings(str(_cc_switch_db_path()), app_type)

    for settings in settings_rows:
        if _settings_matches_provider_name(settings, provider):
            return settings

    for settings in settings_rows:
        if _settings_matches_provider_url(settings, provider):
            return settings

    return None


def _provider_settings(app_type: str, provider_key: str) -> dict[str, Any] | None:
    matched = _provider_settings_by_match(app_type, provider_key)
    if matched is not None:
        return matched

    if app_type == "codex":
        provider_keys = CODEX_PROVIDER_KEYS
    elif app_type == "claude":
        provider_keys = CLAUDE_PROVIDER_KEYS
    else:
        return None

    try:
        provider_index = provider_keys.index(provider_key)
    except ValueError:
        return None

    settings_rows = _load_provider_settings(str(_cc_switch_db_path()), app_type)
    if provider_index >= len(settings_rows):
        return None
    return settings_rows[provider_index]


def _first_env_value(env_names: tuple[str, ...]) -> str | None:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def resolve_codex_api_key(
    provider_key: str,
    env_names: tuple[str, ...],
) -> str | None:
    """Return the best available OpenAI-style key for a Codex provider."""
    if provider_key == "ccswitch":
        return "not-needed"

    api_key = _first_env_value(env_names)
    if api_key:
        return api_key

    settings = _provider_settings("codex", provider_key)
    if not settings:
        return None

    auth = settings.get("auth")
    if isinstance(auth, dict):
        value = auth.get("OPENAI_API_KEY")
        if isinstance(value, str) and value:
            return value

    env = settings.get("env")
    if isinstance(env, dict):
        value = env.get("OPENAI_API_KEY")
        if isinstance(value, str) and value:
            return value

    return None


def resolve_claude_api_key(
    provider_key: str,
    auth_token_envs: tuple[str, ...],
    api_key_envs: tuple[str, ...],
) -> str | None:
    """Return the best available key/token for a Claude provider."""
    api_key = _first_env_value(auth_token_envs) or _first_env_value(api_key_envs)
    if api_key:
        return api_key

    settings = _provider_settings("claude", provider_key)
    if not settings:
        return None

    env = settings.get("env")
    if isinstance(env, dict):
        for key in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            value = env.get(key)
            if isinstance(value, str) and value:
                return value

    auth = settings.get("auth")
    if isinstance(auth, dict):
        for key in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            value = auth.get(key)
            if isinstance(value, str) and value:
                return value

    return None
