"""OpenRouter request header helpers.

OpenRouter-specific transport features should stay opt-in so Hermes does not
change behaviour for existing users by default.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Dict, Optional


OPENROUTER_ATTRIBUTION_HEADERS: Dict[str, str] = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-OpenRouter-Title": "Hermes Agent",
    "X-OpenRouter-Categories": "productivity,cli-agent",
}

OPENROUTER_RESPONSE_CACHE_HEADER = "X-OpenRouter-Cache"
OPENROUTER_RESPONSE_CACHE_TTL_HEADER = "X-OpenRouter-Cache-TTL"
OPENROUTER_RESPONSE_CACHE_ENV = "HERMES_OPENROUTER_RESPONSE_CACHE"
OPENROUTER_RESPONSE_CACHE_TTL_ENV = "HERMES_OPENROUTER_RESPONSE_CACHE_TTL"
OPENROUTER_RESPONSE_CACHE_MAX_TTL_SECONDS = 24 * 60 * 60

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    """Return a bool for known bool-ish values, otherwise ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        return None
    return bool(value)


def _coerce_ttl_seconds(value: Any) -> Optional[int]:
    """Return a valid OpenRouter response-cache TTL in seconds.

    OpenRouter's documented maximum is 24 hours. Invalid values are ignored so a
    bad optional cache setting never prevents client creation.
    """
    if value in (None, ""):
        return None
    try:
        ttl = int(value)
    except (TypeError, ValueError):
        return None
    if ttl < 1 or ttl > OPENROUTER_RESPONSE_CACHE_MAX_TTL_SECONDS:
        return None
    return ttl


def _load_openrouter_config() -> Dict[str, Any]:
    """Best-effort read of ``config.yaml``'s ``openrouter`` section."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("openrouter", {})
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def openrouter_response_cache_headers(
    config: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return OpenRouter response-cache headers when explicitly enabled.

    Supported config shape::

        openrouter:
          response_cache:
            enabled: true
            ttl_seconds: 600

    Environment variables override config:
    ``HERMES_OPENROUTER_RESPONSE_CACHE`` and
    ``HERMES_OPENROUTER_RESPONSE_CACHE_TTL``.
    """
    env = os.environ if environ is None else environ
    openrouter_config: Mapping[str, Any]
    if config is None:
        openrouter_config = _load_openrouter_config()
    else:
        if "openrouter" in config:
            section = config.get("openrouter")
            openrouter_config = section if isinstance(section, Mapping) else {}
        else:
            openrouter_config = config

    response_cache = openrouter_config.get("response_cache", {})
    if not isinstance(response_cache, Mapping):
        response_cache = {}

    enabled = _coerce_optional_bool(response_cache.get("enabled")) or False
    if OPENROUTER_RESPONSE_CACHE_ENV in env:
        env_enabled = _coerce_optional_bool(env.get(OPENROUTER_RESPONSE_CACHE_ENV))
        if env_enabled is not None:
            enabled = env_enabled

    if not enabled:
        return {}

    ttl_value = response_cache.get("ttl_seconds")
    if OPENROUTER_RESPONSE_CACHE_TTL_ENV in env:
        ttl_value = env.get(OPENROUTER_RESPONSE_CACHE_TTL_ENV)
    ttl = _coerce_ttl_seconds(ttl_value)

    headers = {OPENROUTER_RESPONSE_CACHE_HEADER: "true"}
    if ttl is not None:
        headers[OPENROUTER_RESPONSE_CACHE_TTL_HEADER] = str(ttl)
    return headers


def openrouter_default_headers(
    config: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return attribution headers plus optional opt-in response-cache headers."""
    headers = dict(OPENROUTER_ATTRIBUTION_HEADERS)
    headers.update(openrouter_response_cache_headers(config=config, environ=environ))
    return headers
