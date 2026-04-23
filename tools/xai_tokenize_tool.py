#!/usr/bin/env python3
"""
xAI Tokenize Text Tool — inspect exact model-specific tokens.

Use this for exact token counts, prompt debugging, and context-budget checks.
Requires ``XAI_API_KEY`` in ``~/.hermes/.env``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.20-reasoning"
DEFAULT_TIMEOUT_SECONDS = 30


def _get_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("xai_tokenize", {})
    except Exception:
        return {}


def _get_model() -> str:
    return (_load_config().get("model") or DEFAULT_MODEL).strip()


def _get_timeout() -> int:
    raw = _load_config().get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        return max(5, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def check_xai_tokenize_requirements() -> bool:
    """Return True if XAI_API_KEY is set."""
    return bool(os.getenv("XAI_API_KEY"))


def _http_error_message(exc: requests.HTTPError) -> str:
    resp = exc.response
    if resp is None:
        return str(exc)
    try:
        body = resp.json()
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or resp.text[:300])
        if isinstance(error, str):
            return error
        return resp.text[:300]
    except Exception:
        return resp.text[:300]


def xai_tokenize_tool(text: str, model: str = "") -> str:
    """Tokenize text using xAI's model-specific tokenizer."""
    if not text or not text.strip():
        return tool_error("text is required for xai_tokenize")

    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        return tool_error("XAI_API_KEY not set — add it to ~/.hermes/.env")

    clean_text = text.strip()
    selected_model = (model or _get_model()).strip()
    payload = {"text": clean_text, "model": selected_model}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }

    try:
        response = requests.post(
            f"{_get_base_url()}/tokenize-text",
            headers=headers,
            json=payload,
            timeout=_get_timeout(),
        )
        response.raise_for_status()
        try:
            response_payload = response.json()
        except Exception:
            return tool_error("xAI tokenize-text returned invalid JSON")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return tool_error(f"xAI tokenize-text error ({status}): {_http_error_message(exc)}")
    except requests.Timeout:
        return tool_error(f"xAI tokenize-text timed out after {_get_timeout()}s")
    except requests.ConnectionError as exc:
        return tool_error(f"xAI tokenize-text connection error: {exc}")
    except Exception as exc:
        logger.error("xai_tokenize failed: %s", exc, exc_info=True)
        return tool_error(f"xAI tokenize-text failed: {exc}")

    tokens = response_payload.get("token_ids") or []
    result = {
        "tool": "xai_tokenize",
        "text": clean_text,
        "model": selected_model,
        "token_count": len(tokens),
        "tokens": tokens,
    }
    return json.dumps(result, ensure_ascii=False)


XAI_TOKENIZE_SCHEMA = {
    "name": "xai_tokenize",
    "description": (
        "Tokenize text with xAI's model-specific tokenizer. Returns exact token_count "
        "and per-token metadata (token_id, string_token, token_bytes). Useful for prompt debugging and context budgeting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to tokenize.",
            },
            "model": {
                "type": "string",
                "description": "xAI model tokenizer to use. Defaults to xai_tokenize.model config or grok-4.20-reasoning.",
            },
        },
        "required": ["text"],
    },
}


def _handle_xai_tokenize(args: Dict[str, Any], **kw: Any) -> str:
    return xai_tokenize_tool(
        text=args.get("text", ""),
        model=args.get("model", ""),
    )


registry.register(
    name="xai_tokenize",
    toolset="xai_tokenize",
    schema=XAI_TOKENIZE_SCHEMA,
    handler=_handle_xai_tokenize,
    check_fn=check_xai_tokenize_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🧮",
    max_result_size_chars=100_000,
)
