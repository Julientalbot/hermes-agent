#!/usr/bin/env python3
"""xAI server-side code execution via the Responses API.

This tool is intentionally separate from Hermes' local ``execute_code`` tool.
It asks Grok to solve a natural-language task while giving it xAI's hosted
``code_interpreter`` sandbox, so code runs on xAI infrastructure rather than on
the user's machine.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_CODE_EXECUTION_MODEL = "grok-4.3"
DEFAULT_XAI_CODE_EXECUTION_TIMEOUT_SECONDS = 180
DEFAULT_XAI_CODE_EXECUTION_RETRIES = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_xai_code_execution_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return load_config().get("xai_code_execution", {}) or {}
    except Exception:
        return {}


def _get_xai_code_execution_model() -> str:
    cfg = _load_xai_code_execution_config()
    return (
        str(cfg.get("model") or "").strip()
        or DEFAULT_XAI_CODE_EXECUTION_MODEL
    )


def _get_xai_code_execution_timeout_seconds() -> int:
    cfg = _load_xai_code_execution_config()
    raw_value = cfg.get(
        "timeout_seconds",
        DEFAULT_XAI_CODE_EXECUTION_TIMEOUT_SECONDS,
    )
    try:
        return max(30, int(raw_value))
    except Exception:
        return DEFAULT_XAI_CODE_EXECUTION_TIMEOUT_SECONDS


def _get_xai_code_execution_retries() -> int:
    cfg = _load_xai_code_execution_config()
    raw_value = cfg.get("retries", DEFAULT_XAI_CODE_EXECUTION_RETRIES)
    try:
        return max(0, int(raw_value))
    except Exception:
        return DEFAULT_XAI_CODE_EXECUTION_RETRIES


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def _resolve_xai_bearer() -> Tuple[str, str, str]:
    """Return ``(api_key, base_url, source)`` or raise on missing credentials."""
    creds = resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "No xAI credentials available. Run `hermes auth add xai-oauth` "
            "to sign in with your SuperGrok subscription, or set XAI_API_KEY."
        )
    base_url = str(creds.get("base_url") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    source = str(creds.get("provider") or "xai")
    return api_key, base_url, source


def check_xai_code_execution_requirements() -> bool:
    """Return True when xAI credentials are available and non-empty."""
    try:
        creds = resolve_xai_http_credentials()
        return bool(str(creds.get("api_key") or "").strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

    parts: List[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if ctype in ("output_text", "text"):
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_code_interpreter_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in ("code_interpreter_call", "code_execution_call"):
            calls.append(item)
    return calls


def _extract_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_calls = payload.get("tool_calls") or []
    if isinstance(raw_calls, dict):
        return [raw_calls]
    if isinstance(raw_calls, list):
        return [call for call in raw_calls if isinstance(call, dict)]
    return []


def _http_error_message(exc: requests.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        code = str(payload.get("code") or "").strip()
        error_value = payload.get("error")
        if isinstance(error_value, dict):
            error = str(error_value.get("message") or "").strip()
        else:
            error = str(error_value or "").strip()
        message = error or str(payload)
        if code and code not in message:
            message = f"{code}: {message}"
        return message or str(exc)

    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text[:500]
    return str(exc)


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def xai_code_execution_tool(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return tool_error("prompt is required for xai_code_execution")

    try:
        api_key, base_url, source = _resolve_xai_bearer()
    except RuntimeError as exc:
        return tool_error(str(exc))

    try:
        payload = {
            "model": _get_xai_code_execution_model(),
            "input": [
                {
                    "role": "user",
                    "content": prompt.strip(),
                }
            ],
            "tools": [{"type": "code_interpreter"}],
            "store": False,
        }

        timeout_seconds = _get_xai_code_execution_timeout_seconds()
        max_retries = _get_xai_code_execution_retries()
        response: Optional[requests.Response] = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": hermes_xai_user_agent(),
                    },
                    json=payload,
                    timeout=timeout_seconds,
                )
                response.raise_for_status()
                break
            except requests.HTTPError as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code is None or status_code < 500 or attempt >= max_retries:
                    raise
                logger.warning(
                    "xai_code_execution upstream failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    _http_error_message(e),
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "xai_code_execution transient failure on attempt %s/%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                time.sleep(min(5.0, 1.5 * (attempt + 1)))

        if response is None:
            raise RuntimeError("xai_code_execution request did not return a response")

        data = response.json()
        answer = _extract_response_text(data)

        return json.dumps(
            {
                "success": True,
                "provider": "xai",
                "credential_source": source,
                "tool": "xai_code_execution",
                "xai_tool": "code_interpreter",
                "model": payload["model"],
                "prompt": prompt.strip(),
                "answer": answer,
                "tool_calls": _extract_tool_calls(data),
                "code_interpreter_calls": _extract_code_interpreter_calls(data),
                "server_side_tool_usage": data.get("server_side_tool_usage") or {},
                "usage": data.get("usage") or {},
            },
            ensure_ascii=False,
        )
    except requests.HTTPError as e:
        logger.error("xai_code_execution failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_code_execution",
                "xai_tool": "code_interpreter",
                "error": _http_error_message(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except requests.Timeout as e:
        logger.error("xai_code_execution timed out: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_code_execution",
                "xai_tool": "code_interpreter",
                "error": (
                    "xAI code execution timed out after "
                    f"{_get_xai_code_execution_timeout_seconds()} seconds"
                ),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("xai_code_execution failed: %s", e, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "xai_code_execution",
                "xai_tool": "code_interpreter",
                "error": str(e),
                "error_type": type(e).__name__,
            },
            ensure_ascii=False,
        )


XAI_CODE_EXECUTION_SCHEMA = {
    "name": "xai_code_execution",
    "description": (
        "Ask Grok to solve a task using xAI's hosted code_interpreter "
        "sandbox. Use this for calculations, data analysis, simulations, or "
        "Python-backed verification when the user wants xAI/Grok server-side "
        "code execution rather than local Hermes execute_code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Natural-language task or data for Grok to analyze using "
                    "xAI's server-side Python sandbox."
                ),
            },
        },
        "required": ["prompt"],
    },
}


def _handle_xai_code_execution(args, **kw):
    if not isinstance(args, dict):
        args = {}
    return xai_code_execution_tool(prompt=args.get("prompt", ""))


registry.register(
    name="xai_code_execution",
    toolset="xai_code_execution",
    schema=XAI_CODE_EXECUTION_SCHEMA,
    handler=_handle_xai_code_execution,
    check_fn=check_xai_code_execution_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🧮",
    max_result_size_chars=100_000,
)
