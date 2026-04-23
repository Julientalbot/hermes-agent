#!/usr/bin/env python3
"""
X Search Tool — Search X (Twitter) via xAI's native x_search Responses API.

xAI is the only provider with native access to X/Twitter data.
This tool uses the Responses API `x_search` built-in tool to search
posts, profiles, and threads.

Requires ``XAI_API_KEY`` in ``~/.hermes/.env``.

Configuration (optional, in config.yaml)::

    x_search:
      model: grok-4.20-reasoning   # default
      timeout_seconds: 180          # default
      retries: 2                    # default

Usage::

    from tools.x_search_tool import x_search_tool, check_x_search_requirements

    result = x_search_tool(query="latest news about Hermes Agent")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.20-reasoning"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_RETRIES = 2
MAX_HANDLES = 10

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("x_search", {})
    except Exception:
        return {}


def _get_model() -> str:
    return (_load_config().get("model") or DEFAULT_MODEL).strip()


def _get_timeout() -> int:
    raw = _load_config().get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        return max(10, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def _get_retries() -> int:
    raw = _load_config().get("retries", DEFAULT_RETRIES)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# Requirement check
# ---------------------------------------------------------------------------


def check_x_search_requirements() -> bool:
    """Return True if XAI_API_KEY is set."""
    return bool(os.getenv("XAI_API_KEY"))


# ---------------------------------------------------------------------------
# Handle normalisation
# ---------------------------------------------------------------------------


def _normalize_handles(
    handles: Optional[List[str]], field_name: str
) -> List[str]:
    """Strip @ prefixes, deduplicate, enforce MAX_HANDLES."""
    if not handles:
        return []
    cleaned: List[str] = []
    seen: set = set()
    for h in handles:
        if not isinstance(h, str):
            continue
        handle = h.strip().lstrip("@")
        if not handle or handle in seen:
            continue
        seen.add(handle)
        cleaned.append(handle)
        if len(cleaned) >= MAX_HANDLES:
            break
    return cleaned


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Pull the assistant message text from a Responses API payload."""
    # Fast path: top-level output_text (some SDK versions)
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

    # Standard path: walk output items
    parts: List[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if block.get("type") == "output_text":
                parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def _extract_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract inline URL citations from the response."""
    citations: List[Dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            for ann in block.get("annotations", []) or []:
                if ann.get("type") != "url_citation":
                    continue
                citations.append(
                    {
                        "url": ann.get("url", ""),
                        "title": ann.get("title", ""),
                        "start_index": ann.get("start_index"),
                        "end_index": ann.get("end_index"),
                    }
                )
    return citations


def _http_error_message(exc: requests.HTTPError) -> str:
    """Extract a readable error from an HTTPError."""
    resp = exc.response
    if resp is None:
        return str(exc)
    try:
        body = resp.json()
        return body.get("error", {}).get("message", resp.text[:300])
    except Exception:
        return resp.text[:300]


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


def x_search_tool(
    query: str,
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: str = "",
    to_date: str = "",
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
) -> str:
    """Search X (Twitter) via xAI's native x_search tool.

    Returns JSON with ``text``, ``citations``, and metadata.
    """
    if not query or not query.strip():
        return tool_error("query is required for x_search")

    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        return tool_error("XAI_API_KEY not set — add it to ~/.hermes/.env")

    # Build the x_search tool definition
    tool_def: Dict[str, Any] = {"type": "x_search"}

    allowed = _normalize_handles(allowed_x_handles, "allowed_x_handles")
    excluded = _normalize_handles(excluded_x_handles, "excluded_x_handles")

    if allowed:
        tool_def["allowed_x_handles"] = allowed
    if excluded:
        tool_def["excluded_x_handles"] = excluded
    if from_date:
        tool_def["from_date"] = from_date.strip()
    if to_date:
        tool_def["to_date"] = to_date.strip()
    if enable_image_understanding:
        tool_def["enable_image_understanding"] = True
    if enable_video_understanding:
        tool_def["enable_video_understanding"] = True

    # Build payload
    payload = {
        "model": _get_model(),
        "input": [{"role": "user", "content": query.strip()}],
        "tools": [tool_def],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }

    timeout = _get_timeout()
    max_retries = _get_retries()
    url = f"{_get_base_url()}/responses"

    # Retry loop
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            last_error = None
            break
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response else 0
            if status in (400, 401, 403):
                # Non-retryable
                logger.error("x_search auth/client error: %s", _http_error_message(exc))
                return tool_error(f"xSearch error ({status}): {_http_error_message(exc)}")
            logger.warning(
                "x_search HTTP %s on attempt %d/%d: %s",
                status, attempt + 1, max_retries + 1, _http_error_message(exc),
            )
        except requests.ConnectionError as exc:
            last_error = exc
            logger.warning(
                "x_search connection error on attempt %d/%d: %s",
                attempt + 1, max_retries + 1, exc,
            )
        except requests.Timeout as exc:
            last_error = exc
            logger.warning(
                "x_search timeout on attempt %d/%d: %s",
                attempt + 1, max_retries + 1, exc,
            )

    # All retries exhausted
    if last_error is not None:
        if isinstance(last_error, requests.Timeout):
            return tool_error(
                f"xSearch timed out after {timeout}s ({max_retries + 1} attempts)"
            )
        return tool_error(f"xSearch failed after {max_retries + 1} attempts: {last_error}")

    # Parse response
    try:
        payload = response.json()
    except Exception as exc:
        logger.error("x_search JSON decode failed: %s", exc)
        return tool_error(f"xSearch: invalid JSON response from xAI")

    text = _extract_response_text(payload)
    citations = _extract_citations(payload)

    if not text and not citations:
        return tool_error("xSearch returned empty response")

    result = {
        "tool": "x_search",
        "query": query.strip(),
        "text": text,
        "citations": citations,
        "citation_count": len(citations),
        "model": _get_model(),
    }

    logger.info(
        "x_search completed: query=%r, text_len=%d, citations=%d",
        query, len(text), len(citations),
    )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

X_SEARCH_SCHEMA = {
    "name": "x_search",
    "description": (
        "Search X (Twitter) posts, profiles, and threads using xAI's built-in "
        "X Search tool. Use this for current discussion, reactions, or claims "
        "on X rather than general web pages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up on X.",
            },
            "allowed_x_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of X handles to search exclusively "
                    f"(max {MAX_HANDLES})."
                ),
            },
            "excluded_x_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of X handles to exclude "
                    f"(max {MAX_HANDLES})."
                ),
            },
            "from_date": {
                "type": "string",
                "description": "Optional start date in YYYY-MM-DD format.",
            },
            "to_date": {
                "type": "string",
                "description": "Optional end date in YYYY-MM-DD format.",
            },
            "enable_image_understanding": {
                "type": "boolean",
                "description": "Analyze images attached to matching X posts.",
                "default": False,
            },
            "enable_video_understanding": {
                "type": "boolean",
                "description": "Analyze videos attached to matching X posts.",
                "default": False,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Handler + registration
# ---------------------------------------------------------------------------


def _handle_x_search(args: Dict[str, Any], **kw: Any) -> str:
    return x_search_tool(
        query=args.get("query", ""),
        allowed_x_handles=args.get("allowed_x_handles"),
        excluded_x_handles=args.get("excluded_x_handles"),
        from_date=args.get("from_date", ""),
        to_date=args.get("to_date", ""),
        enable_image_understanding=args.get("enable_image_understanding", False),
        enable_video_understanding=args.get("enable_video_understanding", False),
    )


registry.register(
    name="x_search",
    toolset="x_search",
    schema=X_SEARCH_SCHEMA,
    handler=_handle_x_search,
    check_fn=check_x_search_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🐦",
    max_result_size_chars=100_000,
)
