#!/usr/bin/env python3
"""
xAI Web Search Tool — Search the web via xAI's native web_search Responses API.

Grok searches, browses, and extracts web information server-side.
No intermediate services (Parallel, Firecrawl) needed.

Requires ``XAI_API_KEY`` in ``~/.hermes/.env``.

Configuration (optional, in config.yaml)::

    xai_web_search:
      model: grok-4.20-reasoning   # default
      timeout_seconds: 180          # default
      retries: 2                    # default

Usage::

    from tools.xai_web_search_tool import xai_web_search_tool, check_xai_web_search_requirements

    result = xai_web_search_tool(query="latest ergonomic research 2026")
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
MAX_DOMAINS = 5

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("xai_web_search", {})
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


def check_xai_web_search_requirements() -> bool:
    """Return True if XAI_API_KEY is set."""
    return bool(os.getenv("XAI_API_KEY"))


# ---------------------------------------------------------------------------
# Domain normalisation
# ---------------------------------------------------------------------------


def _normalize_domains(
    domains: Optional[List[str]], field_name: str
) -> List[str]:
    """Strip protocols, deduplicate, enforce MAX_DOMAINS."""
    if not domains:
        return []
    cleaned: List[str] = []
    seen: set = set()
    for d in domains:
        if not isinstance(d, str):
            continue
        domain = d.strip()
        # Strip protocol if accidentally included
        for prefix in ("https://", "http://", "www."):
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        domain = domain.rstrip("/")
        if not domain or domain in seen:
            continue
        seen.add(domain)
        cleaned.append(domain)
        if len(cleaned) >= MAX_DOMAINS:
            break
    return cleaned


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Pull the assistant message text from a Responses API payload."""
    # Fast path: top-level output_text
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


def _extract_server_tool_usage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract server-side tool usage metadata (searches, page views, etc.)."""
    usage: Dict[str, Any] = {}
    for key in ("server_side_tool_usage", "metadata"):
        data = payload.get(key)
        if data:
            usage.update(data)
    return usage


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


def xai_web_search_tool(
    query: str,
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    enable_image_understanding: bool = False,
) -> str:
    """Search the web via xAI's native web_search tool.

    Returns JSON with ``text``, ``citations``, and metadata.
    """
    if not query or not query.strip():
        return tool_error("query is required for xai_web_search")

    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        return tool_error("XAI_API_KEY not set — add it to ~/.hermes/.env")

    # Build the web_search tool definition
    tool_def: Dict[str, Any] = {"type": "web_search"}

    allowed = _normalize_domains(allowed_domains, "allowed_domains")
    excluded = _normalize_domains(excluded_domains, "excluded_domains")

    # allowed_domains and excluded_domains are mutually exclusive per API docs
    if allowed and excluded:
        return tool_error(
            "allowed_domains and excluded_domains cannot be set together"
        )

    if allowed:
        tool_def["filters"] = {"allowed_domains": allowed}
    elif excluded:
        tool_def["filters"] = {"excluded_domains": excluded}

    if enable_image_understanding:
        tool_def["enable_image_understanding"] = True

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
                logger.error(
                    "xai_web_search auth/client error: %s",
                    _http_error_message(exc),
                )
                return tool_error(
                    f"WebSearch error ({status}): {_http_error_message(exc)}"
                )
            logger.warning(
                "xai_web_search HTTP %s on attempt %d/%d: %s",
                status, attempt + 1, max_retries + 1, _http_error_message(exc),
            )
        except requests.ConnectionError as exc:
            last_error = exc
            logger.warning(
                "xai_web_search connection error on attempt %d/%d: %s",
                attempt + 1, max_retries + 1, exc,
            )
        except requests.Timeout as exc:
            last_error = exc
            logger.warning(
                "xai_web_search timeout on attempt %d/%d: %s",
                attempt + 1, max_retries + 1, exc,
            )

    # All retries exhausted
    if last_error is not None:
        if isinstance(last_error, requests.Timeout):
            return tool_error(
                f"WebSearch timed out after {timeout}s "
                f"({max_retries + 1} attempts)"
            )
        return tool_error(
            f"WebSearch failed after {max_retries + 1} attempts: {last_error}"
        )

    # Parse response
    try:
        resp_payload = response.json()
    except Exception as exc:
        logger.error("xai_web_search JSON decode failed: %s", exc)
        return tool_error("WebSearch: invalid JSON response from xAI")

    text = _extract_response_text(resp_payload)
    citations = _extract_citations(resp_payload)
    tool_usage = _extract_server_tool_usage(resp_payload)

    if not text and not citations:
        return tool_error("WebSearch returned empty response")

    result = {
        "tool": "xai_web_search",
        "query": query.strip(),
        "text": text,
        "citations": citations,
        "citation_count": len(citations),
        "model": _get_model(),
    }
    if tool_usage:
        result["tool_usage"] = tool_usage

    logger.info(
        "xai_web_search completed: query=%r, text_len=%d, citations=%d",
        query, len(text), len(citations),
    )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

XAI_WEB_SEARCH_SCHEMA = {
    "name": "xai_web_search",
    "description": (
        "Search the entire web using xAI's built-in Web Search tool. "
        "Grok searches, browses pages, and extracts information server-side. "
        "Returns results with source citations. "
        "Use this for general web research, current events, documentation, "
        "or any information not on X/Twitter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for on the web.",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    f"Restrict search to these domains only "
                    f"(max {MAX_DOMAINS}). Cannot be combined "
                    f"with excluded_domains."
                ),
            },
            "excluded_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    f"Exclude these domains from search "
                    f"(max {MAX_DOMAINS}). Cannot be combined "
                    f"with allowed_domains."
                ),
            },
            "enable_image_understanding": {
                "type": "boolean",
                "description": (
                    "Enable analysis of images found during web browsing. "
                    "Also enables image understanding for x_search if used "
                    "in the same session."
                ),
                "default": False,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Handler + registration
# ---------------------------------------------------------------------------


def _handle_xai_web_search(args: Dict[str, Any], **kw: Any) -> str:
    return xai_web_search_tool(
        query=args.get("query", ""),
        allowed_domains=args.get("allowed_domains"),
        excluded_domains=args.get("excluded_domains"),
        enable_image_understanding=args.get(
            "enable_image_understanding", False
        ),
    )


registry.register(
    name="xai_web_search",
    toolset="xai_web_search",
    schema=XAI_WEB_SEARCH_SCHEMA,
    handler=_handle_xai_web_search,
    check_fn=check_xai_web_search_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🌐",
    max_result_size_chars=100_000,
)
