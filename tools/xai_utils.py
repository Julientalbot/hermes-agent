"""Shared xAI utilities for credential resolution and availability checks."""

import os


def resolve_xai_credentials():
    """Return (api_key, base_url) for xAI."""
    key = os.getenv("XAI_API_KEY")
    base = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    return key, base


def check_xai_tool_available(tool_name: str = None) -> bool:
    """Check if an xAI tool is available (XAI_API_KEY set).

    Args:
        tool_name: Optional tool name for future per-tool gating.
                   Currently all xAI tools share the same key check.
    """
    return bool(os.getenv("XAI_API_KEY"))
