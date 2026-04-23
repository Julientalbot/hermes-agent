#!/usr/bin/env python3
"""Tests for tools/x_search_tool.py."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure XAI_API_KEY is set for all tests."""
    monkeypatch.setenv("XAI_API_KEY", "test-key-12345")


@pytest.fixture()
def mock_post():
    """Patch requests.post for x_search calls."""
    with patch("tools.x_search_tool.requests.post") as m:
        yield m


def _make_response(
    text: str = "Test result",
    citations: list | None = None,
    status_code: int = 200,
) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        http_err = requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
        resp.json.return_value = {
            "error": {"message": f"HTTP {status_code}"}
        }
        return resp

    output_items = []
    content_blocks = [{"type": "output_text", "text": text}]
    if citations:
        content_blocks[0]["annotations"] = [
            {"type": "url_citation", **c} for c in citations
        ]
    output_items.append({"type": "message", "content": content_blocks})
    resp.json.return_value = {"output": output_items}
    return resp


# ---------------------------------------------------------------------------
# check_x_search_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_returns_true_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-xxx")
        from tools.x_search_tool import check_x_search_requirements
        assert check_x_search_requirements() is True

    def test_returns_false_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.x_search_tool import check_x_search_requirements
        assert check_x_search_requirements() is False


# ---------------------------------------------------------------------------
# _normalize_handles
# ---------------------------------------------------------------------------


class TestNormalizeHandles:
    def test_strips_at_prefix(self):
        from tools.x_search_tool import _normalize_handles
        result = _normalize_handles(["@elonmusk", "@xaboratory"], "test")
        assert result == ["elonmusk", "xaboratory"]

    def test_deduplicates(self):
        from tools.x_search_tool import _normalize_handles
        result = _normalize_handles(["@a", "a", "@a"], "test")
        assert result == ["a"]

    def test_empty_input(self):
        from tools.x_search_tool import _normalize_handles
        assert _normalize_handles(None, "test") == []
        assert _normalize_handles([], "test") == []

    def test_max_handles(self):
        from tools.x_search_tool import _normalize_handles, MAX_HANDLES
        handles = [f"@user{i}" for i in range(MAX_HANDLES + 5)]
        result = _normalize_handles(handles, "test")
        assert len(result) == MAX_HANDLES

    def test_skips_non_strings(self):
        from tools.x_search_tool import _normalize_handles
        result = _normalize_handles(["@valid", 123, None, "@ok"], "test")
        assert result == ["valid", "ok"]


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------


class TestExtractResponseText:
    def test_from_output_text(self):
        from tools.x_search_tool import _extract_response_text
        payload = {"output_text": "Hello world"}
        assert _extract_response_text(payload) == "Hello world"

    def test_from_output_items(self):
        from tools.x_search_tool import _extract_response_text
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Found it"}],
                }
            ]
        }
        assert _extract_response_text(payload) == "Found it"

    def test_empty(self):
        from tools.x_search_tool import _extract_response_text
        assert _extract_response_text({}) == ""
        assert _extract_response_text({"output": []}) == ""


# ---------------------------------------------------------------------------
# _extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_extracts_url_citations(self):
        from tools.x_search_tool import _extract_citations
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "result",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://x.com/user/status/123",
                                    "title": "Post",
                                    "start_index": 0,
                                    "end_index": 10,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        cites = _extract_citations(payload)
        assert len(cites) == 1
        assert cites[0]["url"] == "https://x.com/user/status/123"

    def test_empty(self):
        from tools.x_search_tool import _extract_citations
        assert _extract_citations({}) == []
        assert _extract_citations({"output": []}) == []


# ---------------------------------------------------------------------------
# x_search_tool (main function)
# ---------------------------------------------------------------------------


class TestXSearchTool:
    def test_missing_query(self):
        from tools.x_search_tool import x_search_tool
        result = x_search_tool(query="")
        parsed = json.loads(result)
        assert "error" in parsed

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.x_search_tool import x_search_tool
        result = x_search_tool(query="test")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "XAI_API_KEY" in parsed.get("error", "")

    def test_successful_search(self, mock_post):
        mock_post.return_value = _make_response(
            text="xAI announced Grok 4.3",
            citations=[
                {"url": "https://x.com/xai/status/1", "title": "xAI", "start_index": 0, "end_index": 5}
            ],
        )
        from tools.x_search_tool import x_search_tool
        result = x_search_tool(query="xAI Grok 4.3")
        parsed = json.loads(result)
        assert parsed["text"] == "xAI announced Grok 4.3"
        assert parsed["citation_count"] == 1
        assert parsed["tool"] == "x_search"
        assert mock_post.call_count == 1

    def test_handles_passed_to_payload(self, mock_post):
        mock_post.return_value = _make_response(text="ok")
        from tools.x_search_tool import x_search_tool
        x_search_tool(
            query="test",
            allowed_x_handles=["@xaboratory"],
            excluded_x_handles=["@spam"],
        )
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        tool_def = payload["tools"][0]
        assert tool_def["allowed_x_handles"] == ["xaboratory"]
        assert tool_def["excluded_x_handles"] == ["spam"]

    def test_date_filters_passed(self, mock_post):
        mock_post.return_value = _make_response(text="ok")
        from tools.x_search_tool import x_search_tool
        x_search_tool(query="test", from_date="2026-01-01", to_date="2026-04-23")
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        tool_def = payload["tools"][0]
        assert tool_def["from_date"] == "2026-01-01"
        assert tool_def["to_date"] == "2026-04-23"

    def test_auth_header(self, mock_post):
        mock_post.return_value = _make_response(text="ok")
        from tools.x_search_tool import x_search_tool
        x_search_tool(query="test")
        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Bearer test-key-12345" in headers["Authorization"]
        assert "Hermes-Agent" in headers["User-Agent"]

    def test_retry_on_server_error(self, mock_post):
        import requests as req_lib

        # First call: server error
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.text = "Internal Server Error"
        error_resp.json.return_value = {"error": {"message": "Internal error"}}
        http_err = req_lib.HTTPError(response=error_resp)

        # Second call: success
        ok_resp = _make_response(text="recovered")

        call_count = [0]
        def post_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                resp = MagicMock()
                resp.status_code = 500
                resp.raise_for_status.side_effect = http_err
                resp.text = "Internal Server Error"
                resp.json.return_value = {"error": {"message": "Internal error"}}
                return resp
            return ok_resp

        mock_post.side_effect = post_side_effect

        from tools.x_search_tool import x_search_tool
        result = x_search_tool(query="test")
        parsed = json.loads(result)
        assert parsed["text"] == "recovered"
        assert mock_post.call_count == 2

    def test_no_retry_on_auth_error(self, mock_post):
        import requests as req_lib
        error_resp = MagicMock()
        error_resp.status_code = 401
        error_resp.text = "Unauthorized"
        error_resp.json.return_value = {"error": {"message": "Unauthorized"}}
        http_err = req_lib.HTTPError(response=error_resp)
        error_resp.raise_for_status.side_effect = http_err
        mock_post.return_value = error_resp

        from tools.x_search_tool import x_search_tool
        result = x_search_tool(query="test")
        parsed = json.loads(result)
        assert "error" in parsed
        assert mock_post.call_count == 1  # No retry


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self):
        from tools.x_search_tool import _get_model
        assert _get_model() == "grok-4.20-reasoning"

    def test_default_timeout(self):
        from tools.x_search_tool import _get_timeout
        assert _get_timeout() == 180

    def test_default_retries(self):
        from tools.x_search_tool import _get_retries
        assert _get_retries() == 2

    def test_custom_config(self, monkeypatch):
        """Verify config.yaml overrides work by patching _load_config."""
        with patch("tools.x_search_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {
                "model": "grok-4-fast",
                "timeout_seconds": 60,
                "retries": 0,
            }
            from tools.x_search_tool import _get_model, _get_timeout, _get_retries
            assert _get_model() == "grok-4-fast"
            assert _get_timeout() == 60
            assert _get_retries() == 0
