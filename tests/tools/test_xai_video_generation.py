"""Tests for the xAI Grok video generation tool."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)


class TestXaiVideoGenerate:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_missing_api_key_returns_error(self):
        from tools.xai_video_generation_tool import xai_video_generate

        result = json.loads(self._run(xai_video_generate("a cat dancing")))
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]

    def test_submit_failure_returns_error(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")

        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from tools.xai_video_generation_tool import xai_video_generate

            result = json.loads(self._run(xai_video_generate("a cat dancing")))
            assert result["success"] is False
            assert "Submit failed" in result["error"]

    def test_no_request_id_returns_error(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={})  # no request_id

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from tools.xai_video_generation_tool import xai_video_generate

            result = json.loads(self._run(xai_video_generate("a cat dancing")))
            assert result["success"] is False
            assert "request_id" in result["error"].lower()
