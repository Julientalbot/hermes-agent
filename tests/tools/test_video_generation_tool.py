#!/usr/bin/env python3
"""Tests for tools/video_generation_tool.py."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure XAI_API_KEY is set for all tests."""
    monkeypatch.setenv("XAI_API_KEY", "test-key-12345")


# ---------------------------------------------------------------------------
# check_video_generation_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_returns_true_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-xxx")
        from tools.video_generation_tool import check_video_generation_requirements
        assert check_video_generation_requirements() is True

    def test_returns_false_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.video_generation_tool import check_video_generation_requirements
        assert check_video_generation_requirements() is False

    def test_returns_false_empty_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "   ")
        from tools.video_generation_tool import check_video_generation_requirements
        assert check_video_generation_requirements() is False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizeOperation:
    def test_default(self):
        from tools.video_generation_tool import _normalize_operation
        assert _normalize_operation(None) == "generate"
        assert _normalize_operation("") == "generate"

    def test_valid_operations(self):
        from tools.video_generation_tool import _normalize_operation
        assert _normalize_operation("generate") == "generate"
        assert _normalize_operation("edit") == "edit"
        assert _normalize_operation("extend") == "extend"

    def test_case_insensitive(self):
        from tools.video_generation_tool import _normalize_operation
        assert _normalize_operation("Generate") == "generate"
        assert _normalize_operation("EDIT") == "edit"

    def test_invalid_raises(self):
        from tools.video_generation_tool import _normalize_operation
        with pytest.raises(ValueError, match="operation must be"):
            _normalize_operation("invalid")


class TestNormalizeDuration:
    def test_default_generate(self):
        from tools.video_generation_tool import _normalize_duration
        assert _normalize_duration(None, "generate") == 8

    def test_default_extend(self):
        from tools.video_generation_tool import _normalize_duration
        assert _normalize_duration(None, "extend") == 6

    def test_clamps_max_generate(self):
        from tools.video_generation_tool import _normalize_duration
        assert _normalize_duration(100, "generate") == 15

    def test_clamps_max_extend(self):
        from tools.video_generation_tool import _normalize_duration
        assert _normalize_duration(100, "extend") == 10

    def test_clamps_min(self):
        from tools.video_generation_tool import _normalize_duration
        assert _normalize_duration(0, "generate") == 1


class TestNormalizeAspectRatio:
    def test_default(self):
        from tools.video_generation_tool import _normalize_aspect_ratio
        assert _normalize_aspect_ratio(None) == "16:9"

    def test_valid(self):
        from tools.video_generation_tool import _normalize_aspect_ratio
        assert _normalize_aspect_ratio("1:1") == "1:1"
        assert _normalize_aspect_ratio("9:16") == "9:16"

    def test_invalid_raises(self):
        from tools.video_generation_tool import _normalize_aspect_ratio
        with pytest.raises(ValueError, match="aspect_ratio must be"):
            _normalize_aspect_ratio("21:9")


class TestNormalizeResolution:
    def test_default(self):
        from tools.video_generation_tool import _normalize_resolution
        assert _normalize_resolution(None) == "720p"

    def test_valid(self):
        from tools.video_generation_tool import _normalize_resolution
        assert _normalize_resolution("480p") == "480p"
        assert _normalize_resolution("720p") == "720p"

    def test_invalid_raises(self):
        from tools.video_generation_tool import _normalize_resolution
        with pytest.raises(ValueError, match="resolution must be"):
            _normalize_resolution("1080p")


class TestNormalizeSize:
    def test_none(self):
        from tools.video_generation_tool import _normalize_size
        assert _normalize_size(None) is None
        assert _normalize_size("") is None

    def test_valid(self):
        from tools.video_generation_tool import _normalize_size
        assert _normalize_size("1280x720") == "1280x720"

    def test_invalid_raises(self):
        from tools.video_generation_tool import _normalize_size
        with pytest.raises(ValueError, match="size must be"):
            _normalize_size("1920x1080p")


class TestNormalizeReferenceImages:
    def test_empty(self):
        from tools.video_generation_tool import _normalize_reference_images
        assert _normalize_reference_images(None) == []
        assert _normalize_reference_images([]) == []

    def test_strips_whitespace(self):
        from tools.video_generation_tool import _normalize_reference_images
        result = _normalize_reference_images([" https://example.com/img.png "])
        assert result == ["https://example.com/img.png"]

    def test_max_5(self):
        from tools.video_generation_tool import _normalize_reference_images
        images = [f"https://example.com/{i}.png" for i in range(10)]
        result = _normalize_reference_images(images)
        assert len(result) == 5

    def test_skips_non_strings(self):
        from tools.video_generation_tool import _normalize_reference_images
        result = _normalize_reference_images(["https://ok.png", 123, None, ""])
        assert result == ["https://ok.png"]


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestXaiHeaders:
    def test_returns_headers(self):
        from tools.video_generation_tool import _xai_headers
        headers = _xai_headers()
        assert "Bearer test-key-12345" in headers["Authorization"]
        assert "application/json" in headers["Content-Type"]
        assert "Hermes-Agent" in headers["User-Agent"]

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.video_generation_tool import _xai_headers
        with pytest.raises(ValueError, match="XAI_API_KEY not set"):
            _xai_headers()


# ---------------------------------------------------------------------------
# video_generate_tool (async main function)
# ---------------------------------------------------------------------------


class TestVideoGenerateTool:
    @pytest.mark.asyncio
    async def test_missing_prompt_for_generate(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="", operation="generate")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "prompt or image_url" in parsed["error"]

    @pytest.mark.asyncio
    async def test_edit_requires_video_url(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test", operation="edit")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "video_url" in parsed["error"]

    @pytest.mark.asyncio
    async def test_extend_requires_video_url(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test", operation="extend")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "video_url" in parsed["error"]

    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test", operation="invalid")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_invalid_aspect_ratio(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test", aspect_ratio="21:9")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_invalid_resolution(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test", resolution="4k")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_timeout_too_low(self):
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test", timeout_seconds=5)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.video_generation_tool import video_generate_tool
        result = await video_generate_tool(prompt="test")
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful video generation with mocked httpx."""
        from tools.video_generation_tool import video_generate_tool

        # Mock submit response
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.raise_for_status = MagicMock()
        submit_resp.json = MagicMock(return_value={"id": "job-123"})

        # Mock poll response (done)
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json = MagicMock(return_value={
            "status": "done",
            "video": {"url": "https://xai.video/result.mp4"},
            "model": "grok-imagine-video",
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.video_generation_tool.httpx.AsyncClient", return_value=mock_client):
            result = await video_generate_tool(
                prompt="A cat playing piano",
                timeout_seconds=30,
                poll_interval_seconds=1,
            )

        parsed = json.loads(result)
        assert parsed["status"] == "done"
        assert parsed["video_url"] == "https://xai.video/result.mp4"
        assert parsed["tool"] == "video_generate"

    @pytest.mark.asyncio
    async def test_failed_generation(self):
        """Test failed video generation."""
        from tools.video_generation_tool import video_generate_tool

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.raise_for_status = MagicMock()
        submit_resp.json = MagicMock(return_value={"id": "job-456"})

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json = MagicMock(return_value={
            "status": "failed",
            "error": {"message": "Content policy violation"},
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.video_generation_tool.httpx.AsyncClient", return_value=mock_client):
            result = await video_generate_tool(
                prompt="test",
                timeout_seconds=30,
                poll_interval_seconds=1,
            )

        parsed = json.loads(result)
        assert "error" in parsed
        assert "Content policy" in parsed["error"]

    @pytest.mark.asyncio
    async def test_submit_auth_error(self):
        """Test auth error on submit (no retry)."""
        import httpx as httpx_lib
        from tools.video_generation_tool import video_generate_tool

        error_resp = MagicMock()
        error_resp.status_code = 401
        error_resp.text = "Unauthorized"
        error_resp.json = MagicMock(return_value={"error": {"message": "Invalid API key"}})
        error_resp.raise_for_status = MagicMock(
            side_effect=httpx_lib.HTTPStatusError("401", request=MagicMock(), response=error_resp)
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=error_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.video_generation_tool.httpx.AsyncClient", return_value=mock_client):
            result = await video_generate_tool(prompt="test")

        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_image_to_video(self):
        """Test image-to-video generation."""
        from tools.video_generation_tool import video_generate_tool

        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.raise_for_status = MagicMock()
        submit_resp.json = MagicMock(return_value={"id": "job-789"})

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.raise_for_status = MagicMock()
        poll_resp.json = MagicMock(return_value={
            "status": "done",
            "video": {"url": "https://xai.video/img2vid.mp4"},
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(return_value=poll_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.video_generation_tool.httpx.AsyncClient", return_value=mock_client):
            result = await video_generate_tool(
                image_url="https://example.com/cat.png",
                timeout_seconds=30,
                poll_interval_seconds=1,
            )

        parsed = json.loads(result)
        assert parsed["status"] == "done"
        assert parsed["video_url"] == "https://xai.video/img2vid.mp4"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self):
        from tools.video_generation_tool import _get_model
        assert _get_model() == "grok-imagine-video"

    def test_default_timeout(self):
        from tools.video_generation_tool import _get_timeout
        assert _get_timeout() == 240

    def test_default_poll_interval(self):
        from tools.video_generation_tool import _get_poll_interval
        assert _get_poll_interval() == 5

    def test_custom_config(self):
        with patch("tools.video_generation_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {
                "model": "grok-video-fast",
                "timeout_seconds": 120,
                "poll_interval_seconds": 2,
            }
            from tools.video_generation_tool import _get_model, _get_timeout, _get_poll_interval
            assert _get_model() == "grok-video-fast"
            assert _get_timeout() == 120
            assert _get_poll_interval() == 2
