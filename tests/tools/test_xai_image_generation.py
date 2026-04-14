"""Tests for the xAI Grok image generation tool."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    sys.modules.pop("tools.xai_image_generation_tool", None)


class TestXaiImageGenerate:
    def test_missing_api_key_returns_error(self):
        from tools.xai_image_generation_tool import xai_image_generate

        result = json.loads(xai_image_generate("a sunset"))
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]

    def test_successful_generation(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XAI_API_KEY", "test-key")

        fake_image_url = "https://example.com/fake.jpg"
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url=fake_image_url)]

        mock_client = MagicMock()
        mock_client.images.generate.return_value = mock_response
        mock_openai_cls = MagicMock(return_value=mock_client)

        fake_dl_response = MagicMock()
        fake_dl_response.content = b"fake-image-bytes"

        with patch("openai.OpenAI", mock_openai_cls), \
             patch("requests.get", return_value=fake_dl_response), \
             patch("hermes_constants.get_hermes_home", return_value=str(tmp_path)), \
             patch("hermes_constants.display_hermes_home", return_value="~/test"):
            from tools.xai_image_generation_tool import xai_image_generate

            result = json.loads(xai_image_generate("a sunset"))

        assert result["success"] is True
        assert "image" in result
        assert "media_tag" in result

    def test_api_error_returns_error(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.images.generate.side_effect = Exception("API down")
        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch("openai.OpenAI", mock_openai_cls):
            from tools.xai_image_generation_tool import xai_image_generate

            result = json.loads(xai_image_generate("a sunset"))
            assert result["success"] is False
            assert "API down" in result["error"]
