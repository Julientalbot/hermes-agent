"""Tests for xAI tokenize-text tool."""

import json
from unittest.mock import MagicMock

from tools.xai_tokenize_tool import (
    _handle_xai_tokenize,
    check_xai_tokenize_requirements,
    xai_tokenize_tool,
)


def _response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(payload)
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _http_error_response(payload, status_code=400):
    resp = _response(payload, status_code=status_code)
    import requests
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestXAITokenizeRequirements:
    def test_requirements_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        assert check_xai_tokenize_requirements() is True

    def test_requirements_false_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert check_xai_tokenize_requirements() is False


class TestXAITokenizeValidation:
    def test_missing_key_returns_tool_error(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        parsed = json.loads(xai_tokenize_tool(text="Hello"))
        assert "error" in parsed
        assert "XAI_API_KEY" in parsed["error"]

    def test_empty_text_returns_tool_error(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        parsed = json.loads(xai_tokenize_tool(text="   "))
        assert "error" in parsed
        assert "text is required" in parsed["error"]


class TestXAITokenizeHTTP:
    def test_posts_text_and_model_to_tokenize_endpoint(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return _response({
                "token_ids": [
                    {"token_id": 13902, "string_token": "Hello", "token_bytes": [72, 101, 108, 108, 111]},
                    {"token_id": 1749, "string_token": " world", "token_bytes": [32, 119, 111, 114, 108, 100]},
                ]
            })

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_tokenize_tool(text="Hello world", model="grok-4.20-reasoning"))

        assert parsed["tool"] == "xai_tokenize"
        assert parsed["model"] == "grok-4.20-reasoning"
        assert parsed["text"] == "Hello world"
        assert parsed["token_count"] == 2
        assert parsed["tokens"][0]["string_token"] == "Hello"
        assert captured["url"] == "https://api.x.ai/v1/tokenize-text"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["json"] == {"text": "Hello world", "model": "grok-4.20-reasoning"}

    def test_uses_configured_default_model(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["json"] = json
            return _response({"token_ids": []})

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr("tools.xai_tokenize_tool._load_config", lambda: {"model": "grok-4-fast"})

        parsed = json.loads(xai_tokenize_tool(text="Bonjour"))

        assert parsed["model"] == "grok-4-fast"
        assert captured["json"]["model"] == "grok-4-fast"

    def test_http_error_message_is_returned(self, monkeypatch):
        def fake_post(url, headers, json, timeout):
            return _http_error_response({"error": {"message": "bad model"}}, status_code=400)

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_tokenize_tool(text="Hello", model="bad-model"))
        assert "error" in parsed
        assert "bad model" in parsed["error"]

    def test_invalid_json_response_returns_tool_error(self, monkeypatch):
        def fake_post(url, headers, json, timeout):
            resp = _response({})
            resp.json.side_effect = ValueError("invalid json")
            return resp

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_tokenize_tool(text="Hello"))
        assert "error" in parsed
        assert "invalid JSON" in parsed["error"]


class TestXAITokenizeHandler:
    def test_handler_forwards_arguments(self, monkeypatch):
        seen = {}

        def fake_tool(**kwargs):
            seen.update(kwargs)
            return json.dumps({"ok": True})

        monkeypatch.setattr("tools.xai_tokenize_tool.xai_tokenize_tool", fake_tool)

        parsed = json.loads(_handle_xai_tokenize({"text": "Hello", "model": "grok-test"}))

        assert parsed == {"ok": True}
        assert seen == {"text": "Hello", "model": "grok-test"}
