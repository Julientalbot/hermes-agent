"""Tests for xAI Batch API tool."""

import json
from unittest.mock import MagicMock

import pytest

from tools.xai_batches_tool import (
    _handle_xai_batches,
    check_xai_batches_requirements,
    xai_batches_tool,
)


def _response(payload, status_code=200, text=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(payload) if text is None else text
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _http_error_response(payload, status_code=400):
    resp = _response(payload, status_code=status_code)
    import requests
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestXAIBatchesRequirements:
    def test_requirements_true_when_key_set(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        assert check_xai_batches_requirements() is True

    def test_requirements_false_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert check_xai_batches_requirements() is False


class TestXAIBatchesValidation:
    def test_missing_api_key_returns_tool_error(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        parsed = json.loads(xai_batches_tool(action="list"))
        assert "error" in parsed
        assert "XAI_API_KEY" in parsed["error"]

    def test_unknown_action_returns_tool_error(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        parsed = json.loads(xai_batches_tool(action="explode"))
        assert "error" in parsed
        assert "unsupported action" in parsed["error"]

    def test_batch_id_required_for_status(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        parsed = json.loads(xai_batches_tool(action="status"))
        assert "error" in parsed
        assert "batch_id" in parsed["error"]


class TestXAIBatchesHTTP:
    def test_create_batch_posts_name_and_input_file_id(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return _response({"batch_id": "batch-123", "name": "nightly"})

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_batches_tool(
            action="create", name="nightly", input_file_id="file-abc"
        ))

        assert parsed["tool"] == "xai_batches"
        assert parsed["action"] == "create"
        assert parsed["batch"]["batch_id"] == "batch-123"
        assert captured["url"] == "https://api.x.ai/v1/batches"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["json"] == {"name": "nightly", "input_file_id": "file-abc"}

    def test_add_requests_posts_batch_requests(self, monkeypatch):
        captured = {}
        batch_requests = [
            {
                "batch_request_id": "req-1",
                "batch_request": {
                    "responses": {
                        "model": "grok-4.20-reasoning",
                        "input": [{"role": "user", "content": "Classify A"}],
                    }
                },
            }
        ]

        def fake_post(url, headers, json, timeout):
            captured.update(url=url, json=json)
            response = _response({}, text="")
            response.json.side_effect = ValueError("empty body")
            return response

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_batches_tool(
            action="add_requests", batch_id="batch-123", batch_requests=batch_requests
        ))

        assert parsed["batch_id"] == "batch-123"
        assert parsed["result"] == {}
        assert captured["url"] == "https://api.x.ai/v1/batches/batch-123/requests"
        assert captured["json"] == {"batch_requests": batch_requests}

    def test_create_responses_batch_creates_then_adds_prompts(self, monkeypatch):
        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append((url, json))
            if url.endswith("/batches"):
                return _response({"batch_id": "batch-123", "name": "leads"})
            return _response({"num_requests": 2})

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_batches_tool(
            action="create_responses_batch",
            name="leads",
            prompts=["Classify lead A", "Classify lead B"],
            model="grok-4.20-reasoning",
            system_prompt="Return JSON only.",
        ))

        assert parsed["batch_id"] == "batch-123"
        assert parsed["request_count"] == 2
        assert calls[0] == ("https://api.x.ai/v1/batches", {"name": "leads"})
        added = calls[1][1]["batch_requests"]
        assert added[0]["batch_request_id"] == "req-000001"
        assert added[0]["batch_request"]["responses"]["model"] == "grok-4.20-reasoning"
        assert added[0]["batch_request"]["responses"]["input"][0] == {
            "role": "system", "content": "Return JSON only."
        }
        assert added[0]["batch_request"]["responses"]["input"][1] == {
            "role": "user", "content": "Classify lead A"
        }

    def test_status_gets_batch(self, monkeypatch):
        captured = {}

        def fake_get(url, headers, params, timeout):
            captured.update(url=url, params=params)
            return _response({"batch_id": "batch-123", "state": {"num_pending": 0}})

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.get", fake_get)

        parsed = json.loads(xai_batches_tool(action="status", batch_id="batch-123"))
        assert parsed["batch"]["state"]["num_pending"] == 0
        assert captured["url"] == "https://api.x.ai/v1/batches/batch-123"
        assert captured["params"] is None

    def test_results_gets_paginated_results(self, monkeypatch):
        captured = {}

        def fake_get(url, headers, params, timeout):
            captured.update(url=url, params=params)
            return _response({"results": [{"batch_request_id": "req-1"}]})

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.get", fake_get)

        parsed = json.loads(xai_batches_tool(
            action="results", batch_id="batch-123", limit=50, pagination_token="next"
        ))
        assert parsed["results"]["results"][0]["batch_request_id"] == "req-1"
        assert captured["url"] == "https://api.x.ai/v1/batches/batch-123/results"
        assert captured["params"] == {"limit": 50, "pagination_token": "next"}

    def test_cancel_posts_cancel_endpoint(self, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured.update(url=url, json=json)
            return _response({"batch_id": "batch-123", "cancelled": True})

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        parsed = json.loads(xai_batches_tool(action="cancel", batch_id="batch-123"))
        assert parsed["batch"]["cancelled"] is True
        assert captured["url"] == "https://api.x.ai/v1/batches/batch-123:cancel"
        assert captured["json"] == {}

    def test_http_error_message_is_returned(self, monkeypatch):
        def fake_get(url, headers, params, timeout):
            return _http_error_response({"error": {"message": "bad batch"}}, status_code=404)

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.get", fake_get)

        parsed = json.loads(xai_batches_tool(action="status", batch_id="missing"))
        assert "error" in parsed
        assert "bad batch" in parsed["error"]


class TestXAIBatchesHandler:
    def test_handler_forwards_arguments(self, monkeypatch):
        seen = {}

        def fake_tool(**kwargs):
            seen.update(kwargs)
            return json.dumps({"ok": True})

        monkeypatch.setattr("tools.xai_batches_tool.xai_batches_tool", fake_tool)

        result = json.loads(_handle_xai_batches({
            "action": "status",
            "batch_id": "batch-123",
            "limit": 10,
        }))

        assert result == {"ok": True}
        assert seen["action"] == "status"
        assert seen["batch_id"] == "batch-123"
        assert seen["limit"] == 10
