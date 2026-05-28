"""Tests for metadata-only xAI request trace capture."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

from agent import xai_trace_capture


@dataclass
class FakeRequest:
    url: str
    method: str = "POST"
    headers: dict[str, str] | None = None
    content: bytes = b""


@dataclass
class FakeResponse:
    headers: dict[str, str]
    status_code: int = 200


class StreamingBodyRequest:
    def __init__(
        self,
        *,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.method = method
        self.headers = headers

    @property
    def content(self):  # noqa: D102
        raise RuntimeError("body stream was not read")


def _read_one_jsonl(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def test_capture_writes_metadata_only_ledger_row(tmp_path, monkeypatch):
    ledger = tmp_path / "fso" / "xai-request-ledger.jsonl"
    monkeypatch.setenv("HERMES_XAI_TRACE_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("XAI_KEY_SURFACE", "xai-beta-fso-logged")
    monkeypatch.setenv("CLIENT_SLUG", "audit-client")

    raw_prompt = "private customer prompt that must never be written"
    raw_tool = "raw tool arguments that must never be written"
    request = FakeRequest(
        url="https://api.x.ai/v1/responses",
        headers={"x-grok-conv-id": "session-123"},
        content=json.dumps(
            {
                "model": "grok-build-0.1",
                "input": raw_prompt,
                "tools": [{"function": {"arguments": raw_tool}}],
                "extra_body": {"prompt_cache_key": "cache-key-ignored"},
            }
        ).encode("utf-8"),
    )
    response = FakeResponse(
        headers={
            "x-request-id": "req-123",
            "x-response-id": "resp-123",
            "x-trace-id": "trace-123",
        }
    )

    xai_trace_capture._capture(request, response, None, source="test")

    row = _read_one_jsonl(ledger)
    assert row["schema"] == "xai-request-ledger-v0.2"
    assert row["provider"] == "xai"
    assert row["key_surface"] == "xai-beta-fso-logged"
    assert row["tenant_instance"] == "audit-client"
    assert row["local_session_id"] == "session-123"
    assert row["model"] == "grok-build-0.1"
    assert row["endpoint"] == "responses / https://api.x.ai/v1/responses"
    assert row["http_status"] == 200
    assert row["request_id"] == "req-123"
    assert row["response_id"] == "resp-123"
    assert row["trace_id"] == "trace-123"
    assert row["privacy_tier"] == "metadata_only_no_raw_content"
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600

    serialized = json.dumps(row, sort_keys=True)
    assert raw_prompt not in serialized
    assert raw_tool not in serialized
    assert "input" not in row
    assert "tools" not in row


def test_capture_uses_body_session_when_header_missing(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("HERMES_XAI_TRACE_LEDGER_PATH", str(ledger))

    request = FakeRequest(
        url="https://api.x.ai/v1/chat/completions",
        content=json.dumps(
            {
                "model": "grok-4.3",
                "extra_body": {"prompt_cache_key": "body-session-123"},
            }
        ).encode("utf-8"),
    )

    xai_trace_capture._capture(request, FakeResponse(headers={}), None, source="test")

    row = _read_one_jsonl(ledger)
    assert row["local_session_id"] == "body-session-123"
    assert row["model"] == "grok-4.3"
    assert row["endpoint"] == "chat.completions / https://api.x.ai/v1/chat/completions"


def test_capture_survives_unread_streaming_body(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("HERMES_XAI_TRACE_LEDGER_PATH", str(ledger))

    request = StreamingBodyRequest(
        url="https://api.x.ai/v1/responses",
        headers={"x-grok-conv-id": "stream-session-123"},
    )

    xai_trace_capture._capture(
        request,
        FakeResponse(headers={"x-request-id": "req-stream"}),
        None,
        source="test",
    )

    row = _read_one_jsonl(ledger)
    assert row["local_session_id"] == "stream-session-123"
    assert row["model"] == "unknown"
    assert row["request_id"] == "req-stream"


def test_capture_ignores_non_xai_or_unsupported_endpoints(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("HERMES_XAI_TRACE_LEDGER_PATH", str(ledger))

    xai_trace_capture._capture(
        FakeRequest(url="https://api.openai.com/v1/responses"),
        FakeResponse(headers={}),
        None,
        source="test",
    )
    xai_trace_capture._capture(
        FakeRequest(url="https://api.x.ai/v1/models"),
        FakeResponse(headers={}),
        None,
        source="test",
    )

    assert not ledger.exists()
