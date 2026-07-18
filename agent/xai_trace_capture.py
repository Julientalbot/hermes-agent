"""Metadata-only xAI request trace capture.

This module records provider join keys for xAI API calls without copying raw
prompts, responses, tool arguments, or client transcripts. It is intentionally
implemented as a small HTTPX hook because both the OpenAI SDK and Hermes
Responses paths eventually cross that boundary.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

_LOCK = Lock()
_PATCHED = False
_ORIGINAL_SYNC_SEND = None
_ORIGINAL_ASYNC_SEND = None

_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_FALSY = {"", "0", "false", "no", "off", "disabled"}


def _truthy(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in _FALSY:
        return False
    return text in _TRUTHY


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _header(headers: Any, *names: str) -> str | None:
    for name in names:
        try:
            value = headers.get(name) or headers.get(name.lower()) or headers.get(name.upper())
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def _safe_text(value: Any, *, max_len: int = 200) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\n", " ").replace("\r", " ")
    return text[:max_len]


def _body_metadata(request: Any) -> dict[str, str | None]:
    """Extract non-content metadata from a JSON request body.

    Only allowlisted scalar routing fields are returned. Raw input/messages,
    instructions, tools, and arguments are deliberately ignored.
    """
    try:
        content = getattr(request, "content", b"") or b""
    except Exception:
        return {"model": None, "body_session_id": None}
    if not content:
        return {"model": None, "body_session_id": None}
    try:
        payload = json.loads(content.decode("utf-8", "ignore"))
    except Exception:
        return {"model": None, "body_session_id": None}
    if not isinstance(payload, dict):
        return {"model": None, "body_session_id": None}
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    return {
        "model": _safe_text(payload.get("model")),
        "body_session_id": _safe_text(
            payload.get("prompt_cache_key")
            or payload.get("session_id")
            or extra_body.get("prompt_cache_key")
        ),
    }


def _endpoint_kind(url: str) -> str | None:
    if "/v1/responses" in url:
        return "responses / https://api.x.ai/v1/responses"
    if "/v1/chat/completions" in url:
        return "chat.completions / https://api.x.ai/v1/chat/completions"
    return None


def _ledger_path() -> Path | None:
    explicit = os.environ.get("HERMES_XAI_TRACE_LEDGER_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HERMES_HOME", "").strip()
    if not home:
        return None
    return Path(home).expanduser() / "fso" / "xai-request-ledger.jsonl"


def _write_row(row: dict[str, Any]) -> None:
    path = _ledger_path()
    if path is None:
        return
    try:
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with _LOCK:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(payload)
            os.chmod(path, 0o600)
    except Exception:
        # Capture is observability only; never break an agent request.
        return


def _capture(request: Any, response: Any = None, error: BaseException | None = None, *, source: str) -> None:
    try:
        url = str(getattr(request, "url", "") or "")
        if "api.x.ai" not in url:
            return
        endpoint = _endpoint_kind(url)
        if endpoint is None:
            return

        headers = getattr(response, "headers", {}) if response is not None else {}
        request_headers = getattr(request, "headers", {}) or {}
        body = _body_metadata(request)
        local_session_id = (
            _header(request_headers, "x-grok-conv-id", "x-client-request-id", "session_id")
            or body.get("body_session_id")
            or _safe_text(os.environ.get("HERMES_SESSION_ID"))
        )

        row = {
            "schema": "xai-request-ledger-v0.2",
            "captured_at_utc": _now_utc(),
            "capture_source": source,
            "provider": "xai",
            "key_surface": _safe_text(os.environ.get("XAI_KEY_SURFACE")) or "unknown",
            "tenant_instance": _safe_text(
                os.environ.get("CLIENT_SLUG")
                or os.environ.get("HERMES_CLIENT_SLUG")
                or os.environ.get("HERMES_TENANT")
            )
            or "unknown",
            "local_session_id": local_session_id or "missing_not_available",
            "model": body.get("model") or _safe_text(os.environ.get("MODEL_DEFAULT")) or "unknown",
            "endpoint": endpoint,
            "method": _safe_text(getattr(request, "method", "POST"), max_len=16) or "POST",
            "http_status": getattr(response, "status_code", None) if response is not None else None,
            "privacy_tier": "metadata_only_no_raw_content",
            "request_id": _header(
                headers,
                "x-request-id",
                "request-id",
                "x-requestid",
                "xai-request-id",
            ),
            "response_id": _header(headers, "x-response-id", "response-id"),
            "trace_id": _header(headers, "x-trace-id", "trace-id", "xai-trace-id"),
            "error_type": type(error).__name__ if error is not None else None,
        }
        _write_row(row)
    except Exception:
        return


def install_xai_trace_capture_from_env() -> bool:
    """Install the HTTPX hook when HERMES_XAI_TRACE_CAPTURE is truthy."""
    global _PATCHED, _ORIGINAL_ASYNC_SEND, _ORIGINAL_SYNC_SEND
    if _PATCHED:
        return True
    if not _truthy(os.environ.get("HERMES_XAI_TRACE_CAPTURE")):
        return False
    try:
        import httpx
    except Exception:
        return False

    _ORIGINAL_SYNC_SEND = httpx.Client.send
    _ORIGINAL_ASYNC_SEND = httpx.AsyncClient.send

    def patched_sync_send(self, request, *args, **kwargs):
        try:
            response = _ORIGINAL_SYNC_SEND(self, request, *args, **kwargs)
        except Exception as exc:
            _capture(request, None, exc, source="httpx_sync")
            raise
        _capture(request, response, None, source="httpx_sync")
        return response

    async def patched_async_send(self, request, *args, **kwargs):
        try:
            response = await _ORIGINAL_ASYNC_SEND(self, request, *args, **kwargs)
        except Exception as exc:
            _capture(request, None, exc, source="httpx_async")
            raise
        _capture(request, response, None, source="httpx_async")
        return response

    httpx.Client.send = patched_sync_send
    httpx.AsyncClient.send = patched_async_send
    _PATCHED = True
    return True


__all__ = ["install_xai_trace_capture_from_env"]
