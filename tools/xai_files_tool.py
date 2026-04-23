#!/usr/bin/env python3
"""
xAI Files API Tool — Upload and analyze documents with Grok.

Send local files or public URLs to Grok for analysis.
Grok autonomously searches the document via the ``attachment_search``
server-side tool and returns a reasoned answer with citations.

Supports: PDF, TXT, CSV, JSON, MD, DOCX, XLSX, and more.

Requires ``XAI_API_KEY`` in ``~/.hermes/.env``.

Usage::

    from tools.xai_files_tool import xai_files_tool

    # Local file
    result = xai_files_tool(
        file_path="/path/to/audit.pdf",
        query="Résume les recommandations ergonomiques",
    )

    # Public URL
    result = xai_files_tool(
        file_url="https://example.com/report.pdf",
        query="Quels sont les points clés ?",
    )
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
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
DEFAULT_TIMEOUT_SECONDS = 300  # Files take longer to process
DEFAULT_RETRIES = 2
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# MIME types the API accepts (common document formats)
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/markdown",
    "text/html",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/msword",
    "application/vnd.ms-excel",
    "application/xml",
    "text/xml",
}

# File extension → MIME fallback
EXTENSION_MIME_MAP = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".doc": "application/msword",
    ".xls": "application/vnd.ms-excel",
    ".xml": "application/xml",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("xai_files", {})
    except Exception:
        return {}


def _get_model() -> str:
    return (_load_config().get("model") or DEFAULT_MODEL).strip()


def _get_timeout() -> int:
    raw = _load_config().get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        return max(30, int(raw))
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


def check_xai_files_requirements() -> bool:
    """Return True if XAI_API_KEY is set."""
    return bool(os.getenv("XAI_API_KEY"))


# ---------------------------------------------------------------------------
# MIME detection
# ---------------------------------------------------------------------------


def _detect_mime(file_path: Path) -> str:
    """Detect MIME type from file extension."""
    ext = file_path.suffix.lower()
    if ext in EXTENSION_MIME_MAP:
        return EXTENSION_MIME_MAP[ext]
    # Fallback to stdlib
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _upload_file(
    file_path: Path,
    api_key: str,
    timeout: int,
) -> Dict[str, Any]:
    """Upload a local file to xAI and return the file metadata.

    Uses multipart/form-data upload.
    """
    file_data = file_path.read_bytes()
    if len(file_data) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {len(file_data)} bytes "
            f"(max {MAX_FILE_SIZE_BYTES})"
        )

    mime = _detect_mime(file_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": hermes_xai_user_agent(),
    }

    url = f"{_get_base_url()}/files"
    resp = requests.post(
        url,
        headers=headers,
        files={"file": (file_path.name, file_data, mime)},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _delete_file(file_id: str, api_key: str) -> None:
    """Delete an uploaded file. Best-effort, errors are logged."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": hermes_xai_user_agent(),
        }
        url = f"{_get_base_url()}/files/{file_id}"
        requests.delete(url, headers=headers, timeout=30)
    except Exception as exc:
        logger.warning("Failed to delete file %s: %s", file_id, exc)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Pull the assistant message text from a Responses API payload."""
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

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


def xai_files_tool(
    query: str,
    file_path: str = "",
    file_url: str = "",
) -> str:
    """Analyze a document with Grok via xAI Files API.

    Provide either ``file_path`` (local) or ``file_url`` (public).
    Grok autonomously searches the document and answers the query.

    Returns JSON with ``text``, ``citations``, and metadata.
    """
    if not query or not query.strip():
        return tool_error("query is required — what should Grok analyze?")

    if not file_path and not file_url:
        return tool_error("provide either file_path or file_url")

    if file_path and file_url:
        return tool_error("provide only one: file_path or file_url")

    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        return tool_error("XAI_API_KEY not set — add it to ~/.hermes/.env")

    timeout = _get_timeout()
    max_retries = _get_retries()
    uploaded_file_id: Optional[str] = None

    # --- Upload local file if needed ---
    file_ref: Dict[str, Any] = {}
    source_label = ""

    if file_path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return tool_error(f"file not found: {file_path}")
        if not path.is_file():
            return tool_error(f"not a file: {file_path}")

        try:
            logger.info("Uploading %s (%s)", path.name, _detect_mime(path))
            upload_result = _upload_file(path, api_key, timeout)
            uploaded_file_id = upload_result.get("id") or upload_result.get("file_id")
            if not uploaded_file_id:
                return tool_error(
                    f"Upload returned no file ID: {json.dumps(upload_result)}"
                )
            file_ref = {"type": "input_file", "file_id": uploaded_file_id}
            source_label = path.name
            logger.info(
                "Uploaded %s -> %s (%s bytes)",
                path.name,
                uploaded_file_id,
                upload_result.get("bytes") or upload_result.get("size_bytes"),
            )
        except ValueError as exc:
            return tool_error(str(exc))
        except requests.HTTPError as exc:
            return tool_error(
                f"Upload failed ({exc.response.status_code}): "
                f"{_http_error_message(exc)}"
            )
        except Exception as exc:
            return tool_error(f"Upload failed: {exc}")

    elif file_url:
        file_ref = {"type": "input_file", "file_url": file_url.strip()}
        source_label = file_url.strip()

    # --- Build Responses API payload ---
    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": query.strip()},
        file_ref,
    ]

    payload = {
        "model": _get_model(),
        "input": [{"role": "user", "content": user_content}],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }

    url = f"{_get_base_url()}/responses"

    # --- Retry loop ---
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout,
            )
            response.raise_for_status()
            last_error = None
            break
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response else 0
            if status in (400, 401, 403):
                logger.error(
                    "xai_files error: %s", _http_error_message(exc)
                )
                return tool_error(
                    f"FilesAPI error ({status}): {_http_error_message(exc)}"
                )
            logger.warning(
                "xai_files HTTP %s attempt %d/%d: %s",
                status, attempt + 1, max_retries + 1, _http_error_message(exc),
            )
        except requests.ConnectionError as exc:
            last_error = exc
            logger.warning(
                "xai_files connection error attempt %d/%d: %s",
                attempt + 1, max_retries + 1, exc,
            )
        except requests.Timeout as exc:
            last_error = exc
            logger.warning(
                "xai_files timeout attempt %d/%d: %s",
                attempt + 1, max_retries + 1, exc,
            )

    # --- All retries exhausted ---
    if last_error is not None:
        if isinstance(last_error, requests.Timeout):
            return tool_error(
                f"FilesAPI timed out after {timeout}s "
                f"({max_retries + 1} attempts)"
            )
        return tool_error(
            f"FilesAPI failed after {max_retries + 1} attempts: {last_error}"
        )

    # --- Parse response ---
    try:
        resp_payload = response.json()
    except Exception as exc:
        logger.error("xai_files JSON decode failed: %s", exc)
        return tool_error("FilesAPI: invalid JSON response from xAI")

    text = _extract_response_text(resp_payload)
    citations = _extract_citations(resp_payload)

    # --- Cleanup uploaded file ---
    if uploaded_file_id:
        _delete_file(uploaded_file_id, api_key)

    if not text and not citations:
        return tool_error("FilesAPI returned empty response")

    result = {
        "tool": "xai_files",
        "query": query.strip(),
        "source": source_label,
        "text": text,
        "citations": citations,
        "citation_count": len(citations),
        "model": _get_model(),
    }

    logger.info(
        "xai_files completed: source=%r, text_len=%d, citations=%d",
        source_label, len(text), len(citations),
    )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

XAI_FILES_SCHEMA = {
    "name": "xai_files",
    "description": (
        "Upload and analyze documents with Grok via xAI Files API. "
        "Grok autonomously searches the document and answers your query. "
        "Supports PDF, TXT, CSV, DOCX, XLSX, and more. "
        "Provide either a local file_path or a public file_url. "
        "Ideal for audit reports, technical documents, spreadsheets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to analyze or extract from the document. "
                    "Be specific for better results."
                ),
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Absolute path to a local file to upload and analyze. "
                    "Mutually exclusive with file_url."
                ),
            },
            "file_url": {
                "type": "string",
                "description": (
                    "Public URL of a document to analyze (no upload needed). "
                    "Mutually exclusive with file_path."
                ),
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Handler + registration
# ---------------------------------------------------------------------------


def _handle_xai_files(args: Dict[str, Any], **kw: Any) -> str:
    return xai_files_tool(
        query=args.get("query", ""),
        file_path=args.get("file_path", ""),
        file_url=args.get("file_url", ""),
    )


registry.register(
    name="xai_files",
    toolset="xai_files",
    schema=XAI_FILES_SCHEMA,
    handler=_handle_xai_files,
    check_fn=check_xai_files_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="📄",
    max_result_size_chars=100_000,
)
