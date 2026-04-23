#!/usr/bin/env python3
"""
xAI Batch API Tool — create, populate, monitor, and retrieve xAI batches.

Use batches for large non-interactive workloads where asynchronous execution
and discounted text processing are more important than immediate answers.

Requires ``XAI_API_KEY`` in ``~/.hermes/.env``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.20-reasoning"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RESULTS_LIMIT = 100
MAX_DIRECT_PROMPTS = 1000


def _get_base_url() -> str:
    return (os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config().get("xai_batches", {})
    except Exception:
        return {}


def _get_model() -> str:
    return (_load_config().get("model") or DEFAULT_MODEL).strip()


def _get_timeout() -> int:
    raw = _load_config().get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        return max(10, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def check_xai_batches_requirements() -> bool:
    """Return True if XAI_API_KEY is set."""
    return bool(os.getenv("XAI_API_KEY"))


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }


def _http_error_message(exc: requests.HTTPError) -> str:
    resp = exc.response
    if resp is None:
        return str(exc)
    try:
        body = resp.json()
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or resp.text[:300])
        if isinstance(error, str):
            return error
        return resp.text[:300]
    except Exception:
        return resp.text[:300]


def _request_id(index: int) -> str:
    return f"req-{index:06d}"


def _build_responses_batch_requests(
    prompts: List[str],
    model: str,
    system_prompt: str = "",
) -> List[Dict[str, Any]]:
    if not prompts:
        raise ValueError("prompts are required for create_responses_batch")
    if len(prompts) > MAX_DIRECT_PROMPTS:
        raise ValueError(
            f"too many prompts for direct mode: {len(prompts)} "
            f"(max {MAX_DIRECT_PROMPTS}; use Files API JSONL for larger batches)"
        )

    batch_requests: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, start=1):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt #{idx} is empty or not a string")
        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        batch_requests.append(
            {
                "batch_request_id": _request_id(idx),
                "batch_request": {
                    "responses": {
                        "model": model,
                        "input": messages,
                    }
                },
            }
        )
    return batch_requests


def _post_json(url: str, api_key: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    response.raise_for_status()
    if not response.text.strip():
        return {}
    return response.json()


def _get_json(
    url: str,
    api_key: str,
    timeout: int,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = requests.get(url, headers=_headers(api_key), params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _create_batch(api_key: str, timeout: int, name: str, input_file_id: str = "") -> Dict[str, Any]:
    if not name or not name.strip():
        raise ValueError("name is required for create")
    payload: Dict[str, Any] = {"name": name.strip()}
    if input_file_id and input_file_id.strip():
        payload["input_file_id"] = input_file_id.strip()
    return _post_json(f"{_get_base_url()}/batches", api_key, payload, timeout)


def _add_requests(
    api_key: str,
    timeout: int,
    batch_id: str,
    batch_requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not batch_id or not batch_id.strip():
        raise ValueError("batch_id is required for add_requests")
    if not batch_requests:
        raise ValueError("batch_requests are required for add_requests")
    payload = {"batch_requests": batch_requests}
    return _post_json(
        f"{_get_base_url()}/batches/{batch_id.strip()}/requests",
        api_key,
        payload,
        timeout,
    )


def xai_batches_tool(
    action: str,
    batch_id: str = "",
    name: str = "",
    input_file_id: str = "",
    batch_requests: Optional[List[Dict[str, Any]]] = None,
    prompts: Optional[List[str]] = None,
    model: str = "",
    system_prompt: str = "",
    limit: Optional[int] = None,
    pagination_token: str = "",
) -> str:
    """Manage xAI Batch API jobs.

    Supported actions:
    - create: create an empty batch or file-based batch
    - add_requests: add direct structured batch requests to an existing batch
    - create_responses_batch: create a batch and add simple Responses prompts
    - list: list batches
    - status: get one batch
    - requests: list request metadata for one batch
    - results: get paginated results for one batch
    - cancel: cancel a batch
    """
    api_key = os.getenv("XAI_API_KEY", "")
    if not api_key:
        return tool_error("XAI_API_KEY not set — add it to ~/.hermes/.env")

    action_key = (action or "").strip().lower()
    timeout = _get_timeout()
    base_url = _get_base_url()

    try:
        if action_key == "create":
            batch = _create_batch(api_key, timeout, name, input_file_id)
            return json.dumps({"tool": "xai_batches", "action": action_key, "batch": batch}, ensure_ascii=False)

        if action_key == "add_requests":
            added = _add_requests(api_key, timeout, batch_id, batch_requests or [])
            return json.dumps(
                {"tool": "xai_batches", "action": action_key, "batch_id": batch_id.strip(), "result": added},
                ensure_ascii=False,
            )

        if action_key == "create_responses_batch":
            batch = _create_batch(api_key, timeout, name, input_file_id="")
            new_batch_id = str(batch.get("batch_id") or batch.get("id") or "").strip()
            if not new_batch_id:
                return tool_error(f"Batch create returned no batch_id: {json.dumps(batch)}")
            request_model = (model or _get_model()).strip()
            built_requests = _build_responses_batch_requests(prompts or [], request_model, system_prompt)
            added = _add_requests(api_key, timeout, new_batch_id, built_requests)
            return json.dumps(
                {
                    "tool": "xai_batches",
                    "action": action_key,
                    "batch_id": new_batch_id,
                    "batch": batch,
                    "add_requests_result": added,
                    "request_count": len(built_requests),
                    "model": request_model,
                },
                ensure_ascii=False,
            )

        if action_key == "list":
            params: Dict[str, Any] = {}
            if limit is not None:
                params["limit"] = int(limit)
            if pagination_token:
                params["pagination_token"] = pagination_token
            batches = _get_json(f"{base_url}/batches", api_key, timeout, params or None)
            return json.dumps({"tool": "xai_batches", "action": action_key, "batches": batches}, ensure_ascii=False)

        if action_key in ("status", "requests", "results", "cancel"):
            if not batch_id or not batch_id.strip():
                return tool_error(f"batch_id is required for {action_key}")
            clean_id = batch_id.strip()

            if action_key == "status":
                batch = _get_json(f"{base_url}/batches/{clean_id}", api_key, timeout)
                return json.dumps({"tool": "xai_batches", "action": action_key, "batch": batch}, ensure_ascii=False)

            if action_key == "requests":
                params = {}
                if limit is not None:
                    params["limit"] = int(limit)
                if pagination_token:
                    params["pagination_token"] = pagination_token
                reqs = _get_json(f"{base_url}/batches/{clean_id}/requests", api_key, timeout, params or None)
                return json.dumps(
                    {"tool": "xai_batches", "action": action_key, "batch_id": clean_id, "requests": reqs},
                    ensure_ascii=False,
                )

            if action_key == "results":
                params = {}
                if limit is not None:
                    params["limit"] = int(limit)
                if pagination_token:
                    params["pagination_token"] = pagination_token
                results = _get_json(f"{base_url}/batches/{clean_id}/results", api_key, timeout, params or None)
                return json.dumps(
                    {"tool": "xai_batches", "action": action_key, "batch_id": clean_id, "results": results},
                    ensure_ascii=False,
                )

            cancelled = _post_json(f"{base_url}/batches/{clean_id}:cancel", api_key, {}, timeout)
            return json.dumps({"tool": "xai_batches", "action": action_key, "batch": cancelled}, ensure_ascii=False)

        return tool_error(
            "unsupported action for xai_batches; use create, add_requests, "
            "create_responses_batch, list, status, requests, results, or cancel"
        )

    except ValueError as exc:
        return tool_error(str(exc))
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return tool_error(f"xAI Batch API error ({status}): {_http_error_message(exc)}")
    except requests.Timeout:
        return tool_error(f"xAI Batch API timed out after {timeout}s")
    except requests.ConnectionError as exc:
        return tool_error(f"xAI Batch API connection error: {exc}")
    except Exception as exc:
        logger.error("xai_batches failed: %s", exc, exc_info=True)
        return tool_error(f"xAI Batch API failed: {exc}")


XAI_BATCHES_SCHEMA = {
    "name": "xai_batches",
    "description": (
        "Manage xAI Batch API jobs for large asynchronous workloads. "
        "Use create_responses_batch for simple bulk prompt processing, or create/add_requests/status/results/cancel for lower-level batch control."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: create, add_requests, create_responses_batch, list, status, requests, results, cancel.",
            },
            "batch_id": {"type": "string", "description": "Batch ID for status, requests, results, cancel, or add_requests."},
            "name": {"type": "string", "description": "Human-readable batch name for create/create_responses_batch."},
            "input_file_id": {"type": "string", "description": "Optional Files API JSONL input file ID when creating a file-based batch."},
            "batch_requests": {
                "type": "array",
                "description": "Direct xAI batch_requests payload for add_requests.",
                "items": {"type": "object"},
            },
            "prompts": {
                "type": "array",
                "description": "Simple prompt strings for create_responses_batch; each becomes one Responses API request.",
                "items": {"type": "string"},
            },
            "model": {"type": "string", "description": "Model for create_responses_batch; defaults to xai_batches.model config or grok-4.20-reasoning."},
            "system_prompt": {"type": "string", "description": "Optional system instruction prepended to every create_responses_batch request."},
            "limit": {"type": "integer", "description": "Optional pagination limit for list/requests/results."},
            "pagination_token": {"type": "string", "description": "Optional pagination token for list/requests/results."},
        },
        "required": ["action"],
    },
}


def _handle_xai_batches(args: Dict[str, Any], **kw: Any) -> str:
    return xai_batches_tool(
        action=args.get("action", ""),
        batch_id=args.get("batch_id", ""),
        name=args.get("name", ""),
        input_file_id=args.get("input_file_id", ""),
        batch_requests=args.get("batch_requests"),
        prompts=args.get("prompts"),
        model=args.get("model", ""),
        system_prompt=args.get("system_prompt", ""),
        limit=args.get("limit"),
        pagination_token=args.get("pagination_token", ""),
    )


registry.register(
    name="xai_batches",
    toolset="xai_batches",
    schema=XAI_BATCHES_SCHEMA,
    handler=_handle_xai_batches,
    check_fn=check_xai_batches_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="📦",
    max_result_size_chars=100_000,
)
