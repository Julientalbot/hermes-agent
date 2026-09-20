"""Agent-facing, non-blocking request for a human to take over Bot Desktop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tools.registry import registry

logger = logging.getLogger(__name__)


def _public_url() -> str:
    from gateway.screen_handoff_config import public_url as resolve_public_url
    return str(resolve_public_url() or "").rstrip("/")


def _check_screen_handoff() -> bool:
    """Expose the tool only when the configured web service and Bot Desktop exist."""
    if not _public_url():
        return False
    try:
        from tools.bot_desktop.runtime import status
        from gateway.screen_handoff_config import service_ready
        # Discovery is read-only. Bot Desktop owns its configured lazy start.
        return bool(status().installed and service_ready())
    except Exception:
        return False


def _session_source(session_id: str) -> tuple[str, str, str] | None:
    """Return (origin_json, profile_home, session_key) from the durable session row."""
    if not session_id:
        return None
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    home = Path(get_hermes_home())
    db = SessionDB(home / "state.db", read_only=True)
    try:
        row = db.get_session(str(session_id))
    finally:
        db.close()
    origin = str((row or {}).get("origin_json") or "")
    if not origin:
        return None
    try:
        parsed = json.loads(origin)
    except (TypeError, ValueError):
        return None
    platform = str(parsed.get("platform") or "").lower()
    if platform not in {"telegram", "discord"} or not parsed.get("user_id"):
        return None
    session_key = str((row or {}).get("session_key") or "")
    return origin, str((row or {}).get("profile_home") or home), session_key


def request_screen_access(args: dict[str, Any], *, session_id: str = "", **_kwargs: Any) -> str:
    reason = str((args or {}).get("reason") or "").strip()
    if not reason:
        return json.dumps({"success": False, "error": "reason is required"})
    if not _public_url():
        return json.dumps({"success": False, "error": "screen handoff is not configured"})
    from gateway.screen_handoff import (
        ScreenHandoffStore, has_screen_handoff_notify, notify_screen_handoff,
    )

    source_info = _session_source(session_id)
    if source_info is None:
        return json.dumps({"success": False, "error": "screen handoff requires a Telegram or Discord private identity"})
    origin_json, profile_home, session_key = source_info
    if not session_key or not has_screen_handoff_notify(session_key):
        return json.dumps({"success": False, "error": "screen handoff is unavailable for this turn"})
    if not _check_screen_handoff():
        return json.dumps({"success": False, "error": "the local screen service is unavailable"})
    from tools.bot_desktop.runtime import ensure_started_for_tool, status
    ensure_started_for_tool()
    if not status().running:
        return json.dumps({"success": False, "error": "Bot Desktop is stopped; check auto_start and its resource requirements"})
    store = ScreenHandoffStore(profile_home)
    handoff, created = None, False
    try:
        # Private recovery uses the existing owner-bound protocol and preserves
        # the original conversation, even if the owner starts a new DM session.
        source = json.loads(origin_json)
        if source.get("chat_type") in {"dm", "private"}:
            handoff = store.reissue(session_id=session_id, source_json=origin_json, reason=reason)
        if handoff is None:
            handoff, created = store.create_or_get(session_id=session_id, source_json=origin_json, reason=reason)
            if not created:
                if handoff.state in {"returning", "returned", "queued", "resuming", "needs_attention"}:
                    return json.dumps({"success": False, "state": handoff.state,
                                       "error": "The previous return is being processed or needs attention; no new invitation was sent."})
                handoff = store.reissue(session_id=session_id, source_json=origin_json, reason=reason)
                if handoff is None:
                    return json.dumps({"success": False, "error": "Screen recovery is unavailable for this conversation."})
        if created:
            from tools.bot_desktop.browser import present_running_browser
            from tools.browser_tool_session import run_fenced
            prepared = run_fenced({"features": {"local": True}}, present_running_browser)
            if not prepared.get("success"):
                store.revoke(handoff.request_id)
                return json.dumps(prepared)
        # Recovery never touches the browser or releases its human ownership.
        public_url = _public_url()
        delivered = notify_screen_handoff(session_key, {
            **handoff.public(), "invite_token": handoff.invite_token,
            "confirmation_code": handoff.confirmation_code,
            "invite_url": f"{public_url}/screen-handoff/{handoff.invite_token}",
        })
        if not delivered:
            store.revoke(handoff.request_id)
            return json.dumps({"success": False, "error": "private delivery is unavailable"})
        return json.dumps({
            "success": True, "state": handoff.state, "request_id": handoff.request_id,
            "expires_at": handoff.invite_expires_at, "delivery": "private",
            "reused": not created,
            "next_step": "End this turn now. Do not poll, wait, or perform further browser actions. Resume only after the user returns control, then reobserve the page.",
        })
    except Exception as exc:
        if created and handoff is not None:
            store.revoke(handoff.request_id)
        logger.warning("screen handoff request failed (%s)", type(exc).__name__)
        return json.dumps({"success": False, "error": f"screen handoff failed: {type(exc).__name__}"})


registry.register(
    name="request_screen_access",
    toolset="screen_handoff",
    schema={
        "name": "request_screen_access",
        "description": (
            "Ask the authenticated user to take over the Bot Desktop browser in a private Telegram "
            "or Discord message, only for an observed human-only blocker such as login or two-factor authentication. "
            "For a new intervention, first put the shared browser on the required page. "
            "Also use this tool once when the user asks to recover access or browser actions report human control: "
            "it sends a fresh private recovery button without navigating, reading the screen or releasing control. "
            "Do not ask the user to type a slash command. They authorize the recovered page and explicitly return control there. "
            "Give the task and short intervention needed, without secrets. "
            "This returns immediately and does not take control. After successful private delivery, briefly tell the user "
            "to use their computer and END THIS TURN: no polling, waiting, or further navigation. "
            "If delivery fails, explain the failure; logging in through the user’s personal browser will not authenticate "
            "this shared browser, so do not offer a normal site link as a substitute. "
            "After control is returned, reobserve before continuing the original task; "
            "returning control does not prove login or authorize new actions. Never ask for a password in chat."
        ),
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    },
    handler=request_screen_access,
    check_fn=_check_screen_handoff,
    requires_env=[],
    description="Request a short-lived private human screen takeover without blocking the agent turn.",
)
