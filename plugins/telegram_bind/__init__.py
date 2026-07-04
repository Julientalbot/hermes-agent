"""telegram-bind plugin — auto-approve Telegram /start deep-link payloads.

Activation (per instance):
  1. Ensure this plugin ships in the Hermes image (bundled under ``plugins/telegram_bind/``).
  2. Enable it in gateway config::
       hermes plugins enable telegram-bind
     or add ``telegram-bind`` to ``plugins.enabled`` in ``~/.hermes/config.yaml``.
  3. Issue a code from fleet ops::
       hermes-fleet bind-code <slug> --customer <cus_...>
  4. After the client clicks the deep link, sync env allowlists::
       hermes-fleet bind-sync <slug>

Optional env:
  ``BIND_WELCOME_MESSAGE`` — override the default French welcome text.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from gateway.config import Platform

from .bind_codes import (
    DEFAULT_WELCOME_MESSAGE,
    BindCodesStore,
    _parse_iso_utc,
    _utc_now,
)

logger = logging.getLogger(__name__)

START_PAYLOAD_RE = re.compile(r"^/start\s+(\S+)$", re.IGNORECASE)
PLATFORM_NAME = Platform.TELEGRAM.value


@dataclass(frozen=True)
class BindEvaluation:
    """Internal result from evaluating a /start <code> message."""

    action: Literal["allow", "defer"]
    welcome_message: Optional[str] = None
    log_warning: Optional[str] = None


def _welcome_message() -> str:
    custom = os.getenv("BIND_WELCOME_MESSAGE", "").strip()
    return custom or DEFAULT_WELCOME_MESSAGE


def _schedule_welcome(gateway: Any, chat_id: str, text: str) -> None:
    adapter = gateway.adapters.get(Platform.TELEGRAM)
    if not adapter:
        logger.warning("telegram-bind: Telegram adapter unavailable for welcome send")
        return

    async def _send() -> None:
        try:
            await adapter.send(chat_id, text)
        except Exception as exc:
            logger.warning("telegram-bind: failed to send welcome message: %s", exc)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("telegram-bind: no running event loop for welcome send")
        return
    loop.create_task(_send())


def evaluate_start_bind(
    event: Any,
    gateway: Any,
    *,
    bind_store: Optional[BindCodesStore] = None,
    now: Optional[datetime] = None,
) -> BindEvaluation:
    """Evaluate a Telegram DM ``/start <code>`` against pre-issued bind codes."""
    source = event.source
    platform = getattr(source.platform, "value", source.platform)
    if platform != PLATFORM_NAME:
        return BindEvaluation(action="defer")
    if getattr(source, "chat_type", None) != "dm":
        return BindEvaluation(action="defer")
    if not source.user_id:
        return BindEvaluation(action="defer")

    pairing_store = gateway.pairing_store
    if pairing_store.is_approved(PLATFORM_NAME, source.user_id):
        return BindEvaluation(action="defer")

    text = (event.text or "").strip()
    match = START_PAYLOAD_RE.match(text)
    if not match:
        return BindEvaluation(action="defer")

    code = match.group(1).strip().upper()
    store = bind_store or BindCodesStore()
    entry = store.get(code)
    now = now or _utc_now()

    if not entry:
        return BindEvaluation(
            action="defer",
            log_warning=(
                f"telegram-bind: unknown bind code from user_id={source.user_id} "
                f"chat_id={source.chat_id}"
            ),
        )

    if entry.get("consumed_utc"):
        stored_user_id = str(entry.get("telegram_user_id") or "")
        if stored_user_id and pairing_store._user_ids_match(
            PLATFORM_NAME, stored_user_id, source.user_id
        ):
            return BindEvaluation(action="allow", welcome_message=_welcome_message())

        return BindEvaluation(
            action="defer",
            log_warning=(
                f"telegram-bind: bind code already consumed by another user "
                f"(code={code}, requester={source.user_id}, "
                f"bound_user={stored_user_id or 'unknown'})"
            ),
        )

    expires = _parse_iso_utc(str(entry.get("expires_utc", "")))
    if expires and now >= expires:
        return BindEvaluation(
            action="defer",
            log_warning=(
                f"telegram-bind: expired bind code from user_id={source.user_id} "
                f"chat_id={source.chat_id}"
            ),
        )

    user_name = getattr(source, "user_name", "") or ""
    with pairing_store._lock:
        pairing_store._approve_user(PLATFORM_NAME, source.user_id, user_name)

    consumed = store.consume(
        code,
        telegram_user_id=str(source.user_id),
        telegram_chat_id=str(source.chat_id or source.user_id),
    )
    if not consumed:
        logger.warning(
            "telegram-bind: approved user but failed to mark code consumed (code=%s)",
            code,
        )

    return BindEvaluation(action="allow", welcome_message=_welcome_message())


def on_pre_gateway_dispatch(event, gateway, session_store, **kwargs):
    """``pre_gateway_dispatch`` hook entry point.

    Returns ``{"action": "skip"}`` on a successful bind: the plugin sends the
    welcome itself, and the raw ``/start <code>`` text must not reach the
    agent as a user message.
    """
    del session_store, kwargs
    result = evaluate_start_bind(event, gateway)
    if result.log_warning:
        logger.warning(result.log_warning)
    if result.action != "allow":
        return None
    if result.welcome_message:
        _schedule_welcome(gateway, event.source.chat_id, result.welcome_message)
    return {"action": "skip", "reason": "telegram-bind-consumed"}


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)