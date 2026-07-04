"""Tests for the telegram-bind plugin and bind-code store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from plugins.telegram_bind import evaluate_start_bind, on_pre_gateway_dispatch
from plugins.telegram_bind.bind_codes import (
    BIND_CODE_LENGTH,
    BindCodesStore,
    _iso_utc,
    _utc_now,
)


def _make_telegram_event(
    text: str = "/start ABCDEFGH23",
    *,
    user_id: str = "900001",
    chat_id: str = "900001",
    chat_type: str = "dm",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id=chat_id,
            user_name="tester",
            chat_type=chat_type,
        ),
    )


def _make_gateway(*, approved: bool = False) -> SimpleNamespace:
    pairing_store = MagicMock()
    pairing_store.is_approved.return_value = approved
    pairing_store._user_ids_match.side_effect = lambda platform, left, right: str(left) == str(right)
    pairing_store._lock = MagicMock()
    pairing_store._lock.__enter__ = MagicMock(return_value=None)
    pairing_store._lock.__exit__ = MagicMock(return_value=False)
    adapter = SimpleNamespace(send=AsyncMock())
    return SimpleNamespace(
        pairing_store=pairing_store,
        adapters={Platform.TELEGRAM: adapter},
    )


@pytest.fixture
def bind_store(tmp_path) -> BindCodesStore:
    path = tmp_path / "telegram-bind-codes.json"
    return BindCodesStore(path=path)


@pytest.fixture
def issued_entry(bind_store) -> dict:
    return bind_store.issue("cus_test_001")


class TestBindCodesStore:
    def test_issue_generates_ten_char_code(self, bind_store):
        entry = bind_store.issue("cus_alpha")
        assert len(entry["code"]) == BIND_CODE_LENGTH
        assert entry["customer_ref"] == "cus_alpha"
        assert entry["consumed_utc"] is None

    def test_list_reports_pending_consumed_expired(self, bind_store):
        pending = bind_store.issue("cus_pending")
        consumed = bind_store.issue("cus_consumed")
        bind_store.consume(
            consumed["code"],
            telegram_user_id="111",
            telegram_chat_id="111",
        )
        expired = bind_store.issue("cus_expired")
        data = bind_store._load()
        data[expired["code"]]["expires_utc"] = _iso_utc(_utc_now() - timedelta(hours=1))
        bind_store._save(data)

        statuses = {row["code"]: row["status"] for row in bind_store.list_entries()}
        assert statuses[pending["code"]] == "pending"
        assert statuses[consumed["code"]] == "consumed"
        assert statuses[expired["code"]] == "expired"

    def test_revoke_removes_entry(self, bind_store):
        entry = bind_store.issue("cus_revoke")
        assert bind_store.revoke(entry["code"]) is True
        assert bind_store.get(entry["code"]) is None

    def test_get_first_unsynced_consumed(self, bind_store):
        entry = bind_store.issue("cus_sync")
        bind_store.consume(
            entry["code"],
            telegram_user_id="222",
            telegram_chat_id="222",
        )
        unsynced = bind_store.get_first_unsynced_consumed()
        assert unsynced is not None
        assert unsynced["code"] == entry["code"]
        assert bind_store.mark_env_synced(entry["code"]) is True
        assert bind_store.get_first_unsynced_consumed() is None


class TestEvaluateStartBind:
    def test_nominal_approves_and_consumes(self, bind_store, issued_entry):
        gateway = _make_gateway()
        event = _make_telegram_event(f"/start {issued_entry['code']}")

        result = evaluate_start_bind(event, gateway, bind_store=bind_store)

        assert result.action == "allow"
        assert result.welcome_message
        gateway.pairing_store._approve_user.assert_called_once()
        stored = bind_store.get(issued_entry["code"])
        assert stored["consumed_utc"]
        assert stored["telegram_user_id"] == "900001"

    def test_expired_code_defers_with_warning(self, bind_store, issued_entry):
        gateway = _make_gateway()
        data = bind_store._load()
        data[issued_entry["code"]]["expires_utc"] = _iso_utc(_utc_now() - timedelta(hours=1))
        bind_store._save(data)

        result = evaluate_start_bind(
            event=_make_telegram_event(f"/start {issued_entry['code']}"),
            gateway=gateway,
            bind_store=bind_store,
        )

        assert result.action == "defer"
        assert result.log_warning and "expired" in result.log_warning
        gateway.pairing_store._approve_user.assert_not_called()

    def test_unknown_code_defers_with_warning(self, bind_store):
        gateway = _make_gateway()
        result = evaluate_start_bind(
            event=_make_telegram_event("/start UNKNOWN12"),
            gateway=gateway,
            bind_store=bind_store,
        )
        assert result.action == "defer"
        assert result.log_warning and "unknown bind code" in result.log_warning

    def test_consumed_by_other_user_defers_with_warning(self, bind_store, issued_entry):
        gateway = _make_gateway()
        bind_store.consume(
            issued_entry["code"],
            telegram_user_id="800001",
            telegram_chat_id="800001",
        )

        result = evaluate_start_bind(
            event=_make_telegram_event(
                f"/start {issued_entry['code']}",
                user_id="900002",
                chat_id="900002",
            ),
            gateway=gateway,
            bind_store=bind_store,
        )

        assert result.action == "defer"
        assert result.log_warning and "another user" in result.log_warning

    def test_same_user_reclick_is_idempotent(self, bind_store, issued_entry):
        gateway = _make_gateway()
        bind_store.consume(
            issued_entry["code"],
            telegram_user_id="900001",
            telegram_chat_id="900001",
        )

        result = evaluate_start_bind(
            event=_make_telegram_event(f"/start {issued_entry['code']}"),
            gateway=gateway,
            bind_store=bind_store,
        )

        assert result.action == "allow"
        assert result.welcome_message
        gateway.pairing_store._approve_user.assert_not_called()

    def test_already_authorized_user_defers(self, bind_store, issued_entry):
        gateway = _make_gateway(approved=True)
        result = evaluate_start_bind(
            event=_make_telegram_event(f"/start {issued_entry['code']}"),
            gateway=gateway,
            bind_store=bind_store,
        )
        assert result.action == "defer"
        gateway.pairing_store._approve_user.assert_not_called()


@pytest.mark.asyncio
async def test_hook_returns_allow_and_schedules_welcome(monkeypatch, bind_store, issued_entry):
    gateway = _make_gateway()
    event = _make_telegram_event(f"/start {issued_entry['code']}")
    scheduled = []

    def _capture(gw, chat_id, text):
        scheduled.append((gw, chat_id, text))

    monkeypatch.setattr("plugins.telegram_bind._schedule_welcome", _capture)
    monkeypatch.setattr(
        "plugins.telegram_bind.BindCodesStore", lambda *args, **kwargs: bind_store
    )

    result = on_pre_gateway_dispatch(event, gateway, session_store=MagicMock())

    assert result == {"action": "skip", "reason": "telegram-bind-consumed"}
    assert scheduled
    assert scheduled[0][1] == "900001"