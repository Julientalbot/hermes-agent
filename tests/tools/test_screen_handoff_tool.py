"""An invitation prepares the existing browser once and preserves failed delivery semantics."""
import json
from types import SimpleNamespace

import pytest

from tools import screen_handoff_tool as tool
from tools.bot_desktop import browser, runtime
from gateway import screen_handoff as handoff


@pytest.mark.parametrize("prepared,delivered", [(True, True), (False, True), (True, False)])
def test_invitation_prepares_once_and_reports_failure(tmp_path, monkeypatch, prepared, delivered):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = json.dumps({"platform": "telegram", "user_id": "42", "chat_id": "42", "chat_type": "dm"})
    monkeypatch.setattr(tool, "_public_url", lambda: "https://example.invalid")
    monkeypatch.setattr(tool, "_session_source", lambda _: (source, str(home), "key"))
    monkeypatch.setattr(tool, "_check_screen_handoff", lambda: True)
    monkeypatch.setattr(runtime, "ensure_started_for_tool", lambda: None)
    monkeypatch.setattr(runtime, "status", lambda: SimpleNamespace(running=True))
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":99"})
    monkeypatch.setattr(handoff, "has_screen_handoff_notify", lambda _: True)
    calls = []
    monkeypatch.setattr(browser, "present_running_browser", lambda: calls.append("prepare") or {"success": prepared, "error": "browser absent"})
    monkeypatch.setattr(handoff, "notify_screen_handoff", lambda *args: calls.append("notify") or delivered)
    first = json.loads(tool.request_screen_access({"reason": "Log in to finish the requested task"}, session_id="test"))
    assert first["success"] is (prepared and delivered)
    if first["success"]:
        second = json.loads(tool.request_screen_access({"reason": "same task"}, session_id="test"))
        assert second["reused"] and second["request_id"] == first["request_id"]
        assert calls == ["prepare", "notify"]
        assert "example.invalid" not in json.dumps(first)
    else:
        # Failure must not leave a pending request that blocks the next attempt.
        _, created = handoff.ScreenHandoffStore(home).create_or_get(session_id="test", source_json=source, reason="retry")
        assert created
        assert calls == (["prepare", "notify"] if prepared else ["prepare"])
