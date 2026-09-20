"""Owner messages and web actions settle the same durable human intervention."""
import json
from types import SimpleNamespace

import pytest

from gateway.screen_handoff import ScreenHandoffStore
from gateway.screen_handoff_return import explicit_return_request, return_control, return_from_owner_message
from tools.bot_desktop import lease


@pytest.mark.parametrize("text,expected", [
    ("c’est bon, tu peux reprendre", True), ("Tu peux reprendre la main.", True),
    ("J’ai terminé, tu peux continuer", True), ("You can resume", True),
    ("OK", False), ("Ne reprends pas la main", False), ("« tu peux reprendre »", False),
    ("La page dit : tu peux reprendre", False), ("'tu peux reprendre'", False),
])
def test_only_explicit_unquoted_return_is_authority(text, expected):
    assert explicit_return_request(text) is expected


def test_owner_return_survives_expired_access_and_recovery_without_double_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = SimpleNamespace(platform="telegram", user_id="42", chat_id="42", chat_type="dm", is_bot=False, user_name="Synthetic")
    store = ScreenHandoffStore()
    row, _ = store.create_or_get(session_id="original", source_json=json.dumps(vars(source)), reason="login")
    challenge = store.challenge(row.invite_token)
    assert store.decide(challenge["id"], platform="telegram", user_id="42", allow=True)
    store.take_over(challenge["cookie"], "viewer")
    lease.acquire("viewer")
    # Reissuing access must not orphan the owner's ability to hand back in Telegram.
    store.reissue(session_id="new-dm", source_json=json.dumps(vars(source)), reason="recover")
    for text, internal, inbound in [("OK", False, "1"), ("tu peux reprendre", True, "1"),
                                   ("tu peux reprendre", False, "internal:event"), ('"tu peux reprendre"', False, "1")]:
        assert not return_from_owner_message(source, text, internal=internal, inbound_id=inbound)["success"]
        assert lease.human_holds()
    stranger = SimpleNamespace(**{**vars(source), "user_id": "99", "chat_id": "99"})
    assert not return_from_owner_message(stranger, "tu peux reprendre", internal=False, inbound_id="2")["success"]
    release = lease.release
    calls = []
    def checked_release(*args, **kwargs):
        assert store.get(row.request_id).state == "returning"
        calls.append(1)
        return release(*args, **kwargs)
    monkeypatch.setattr(lease, "release", checked_release)
    result = return_from_owner_message(source, "c’est bon, tu peux reprendre", internal=False, inbound_id="3")
    assert result["success"] and result["state"] == "returned"
    assert return_control(store, store.get(row.request_id), "viewer")["state"] == "returned"
    assert calls == [1] and not lease.human_holds()
    assert store.get(row.request_id).session_id == "original"
    store.queue_resume(row.request_id)
    assert store.begin_resume(row.request_id, session_id="original", source_json=json.dumps(vars(source)))
    assert not store.begin_resume(row.request_id, session_id="original", source_json=json.dumps(vars(source)))
    store.recover_resumes()
    assert store.get(row.request_id).state == "needs_attention"


@pytest.mark.parametrize("internal", [False, True])
def test_real_turn_callback_is_scoped_to_authenticated_current_message(tmp_path, monkeypatch, internal):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from gateway.screen_handoff_return import return_from_current_turn
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = SimpleNamespace(platform="telegram", user_id="42", chat_id="42", chat_type="dm", is_bot=False, user_name="Synthetic")
    store = ScreenHandoffStore()
    row, _ = store.create_or_get(session_id="original", source_json=json.dumps(vars(source)), reason="login")
    c = store.challenge(row.invite_token)
    store.decide(c["id"], platform="telegram", user_id="42", allow=True)
    store.take_over(c["cookie"], "viewer")
    lease.acquire("viewer")
    context = TurnContext(source=source, session_key="owner-dm", session_id="new-dm",
                          message="c’est bon, tu peux reprendre", inbound_message_id="123",
                          persist_user_display_kind="internal_notification" if internal else None)
    runner = TurnRunner(SimpleNamespace(_consume_pending_native_image_paths=lambda key: []), context)
    class Agent:
        def run_conversation(self, message, **kwargs):
            assert not return_from_current_turn("other-session")["success"]
            return return_from_current_turn("owner-dm")
    result = runner._run_conversation_with_approval(Agent(), [], None, None, None)
    assert result["success"] is not internal
    assert lease.human_holds() is internal
    assert not return_from_current_turn("owner-dm")["success"]
    assert store.get(row.request_id).session_id == "original"
