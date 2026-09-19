import json

import pytest

from gateway.screen_handoff import ScreenHandoffStore


def source(user="42", topic="9"):
    return json.dumps({"platform": "telegram", "chat_id": "-10", "chat_type": "group", "user_id": user, "thread_id": topic})


def access(tmp_path):
    store = ScreenHandoffStore(tmp_path)
    request, _ = store.create_or_get(session_id="original", source_json=source(), reason="login")
    challenge = store.challenge(request.invite_token)
    assert store.decide(challenge["id"], platform="telegram", user_id="42", allow=True)
    return store, request, challenge


def test_one_request_per_desktop_and_owner_recovery_preserves_group(tmp_path):
    store, request, challenge = access(tmp_path)
    with pytest.raises(ValueError):
        store.create_or_get(session_id="other", source_json=source(), reason="login")
    assert store.reissue(session_id="dm", source_json=source("intruder"), reason="recovery") is None
    dm = json.dumps({"platform": "telegram", "chat_id": "42", "chat_type": "dm", "user_id": "42"})
    recovered = store.reissue(session_id="dm", source_json=dm, reason="recovery")
    assert recovered.request_id == request.request_id
    assert recovered.session_id == "original"
    assert recovered.source_json == source()
    assert store.web_session(challenge["cookie"]) is None


def test_browser_confirmation_is_bound_and_persists_without_bearers(tmp_path):
    store = ScreenHandoffStore(tmp_path)
    row, _ = store.create_or_get(session_id="s", source_json=source(), reason="login")
    a, b = store.challenge(row.invite_token), store.challenge(row.invite_token)
    assert store.challenge(row.invite_token, a["cookie"])["id"] == a["id"]
    assert store.decide(a["id"], platform="telegram", user_id="42", allow=True)
    assert not store.decide(b["id"], platform="telegram", user_id="42", allow=True)
    assert store.decide(a["id"], platform="telegram", user_id="42", allow=True)
    assert store.web_session(a["cookie"])
    assert store.web_session(b["cookie"]) is None
    raw = store.path.read_bytes()
    assert all(value.encode() not in raw for value in (row.invite_token, a["cookie"], b["cookie"]))


def test_expired_access_preserves_human_request(tmp_path, monkeypatch):
    store, request, c = access(tmp_path)
    store.take_over(c["cookie"], "viewer")
    assert store.stream_valid(request.request_id, "viewer")
    expiry = store.get(request.request_id).web_expires_at
    monkeypatch.setattr("gateway.screen_handoff._now", lambda: expiry + 1)
    assert not store.stream_valid(request.request_id, "viewer")
    assert store.web_session(c["cookie"]) is None
    assert store.get(request.request_id).state == "human"
    assert store.reissue(session_id="dm", source_json=source(), reason="recover").session_id == "original"


def test_restart_between_return_and_admission_does_not_duplicate(tmp_path):
    store, row, c = access(tmp_path)
    store.take_over(c["cookie"], "viewer")
    store.return_to_agent(c["cookie"], "viewer")
    restarted = ScreenHandoffStore(tmp_path)
    assert restarted.returning()[0].request_id == row.request_id
    restarted.complete_return(row.request_id)
    restarted.queue_resume(row.request_id)
    restarted = ScreenHandoffStore(tmp_path)
    assert restarted.claim_returned()[0].request_id == row.request_id
    assert not restarted.begin_resume(row.request_id, session_id="replaced", source_json=source())
    assert restarted.begin_resume(row.request_id, session_id="original", source_json=source())
    assert not restarted.begin_resume(row.request_id, session_id="original", source_json=source())
    restarted.recover_resumes()
    assert restarted.get(row.request_id).state == "needs_attention"
    assert restarted.claim_returned() == []


def test_challenge_deadline_starts_at_browser_request_and_is_rate_limited(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("gateway.screen_handoff._now", lambda: now[0])
    store = ScreenHandoffStore(tmp_path)
    row, _ = store.create_or_get(session_id="s", source_json=source(), reason="login")
    now[0] += 300
    c = store.challenge(row.invite_token)
    assert c["expires"] == now[0] + 120
    assert store.challenge(row.invite_token)
    assert store.challenge(row.invite_token)
    assert store.challenge(row.invite_token) is None
    now[0] += 121
    assert not store.decide(c["id"], platform="telegram", user_id="42", allow=True)


def test_revoke_access_does_not_forget_human_control(tmp_path):
    store, row, c = access(tmp_path)
    store.take_over(c["cookie"], "viewer")
    store.revoke(row.request_id)
    assert store.web_session(c["cookie"]) is None
    assert store.get(row.request_id).state == "human"
    with pytest.raises(ValueError):
        store.create_or_get(session_id="other", source_json=source(), reason="new")


def test_old_owner_cannot_resurrect_request_over_current_owner(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("gateway.screen_handoff._now", lambda: now[0])
    store = ScreenHandoffStore(tmp_path)
    store.create_or_get(session_id="old", source_json=source(), reason="old")
    now[0] += 601
    current, _ = store.create_or_get(session_id="new", source_json=source("99"), reason="new")
    assert store.reissue(session_id="dm", source_json=source(), reason="recover") is None
    assert store.get(current.request_id).state == "pending"
