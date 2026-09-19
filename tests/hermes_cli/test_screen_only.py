import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gateway.screen_handoff import ScreenHandoffStore
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def create(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text("bot_desktop:\n  handoff:\n    enabled: true\n    public_url: https://screen.example/handoff/" + name + "\n")
    scope = set_hermes_home_override(home)
    try:
        from hermes_cli.screen_server import create_screen_app
        app = create_screen_app()
        store = ScreenHandoffStore(home)
        row, _ = store.create_or_get(session_id="original", source_json=json.dumps({
            "platform": "telegram", "user_id": "42", "chat_id": "42", "chat_type": "dm"}), reason="login")
    finally:
        reset_hermes_home_override(scope)
    return app, store, row


def test_http_preview_confirmation_and_cross_profile(tmp_path):
    a, store, row = create(tmp_path, "a")
    b, other, other_row = create(tmp_path, "b")
    headers = {"Origin": "https://screen.example", "X-Hermes-Screen": "1"}
    with TestClient(a, base_url="https://screen.example") as client:
        invite = "/handoff/a/screen-handoff/" + row.invite_token
        assert client.get(invite).status_code == 200
        assert store.get(row.request_id).state == "pending"
        assert store.pending_confirmations() == []
        assert client.post(invite + "/challenge").status_code == 403
        response = client.post(invite + "/challenge", headers=headers)
        assert response.status_code == 200
        assert "Secure" in response.headers["set-cookie"]
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "Path=/handoff/a/screen-handoff" in response.headers["set-cookie"]
        status = "/handoff/a/screen-handoff/r/" + row.request_id + "/status"
        assert client.get(status).json()["state"] == "waiting"
        challenge = store.pending_confirmations()[0]
        assert not other.decide(challenge["id"], platform="telegram", user_id="42", allow=True)
        assert store.decide(challenge["id"], platform="telegram", user_id="42", allow=True)
        assert client.get(status).json()["state"] == "authorized"
        # An authenticated cookie for A cannot operate on a different request.
        assert client.post("/handoff/a/screen-handoff/r/"+other_row.request_id+"/takeover", headers=headers).status_code == 401
        assert client.get("/handoff/a/api/config").status_code == 404
        assert client.get("/handoff/a/docs").status_code == 404
        with TestClient(b, base_url="https://screen.example") as second:
            assert second.get("/handoff/b/screen-handoff/"+row.invite_token).status_code == 410
        assert client.get(status).json()["state"] == "authorized"


def test_return_intent_precedes_release_and_double_click(tmp_path, monkeypatch):
    app, store, row = create(tmp_path, "a")
    from tools.bot_desktop import lease
    from hermes_cli.web_routers.screen_handoff import _viewer_id
    monkeypatch.setattr("tools.bot_desktop.runtime.status", lambda: SimpleNamespace(running=True, installed=True))
    headers = {"Origin": "https://screen.example", "X-Hermes-Screen": "1"}
    with TestClient(app, base_url="https://screen.example") as client:
        base = "/handoff/a/screen-handoff/"
        client.post(base+row.invite_token+"/challenge", headers=headers)
        challenge = store.pending_confirmations()[0]
        store.decide(challenge["id"], platform="telegram", user_id="42", allow=True)
        path = base+"r/"+row.request_id
        monkeypatch.setattr("gateway.drain_control.drain_requested", lambda: True)
        assert client.post(path+"/takeover", headers=headers).status_code == 409
        assert not lease.human_holds(profile_key=store.profile_home)
        monkeypatch.setattr("gateway.drain_control.drain_requested", lambda: False)
        assert client.post(path+"/takeover", headers=headers).status_code == 200
        assert lease.human_holds(profile_key=store.profile_home)
        release = lease.release
        calls = []

        def checked_release(*args, **kwargs):
            assert store.get(row.request_id).state == "returning"
            calls.append(1)
            return release(*args, **kwargs)

        monkeypatch.setattr(lease, "release", checked_release)
        assert client.post(path+"/return", headers=headers).status_code == 200
        assert client.post(path+"/return", headers=headers).status_code == 200
        assert calls == [1]
        assert not lease.human_holds(profile_key=store.profile_home)
