"""Real HTTP + SQLite + signed JWT, with only Telegram's key transport substituted."""
import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from gateway.screen_handoff import ScreenHandoffStore
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def setup_app(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(
        "bot_desktop:\n  handoff:\n    enabled: true\n"
        f"    public_url: https://screen.example/handoff/{name}\n    telegram_login_client_id: '12345'\n")
    scope = set_hermes_home_override(home)
    try:
        from hermes_cli.screen_server import create_screen_app
        app = create_screen_app()
        store = ScreenHandoffStore(home)
        row, _ = store.create_or_get(session_id="original", source_json=json.dumps({
            "platform": "telegram", "user_id": "42", "chat_id": "42", "chat_type": "dm"}),
            reason="Connect to the test site", protocol=3)
    finally:
        reset_hermes_home_override(scope)
    return app, store, row


def signer(monkeypatch):
    from gateway import screen_handoff_login as login
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(login, "_keys", lambda: SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=key.public_key())))

    def sign(expected_nonce, **changes):
        claims = {"iss": login.ISSUER, "aud": "12345", "id": 42,
                  "iat": int(time.time()), "exp": int(time.time()) + 120, "nonce": expected_nonce}
        claims.update(changes)
        return jwt.encode(claims, key, algorithm="RS256")
    return sign


@pytest.mark.parametrize("failure", ["signature", "keys_unavailable", "attempt_expired", "reissued"])
def test_failed_or_obsolete_login_keeps_control_protected(tmp_path, monkeypatch, failure):
    from gateway import screen_handoff_login as login
    from tools.bot_desktop import lease
    app, store, row = setup_app(tmp_path, "a")
    sign = signer(monkeypatch)
    with TestClient(app, base_url="https://screen.example") as client:
        base = "/handoff/a/screen-handoff/"
        client.get(base + row.invite_token)
        nonce = client.post(base + row.invite_token + "/telegram/begin", headers=HEADERS).json()["nonce"]
        token = sign(nonce)
        if failure == "signature":
            signer(monkeypatch)  # Replace the public key, retaining the signed token.
        elif failure == "keys_unavailable":
            def unavailable():
                raise jwt.PyJWKClientError("provider unavailable")
            monkeypatch.setattr(login, "_keys", unavailable)
        elif failure == "attempt_expired":
            now = login._now()
            monkeypatch.setattr(login, "_now", lambda: now + 121)
        else:
            replacement = store.reissue(session_id="another", source_json=row.source_json, reason="recover")
            assert replacement.protocol == 3
        response = client.post(base + "r/" + row.request_id + "/telegram/complete", headers=HEADERS,
                               json={"nonce": nonce, "id_token": token})
        assert response.status_code == 401
        assert store.get(row.request_id).state in {"pending", "opened"}
        assert not lease.human_holds(profile_key=store.profile_home)


def test_return_wins_over_inflight_takeover_without_reacquiring(tmp_path, monkeypatch):
    from tools.bot_desktop import lease
    from gateway.screen_handoff_return import return_control
    app, store, row = setup_app(tmp_path, "a")
    sign = signer(monkeypatch)
    monkeypatch.setattr("tools.bot_desktop.runtime.status", lambda: SimpleNamespace(running=True))
    original = ScreenHandoffStore.take_over
    def interrupted_takeover(self, cookie, viewer):
        handoff = original(self, cookie, viewer)
        assert handoff
        return_control(self, handoff, viewer)
        return handoff
    monkeypatch.setattr(ScreenHandoffStore, "take_over", interrupted_takeover)
    with TestClient(app, base_url="https://screen.example") as client:
        base = "/handoff/a/screen-handoff/"
        client.get(base + row.invite_token)
        nonce = client.post(base + row.invite_token + "/telegram/begin", headers=HEADERS).json()["nonce"]
        path = base + "r/" + row.request_id
        assert client.post(path + "/telegram/complete", headers=HEADERS,
                           json={"nonce": nonce, "id_token": sign(nonce)}).status_code == 200
        assert client.post(path + "/takeover", headers=HEADERS).status_code == 409
        assert store.get(row.request_id).state == "returned"
        assert not lease.human_holds(profile_key=store.profile_home)


HEADERS = {"Origin": "https://screen.example", "X-Hermes-Screen": "1"}


def test_login_binds_identity_browser_request_and_profile_then_consumes_once(tmp_path, monkeypatch):
    app, store, row = setup_app(tmp_path, "a")
    other_app, other_store, other_row = setup_app(tmp_path, "b")
    sign = signer(monkeypatch)
    from tools.bot_desktop import lease
    monkeypatch.setattr("tools.bot_desktop.runtime.status", lambda: SimpleNamespace(running=True))
    with TestClient(app, base_url="https://screen.example") as client:
        base = "/handoff/a/screen-handoff/"
        invitation = base + row.invite_token
        page = client.get(invitation)
        assert page.status_code == 200 and "Secure" in page.headers["set-cookie"]
        assert "HttpOnly" in page.headers["set-cookie"]
        assert store.get(row.request_id).state == "pending"
        assert not lease.human_holds(profile_key=store.profile_home)
        assert store.pending_confirmations() == []
        with store._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM screen_login_attempts").fetchone()[0] == 0
        assert client.post(invitation + "/challenge", headers=HEADERS).status_code == 409
        assert client.post(invitation + "/telegram/begin").status_code == 403
        begin = client.post(invitation + "/telegram/begin", headers=HEADERS).json()
        nonce = begin["nonce"]
        body = {"nonce": nonce, "id_token": sign(nonce)}
        complete = base + "r/" + row.request_id + "/telegram/complete"
        with TestClient(app, base_url="https://screen.example") as stranger:
            assert stranger.post(complete, headers=HEADERS, json=body).status_code == 401
        with TestClient(other_app, base_url="https://screen.example") as other:
            assert other.post("/handoff/b/screen-handoff/r/"+other_row.request_id+"/telegram/complete",
                              headers=HEADERS, json=body).status_code == 401
        assert client.post(complete, headers=HEADERS, json=body).status_code == 200
        assert store.get(row.request_id).state == "authorized"
        assert not lease.human_holds(profile_key=store.profile_home)
        assert client.post(complete, headers=HEADERS, json=body).status_code == 401
        path = base + "r/" + row.request_id
        assert client.post(path+"/takeover", headers=HEADERS).status_code == 200
        assert lease.human_holds(profile_key=store.profile_home)
        assert client.get("/handoff/a/api/config").status_code == 404
        assert client.post(path+"/return", headers=HEADERS).status_code == 200
        assert client.post(path+"/return", headers=HEADERS).status_code == 200
        assert not lease.human_holds(profile_key=store.profile_home)
        assert other_store.get(other_row.request_id).state == "pending"
        assert body["id_token"].encode() not in store.path.read_bytes()
        assert nonce.encode() not in store.path.read_bytes()


@pytest.mark.parametrize("changes", [{"id": 99}, {"aud": "wrong"}, {"iss": "https://wrong.invalid"},
                                    {"nonce": "another-browser"}, {"exp": 1}, {"id": True}])
def test_bad_claims_never_authorize_or_downgrade(tmp_path, monkeypatch, changes):
    app, store, row = setup_app(tmp_path, "a")
    sign = signer(monkeypatch)
    with TestClient(app, base_url="https://screen.example") as client:
        base = "/handoff/a/screen-handoff/"
        client.get(base+row.invite_token)
        begin = client.post(base+row.invite_token+"/telegram/begin", headers=HEADERS).json()
        response = client.post(base+"r/"+row.request_id+"/telegram/complete", headers=HEADERS,
                               json={"nonce": begin["nonce"], "id_token": sign(begin["nonce"], **changes)})
        assert response.status_code == 401
        assert store.get(row.request_id).state == "opened"
        assert store.get(row.request_id).protocol == 3
        assert store.pending_confirmations() == []
