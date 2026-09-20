"""Telegram Login authorizes one browser, not possession of an invitation URL."""
from __future__ import annotations

import secrets
import hashlib
import hmac
from functools import lru_cache

import jwt

from gateway.screen_handoff import CONFIRM_TTL_SECONDS, WEB_SESSION_TTL_SECONDS, _digest, _now

ISSUER = "https://oauth.telegram.org"


@lru_cache(maxsize=1)
def _keys():
    # Only provider public keys are cached globally; no profile credentials or claims.
    return jwt.PyJWKClient(ISSUER + "/.well-known/jwks.json", timeout=5, lifespan=300)


def verify_identity(token: str, *, client_id: str, nonce: str) -> str:
    if not isinstance(token, str) or len(token) > 16384 or not client_id or not nonce:
        raise ValueError("invalid Telegram login")
    claims = jwt.decode(token, _keys().get_signing_key_from_jwt(token).key,
                        algorithms=["RS256"], audience=str(client_id), issuer=ISSUER,
                        options={"require": ["exp", "iat", "iss", "aud", "nonce", "id"]}, leeway=15)
    if not isinstance(claims["nonce"], str) or not secrets.compare_digest(claims["nonce"], nonce):
        raise ValueError("invalid Telegram nonce")
    owner = claims["id"]
    if type(owner) is not int or owner <= 0:
        raise ValueError("invalid Telegram identity")
    return str(owner)


def login_nonce(cookie: str, request_id: str, profile_home: str) -> str:
    # A GET can prepare the SDK without a journal mutation. The random HttpOnly
    # cookie is the preimage: a leaked ID token's nonce cannot seed another browser.
    return hmac.new(cookie.encode(), (profile_home + "\0" + request_id).encode(), hashlib.sha256).hexdigest()


def begin_login(store, invite: str, cookie: str) -> dict | None:
    if not cookie or len(cookie) > 128:
        return None
    now = _now()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM screen_handoffs WHERE invite_digest=? AND profile_home=?",
                           (_digest(invite), store.profile_home)).fetchone()
        if (not row or row["protocol"] != 3 or row["invite_expires_at"] <= now
                or row["state"] not in {"pending", "opened"}):
            return None
        nonce = login_nonce(cookie, row["request_id"], store.profile_home)
        existing = conn.execute("SELECT * FROM screen_login_attempts WHERE nonce_digest=?", (_digest(nonce),)).fetchone()
        if existing:
            if existing["consumed_at"] is not None or existing["expires"] <= now:
                return None
            return {"nonce": nonce, "expires": existing["expires"], "request_id": row["request_id"]}
        count = conn.execute("SELECT COUNT(*) FROM screen_login_attempts WHERE request_id=? AND created>?",
                             (row["request_id"], now - 600)).fetchone()[0]
        if count >= 3:
            return None
        expires = min(now + CONFIRM_TTL_SECONDS, row["invite_expires_at"])
        conn.execute("INSERT INTO screen_login_attempts VALUES(?,?,?,?,?,NULL)",
                     (_digest(nonce), row["request_id"], _digest(cookie), now, expires))
        conn.execute("UPDATE screen_handoffs SET state='opened', opened_at=?, confirmation_expires_at=? WHERE request_id=?",
                     (now, expires, row["request_id"]))
        conn.commit()
    return {"nonce": nonce, "expires": expires, "request_id": row["request_id"]}


def complete_login(store, request_id: str, cookie: str, nonce: str, token: str, *, client_id: str) -> bool:
    # Check binding before contacting Telegram, then recheck under the write lock.
    with store._connect() as conn:
        attempt = conn.execute("SELECT * FROM screen_login_attempts WHERE nonce_digest=? AND request_id=? AND browser_digest=?",
                               (_digest(nonce), request_id, _digest(cookie))).fetchone()
    if not attempt or attempt["consumed_at"] is not None or attempt["expires"] <= _now():
        return False
    owner = verify_identity(token, client_id=client_id, nonce=nonce)
    now = _now()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=? AND profile_home=?",
                           (request_id, store.profile_home)).fetchone()
        if (not row or row["protocol"] != 3 or row["state"] not in {"pending", "opened"}
                or row["invite_expires_at"] <= now or store._owner(row["source_json"]) != ("telegram", owner)):
            return False
        changed = conn.execute("UPDATE screen_login_attempts SET consumed_at=? WHERE nonce_digest=? AND request_id=? "
                               "AND browser_digest=? AND consumed_at IS NULL AND expires>?",
                               (now, _digest(nonce), request_id, _digest(cookie), now)).rowcount
        if changed != 1:
            return False
        conn.execute("UPDATE screen_login_attempts SET consumed_at=? WHERE request_id=? AND consumed_at IS NULL", (now, request_id))
        conn.execute("UPDATE screen_handoffs SET state='authorized', authorized_at=?, web_expires_at=?, "
                     "web_session_digest=? WHERE request_id=?", (now, now + WEB_SESSION_TTL_SECONDS, _digest(cookie), request_id))
        conn.commit()
    return True
