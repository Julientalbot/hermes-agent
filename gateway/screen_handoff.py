"""Durable, profile-scoped browser screen handoffs.

The Bot Desktop lease remains the only input lock.  This module only stores the
short-lived invitation and the explicit return-to-agent intent.  Tokens are
returned to the delivery adapter once and are never written to disk or logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home, secure_parent_dir

logger = logging.getLogger(__name__)

INVITE_TTL_SECONDS = 10 * 60
CONFIRM_TTL_SECONDS = 2 * 60
WEB_SESSION_TTL_SECONDS = 30 * 60
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_CALLBACKS: dict[str, Callable[[dict[str, Any]], Any]] = {}
_CALLBACK_LOCK = threading.RLock()


def _now() -> float:
    return time.time()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))


def _db_path(profile_home: Optional[str | Path] = None) -> Path:
    home = Path(profile_home) if profile_home else get_hermes_home()
    return home / "state" / "screen-handoffs.db"


def _private_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)


@dataclass(frozen=True)
class Handoff:
    request_id: str
    invite_token: str
    confirmation_code: str
    state: str
    session_id: str
    profile_home: str
    source_json: str
    reason: str
    created_at: float
    invite_expires_at: float
    confirmation_expires_at: float
    web_expires_at: Optional[float]
    viewer_id: Optional[str]
    # One-time response value; never persisted and never included by public().
    web_session_token: str = ""
    protocol: int = 2

    def public(self) -> dict[str, Any]:
        """Model/gateway-safe representation; excludes the invite and confirmation secrets."""
        return {
            "request_id": self.request_id,
            "state": self.state,
            "session_id": self.session_id,
            "profile_home": self.profile_home,
            "reason": self.reason,
            "created_at": self.created_at,
            "invite_expires_at": self.invite_expires_at,
            "confirmation_expires_at": self.confirmation_expires_at,
            "web_expires_at": self.web_expires_at,
            "viewer_id": self.viewer_id,
            "protocol": self.protocol,
        }


class ScreenHandoffStore:
    """SQLite-backed state machine for one profile's screen invitations."""

    def __init__(self, profile_home: Optional[str | Path] = None, *, db_path: Optional[Path] = None):
        self.profile_home = str(Path(profile_home) if profile_home else get_hermes_home())
        self.path = Path(db_path) if db_path else _db_path(self.profile_home)
        _private_db(self.path)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS screen_handoffs (
                    request_id TEXT PRIMARY KEY,
                    invite_digest TEXT NOT NULL UNIQUE,
                    confirmation_digest TEXT NOT NULL,
                    web_session_digest TEXT,
                    state TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    profile_home TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    invite_expires_at REAL NOT NULL,
                    confirmation_expires_at REAL NOT NULL,
                    web_expires_at REAL,
                    viewer_id TEXT,
                    opened_at REAL,
                    authorized_at REAL,
                    human_at REAL,
                    returned_at REAL,
                    resumed_at REAL,
                    revoked_at REAL,
                    last_error TEXT
                )"""
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(screen_handoffs)")}
            if "protocol" not in columns:
                conn.execute("ALTER TABLE screen_handoffs ADD COLUMN protocol INTEGER NOT NULL DEFAULT 1")
            conn.execute("""CREATE TABLE IF NOT EXISTS screen_confirmations (
                id TEXT PRIMARY KEY, request_id TEXT NOT NULL, browser_digest TEXT NOT NULL,
                code TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                decision TEXT, delivered REAL, attempts INTEGER NOT NULL DEFAULT 0,
                next_delivery REAL NOT NULL DEFAULT 0
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS screen_login_attempts (
                nonce_digest TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                browser_digest TEXT NOT NULL, created REAL NOT NULL,
                expires REAL NOT NULL, consumed_at REAL
            )""")
            # Codes are visual comparison labels, never authorization credentials.
            # Browser/invitation/session bearer values persist only as digests.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS screen_handoffs_active_idx "
                "ON screen_handoffs(profile_home, session_id, state)"
            )

    @staticmethod
    def _row(row: Optional[sqlite3.Row], *, invite_token: str = "", confirmation_code: str = "",
             web_session_token: str = "") -> Optional[Handoff]:
        if row is None:
            return None
        return Handoff(
            request_id=row["request_id"], invite_token=invite_token, confirmation_code=confirmation_code,
            state=row["state"], session_id=row["session_id"], profile_home=row["profile_home"],
            source_json=row["source_json"], reason=row["reason"], created_at=float(row["created_at"]),
            invite_expires_at=float(row["invite_expires_at"]),
            confirmation_expires_at=float(row["confirmation_expires_at"]),
            web_expires_at=float(row["web_expires_at"]) if row["web_expires_at"] is not None else None,
            viewer_id=row["viewer_id"], web_session_token=web_session_token,
            protocol=row["protocol"],
        )

    def _expire(self, conn: sqlite3.Connection, now: float) -> None:
        # Expiring access never releases a human-held screen or its origin.
        conn.execute("UPDATE screen_handoffs SET state='expired' WHERE state IN ('pending','opened') "
                     "AND human_at IS NULL AND invite_expires_at <= ?", (now,))

    @staticmethod
    def _owner(source_json: str) -> tuple[str, str]:
        source = json.loads(source_json)
        return str(source.get("platform") or ""), str(source.get("user_id") or "")

    def get(self, request_id: str) -> Optional[Handoff]:
        with self._connect() as conn:
            return self._row(conn.execute("SELECT * FROM screen_handoffs WHERE request_id=? AND profile_home=?",
                                          (request_id, self.profile_home)).fetchone())

    def create_or_get(self, *, session_id: str, source_json: str, reason: str, protocol: int = 2) -> tuple[Handoff, bool]:
        if not session_id:
            raise ValueError("screen handoff requires a session id")
        if protocol not in {2, 3} or (protocol == 3 and self._owner(source_json)[0] != "telegram"):
            raise ValueError("unsupported screen authorization protocol")
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            row = conn.execute(
                "SELECT * FROM screen_handoffs WHERE profile_home=? "
                "AND state IN ('pending','opened','authorized','human','returning','returned','queued','resuming','needs_attention') "
                "ORDER BY created_at DESC LIMIT 1",
                (self.profile_home,),
            ).fetchone()
            if row is not None:
                conn.commit()
                if row["session_id"] != str(session_id) or self._owner(row["source_json"]) != self._owner(source_json):
                    raise ValueError("another conversation already owns the screen request")
                return self._row(row), False
            request_id = secrets.token_urlsafe(18)
            invite_token = secrets.token_urlsafe(32)
            code = ""
            invite_expires = now + INVITE_TTL_SECONDS
            confirmation_expires = 0
            conn.execute(
                "INSERT INTO screen_handoffs(request_id, invite_digest, confirmation_digest, state, session_id, "
                "profile_home, source_json, reason, created_at, invite_expires_at, confirmation_expires_at, protocol) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, _digest(invite_token), _digest(code), "pending", str(session_id), self.profile_home,
                 source_json, str(reason or "").strip()[:500], now, invite_expires, confirmation_expires, protocol),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (request_id,)).fetchone()
        return self._row(row, invite_token=invite_token, confirmation_code=code), True

    def reissue(self, *, session_id: str, source_json: str, reason: str) -> Optional[Handoff]:
        """Owner's private /screen rotates access while preserving the original conversation."""
        now, invite_token = _now(), secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            rows = conn.execute("SELECT * FROM screen_handoffs WHERE profile_home=? "
                                "AND state NOT IN ('revoked','resumed') ORDER BY created_at DESC", (self.profile_home,)).fetchall()
            row = next((r for r in rows if self._owner(r["source_json"]) == self._owner(source_json)), None)
            if any(r["state"] != "expired" and (row is None or r["request_id"] != row["request_id"]) for r in rows):
                conn.commit()
                return None
            if row is None or row["state"] in {"returning", "returned", "queued", "resuming", "needs_attention"}:
                conn.commit()
                return None
            conn.execute("UPDATE screen_confirmations SET decision='superseded' WHERE request_id=? AND decision IS NULL", (row["request_id"],))
            conn.execute("UPDATE screen_login_attempts SET consumed_at=? WHERE request_id=? AND consumed_at IS NULL",
                         (now, row["request_id"]))
            conn.execute("UPDATE screen_handoffs SET invite_digest=?, web_session_digest=NULL, "
                         "state='pending', invite_expires_at=?, confirmation_expires_at=0, web_expires_at=NULL, "
                         "opened_at=NULL, authorized_at=NULL, last_error=NULL WHERE request_id=?",
                         (_digest(invite_token), now + INVITE_TTL_SECONDS, row["request_id"]))
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row, invite_token=invite_token)

    def by_token(self, token: str, *, mark_opened: bool = False) -> Optional[Handoff]:
        if not token:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE") if mark_opened else None
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE invite_digest=?", (_digest(token),)).fetchone()
            if row is None or row["protocol"] not in {2, 3} or row["profile_home"] != self.profile_home or row["invite_expires_at"] <= now or row["state"] in {"expired", "revoked", "resumed"}:
                if mark_opened:
                    conn.commit()
                return None
            if mark_opened and row["state"] == "pending":
                conn.execute("UPDATE screen_handoffs SET state='opened', opened_at=? WHERE request_id=?", (now, row["request_id"]))
                row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            if mark_opened:
                conn.commit()
            return self._row(row, invite_token=token)

    def challenge(self, invite: str, browser_token: str = "") -> Optional[dict]:
        """Interactive browser request. GET previews never call this method."""
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM screen_handoffs WHERE invite_digest=? AND profile_home=?",
                               (_digest(invite), self.profile_home)).fetchone()
            if row is None or row["protocol"] != 2 or row["invite_expires_at"] <= now or row["state"] not in {"pending", "opened"}:
                conn.commit()
                return None
            if browser_token:
                existing = conn.execute("SELECT * FROM screen_confirmations WHERE request_id=? AND browser_digest=? "
                                        "AND expires>? AND decision IS NULL", (row["request_id"], _digest(browser_token), now)).fetchone()
                if existing:
                    conn.commit()
                    return {"id": existing["id"], "code": existing["code"], "expires": existing["expires"], "cookie": browser_token}
            count = conn.execute("SELECT COUNT(*) FROM screen_confirmations WHERE request_id=? AND created>?",
                                 (row["request_id"], now - INVITE_TTL_SECONDS)).fetchone()[0]
            if count >= 3:
                conn.commit()
                return None
            browser_token = secrets.token_urlsafe(32)
            cid, code = secrets.token_urlsafe(18), _short_code()
            expires = min(now + CONFIRM_TTL_SECONDS, row["invite_expires_at"])
            conn.execute("INSERT INTO screen_confirmations(id,request_id,browser_digest,code,created,expires) VALUES(?,?,?,?,?,?)",
                         (cid, row["request_id"], _digest(browser_token), code, now, expires))
            conn.execute("UPDATE screen_handoffs SET state='opened', opened_at=?, confirmation_expires_at=? WHERE request_id=?",
                         (now, expires, row["request_id"]))
            conn.commit()
        return {"id": cid, "code": code, "expires": expires, "cookie": browser_token}

    def pending_confirmations(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT c.*, h.source_json FROM screen_confirmations c JOIN screen_handoffs h ON h.request_id=c.request_id "
                "WHERE h.profile_home=? AND h.state IN ('pending','opened') AND c.decision IS NULL "
                "AND c.delivered IS NULL AND c.expires>? AND c.next_delivery<=? AND c.attempts<3",
                (self.profile_home, _now(), _now()))]

    def confirmation_delivered(self, cid: str, success: bool) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE screen_confirmations SET attempts=attempts+1, next_delivery=?, delivered=? WHERE id=?",
                         (_now()+15, _now() if success else None, cid))

    def decide(self, cid: str, *, platform: str, user_id: str, allow: bool) -> bool:
        """Called only from the authenticated messaging adapter, never from the web page."""
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            c = conn.execute("SELECT * FROM screen_confirmations WHERE id=?", (cid,)).fetchone()
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=? AND profile_home=?",
                               (c["request_id"], self.profile_home)).fetchone() if c else None
            if not row or self._owner(row["source_json"]) != (str(platform), str(user_id)):
                conn.commit()
                return False
            decision = "allow" if allow else "deny"
            if c["decision"] is not None:
                conn.commit()
                return c["decision"] == decision
            if c["expires"] <= now or row["invite_expires_at"] <= now or row["state"] not in {"pending", "opened"}:
                conn.commit()
                return False
            conn.execute("UPDATE screen_confirmations SET decision=? WHERE id=?", (decision, cid))
            if allow:
                conn.execute("UPDATE screen_handoffs SET state='authorized', authorized_at=?, web_expires_at=?, "
                             "web_session_digest=? WHERE request_id=?", (now, now+WEB_SESSION_TTL_SECONDS, c["browser_digest"], row["request_id"]))
                conn.execute("UPDATE screen_confirmations SET decision='superseded' WHERE request_id=? AND id!=? AND decision IS NULL",
                             (row["request_id"], cid))
            conn.commit()
        return True

    def access_status(self, request_id: str, cookie: str) -> dict:
        with self._connect() as conn:
            c = conn.execute("SELECT * FROM screen_confirmations WHERE request_id=? AND browser_digest=? ORDER BY created DESC LIMIT 1",
                             (request_id, _digest(cookie))).fetchone()
        session = self.any_web_session(cookie)
        if session and session.request_id == request_id:
            return {"state": session.state}
        if c:
            return {"state": c["decision"] if c["decision"] and c["decision"] != "allow" else ("expired" if c["expires"] <= _now() else "waiting"),
                    "code": c["code"]}
        return {"state": "unauthorized"}

    def stream_valid(self, request_id: str, viewer_id: str) -> bool:
        row = self.get(request_id)
        return bool(row and row.state == "human" and row.viewer_id == viewer_id and row.web_expires_at and row.web_expires_at > _now())

    def web_session(self, token: str) -> Optional[Handoff]:
        if not token:
            return None
        now = _now()
        with self._connect() as conn:
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
        if row is None or row["protocol"] not in {2, 3} or row["profile_home"] != self.profile_home or row["state"] not in {"authorized", "human"} or row["web_expires_at"] is None or float(row["web_expires_at"]) <= now:
            return None
        return self._row(row)

    def any_web_session(self, token: str) -> Optional[Handoff]:
        """Lookup a still-valid cookie, including a returned session for idempotent UI actions."""
        if not token:
            return None
        now = _now()
        with self._connect() as conn:
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
        if row is None or row["protocol"] not in {2, 3} or row["profile_home"] != self.profile_home or row["state"] in {"expired", "revoked"} or row["web_expires_at"] is None or float(row["web_expires_at"]) <= now:
            return None
        return self._row(row)

    def take_over(self, token: str, viewer_id: str) -> Optional[Handoff]:
        if not token or not viewer_id:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
            if row is None or row["state"] not in {"authorized", "human"} or not row["web_expires_at"] or row["web_expires_at"] <= now:
                conn.commit()
                return None
            conn.execute("UPDATE screen_handoffs SET state='human', human_at=?, viewer_id=? WHERE request_id=?", (now, viewer_id, row["request_id"]))
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row)

    def return_to_agent(self, token: str, viewer_id: str) -> Optional[Handoff]:
        if not token or not viewer_id:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
            if row is None or row["state"] not in {"human", "returning"} or not row["web_expires_at"] or row["web_expires_at"] <= now:
                conn.commit()
                return None
            if row["viewer_id"] != viewer_id:
                conn.commit()
                return None
            conn.execute("UPDATE screen_handoffs SET state='returning', returned_at=? WHERE request_id=?", (now, row["request_id"]))
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row)

    def complete_return(self, request_id: str) -> None:
        """Only after the existing human lease was released (or observed released)."""
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state='returned' WHERE request_id=? AND state='returning'", (request_id,))

    def returning(self) -> list[Handoff]:
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute("SELECT * FROM screen_handoffs WHERE profile_home=? AND state='returning'", (self.profile_home,))]

    def revoke(self, request_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE screen_handoffs SET state=CASE WHEN human_at IS NOT NULL THEN 'human' ELSE 'revoked' END, "
                "web_session_digest=NULL, web_expires_at=NULL, revoked_at=? WHERE request_id=? "
                "AND state NOT IN ('resumed','revoked','expired')", (_now(), request_id),
            )
            return cur.rowcount == 1

    def refuse(self, token: str) -> bool:
        """Refuse a still-pending invitation using its locator, without authorizing it."""
        handoff = self.by_token(token)
        if handoff is None or handoff.state not in {"pending", "opened"}:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE screen_handoffs SET state='revoked', revoked_at=? WHERE request_id=? AND state IN ('pending','opened')",
                (_now(), handoff.request_id),
            )
            return cur.rowcount == 1

    def claim_returned(self, limit: int = 10) -> list[Handoff]:
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute(
                "SELECT * FROM screen_handoffs WHERE profile_home=? AND state IN ('returned','queued') ORDER BY returned_at LIMIT ?",
                (self.profile_home, int(limit)))]

    def queue_resume(self, request_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state='queued' WHERE request_id=? AND state='returned'", (request_id,))

    def begin_resume(self, request_id: str, *, session_id: str, source_json: str) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=? AND profile_home=?",
                               (request_id, self.profile_home)).fetchone()
            if not row or row["state"] != "queued" or row["session_id"] != session_id or json.loads(row["source_json"]) != json.loads(source_json):
                conn.commit()
                return False
            conn.execute("UPDATE screen_handoffs SET state='resuming' WHERE request_id=?", (request_id,))
            conn.commit()
            return True

    def recover_resumes(self) -> None:
        # Dispatch may already have produced side effects. Never repeat it after a crash.
        # Native gateway interrupted-turn recovery remains the owner of that turn.
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state='needs_attention', last_error='resume_outcome_unknown' "
                         "WHERE profile_home=? AND state='resuming'", (self.profile_home,))

    def finish_resume(self, request_id: str, *, error: str = "") -> None:
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state=?, last_error=?, resumed_at=? "
                         "WHERE request_id=? AND state='resuming'",
                         ("needs_attention" if error else "resumed", error[:500], _now(), request_id))

    def abandon_resume(self, request_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state='revoked', last_error=? WHERE request_id=? AND state IN ('returned','queued','resuming')", (str(error)[:500], request_id))

    def source(self, handoff: Handoff) -> Any:
        from gateway.session import SessionSource
        return SessionSource.from_dict(json.loads(handoff.source_json))


def register_screen_handoff_notify(session_key: str, callback: Callable[[dict[str, Any]], Any]) -> None:
    if session_key:
        with _CALLBACK_LOCK:
            _CALLBACKS[session_key] = callback


def unregister_screen_handoff_notify(session_key: str) -> None:
    with _CALLBACK_LOCK:
        _CALLBACKS.pop(session_key, None)


def notify_screen_handoff(session_key: str, record: dict[str, Any]) -> bool:
    with _CALLBACK_LOCK:
        callback = _CALLBACKS.get(session_key)
    if callback is None:
        return False
    try:
        callback(record)
        return True
    except Exception:
        logger.warning("screen handoff private delivery failed")
        return False


def has_screen_handoff_notify(session_key: str) -> bool:
    with _CALLBACK_LOCK:
        return bool(session_key and session_key in _CALLBACKS)


def _reset_for_tests() -> None:
    with _CALLBACK_LOCK:
        _CALLBACKS.clear()
