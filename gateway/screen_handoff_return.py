"""One durable return operation for the browser and authenticated owner messages."""
from __future__ import annotations

import re
import threading
import unicodedata

from gateway.screen_handoff import ScreenHandoffStore, _now

_callbacks = {}
_lock = threading.RLock()
RETURNED_STATES = {"returned", "queued", "resuming", "resumed", "needs_attention"}


def explicit_return_request(text: str) -> bool:
    """Only an unquoted, affirmative instruction in the current inbound message.

    The model selects the tool, but cannot turn a bare acknowledgement, quotation,
    negation or browser content into permission to end a human lease.
    """
    text = str(text).strip()
    if any(c in text for c in '\n"«»“”`') or text.startswith("'") or text.endswith("'"):
        return False
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[’']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return bool(re.fullmatch(
        r"(?:(?:cest bon|jai fini|jai termine|ca y est|merci)[,!. ]+)?"
        r"(?:tu peux (?:reprendre|continuer)(?: (?:la main|le controle|ta tache))?"
        r"|reprends(?: la main| le controle| ta tache)?"
        r"|(?:you (?:can|may) (?:resume|continue|take over)))[.! ]*", text))


def owner_handoff(store, source):
    if (str(getattr(source.platform, "value", source.platform)) != "telegram"
            or source.chat_type not in {"dm", "private"} or not source.user_id
            or str(source.chat_id) != str(source.user_id) or getattr(source, "is_bot", False)):
        return None
    with store._connect() as conn:
        rows = conn.execute("SELECT * FROM screen_handoffs WHERE profile_home=? AND human_at IS NOT NULL "
                            "AND state NOT IN ('revoked','expired','resumed')", (store.profile_home,)).fetchall()
    rows = [r for r in rows if store._owner(r["source_json"]) == ("telegram", str(source.user_id))]
    return store._row(rows[0]) if len(rows) == 1 else None


def return_control(store, handoff, viewer_id: str) -> dict:
    from tools.bot_desktop import lease
    if handoff.state in RETURNED_STATES:
        return {"state": handoff.state}
    held = lease.get(profile_key=store.profile_home)
    if held.holder == lease.HUMAN and held.viewer_id != viewer_id:
        raise ValueError("Un autre navigateur détient le contrôle.")
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=? AND profile_home=?",
                           (handoff.request_id, store.profile_home)).fetchone()
        if row and row["state"] in RETURNED_STATES:
            return {"state": row["state"]}
        if (not row or not row["human_at"] or row["viewer_id"] != viewer_id
                or row["state"] not in {"human", "returning", "pending", "opened", "authorized"}):
            raise ValueError("Restitution indisponible.")
        conn.execute("UPDATE screen_handoffs SET state='returning', returned_at=COALESCE(returned_at,?) WHERE request_id=?",
                     (_now(), handoff.request_id))
        conn.commit()
    released = lease.release(viewer_id, profile_key=store.profile_home)
    if released.holder != lease.AGENT:
        raise ValueError("Restitution en attente : un autre navigateur détient le contrôle.")
    store.complete_return(handoff.request_id)
    return {"state": "returned"}


def register_return(session_key, callback):
    with _lock:
        _callbacks[session_key] = callback


def unregister_return(session_key):
    with _lock:
        _callbacks.pop(session_key, None)


def return_from_current_turn(session_key):
    with _lock:
        callback = _callbacks.get(session_key)
    if callback is None:
        return {"success": False, "error": "No authenticated owner message in this turn."}
    return callback()


def return_from_owner_message(source, message, *, internal, inbound_id):
    if internal or not str(inbound_id or "").isdigit() or not explicit_return_request(message):
        return {"success": False, "error": "An explicit current owner request to return control is required; an acknowledgement is not enough."}
    store = ScreenHandoffStore()
    row = owner_handoff(store, source)
    if not row:
        return {"success": False, "error": "No matching human intervention for this private owner."}
    try:
        result = return_control(store, row, row.viewer_id)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, **result, "next_step": "End this turn now. The durable return resumes the original task once; do not navigate or continue it in this turn."}
