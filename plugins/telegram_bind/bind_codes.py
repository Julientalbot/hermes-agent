"""Pre-issued Telegram bind codes stored alongside gateway pairing data.

Fleet operators issue codes via ``hermes-fleet bind-code``; this module owns
the JSON file and exposes a small CLI for in-container use.
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import shlex
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from gateway.pairing import ALPHABET, PAIRING_DIR, _secure_write

logger = logging.getLogger(__name__)

BIND_CODES_FILENAME = "telegram-bind-codes.json"
BIND_CODE_LENGTH = 10
BIND_CODE_TTL_HOURS = 72

DEFAULT_WELCOME_MESSAGE = (
    "Bienvenue ! Votre agent Hermes est maintenant connecté. "
    "Vous pouvez commencer à lui parler."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(value: str) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class BindCodesStore:
    """JSON-backed store for pre-issued Telegram bind codes."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or (PAIRING_DIR / BIND_CODES_FILENAME)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _secure_write(self._path, json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def _normalize_code(code: str) -> str:
        return str(code or "").strip().upper()

    def _generate_unique_code(self, data: dict[str, dict[str, Any]]) -> str:
        for _ in range(32):
            code = "".join(secrets.choice(ALPHABET) for _ in range(BIND_CODE_LENGTH))
            if code not in data:
                return code
        raise RuntimeError("Failed to generate a unique bind code")

    @staticmethod
    def entry_status(entry: dict[str, Any], *, now: Optional[datetime] = None) -> str:
        now = now or _utc_now()
        if entry.get("consumed_utc"):
            return "consumed"
        expires = _parse_iso_utc(str(entry.get("expires_utc", "")))
        if expires and now >= expires:
            return "expired"
        return "pending"

    def issue(self, customer_ref: str) -> dict[str, Any]:
        customer_ref = str(customer_ref or "").strip()
        if not customer_ref:
            raise ValueError("customer_ref is required")

        now = _utc_now()
        expires = now + timedelta(hours=BIND_CODE_TTL_HOURS)
        with self._lock:
            data = self._load()
            code = self._generate_unique_code(data)
            entry = {
                "code": code,
                "customer_ref": customer_ref,
                "issued_utc": _iso_utc(now),
                "expires_utc": _iso_utc(expires),
                "consumed_utc": None,
                "telegram_user_id": None,
                "telegram_chat_id": None,
                "env_synced_utc": None,
            }
            data[code] = entry
            self._save(data)
            return dict(entry)

    def list_entries(self) -> list[dict[str, Any]]:
        now = _utc_now()
        with self._lock:
            data = self._load()
        rows = []
        for code in sorted(data):
            entry = dict(data[code])
            entry.setdefault("code", code)
            entry["status"] = self.entry_status(entry, now=now)
            rows.append(entry)
        return rows

    def revoke(self, code: str) -> bool:
        code = self._normalize_code(code)
        with self._lock:
            data = self._load()
            if code not in data:
                return False
            del data[code]
            self._save(data)
            return True

    def get(self, code: str) -> Optional[dict[str, Any]]:
        code = self._normalize_code(code)
        with self._lock:
            data = self._load()
            entry = data.get(code)
            return dict(entry) if entry else None

    def consume(
        self,
        code: str,
        *,
        telegram_user_id: str,
        telegram_chat_id: str,
    ) -> Optional[dict[str, Any]]:
        code = self._normalize_code(code)
        user_id = str(telegram_user_id or "").strip()
        chat_id = str(telegram_chat_id or "").strip()
        if not user_id or not chat_id:
            raise ValueError("telegram_user_id and telegram_chat_id are required")

        now = _utc_now()
        with self._lock:
            data = self._load()
            entry = data.get(code)
            if not entry:
                return None
            entry = dict(entry)
            entry["consumed_utc"] = _iso_utc(now)
            entry["telegram_user_id"] = user_id
            entry["telegram_chat_id"] = chat_id
            data[code] = entry
            self._save(data)
            return entry

    def get_first_unsynced_consumed(self) -> Optional[dict[str, Any]]:
        with self._lock:
            data = self._load()
        for code in sorted(data):
            entry = data[code]
            if not entry.get("consumed_utc"):
                continue
            if entry.get("env_synced_utc"):
                continue
            if not entry.get("telegram_user_id") or not entry.get("telegram_chat_id"):
                continue
            result = dict(entry)
            result.setdefault("code", code)
            return result
        return None

    def mark_env_synced(self, code: str) -> bool:
        code = self._normalize_code(code)
        now = _utc_now()
        with self._lock:
            data = self._load()
            entry = data.get(code)
            if not entry or not entry.get("consumed_utc"):
                return False
            entry = dict(entry)
            entry["env_synced_utc"] = _iso_utc(now)
            data[code] = entry
            self._save(data)
            return True


def _print_kv_pairs(pairs: dict[str, str]) -> None:
    for key, value in pairs.items():
        print(f"{key}={shlex.quote(value)}")


def _cmd_issue(store: BindCodesStore, customer_ref: str) -> int:
    entry = store.issue(customer_ref)
    _print_kv_pairs(
        {
            "code": entry["code"],
            "customer_ref": entry["customer_ref"],
            "issued_utc": entry["issued_utc"],
            "expires_utc": entry["expires_utc"],
        }
    )
    return 0


def _cmd_list(store: BindCodesStore) -> int:
    for entry in store.list_entries():
        print(
            "{code}\t{status}\t{customer_ref}\t{issued_utc}\t{expires_utc}\t"
            "{consumed_utc}\t{telegram_user_id}\t{telegram_chat_id}\t{env_synced_utc}".format(
                code=entry.get("code", ""),
                status=entry.get("status", ""),
                customer_ref=entry.get("customer_ref", ""),
                issued_utc=entry.get("issued_utc", ""),
                expires_utc=entry.get("expires_utc", ""),
                consumed_utc=entry.get("consumed_utc") or "",
                telegram_user_id=entry.get("telegram_user_id") or "",
                telegram_chat_id=entry.get("telegram_chat_id") or "",
                env_synced_utc=entry.get("env_synced_utc") or "",
            )
        )
    return 0


def _cmd_revoke(store: BindCodesStore, code: str) -> int:
    if store.revoke(code):
        print(f"revoked={shlex.quote(store._normalize_code(code))}")
        return 0
    print(f"Bind code not found: {code}", file=sys.stderr)
    return 1


def _cmd_get_unsynced(store: BindCodesStore) -> int:
    entry = store.get_first_unsynced_consumed()
    if not entry:
        print("No consumed bind code pending env sync.", file=sys.stderr)
        return 2
    _print_kv_pairs(
        {
            "code": str(entry.get("code", "")),
            "telegram_user_id": str(entry.get("telegram_user_id", "")),
            "telegram_chat_id": str(entry.get("telegram_chat_id", "")),
            "customer_ref": str(entry.get("customer_ref", "")),
        }
    )
    return 0


def _cmd_mark_synced(store: BindCodesStore, code: str) -> int:
    if store.mark_env_synced(code):
        print(f"synced={shlex.quote(store._normalize_code(code))}")
        return 0
    print(f"Unable to mark bind code synced: {code}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telegram bind-code store CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    issue = sub.add_parser("issue", help="Issue a new bind code")
    issue.add_argument("--customer", required=True, dest="customer_ref")

    sub.add_parser("list", help="List bind codes")

    revoke = sub.add_parser("revoke", help="Revoke a bind code")
    revoke.add_argument("--code", required=True)

    sub.add_parser("get-unsynced", help="Print first consumed code pending env sync")

    mark = sub.add_parser("mark-synced", help="Mark a consumed code as env-synced")
    mark.add_argument("--code", required=True)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = BindCodesStore()

    if args.command == "issue":
        return _cmd_issue(store, args.customer_ref)
    if args.command == "list":
        return _cmd_list(store)
    if args.command == "revoke":
        return _cmd_revoke(store, args.code)
    if args.command == "get-unsynced":
        return _cmd_get_unsynced(store)
    if args.command == "mark-synced":
        return _cmd_mark_synced(store, args.code)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())