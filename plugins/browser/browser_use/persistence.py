"""Opt-in private-chat browser lease. No credentials or CDP URLs are persisted."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime

from hermes_constants import get_hermes_home


def settings():
    from hermes_cli.config import load_config
    return (load_config().get('browser') or {}).get('browser_use') or {}


def owner():
    from gateway.session_context import get_session_env as get
    chat, user = get('HERMES_SESSION_CHAT_ID'), get('HERMES_SESSION_USER_ID')
    if (get('HERMES_SESSION_PLATFORM') != 'telegram'
            or get('HERMES_SESSION_CHAT_TYPE') not in ('dm', 'private')
            or not chat or chat != user or get('HERMES_CRON_SESSION')):
        raise RuntimeError('Persistent browser requires an authenticated private Telegram owner')
    value = [chat, user, get('HERMES_SESSION_THREAD_ID'), get('HERMES_SESSION_KEY')]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


class Lease:
    """The OS lease lives as long as the browser is tracked by Hermes' idle reaper."""
    def __init__(self, profile_id, owner_id):
        # The production target is Linux; fail explicitly rather than omit exclusion elsewhere.
        import fcntl
        root = get_hermes_home() / 'browser' / 'cloud'
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = hashlib.sha256(profile_id.encode()).hexdigest()
        self.path = root / (name + '.json')
        self.fd = os.open(root / (name + '.lock'), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError('Browser profile is already in use') from None
        self.profile_id, self.owner_id = profile_id, owner_id
        self.record = {}
        try:
            if self.path.exists():
                self.record = json.loads(self.path.read_text())
        except Exception:
            self.release()
            raise

    def write(self):
        tmp = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.record, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def acquire(self, provider, config):
        import requests
        headers = provider._headers(config)
        base = config['base_url']
        old = self.record
        if old.get('status') == 'creating':
            raise RuntimeError('Browser creation outcome unknown; reconcile the recorded operation before retrying')
        if old:
            response = requests.get(base + '/browsers/' + old['browser_id'], headers=headers, timeout=15)
            if response.status_code == 404:
                current = {'status': 'stopped'}
            else:
                response.raise_for_status()
                current = response.json()
            if current.get('status') == 'active':
                if old.get('owner') != self.owner_id:
                    raise RuntimeError('Browser profile belongs to another pending conversation')
                expiry = datetime.fromisoformat(current['timeoutAt'].replace('Z', '+00:00')).timestamp()
                if time.time() < expiry and time.time() - old['last_used_at'] < 600:
                    return current
                stopped = provider._release(config, old['browser_id'], 15)
                stopped.raise_for_status()
                check = requests.get(base + '/browsers/' + old['browser_id'], headers=headers, timeout=15)
                check.raise_for_status()
                if check.json().get('status') != 'stopped':
                    raise RuntimeError('Previous browser is not confirmed stopped')
            elif current.get('status') != 'stopped':
                raise RuntimeError('Browser status unknown; no replacement was created')
        self.record = {'status': 'creating', 'profile_id': self.profile_id, 'owner': self.owner_id,
                       'operation_id': uuid.uuid4().hex, 'last_used_at': time.time()}
        self.write()
        response = provider._post_create(base + '/browsers', headers,
            {'profileId': self.profile_id, 'timeout': 30, 'solveCaptchas': True,
             'enableRecording': False, 'metadata': {'hermes_operation': self.record['operation_id']}})
        if 400 <= response.status_code < 500 and response.status_code not in (408, 409, 429):
            self.path.unlink(missing_ok=True)
        provider._check_created(response)
        current = response.json()
        if not current.get('id') or not (current.get('cdpUrl') or current.get('connectUrl')):
            raise RuntimeError('Browser creation response incomplete; reconcile before retrying')
        self.record.update(status='active', browser_id=current['id'], timeout_at=current['timeoutAt'])
        self.write()
        return current

    def checkpoint(self):
        self.record['last_used_at'] = time.time()
        self.write()

    def release(self, *, stopped=False):
        if stopped:
            self.path.unlink(missing_ok=True)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
