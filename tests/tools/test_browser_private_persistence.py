"""Private browser continuity: real files/locks, mocked provider transport only."""
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

import pytest
import requests

from plugins.browser.browser_use.persistence import Lease
from plugins.browser.browser_use.provider import BrowserUseBrowserProvider


def session():
    return {'id': 'browser-1', 'status': 'active', 'cdpUrl': 'wss://example.invalid/cdp',
            'timeoutAt': (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()}


def provider():
    p = BrowserUseBrowserProvider()
    p._post_create = Mock(return_value=Mock(ok=True, status_code=200, json=lambda: session()))
    return p


def test_restart_recovers_same_browser_and_locks_profile(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    p = provider(); cfg = {'api_key': 'synthetic', 'base_url': 'https://example.invalid'}
    lease = Lease('profile-a', 'owner-a')
    try:
        first = lease.acquire(p, cfg)
        with pytest.raises(RuntimeError, match='already in use'):
            Lease('profile-a', 'owner-a')
        assert p._post_create.call_args.args[2]['profileId'] == 'profile-a'
        lease.checkpoint()
    finally:
        lease.release()  # process crash releases OS lock, not browser
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: Mock(status_code=200, json=lambda: first))
    resumed = Lease('profile-a', 'owner-a')
    try:
        assert resumed.acquire(p, cfg)['id'] == first['id']
        assert p._post_create.call_count == 1
        assert 'cdpUrl' not in resumed.path.read_text()
    finally:
        resumed.release()
    wrong_owner = Lease('profile-a', 'owner-b')
    try:
        with pytest.raises(RuntimeError, match='another pending conversation'):
            wrong_owner.acquire(p, cfg)
        assert p._post_create.call_count == 1
    finally:
        wrong_owner.release()


def test_unknown_create_never_repeats_and_other_profile_is_independent(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    p = provider(); p._post_create.side_effect = requests.Timeout()
    cfg = {'api_key': 'synthetic', 'base_url': 'https://example.invalid'}
    lease = Lease('profile-a', 'owner-a')
    try:
        with pytest.raises(requests.Timeout): lease.acquire(p, cfg)
    finally:
        lease.release()
    lease = Lease('profile-a', 'owner-a')
    try:
        with pytest.raises(RuntimeError, match='outcome unknown'): lease.acquire(p, cfg)
        assert p._post_create.call_count == 1
        other = Lease('profile-b', 'owner-b')
        try: assert not other.record
        finally: other.release()
    finally:
        lease.release()


def test_configured_provider_failure_does_not_fall_back(monkeypatch):
    from tools import browser_tool_session as sessions
    p = provider()
    monkeypatch.setattr(p, 'requires_explicit_recovery', lambda: True)
    monkeypatch.setattr(p, 'create_session', Mock(side_effect=requests.Timeout()))
    local = Mock()
    monkeypatch.setattr(sessions, '_create_local_session', local)
    with pytest.raises(requests.Timeout):
        sessions._create_cloud_session_or_fallback('task', p)
    local.assert_not_called()
