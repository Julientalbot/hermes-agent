"""Authorization ending closes an existing stream, but never returns human control."""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from gateway.screen_handoff import ScreenHandoffStore
from hermes_cli.web_routers import display
from tools.bot_desktop import lease


@pytest.mark.parametrize("ending", ["expiry", "revoke", "close"])
def test_stream_access_ends_without_releasing(ending, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("gateway.screen_handoff._now", lambda: now[0])

    async def run(home):
        store = ScreenHandoffStore(home)
        row, _ = store.create_or_get(session_id="original", source_json=json.dumps({
            "platform": "telegram", "user_id": "42"}), reason="login")
        challenge = store.challenge(row.invite_token)
        store.decide(challenge["id"], platform="telegram", user_id="42", allow=True)
        store.take_over(challenge["cookie"], "viewer")
        lease.acquire("viewer", profile_key=home)
        connected, finish = asyncio.Event(), asyncio.Event()

        async def framebuffer(reader, writer):
            connected.set()
            await reader.read()
            writer.close()
            await writer.wait_closed()

        class Socket:
            code = None
            async def receive(self):
                await finish.wait()
                return {"type": "websocket.disconnect", "code": 1000}
            async def send_bytes(self, data):
                pass
            async def close(self, code=1000, reason=""):
                if self.code is None:
                    self.code = code

        server = await asyncio.start_unix_server(framebuffer, path=str(Path(home)/"bot-desktop/rfb.sock"))
        ws = Socket()
        bridge = asyncio.create_task(display._bridge(ws, {
            "provider": "bot-desktop-handoff", "hermes_home": home,
            "viewer_id": "viewer", "handoff_id": row.request_id, "retain_on_disconnect": True}))
        try:
            await asyncio.wait_for(connected.wait(), 2)
            if ending == "expiry":
                now[0] += 1801
            elif ending == "revoke":
                store.revoke(row.request_id)
            else:
                finish.set()
            await asyncio.wait_for(bridge, 2)
            if ending != "close":
                assert ws.code == 4401
            assert lease.human_holds(profile_key=home)
        finally:
            finish.set()
            bridge.cancel()
            await asyncio.gather(bridge, return_exceptions=True)
            server.close()
            await server.wait_closed()

    # macOS AF_UNIX has a short path limit; pytest's nested tmp_path exceeds it.
    with tempfile.TemporaryDirectory(prefix="screen-", dir="/tmp") as home:
        asyncio.run(run(home))
