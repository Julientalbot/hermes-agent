import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.screen_handoff import ScreenHandoffStore
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.parametrize("platform", ["telegram", "discord"])
def test_persistent_confirmation_checks_actor_and_private_channel(tmp_path, platform):
    store = ScreenHandoffStore(tmp_path)
    row, _ = store.create_or_get(session_id="original", source_json=json.dumps({
        "platform": platform, "user_id": "42", "chat_id": "group", "chat_type": "group"}), reason="login")
    challenge = store.challenge(row.invite_token)
    # A new adapter instance has no per-turn callback or pending confirmation map.
    cls = TelegramAdapter if platform == "telegram" else DiscordAdapter
    adapter = object.__new__(cls)
    adapter._screen_profile_home = str(tmp_path.resolve())

    async def tap(actor, private):
        if platform == "telegram":
            adapter._callback_authorized = AsyncMock(return_value=True)
            query = SimpleNamespace(from_user=SimpleNamespace(id=actor),
                message=SimpleNamespace(chat_id=actor if private else -10),
                answer=AsyncMock(), edit_message_reply_markup=AsyncMock())
            await adapter._handle_screen_handoff_callback(query, "sh:allow:"+challenge["id"], {})
        else:
            interaction = SimpleNamespace(data={"custom_id": "screen:allow:"+challenge["id"]},
                user=SimpleNamespace(id=actor), guild=None if private else object(),
                response=SimpleNamespace(send_message=AsyncMock()), message=SimpleNamespace(edit=AsyncMock()))
            await adapter._handle_screen_handoff_interaction(interaction)

    asyncio.run(tap(43, True))
    assert store.web_session(challenge["cookie"]) is None
    asyncio.run(tap(42, False))
    assert store.web_session(challenge["cookie"]) is None
    asyncio.run(tap(42, True))
    assert store.web_session(challenge["cookie"])
