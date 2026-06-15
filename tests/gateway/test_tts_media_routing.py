"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make speech",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


@pytest.mark.asyncio
async def test_base_adapter_routes_telegram_flac_media_tag_to_document_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata=None,
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_non_voice_telegram_ogg_media_tag_to_document_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata=None,
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_voice_tagged_telegram_ogg_media_tag_to_voice_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(
        return_value=f"[[audio_as_voice]]\nMEDIA:{media_file}"
    )
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata=None,
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_attaches_existing_docx_before_sending_file_claim(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.docx")
    adapter._message_handler = AsyncMock(return_value=f"Here is the file:\nMEDIA:{media_file}")

    calls = []

    async def _send_document(**kwargs):
        calls.append(("document", kwargs))
        return SendResult(success=True, message_id="doc")

    async def _send(**kwargs):
        calls.append(("text", kwargs))
        return SendResult(success=True, message_id="text")

    adapter.send_document = AsyncMock(side_effect=_send_document)
    adapter.send = AsyncMock(side_effect=_send)

    await adapter._process_message_background(event, build_session_key(event.source))

    assert [kind for kind, _ in calls] == ["document", "text"]
    assert calls[0][1]["file_path"] == str(media_file)
    assert "MEDIA:" not in calls[1][1]["content"]
    assert "Here is the file" in calls[1][1]["content"]


@pytest.mark.asyncio
async def test_base_adapter_missing_media_docx_sends_failure_notice_without_internal_path(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    root = tmp_path / "media-cache"
    root.mkdir()
    missing_file = root / "missing.docx"
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    adapter._message_handler = AsyncMock(return_value=f"Here is the file:\nMEDIA:{missing_file}")
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="notice"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_not_awaited()
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.kwargs["content"]
    assert "couldn't attach" in notice
    assert "missing.docx" in notice
    assert "MEDIA:" not in notice
    assert str(missing_file) not in notice
    assert "Here is the file" not in notice


@pytest.mark.asyncio
async def test_base_adapter_failed_document_upload_sends_failure_notice_without_claim(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.docx")
    adapter._message_handler = AsyncMock(return_value=f"Here is the file:\nMEDIA:{media_file}")
    adapter.send_document = AsyncMock(
        return_value=SendResult(
            success=False,
            error=f"File not found: {media_file}",
        )
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="notice"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once()
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.kwargs["content"]
    assert "couldn't attach" in notice
    assert "report.docx" in notice
    assert "MEDIA:" not in notice
    assert str(media_file) not in notice
    assert "Here is the file" not in notice


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_flac_media_tag_to_document_sender(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_non_voice_telegram_ogg_media_tag_to_document_sender(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_mp3_media_tag_to_voice_sender(tmp_path, monkeypatch):
    """MP3 audio on Telegram must go through send_voice (which routes to
    sendAudio internally); Telegram accepts MP3 for the audio player."""
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.mp3")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_blocks_media_path_outside_allowed_roots(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    allowed_root = tmp_path / "media-cache"
    allowed_root.mkdir()
    secret = tmp_path / "outside.pdf"
    secret.write_bytes(b"%PDF secret")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (allowed_root,),
    )
    # This test exercises the strict-allowlist path; force strict mode on
    # and disable recency trust so the freshly-written tmp_path file is not
    # auto-accepted by the trust window. (Recency trust is covered separately
    # in test_platform_base.py. The public default flipped to non-strict in
    # 2026-05; this test pins strict on explicitly.)
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{secret}",
        event,
        adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_failed_document_upload_sends_failure_notice(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.docx")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(
            return_value=SendResult(
                success=False,
                error=f"File not found: {media_file}",
            )
        ),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.kwargs["content"]
    assert "couldn't attach" in notice
    assert "report.docx" in notice
    assert "MEDIA:" not in notice
    assert str(media_file) not in notice
