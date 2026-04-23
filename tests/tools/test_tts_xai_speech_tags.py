"""Tests for xAI TTS speech-tag passthrough."""

from unittest.mock import Mock

from tools.tts_tool import (
    TTS_SCHEMA,
    XAI_INLINE_SPEECH_TAGS,
    XAI_WRAPPING_SPEECH_TAGS,
    _generate_xai_tts,
)


class TestXAISpeechTags:
    def test_xai_tts_preserves_speech_tags_in_payload(self, tmp_path, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            response = Mock()
            response.content = b"mp3-bytes"
            response.raise_for_status = Mock()
            return response

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr("requests.post", fake_post)

        text = "Attendez [pause] <whisper>c'est important</whisper>. [laugh]"
        output_path = tmp_path / "speech.mp3"

        result = _generate_xai_tts(
            text,
            str(output_path),
            {"xai": {"voice_id": "ara", "language": "fr"}},
        )

        assert result == str(output_path)
        assert output_path.read_bytes() == b"mp3-bytes"
        assert captured["url"] == "https://api.x.ai/v1/tts"
        assert captured["json"]["text"] == text
        assert captured["json"]["voice_id"] == "ara"
        assert captured["json"]["language"] == "fr"

    def test_tool_schema_documents_xai_speech_tags(self):
        description = TTS_SCHEMA["parameters"]["properties"]["text"]["description"]

        for tag in ("[pause]", "[laugh]", "[sigh]"):
            assert tag in description
        for tag in ("whisper", "slow", "emphasis"):
            assert f"<{tag}>...</{tag}>" in description

        assert "xAI" in description
        assert set(("[pause]", "[laugh]", "[sigh]")).issubset(XAI_INLINE_SPEECH_TAGS)
        assert set(("whisper", "slow", "emphasis")).issubset(XAI_WRAPPING_SPEECH_TAGS)
