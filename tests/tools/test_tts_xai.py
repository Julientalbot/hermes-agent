"""Tests for the xAI TTS provider in tools/tts_tool.py."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)


class TestGenerateXaiTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_xai_tts

        output_path = str(tmp_path / "test.mp3")
        with pytest.raises(ValueError, match="XAI_API_KEY"):
            _generate_xai_tts("Hello", output_path, {})

    def test_successful_mp3_generation(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        monkeypatch.setenv("XAI_API_KEY", "test-key")

        fake_audio = b"fake-mp3-audio-data"
        mock_response = MagicMock()
        mock_response.content = fake_audio
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / "test.mp3")
            result = _generate_xai_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.mp3").read_bytes() == fake_audio
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["text"] == "Hello world"
        assert payload["voice_id"] == "sal"  # default voice
        assert payload["codec"] == "mp3"

    def test_wav_codec_selection(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.content = b"fake-wav"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            output_path = str(tmp_path / "test.wav")
            _generate_xai_tts("Hello", output_path, {})

        payload = mock_post.call_args[1]["json"]
        assert payload["codec"] == "wav"

    def test_ogg_generates_mp3_then_converts(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.content = b"fake-mp3-for-opus"
        mock_response.raise_for_status = MagicMock()

        opus_path = str(tmp_path / "test.ogg")

        with patch("requests.post", return_value=mock_response), \
             patch("tools.tts_tool._convert_to_opus", return_value=opus_path) as mock_convert:
            result = _generate_xai_tts("Hello", str(tmp_path / "test.ogg"), {})

        # Should have called convert_to_opus with the intermediate mp3
        mock_convert.assert_called_once()
        mp3_arg = mock_convert.call_args[0][0]
        assert mp3_arg.endswith(".mp3")

    def test_custom_voice_from_config(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.raise_for_status = MagicMock()

        config = {"xai": {"voice": "eve", "language": "fr"}}
        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_xai_tts("Bonjour", str(tmp_path / "out.mp3"), config)

        payload = mock_post.call_args[1]["json"]
        assert payload["voice_id"] == "eve"
        assert payload["language"] == "fr"

    def test_text_truncation_at_15k(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts, XAI_MAX_TEXT_LENGTH

        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.content = b"audio"
        mock_response.raise_for_status = MagicMock()

        long_text = "x" * 20000
        with patch("requests.post", return_value=mock_response) as mock_post:
            _generate_xai_tts(long_text, str(tmp_path / "out.mp3"), {})

        payload = mock_post.call_args[1]["json"]
        assert len(payload["text"]) == XAI_MAX_TEXT_LENGTH

    def test_empty_audio_raises_runtime_error(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_xai_tts

        monkeypatch.setenv("XAI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.content = b""
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="empty audio"):
                _generate_xai_tts("Hello", str(tmp_path / "out.mp3"), {})
