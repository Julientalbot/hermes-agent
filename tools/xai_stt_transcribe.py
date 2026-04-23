"""Standalone xAI Grok STT transcription tool.

Fichier autonome — ne necessite aucun patch sur transcription_tools.py.
Pour le STT automatique dans le voice mode du gateway, voir aussi
~/.hermes/scripts/patches/xai-stt-provider.patch
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

from tools.registry import registry

logger = logging.getLogger(__name__)

XAI_STT_BASE_URL = os.getenv("XAI_STT_BASE_URL", "https://api.x.ai/v1")


def _transcribe_xai(
    file_path: str,
    language: str = "fr",
    use_format: bool = True,
    use_diarize: bool = False,
) -> Dict[str, Any]:
    """Transcribe audio using xAI Grok STT API."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "XAI_API_KEY not set"}

    base_url = XAI_STT_BASE_URL.rstrip("/")

    try:
        import requests
        from tools.xai_http import hermes_xai_user_agent

        data = {}
        if language:
            data["language"] = language
        if use_format:
            data["format"] = "true"
        if use_diarize:
            data["diarize"] = "true"

        with open(file_path, "rb") as audio_file:
            response = requests.post(
                f"{base_url}/stt",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": hermes_xai_user_agent(),
                },
                files={"file": (Path(file_path).name, audio_file)},
                data=data,
                timeout=120,
            )

        if response.status_code != 200:
            detail = ""
            try:
                err_body = response.json()
                detail = err_body.get("error", {}).get("message", "") or response.text[:300]
            except Exception:
                detail = response.text[:300]
            return {
                "success": False,
                "transcript": "",
                "error": f"xAI STT API error (HTTP {response.status_code}): {detail}",
            }

        result = response.json()
        transcript_text = result.get("text", "").strip()

        if not transcript_text:
            return {
                "success": False,
                "transcript": "",
                "error": "xAI STT returned empty transcript",
            }

        return {
            "success": True,
            "transcript": transcript_text,
            "language": result.get("language", language),
            "duration": result.get("duration", 0),
        }

    except Exception as e:
        return {
            "success": False,
            "transcript": "",
            "error": f"xAI STT failed: {type(e).__name__}: {e}",
        }


XAI_STT_TRANSCRIBE_SCHEMA = {
    "name": "xai_stt_transcribe",
    "description": "Transcribe an audio file using the xAI Grok Speech-to-Text API. Requires XAI_API_KEY.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the audio file to transcribe.",
            },
            "language": {
                "type": "string",
                "description": "Language code (default: fr).",
            },
            "format": {
                "type": "boolean",
                "description": "Enable Inverse Text Normalization (ITN, default: true).",
            },
            "diarize": {
                "type": "boolean",
                "description": "Enable speaker diarization (default: false).",
            },
        },
        "required": ["file_path"],
    },
}


def _check_xai_stt_env() -> bool:
    return bool(os.getenv("XAI_API_KEY"))


def xai_stt_transcribe(
    file_path: str, language: str = "fr", format: bool = True, diarize: bool = False
) -> Dict[str, Any]:
    """Transcribe an audio file using xAI Grok STT API."""
    return _transcribe_xai(
        file_path, language=language, use_format=format, use_diarize=diarize
    )


registry.register(
    name="xai_stt_transcribe",
    toolset="tts",
    schema=XAI_STT_TRANSCRIBE_SCHEMA,
    handler=xai_stt_transcribe,
    check_fn=_check_xai_stt_env,
    requires_env=["XAI_API_KEY"],
    emoji="🎙",
)
