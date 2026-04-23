"""
Hermes Voice Turn API — route /api/voice/turn

Pipeline : blob audio → STT → AIAgent → TTS → JSON {text, audio_base64}

Usage (curl) :
    curl -X POST http://localhost:9119/api/voice/turn \
      -H "Content-Type: multipart/form-data" \
      -F "audio=@recording.webm"
"""

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# ── Ensure project root is on PYTHONPATH so we can import tools.* / run_agent ──
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Load env (API keys) ──
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home

_hermes_home = get_hermes_home()
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_PROJECT_ROOT / ".env")

# ── Imports Hermes core ──
from run_agent import AIAgent
from tools.transcription_tools import transcribe_audio
from tools.tts_tool import text_to_speech_tool
from hermes_cli.config import load_config

_log = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])


# ── Config helpers ──
def _get_agent_config() -> tuple[str, int]:
    """Return (model, max_iterations) from Hermes config.yaml."""
    cfg = load_config()
    model = cfg.get("model", "openrouter/auto")
    max_iter = cfg.get("max_iterations", 60)
    if isinstance(model, dict):
        model = model.get("name", "openrouter/auto")
    return str(model), int(max_iter)


# ── Main endpoint ──
@router.post("/turn")
async def voice_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(default=None),
    model_override: str = Form(default=None),
):
    """
    Receive an audio file, transcribe it, run through Hermes agent,
    synthesize speech, and return both text + base64 audio.
    """
    session_id = session_id or f"voice-{uuid.uuid4().hex[:8]}"
    _log.info("[voice] Turn started — session=%s file=%s", session_id, audio.filename)

    # ------------------------------------------------------------------
    # 1. Persist uploaded blob to a temp file
    # ------------------------------------------------------------------
    suffix = Path(audio.filename or "blob").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await audio.read())
        audio_path = tmp_in.name

    _log.info("[voice] Received %s bytes from %s", Path(audio_path).stat().st_size, audio.filename)

    # Ensure the voice pipeline can clean up safely even if STT fails
    tts_output_path = None
    wav_path = None

    # Convert uploaded blob to WAV (mono 16kHz) for universal STT compatibility
    wav_path = audio_path + ".wav"
    try:
        import subprocess
        conv = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", audio_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            check=True, capture_output=True, text=True,
        )
        audio_path_for_stt = wav_path
        _log.info("[voice] Converted to WAV: %s", wav_path)
    except subprocess.CalledProcessError as conv_exc:
        _log.warning("[voice] ffmpeg conversion failed (rc=%d): %s — stderr: %s", conv_exc.returncode, conv_exc, conv_exc.stderr)
        audio_path_for_stt = audio_path
    except Exception as conv_exc:
        _log.warning("[voice] ffmpeg conversion failed: %s — using original", conv_exc)
        audio_path_for_stt = audio_path

    try:
        # --------------------------------------------------------------
        # 2. STT  (blocking → thread pool)
        # --------------------------------------------------------------
        _log.info("[voice] STT on %s", audio_path_for_stt)
        stt_result = await asyncio.to_thread(transcribe_audio, audio_path_for_stt)

        if not stt_result.get("success"):
            _log.warning("[voice] STT failed: %s", stt_result.get("error"))
            raise HTTPException(
                status_code=422,
                detail=f"Speech recognition failed: {stt_result.get('error', 'Unknown')}",
            )

        user_text = stt_result.get("transcript", "").strip()
        if not user_text:
            raise HTTPException(status_code=422, detail="Empty transcription.")

        _log.info("[voice] Transcribed (%d chars): %.80s", len(user_text), user_text)

        # --------------------------------------------------------------
        # 3. Agent run  (blocking → thread pool)
        # --------------------------------------------------------------
        model, max_iter = _get_agent_config()
        if model_override:
            model = model_override

        agent_kwargs: Dict[str, Any] = {
            "model": model,
            "max_iterations": max_iter,
            "platform": "voice",
            "session_id": session_id,
            "enabled_toolsets": None,
            "disabled_toolsets": None,
        }

        _log.info("[voice] Agent run — model=%s session=%s", model, session_id)
        agent_result = await asyncio.to_thread(_run_agent_sync, user_text, agent_kwargs)

        response_text = ""
        if isinstance(agent_result, dict):
            response_text = agent_result.get("final_response", "")
        elif isinstance(agent_result, str):
            response_text = agent_result

        if not response_text:
            response_text = "(L'agent n'a pas produit de réponse.)"

        _log.info("[voice] Agent responded (%d chars): %.80s", len(response_text), response_text)

        # --------------------------------------------------------------
        # 4. TTS  (blocking → thread pool)
        # --------------------------------------------------------------
        _log.info("[voice] TTS generation")
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_out:
            tts_output_path = tmp_out.name

        tts_path = await asyncio.to_thread(
            text_to_speech_tool,
            response_text,
            tts_output_path,
        )

        if tts_path and Path(tts_path).exists():
            audio_bytes = Path(tts_path).read_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_format = "ogg" if tts_path.endswith(".ogg") else "mp3"
        else:
            audio_b64 = ""
            audio_format = ""
            _log.warning("[voice] TTS produced no audio file.")

        # --------------------------------------------------------------
        # 5. Response
        # --------------------------------------------------------------
        return JSONResponse({
            "text": response_text,
            "user_text": user_text,
            "audio_base64": audio_b64,
            "audio_format": audio_format,
            "session_id": session_id,
            "model": model,
            "platform": "voice",
        })

    except HTTPException:
        raise
    except Exception as exc:
        _log.error("[voice] Unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice pipeline error: {exc}")

    finally:
        # Clean up temp files including conversion artifacts
        for p in (audio_path, wav_path, tts_output_path):
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass


def _run_agent_sync(message: str, kwargs: Dict[str, Any]):
    """Synchronous wrapper so we can call it via asyncio.to_thread()."""
    agent = AIAgent(**kwargs)
    return agent.run_conversation(message)


# ==========================================================================
# Voice Agent endpoint — direct Grok realtime via WebSocket
# ==========================================================================

@router.post("/grok")
async def voice_grok(
    audio: UploadFile = File(...),
    voice: str = Form(default="eve"),
    instructions: str = Form(default=""),
    session_id: str = Form(default=None),
):
    """
    Direct Voice Agent: audio → Grok realtime → audio response.
    Single WebSocket round-trip. No Hermes agent, no tools.
    Fast and natural, but limited to Grok's own knowledge.
    """
    session_id = session_id or f"grok-{uuid.uuid4().hex[:8]}"
    _log.info("[voice/grok] Turn started — session=%s voice=%s", session_id, voice)

    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "blob").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await audio.read())
        audio_path = tmp_in.name

    tts_output_path = None

    try:
        # Convert to WAV for Voice Agent (PCM16 mono 24kHz)
        wav_path = audio_path + ".wav"
        import subprocess
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", audio_path, "-ar", "24000", "-ac", "1",
                 "-c:a", "pcm_s16le", wav_path],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            wav_path = audio_path  # fallback to original

        # Run Voice Agent
        from tools.xai_voice_agent_tool import voice_turn as grok_voice_turn

        result = await grok_voice_turn(
            audio_file=wav_path,
            instructions=instructions or "Tu es Hermes, assistant IA francophone. Réponds de manière concise et utile.",
            voice=voice,
            language="fr",
        )

        if not result.get("success"):
            _log.warning("[voice/grok] Voice Agent failed: %s", result.get("error"))
            raise HTTPException(
                status_code=502,
                detail=f"Voice Agent error: {result.get('error', 'Unknown')}",
            )

        # Convert WAV response to OGG for Telegram
        audio_path_result = result.get("audio_path", "")
        audio_b64 = ""
        audio_format = ""

        if audio_path_result and Path(audio_path_result).exists():
            tts_output_path = audio_path_result + ".ogg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", audio_path_result, "-c:a", "libopus", "-b:a", "64k",
                     tts_output_path],
                    check=True, capture_output=True, text=True,
                )
                audio_bytes = Path(tts_output_path).read_bytes()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_format = "ogg"
            except Exception as e:
                _log.warning("[voice/grok] OGG conversion failed: %s", e)
                # Fallback: send WAV directly
                audio_bytes = Path(audio_path_result).read_bytes()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_format = "wav"

        return JSONResponse({
            "text": result.get("transcript", ""),
            "user_text": result.get("transcript", ""),
            "audio_base64": audio_b64,
            "audio_format": audio_format,
            "session_id": session_id,
            "model": "grok-realtime",
            "platform": "voice-grok",
            "voice": voice,
        })

    except HTTPException:
        raise
    except Exception as exc:
        _log.error("[voice/grok] Unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice Agent error: {exc}")

    finally:
        for p in (audio_path, tts_output_path):
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
