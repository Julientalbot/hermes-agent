"""xAI Voice Agent — Real-time voice conversations via WebSocket.

Connects to xAI's Realtime Voice API (wss://api.x.ai/v1/realtime) for
bidirectional audio conversations with Grok, including tool calling.

Architecture:
  - Session-based WebSocket connection
  - Audio streaming (base64-encoded chunks)
  - VAD (Voice Activity Detection) or manual turn control
  - Function calling bridged to Hermes tool registry
  - Audio response collection and local save

Usage:
  1. File-based turn (async voice message):
     result = await voice_turn(audio_file="recording.ogg", instructions="...")

  2. Live streaming (PWA / future):
     async with VoiceAgentSession(...) as session:
         session.stream_audio(chunk)
         async for event in session.events():
             ...

Docs: https://docs.x.ai/developers/model-capabilities/audio/voice
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import websockets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REALTIME_URL = "wss://api.x.ai/v1/realtime"
DEFAULT_VOICE = "eve"
DEFAULT_LANGUAGE = "fr"
DEFAULT_MODEL = "grok-4.20-reasoning"
DEFAULT_TIMEOUT = 120  # seconds max per turn

VALID_VOICES = ["eve", "ara", "rex", "sal", "leo"]


# ---------------------------------------------------------------------------
# Session Configuration
# ---------------------------------------------------------------------------

def _build_session_config(
    voice: str = DEFAULT_VOICE,
    instructions: str = "",
    language: str = DEFAULT_LANGUAGE,
    tools: list = None,
    turn_detection: dict = None,
) -> dict:
    """Build the session.update payload."""
    session = {
        "voice": voice,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "grok-stt"},
    }
    if instructions:
        session["instructions"] = instructions
    if language:
        session["language"] = language
    if tools:
        session["tools"] = tools
    if turn_detection is not None:
        session["turn_detection"] = turn_detection
    else:
        # Default: server VAD for automatic turn detection
        session["turn_detection"] = {"type": "server_vad"}
    return {"type": "session.update", "session": session}


# ---------------------------------------------------------------------------
# Audio Helpers
# ---------------------------------------------------------------------------

def _audio_file_to_pcm16_base64(file_path: str, sample_rate: int = 24000) -> tuple[str, int]:
    """Convert an audio file to PCM16 base64 chunks.

    Uses ffmpeg for conversion. Returns (base64_data, sample_rate).
    """
    import subprocess
    import tempfile

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Convert to raw PCM16 mono at target sample rate
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ar", str(sample_rate), "-ac", "1",
                tmp_path,
            ],
            capture_output=True,
            check=True,
        )
        pcm_bytes = Path(tmp_path).read_bytes()
        return base64.b64encode(pcm_bytes).decode(), sample_rate
    finally:
        os.unlink(tmp_path)


def _pcm16_base64_to_audio(b64_data: str, output_path: str, sample_rate: int = 24000):
    """Convert PCM16 base64 data to an audio file (WAV)."""
    import struct
    import wave

    pcm_bytes = base64.b64decode(b64_data)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


# ---------------------------------------------------------------------------
# Voice Agent Session (WebSocket)
# ---------------------------------------------------------------------------

class VoiceAgentSession:
    """Manages a real-time voice session with xAI's Voice Agent API."""

    def __init__(
        self,
        api_key: str = None,
        realtime_url: str = DEFAULT_REALTIME_URL,
        voice: str = DEFAULT_VOICE,
        instructions: str = "",
        language: str = DEFAULT_LANGUAGE,
        tools: list = None,
    ):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.realtime_url = realtime_url
        self.voice = voice
        self.instructions = instructions
        self.language = language
        self.tools = tools or []
        self._ws = None
        self._audio_chunks = []
        self._transcript_parts = []
        self._events = []

    async def connect(self):
        """Open WebSocket connection and configure session."""
        if not self.api_key:
            raise ValueError("XAI_API_KEY not set")

        self._ws = await websockets.connect(
            self.realtime_url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

        # Drain initial events (session.created, conversation.created — order varies)
        import time as _time
        _deadline = _time.time() + 10
        while _time.time() < _deadline:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=2)
                event = json.loads(raw)
                logger.debug("Voice Agent init event: %s", event.get("type"))
                if event.get("type") == "error":
                    raise RuntimeError(f"xAI error during connect: {event}")
            except asyncio.TimeoutError:
                break  # No more init events

        # Configure session
        config = _build_session_config(
            voice=self.voice,
            instructions=self.instructions,
            language=self.language,
            tools=self.tools,
        )
        await self._ws.send(json.dumps(config))

        # Wait for session.updated
        event = json.loads(await self._ws.recv())
        if event.get("type") != "session.updated":
            logger.warning("Expected session.updated, got: %s", event.get("type"))

        logger.info("Voice Agent session connected (voice=%s)", self.voice)

    async def send_audio(self, audio_b64: str):
        """Send base64-encoded PCM16 audio data."""
        if not self._ws:
            raise RuntimeError("Not connected")

        # Split into chunks (~100ms each at 24kHz = 4800 samples = 9600 bytes)
        chunk_size = 9600 * 2  # bytes → base64 chars
        pcm_bytes = base64.b64decode(audio_b64)

        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]
            chunk_b64 = base64.b64encode(chunk).decode()
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk_b64,
            }))

        # Commit the audio buffer
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
        }))

        # Request response
        await self._ws.send(json.dumps({
            "type": "response.create",
        }))

    async def send_text(self, text: str):
        """Send a text message instead of audio."""
        if not self._ws:
            raise RuntimeError("Not connected")

        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }))

        await self._ws.send(json.dumps({
            "type": "response.create",
        }))

    async def receive_response(self, timeout: float = DEFAULT_TIMEOUT) -> dict:
        """Collect all events until response.done.

        Returns dict with 'audio_b64', 'transcript', 'text', 'function_calls'.
        """
        import time
        start = time.time()

        audio_chunks = []
        transcript_parts = []
        text_parts = []
        function_calls = []

        while time.time() - start < timeout:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5)
                event = json.loads(raw)
                self._events.append(event)

                etype = event.get("type", "")

                if etype in ("response.output_audio.delta", "response.audio.delta"):
                    audio_chunks.append(event.get("delta", ""))

                elif etype in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                    transcript_parts.append(event.get("delta", ""))

                elif etype == "response.text.delta":
                    text_parts.append(event.get("delta", ""))

                elif etype == "response.function_call_arguments.done":
                    function_calls.append({
                        "name": event.get("name", ""),
                        "arguments": event.get("arguments", ""),
                        "call_id": event.get("call_id", ""),
                    })

                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript_parts.append(event.get("transcript", ""))

                elif etype == "response.done":
                    # Extract any content from the response object
                    resp = event.get("response", {})
                    for item in resp.get("output", []):
                        for content in item.get("content", []):
                            if content.get("type") == "audio" and content.get("audio"):
                                audio_chunks.append(content["audio"])
                            if content.get("type") == "audio" and content.get("transcript"):
                                transcript_parts.append(content["transcript"])
                            if content.get("type") == "text" and content.get("text"):
                                text_parts.append(content["text"])
                    break

                elif etype == "error":
                    logger.error("Voice Agent error: %s", event)
                    return {"error": event.get("error", {}).get("message", str(event))}

            except asyncio.TimeoutError:
                continue

        # Concatenate audio
        full_audio = "".join(audio_chunks)
        full_transcript = "".join(transcript_parts)
        full_text = "".join(text_parts)

        return {
            "audio_b64": full_audio,
            "transcript": full_transcript,
            "text": full_text,
            "function_calls": function_calls,
            "events_count": len(self._events),
        }

    async def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()


# ---------------------------------------------------------------------------
# Convenience Function — Single Turn
# ---------------------------------------------------------------------------

async def voice_turn(
    audio_file: str = None,
    text: str = None,
    instructions: str = "",
    voice: str = DEFAULT_VOICE,
    language: str = DEFAULT_LANGUAGE,
    output_path: str = None,
) -> dict:
    """Execute a single voice turn: send audio/text, get audio response.

    Args:
        audio_file: Path to audio file (ogg, mp3, wav, etc.)
        text: Text input (alternative to audio_file)
        instructions: System instructions for Grok
        voice: Voice ID (eve, ara, rex, sal, leo)
        language: Language code
        output_path: Where to save response audio (auto-generated if None)

    Returns dict with success, transcript, text, audio_path, duration.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return {"success": False, "error": "XAI_API_KEY not set"}

    if not audio_file and not text:
        return {"success": False, "error": "Provide audio_file or text"}

    async with VoiceAgentSession(
        api_key=api_key,
        voice=voice,
        instructions=instructions,
        language=language,
    ) as session:
        # Send input
        if audio_file:
            audio_b64, sample_rate = _audio_file_to_pcm16_base64(audio_file)
            await session.send_audio(audio_b64)
        else:
            await session.send_text(text)

        # Collect response
        result = await session.receive_response()

    if "error" in result:
        return {"success": False, "error": result["error"]}

    # Save audio response
    if result["audio_b64"]:
        if not output_path:
            from hermes_constants import get_hermes_home
            audio_dir = Path(get_hermes_home()) / "audio_cache"
            audio_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(audio_dir / f"xai_voice_{uuid.uuid4().hex[:12]}.wav")

        _pcm16_base64_to_audio(result["audio_b64"], output_path)
        logger.info("Voice Agent response saved: %s", output_path)

    return {
        "success": True,
        "transcript": result.get("transcript", ""),
        "text": result.get("text", ""),
        "audio_path": output_path if result["audio_b64"] else None,
        "media_tag": f"MEDIA:{output_path}" if result["audio_b64"] and output_path else "",
        "function_calls": result.get("function_calls", []),
    }


# ---------------------------------------------------------------------------
# Sync Wrapper (for tool registration)
# ---------------------------------------------------------------------------

def xai_voice_agent(
    audio_file: str = None,
    text: str = None,
    instructions: str = "",
    voice: str = DEFAULT_VOICE,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Synchronous wrapper for voice_turn()."""
    result = asyncio.run(voice_turn(
        audio_file=audio_file,
        text=text,
        instructions=instructions,
        voice=voice,
        language=language,
    ))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error  # noqa: E402

XAI_VOICE_AGENT_SCHEMA = {
    "name": "xai_voice_agent",
    "description": (
        "Real-time voice conversation with Grok via xAI's Voice Agent API. "
        "Sends audio or text and receives a spoken response. "
        "Supports 5 voices (eve, ara, rex, sal, leo) and 25 languages. "
        "For live streaming, use VoiceAgentSession class directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "audio_file": {
                "type": "string",
                "description": "Path to audio file to send (ogg, mp3, wav, etc.). Mutually exclusive with text.",
            },
            "text": {
                "type": "string",
                "description": "Text input to speak. Mutually exclusive with audio_file.",
            },
            "instructions": {
                "type": "string",
                "description": "System instructions for Grok's behavior.",
            },
            "voice": {
                "type": "string",
                "description": "Voice ID.",
                "enum": VALID_VOICES,
                "default": DEFAULT_VOICE,
            },
            "language": {
                "type": "string",
                "description": "Language code (e.g. 'fr', 'en', 'auto').",
                "default": DEFAULT_LANGUAGE,
            },
        },
    },
}


def _handle_xai_voice_agent(args, **kw):
    audio_file = args.get("audio_file")
    text = args.get("text")
    if not audio_file and not text:
        return tool_error("Provide audio_file or text")
    return xai_voice_agent(
        audio_file=audio_file,
        text=text,
        instructions=args.get("instructions", ""),
        voice=args.get("voice", DEFAULT_VOICE),
        language=args.get("language", DEFAULT_LANGUAGE),
    )


registry.register(
    name="xai_voice_agent",
    toolset="tts",
    schema=XAI_VOICE_AGENT_SCHEMA,
    handler=_handle_xai_voice_agent,
    check_fn=lambda: bool(os.getenv("XAI_API_KEY")),
    requires_env=["XAI_API_KEY"],
    is_async=False,
    emoji="🎙️",
)
