"""xAI Grok video generation tool.

Uses xAI's grok-imagine-video model to generate short videos from text
prompts and optional reference images.  Video generation is asynchronous:
the API returns a request_id which is polled until the video is ready.

Supports:
- Text-to-video generation
- Reference images for style/content guidance
- Duration, aspect ratio, and resolution control

Pricing: $0.05 per second of generated video.
Docs: https://docs.x.ai/developers/model-capabilities/video/generation
"""

import json
import logging
import time
import uuid
from pathlib import Path

from tools.xai_utils import check_xai_tool_available, resolve_xai_credentials

logger = logging.getLogger(__name__)

# Polling constants
_POLL_INTERVAL_SECONDS = 5
_POLL_MAX_WAIT_SECONDS = 300  # 5 minutes max wait


def xai_video_generate(
    prompt: str,
    duration: int = None,
    aspect_ratio: str = None,
    resolution: str = None,
    reference_images: list = None,
) -> str:
    """Generate a video using xAI's grok-imagine-video model.

    This function submits the generation request and polls until the video
    is ready, then downloads it to a local file.

    Args:
        prompt: Text description of the desired video.
        duration: Video duration in seconds (default: API decides).
        aspect_ratio: "16:9" (default).
        resolution: "720p" or "480p".
        reference_images: List of image URLs for style/content guidance.

    Returns:
        JSON string with success status and video path or error.
    """
    import requests

    api_key, base_url = resolve_xai_credentials()
    if not api_key:
        return json.dumps({
            "success": False,
            "error": "XAI_API_KEY not configured (env or credential pool)",
        })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Build generation payload.
    payload = {
        "model": "grok-imagine-video",
        "prompt": prompt,
    }
    if duration is not None:
        payload["duration"] = duration
    if aspect_ratio is not None:
        payload["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        payload["resolution"] = resolution
    if reference_images:
        payload["reference_images"] = [
            {"url": url} if isinstance(url, str) else url
            for url in reference_images
        ]

    # Step 1: Submit generation request.
    try:
        submit_resp = requests.post(
            f"{base_url}/videos/generations",
            json=payload,
            headers=headers,
            timeout=30,
        )
        submit_resp.raise_for_status()
        submit_data = submit_resp.json()
    except Exception as e:
        logger.error("xAI video generation submit failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": f"Submit failed: {e}"})

    request_id = submit_data.get("request_id")
    if not request_id:
        return json.dumps({
            "success": False,
            "error": f"No request_id in response: {json.dumps(submit_data)[:200]}",
        })

    logger.info("xAI video generation submitted: request_id=%s", request_id)

    # Step 2: Poll for completion.
    start_time = time.time()
    last_progress = -1
    while time.time() - start_time < _POLL_MAX_WAIT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        try:
            poll_resp = requests.get(
                f"{base_url}/videos/{request_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
        except Exception as e:
            logger.warning("Poll failed (will retry): %s", e)
            continue

        status = poll_data.get("status", "unknown")
        progress = poll_data.get("progress", 0)

        if progress != last_progress:
            logger.info("Video generation %s: %d%% complete", request_id, progress)
            last_progress = progress

        if status == "done":
            video_url = poll_data.get("video_url") or poll_data.get("url")
            if not video_url:
                # xAI nests the URL under "video": {"url": "..."}
                video_obj = poll_data.get("video", {})
                if isinstance(video_obj, dict):
                    video_url = video_obj.get("url")
            if not video_url:
                # Last resort: check "data" wrapper.
                data = poll_data.get("data", {})
                video_url = data.get("video_url") or data.get("url")
            if not video_url:
                return json.dumps({
                    "success": False,
                    "error": f"Video done but no URL found in response: {json.dumps(poll_data)[:300]}",
                })

            # Download the video.
            from hermes_constants import get_hermes_home
            output_dir = Path(get_hermes_home()) / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            local_path = output_dir / f"xai_vid_{uuid.uuid4().hex[:12]}.mp4"

            try:
                vid_data = requests.get(video_url, timeout=60).content
                local_path.write_bytes(vid_data)
                logger.info("xAI video downloaded: %s (%d bytes)", local_path, len(vid_data))
                media_tag = f"MEDIA:{local_path}"
                return json.dumps({
                    "success": True,
                    "video": str(local_path),
                    "media_tag": media_tag,
                    "duration_seconds": poll_data.get("duration") or (video_obj.get("duration") if isinstance(poll_data.get("video"), dict) else None),
                })
            except Exception as dl_err:
                logger.warning("Could not download video, returning URL: %s", dl_err)
                media_tag = f"MEDIA:{video_url}"
                return json.dumps({
                    "success": True,
                    "video": video_url,
                    "media_tag": media_tag,
                })

        elif status == "failed":
            error_msg = poll_data.get("error", "Unknown generation error")
            return json.dumps({"success": False, "error": f"Generation failed: {error_msg}"})

        elif status == "expired":
            return json.dumps({"success": False, "error": "Generation request expired"})

    return json.dumps({
        "success": False,
        "error": f"Timed out after {_POLL_MAX_WAIT_SECONDS}s waiting for video (request_id={request_id})",
    })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error  # noqa: E402

from hermes_constants import display_hermes_home  # noqa: E402

XAI_VIDEO_GENERATE_SCHEMA = {
    "name": "xai_video_generate",
    "description": (
        "Generate short videos using xAI's Grok video model (grok-imagine-video). "
        "Supports text-to-video generation with optional reference images for "
        "style guidance. Video generation is asynchronous and may take 1-3 minutes. "
        f"Returns the path to the generated video file in {display_hermes_home()}/videos/. "
        "Pricing: $0.05 per second of video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the desired video.",
            },
            "duration": {
                "type": "integer",
                "description": "Video duration in seconds.",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Video aspect ratio (default: '16:9').",
                "enum": ["16:9"],
                "default": "16:9",
            },
            "resolution": {
                "type": "string",
                "description": "Output resolution: '720p' or '480p'.",
                "enum": ["720p", "480p"],
            },
            "reference_images": {
                "type": "array",
                "description": "List of image URLs to use as style/content references.",
                "items": {"type": "string"},
            },
        },
        "required": ["prompt"],
    },
}


def _handle_xai_video_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for video generation")
    return xai_video_generate(
        prompt=prompt,
        duration=args.get("duration"),
        aspect_ratio=args.get("aspect_ratio"),
        resolution=args.get("resolution"),
        reference_images=args.get("reference_images"),
    )


registry.register(
    name="xai_video_generate",
    toolset="image_gen",
    schema=XAI_VIDEO_GENERATE_SCHEMA,
    handler=_handle_xai_video_generate,
    check_fn=lambda: check_xai_tool_available("xai_video_generate"),
    requires_env=[],
    is_async=False,
    emoji="🎬",
)
