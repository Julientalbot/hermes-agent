"""xAI Grok image generation tool.

Uses xAI's grok-imagine-image model to generate images from text prompts.
This is an alternative to the FAL.ai-based image_generate tool for users
who have XAI_API_KEY configured.

Supports:
- Text-to-image generation
- Image editing (provide image_url)
- Aspect ratio control
- Resolution control (1k / 2k)

Docs: https://docs.x.ai/developers/model-capabilities/images/generation
"""

import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _check_xai_image_available() -> bool:
    """Check if xAI image generation is available (XAI_API_KEY set)."""
    return bool(os.getenv("XAI_API_KEY"))


def _resolve_xai_credentials():
    """Return (api_key, base_url) for xAI."""
    key = os.getenv("XAI_API_KEY")
    base = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    return key, base


def xai_image_generate(
    prompt: str,
    model: str = "grok-imagine-image",
    aspect_ratio: str = None,
    resolution: str = None,
    image_url: str = None,
) -> str:
    """Generate an image using xAI's Grok image generation model.

    Returns JSON string with success status and image path or error.

    xAI API params (per docs.x.ai):
      - prompt: text description
      - model: grok-imagine-image or grok-imagine-image-pro
      - n: number of images (always 1 here)
      - aspect_ratio: "1:1", "16:9", "4:3", "3:2", "2:1", "auto", etc.
      - resolution: "1k" or "2k"
      - image_url: source image for editing (base64 data URI or public URL)

    NOTE: xAI does NOT support the `size` or `quality` params from the OpenAI
    SDK.  Only the params listed above are valid.
    """
    from openai import OpenAI
    import requests

    api_key, base_url = _resolve_xai_credentials()
    if not api_key:
        return json.dumps({
            "success": False,
            "error": "XAI_API_KEY not configured (env or credential pool)",
        })

    client = OpenAI(api_key=api_key, base_url=base_url)

    generate_kwargs = {"model": model, "prompt": prompt}
    # xAI-specific params must go through extra_body since the OpenAI SDK
    # doesn't recognize them as valid parameters for images.generate().
    extra_body = {}
    if aspect_ratio is not None:
        extra_body["aspect_ratio"] = aspect_ratio
    if resolution is not None:
        extra_body["resolution"] = resolution
    if image_url is not None:
        extra_body["image_url"] = image_url

    try:
        if extra_body:
            generate_kwargs["extra_body"] = extra_body
        response = client.images.generate(**generate_kwargs)
    except Exception as e:
        logger.error("xAI image generation API call failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": str(e)})

    if not response.data or not getattr(response.data[0], "url", None):
        return json.dumps({"success": False, "error": "No image URL in xAI response"})

    remote_url = response.data[0].url

    # Download to local file — xAI image URLs are temporary per docs.
    from hermes_constants import get_hermes_home
    output_dir = Path(get_hermes_home()) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / f"xai_gen_{uuid.uuid4().hex[:12]}.jpg"

    try:
        img_data = requests.get(remote_url, timeout=30).content
        local_path.write_bytes(img_data)
        logger.info("xAI image generated: %s (%d bytes)", local_path, len(img_data))
        media_tag = f"MEDIA:{local_path}"
        return json.dumps({
            "success": True,
            "image": str(local_path),
            "media_tag": media_tag,
        })
    except Exception as dl_err:
        logger.warning("Could not download xAI image, returning URL: %s", dl_err)
        media_tag = f"MEDIA:{remote_url}"
        return json.dumps({
            "success": True,
            "image": remote_url,
            "media_tag": media_tag,
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error  # noqa: E402

from hermes_constants import display_hermes_home  # noqa: E402

XAI_IMAGE_GENERATE_SCHEMA = {
    "name": "xai_image_generate",
    "description": (
        "Generate images using xAI's Grok image model (grok-imagine-image). "
        "Supports text-to-image generation and image editing with a source image. "
        f"Returns the path to the generated image file in {display_hermes_home()}/images/. "
        "Use this when XAI_API_KEY is available and you need image generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image or edit.",
            },
            "model": {
                "type": "string",
                "description": "Model to use: 'grok-imagine-image' ($0.02/img) or 'grok-imagine-image-pro' ($0.07/img, higher quality).",
                "enum": ["grok-imagine-image", "grok-imagine-image-pro"],
                "default": "grok-imagine-image",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Aspect ratio: '1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3', '2:1', '1:2', 'auto'.",
                "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "auto"],
            },
            "resolution": {
                "type": "string",
                "description": "Output resolution: '1k' or '2k'.",
                "enum": ["1k", "2k"],
            },
            "image_url": {
                "type": "string",
                "description": "Source image for editing (public URL or base64 data URI). Omit for text-to-image.",
            },
        },
        "required": ["prompt"],
    },
}


def _handle_xai_image_generate(args, **kw):
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for image generation")
    return xai_image_generate(
        prompt=prompt,
        model=args.get("model", "grok-imagine-image"),
        aspect_ratio=args.get("aspect_ratio"),
        resolution=args.get("resolution"),
        image_url=args.get("image_url"),
    )


registry.register(
    name="xai_image_generate",
    toolset="image_gen",
    schema=XAI_IMAGE_GENERATE_SCHEMA,
    handler=_handle_xai_image_generate,
    check_fn=_check_xai_image_available,
    requires_env=[],
    is_async=False,
    emoji="🎨",
)
