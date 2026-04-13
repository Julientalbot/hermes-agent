"""xAI Grok image generation tool.

Uses xAI's grok-imagine-image model to generate images from text prompts.
This is an alternative to the FAL.ai-based image_generate tool for users
who have XAI_API_KEY configured.

Supports:
- Text-to-image generation
- Image editing (provide image_url)
- Aspect ratio control
- Resolution control (1k / 2k)
- Base64 output
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _check_xai_image_available() -> bool:
    """Check if xAI image generation is available (XAI_API_KEY set)."""
    # Check credential pool first
    try:
        from agent.auxiliary_client import _select_pool_entry, _pool_runtime_api_key
        pool_present, entry = _select_pool_entry("xai")
        if pool_present and _pool_runtime_api_key(entry):
            return True
    except Exception:
        pass
    return bool(os.getenv("XAI_API_KEY"))


def xai_image_generate(
    prompt: str,
    model: str = "grok-imagine-image",
    aspect_ratio: str = None,
    resolution: str = None,
    image_url: str = None,
    image_format: str = None,
) -> str:
    """Generate an image using xAI's Grok image generation model.

    Args:
        prompt: Text description of the desired image.
        model: grok-imagine-image (standard) or grok-imagine-image-pro (higher quality).
        aspect_ratio: "1:1", "16:9", "4:3", "3:2", "2:1", "auto", etc.
        resolution: "1k" or "2k".
        image_url: Source image URL for editing (base64 data URI or public URL).
        image_format: "url" (default) or "base64".

    Returns:
        JSON string with generation results.
    """
    from agent.auxiliary_client import generate_image

    try:
        result_path = generate_image(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_url=image_url,
            image_format=image_format,
        )
        return json.dumps({
            "success": True,
            "image": result_path,
        })
    except Exception as e:
        logger.error("xAI image generation failed: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

XAI_IMAGE_GENERATE_SCHEMA = {
    "name": "xai_image_generate",
    "description": (
        "Generate images using xAI's Grok image model (grok-imagine-image). "
        "Supports text-to-image generation and image editing with a source image. "
        "Returns the path to the generated image file. "
        "Use this when XAI_API_KEY is available and you need quick image generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The text prompt describing the desired image or edit.",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Aspect ratio: '1:1', '16:9', '4:3', '3:2', '2:1', 'auto'.",
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
        image_format=args.get("image_format"),
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
