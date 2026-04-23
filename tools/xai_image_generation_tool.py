"""xAI Grok image generation tool.

Uses xAI's grok-imagine-image model to generate or edit images.
Inspired by OpenClaw's xAI image-generation-provider (PR #68694).

Supports:
- Text-to-image generation (POST /v1/images/generations)
- Image editing with single or multiple reference images (POST /v1/images/edits)
- Aspect ratio control (10 ratios)
- Resolution control (1k / 2k)
- Both URL and base64 response handling

Docs: https://docs.x.ai/developers/model-capabilities/images/generation
"""

import base64
import json
import logging
import uuid
from pathlib import Path

import requests

from tools.xai_utils import check_xai_tool_available, resolve_xai_credentials

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "grok-imagine-image"
DEFAULT_TIMEOUT = 120  # image gen can be slow
MAX_REFERENCE_IMAGES = 5
MAX_OUTPUT_IMAGES = 4

VALID_ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "auto",
]
VALID_RESOLUTIONS = ["1k", "2k"]
VALID_MODELS = ["grok-imagine-image", "grok-imagine-image-pro"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_output_dir() -> Path:
    from hermes_constants import get_hermes_home
    output_dir = Path(get_hermes_home()) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_image_ref(ref: str) -> dict:
    """Convert an image reference (URL or local path) to the API format.

    Returns dict with 'url' key (either http(s) URL or data: URI).
    """
    if ref.startswith(("http://", "https://")):
        return {"url": ref, "type": "image_url"}

    # Local file path — read and convert to data URI
    path = Path(ref).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref}")

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"url": f"data:{mime};base64,{b64}", "type": "image_url"}


def _save_b64_image(b64_data: str, output_dir: Path) -> Path:
    """Decode base64 image data and save to disk."""
    img_bytes = base64.b64decode(b64_data)
    local_path = output_dir / f"xai_gen_{uuid.uuid4().hex[:12]}.png"
    local_path.write_bytes(img_bytes)
    return local_path


def _save_url_image(url: str, output_dir: Path) -> Path:
    """Download image from URL and save to disk."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Detect extension from content-type
    ct = resp.headers.get("content-type", "image/png")
    ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
    ext = ext_map.get(ct.split(";")[0].strip(), ".png")

    local_path = output_dir / f"xai_gen_{uuid.uuid4().hex[:12]}{ext}"
    local_path.write_bytes(resp.content)
    return local_path


def _extract_images_from_response(resp_json: dict) -> list[dict]:
    """Extract image data from xAI response.

    Handles both url and b64_json response formats.
    Returns list of dicts with 'url' or 'b64_json' keys.
    """
    images = []
    data = resp_json.get("data", [])
    for item in data:
        if item.get("b64_json"):
            images.append({"b64_json": item["b64_json"]})
        elif item.get("url"):
            images.append({"url": item["url"]})
    return images


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def xai_image_generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = None,
    resolution: str = None,
    image_url: str = None,
    reference_images: list = None,
) -> str:
    """Generate or edit an image using xAI's Grok image generation model.

    Returns JSON string with success status, image path(s), and media_tag.

    Modes:
      - Text-to-image: omit image_url and reference_images
      - Single edit: pass image_url (backward compat) or reference_images=[one_url]
      - Multi-edit: pass reference_images=[url1, url2, ...] (up to 5)

    API routing (per xAI docs + OpenClaw pattern):
      - No reference images → POST /v1/images/generations
      - With reference images → POST /v1/images/edits
        - Single image: body.image = {url, type: "image_url"}
        - Multiple images: body.images = [{url, type}, ...]
    """
    api_key, base_url = resolve_xai_credentials()
    if not api_key:
        return json.dumps({
            "success": False,
            "error": "XAI_API_KEY not configured (env or credential pool)",
        })

    # Normalize reference images
    # Backward compat: image_url → single-item reference_images
    refs = list(reference_images) if reference_images else []
    if image_url and not refs:
        refs = [image_url]

    if len(refs) > MAX_REFERENCE_IMAGES:
        return json.dumps({
            "success": False,
            "error": f"Too many reference images ({len(refs)}), max {MAX_REFERENCE_IMAGES}",
        })

    # Resolve local paths to data URIs
    resolved_refs = []
    for ref in refs:
        try:
            resolved_refs.append(_resolve_image_ref(ref))
        except FileNotFoundError as e:
            return json.dumps({"success": False, "error": str(e)})
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to read image {ref}: {e}"})

    # Build request body
    body = {
        "model": model,
        "prompt": prompt,
        "n": 1,
    }
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if resolution:
        body["resolution"] = resolution

    # Route to correct endpoint
    if resolved_refs:
        endpoint = f"{base_url}/images/edits"
        if len(resolved_refs) == 1:
            body["image"] = resolved_refs[0]
        else:
            body["images"] = resolved_refs
    else:
        endpoint = f"{base_url}/images/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent/1.0",
    }

    try:
        logger.info("xAI image gen → %s (model=%s, refs=%d)", endpoint, model, len(resolved_refs))
        resp = requests.post(
            endpoint,
            headers=headers,
            json=body,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_body = ""
        try:
            error_body = e.response.json()
        except Exception:
            error_body = e.response.text[:500] if e.response else ""
        logger.error("xAI image gen HTTP error %s: %s", e.response.status_code if e.response else "?", error_body)
        return json.dumps({
            "success": False,
            "error": f"xAI API error {e.response.status_code if e.response else '?'}: {error_body}",
        })
    except Exception as e:
        logger.error("xAI image gen failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": str(e)})

    # Parse response
    try:
        resp_json = resp.json()
    except Exception:
        return json.dumps({"success": False, "error": "Invalid JSON response from xAI"})

    images = _extract_images_from_response(resp_json)
    if not images:
        return json.dumps({"success": False, "error": "No images in xAI response"})

    output_dir = _get_output_dir()
    saved_paths = []

    for img in images:
        try:
            if img.get("b64_json"):
                local_path = _save_b64_image(img["b64_json"], output_dir)
            else:
                local_path = _save_url_image(img["url"], output_dir)
            saved_paths.append(str(local_path))
            logger.info("xAI image saved: %s", local_path)
        except Exception as e:
            logger.warning("Failed to save image: %s", e)
            if img.get("url"):
                saved_paths.append(img["url"])

    # Build result
    primary = saved_paths[0] if saved_paths else ""
    media_tag = f"MEDIA:{primary}" if primary else ""

    result = {
        "success": True,
        "image": primary,
        "images": saved_paths,
        "media_tag": media_tag,
        "model": model,
        "mode": "edit" if resolved_refs else "generate",
    }
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error  # noqa: E402

from hermes_constants import display_hermes_home  # noqa: E402

XAI_IMAGE_GENERATE_SCHEMA = {
    "name": "xai_image_generate",
    "description": (
        "Generate or edit images using xAI's Grok image model. "
        "Supports text-to-image, single-image editing, and multi-image editing "
        "(up to 5 reference images). Reference images can be URLs or local file paths. "
        f"Returns generated image file(s) in {display_hermes_home()}/images/. "
        "Use this when XAI_API_KEY is available."
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
                "description": "Model: 'grok-imagine-image' ($0.02/img) or 'grok-imagine-image-pro' ($0.07/img, higher quality).",
                "enum": VALID_MODELS,
                "default": DEFAULT_MODEL,
            },
            "aspect_ratio": {
                "type": "string",
                "description": f"Aspect ratio: {', '.join(VALID_ASPECT_RATIOS)}.",
                "enum": VALID_ASPECT_RATIOS,
            },
            "resolution": {
                "type": "string",
                "description": "Output resolution: '1k' or '2k'.",
                "enum": VALID_RESOLUTIONS,
            },
            "image_url": {
                "type": "string",
                "description": "Single source image for editing (URL or local path). For multiple images use reference_images instead.",
            },
            "reference_images": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_REFERENCE_IMAGES,
                "description": (
                    "List of reference images for editing (URLs or local file paths, up to 5). "
                    "When provided, uses /v1/images/edits endpoint. "
                    "Local files are auto-converted to data URIs."
                ),
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
        model=args.get("model", DEFAULT_MODEL),
        aspect_ratio=args.get("aspect_ratio"),
        resolution=args.get("resolution"),
        image_url=args.get("image_url"),
        reference_images=args.get("reference_images"),
    )


registry.register(
    name="xai_image_generate",
    toolset="image_gen",
    schema=XAI_IMAGE_GENERATE_SCHEMA,
    handler=_handle_xai_image_generate,
    check_fn=lambda: check_xai_tool_available("xai_image_generate"),
    requires_env=[],
    is_async=False,
    emoji="🎨",
)
