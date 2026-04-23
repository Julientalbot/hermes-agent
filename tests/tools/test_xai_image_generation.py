"""Tests for xAI Grok image generation tool."""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("XAI_BASE_URL", "https://api.x.ai/v1")


@pytest.fixture()
def mock_post():
    with patch("tools.xai_image_generation_tool.requests.post") as m:
        yield m


@pytest.fixture()
def mock_output_dir(tmp_path):
    with patch("tools.xai_image_generation_tool._get_output_dir", return_value=tmp_path):
        yield tmp_path


def _make_image_response(url=None, b64_json=None):
    """Build a mock xAI image API response."""
    item = {}
    if url:
        item["url"] = url
    if b64_json:
        item["b64_json"] = b64_json
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [item]}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Test: missing API key
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    def test_returns_error_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(prompt="a cat"))
        assert "error" in result
        assert "XAI_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# Test: text-to-image generation (no references)
# ---------------------------------------------------------------------------

class TestTextToImage:
    def test_successful_generation_url(self, mock_post, mock_output_dir):
        mock_post.return_value = _make_image_response(
            url="https://cdn.x.ai/generated/image.png"
        )
        # Mock the download
        with patch("tools.xai_image_generation_tool.requests.get") as mock_get:
            mock_dl = MagicMock()
            mock_dl.content = b"\x89PNG fake image data"
            mock_dl.headers = {"content-type": "image/png"}
            mock_get.return_value = mock_dl

            from tools.xai_image_generation_tool import xai_image_generate
            result = json.loads(xai_image_generate(prompt="a cat"))

        assert result["success"] is True
        assert result["mode"] == "generate"
        assert "media_tag" in result
        assert result["media_tag"].startswith("MEDIA:")
        assert Path(result["image"]).exists()

    def test_successful_generation_b64(self, mock_post, mock_output_dir):
        fake_b64 = base64.b64encode(b"\x89PNG fake").decode()
        mock_post.return_value = _make_image_response(b64_json=fake_b64)

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(prompt="a dog"))

        assert result["success"] is True
        assert result["mode"] == "generate"
        assert Path(result["image"]).exists()

    def test_calls_generations_endpoint(self, mock_post, mock_output_dir):
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        xai_image_generate(prompt="sunset")

        call_args = mock_post.call_args
        assert "/images/generations" in call_args[0][0]

    def test_no_reference_no_image_url(self, mock_post, mock_output_dir):
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        xai_image_generate(prompt="test")

        body = mock_post.call_args[1]["json"]
        assert "image" not in body
        assert "images" not in body


# ---------------------------------------------------------------------------
# Test: single image edit (backward compat via image_url)
# ---------------------------------------------------------------------------

class TestSingleImageEdit:
    def test_image_url_routes_to_edits(self, mock_post, mock_output_dir):
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="make it blue",
            image_url="https://example.com/photo.jpg",
        ))

        assert result["success"] is True
        assert result["mode"] == "edit"
        call_args = mock_post.call_args
        assert "/images/edits" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["image"]["url"] == "https://example.com/photo.jpg"
        assert body["image"]["type"] == "image_url"

    def test_local_file_converted_to_data_uri(self, mock_post, mock_output_dir, tmp_path):
        img_file = tmp_path / "source.png"
        img_file.write_bytes(b"\x89PNG fake")
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="edit this",
            image_url=str(img_file),
        ))

        assert result["success"] is True
        body = mock_post.call_args[1]["json"]
        assert body["image"]["url"].startswith("data:image/png;base64,")

    def test_missing_local_file_returns_error(self, mock_post, mock_output_dir):
        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="edit",
            image_url="/nonexistent/image.png",
        ))
        assert "error" in result
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# Test: multiple reference images
# ---------------------------------------------------------------------------

class TestMultipleReferenceImages:
    def test_two_refs_uses_images_array(self, mock_post, mock_output_dir):
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="combine these",
            reference_images=[
                "https://example.com/a.jpg",
                "https://example.com/b.jpg",
            ],
        ))

        assert result["success"] is True
        assert result["mode"] == "edit"
        body = mock_post.call_args[1]["json"]
        assert "images" in body
        assert len(body["images"]) == 2
        assert "image" not in body  # singular not set when multiple

    def test_single_ref_uses_image_singular(self, mock_post, mock_output_dir):
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        xai_image_generate(
            prompt="edit",
            reference_images=["https://example.com/a.jpg"],
        )

        body = mock_post.call_args[1]["json"]
        assert "image" in body
        assert "images" not in body

    def test_too_many_refs_returns_error(self, mock_post, mock_output_dir):
        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="too many",
            reference_images=[f"https://example.com/{i}.jpg" for i in range(6)],
        ))
        assert "error" in result
        assert "Too many" in result["error"]

    def test_mixed_local_and_remote_refs(self, mock_post, mock_output_dir, tmp_path):
        local_img = tmp_path / "local.jpg"
        local_img.write_bytes(b"\xff\xd8\xff fake jpeg")
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="combine",
            reference_images=[
                str(local_img),
                "https://example.com/remote.png",
            ],
        ))

        assert result["success"] is True
        body = mock_post.call_args[1]["json"]
        assert len(body["images"]) == 2
        assert body["images"][0]["url"].startswith("data:image/jpeg;base64,")
        assert body["images"][1]["url"] == "https://example.com/remote.png"


# ---------------------------------------------------------------------------
# Test: backward compat — image_url still works
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_image_url_not_ignored_when_refs_present(self, mock_post, mock_output_dir):
        """When reference_images is provided, image_url is ignored."""
        mock_post.return_value = _make_image_response(b64_json=base64.b64encode(b"x").decode())

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(
            prompt="test",
            image_url="https://example.com/ignored.jpg",
            reference_images=["https://example.com/used.jpg"],
        ))

        assert result["success"] is True
        body = mock_post.call_args[1]["json"]
        assert body["image"]["url"] == "https://example.com/used.jpg"


# ---------------------------------------------------------------------------
# Test: HTTP errors
# ---------------------------------------------------------------------------

class TestHttpErrors:
    def test_4xx_returns_error(self, mock_post, mock_output_dir):
        from requests.exceptions import HTTPError
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "Invalid prompt"}
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)
        mock_post.return_value = mock_resp

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(prompt="bad"))
        assert "error" in result
        assert "400" in result["error"]

    def test_no_images_in_response(self, mock_post, mock_output_dir):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": []}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        from tools.xai_image_generation_tool import xai_image_generate
        result = json.loads(xai_image_generate(prompt="empty"))
        assert "error" in result
        assert "No images" in result["error"]


# ---------------------------------------------------------------------------
# Test: config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_model(self):
        from tools.xai_image_generation_tool import DEFAULT_MODEL
        assert DEFAULT_MODEL == "grok-imagine-image"

    def test_max_reference_images(self):
        from tools.xai_image_generation_tool import MAX_REFERENCE_IMAGES
        assert MAX_REFERENCE_IMAGES == 5

    def test_valid_aspect_ratios(self):
        from tools.xai_image_generation_tool import VALID_ASPECT_RATIOS
        assert "16:9" in VALID_ASPECT_RATIOS
        assert "auto" in VALID_ASPECT_RATIOS


# ---------------------------------------------------------------------------
# Test: schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_has_reference_images(self):
        from tools.xai_image_generation_tool import XAI_IMAGE_GENERATE_SCHEMA
        props = XAI_IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        assert "reference_images" in props
        assert props["reference_images"]["type"] == "array"

    def test_schema_has_image_url(self):
        from tools.xai_image_generation_tool import XAI_IMAGE_GENERATE_SCHEMA
        props = XAI_IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        assert "image_url" in props
