"""Read-only, profile-scoped configuration for the optional screen web surface."""
from urllib.parse import urlsplit


def public_url() -> str:
    from hermes_cli.config import load_config_readonly
    cfg = (load_config_readonly().get("bot_desktop") or {}).get("handoff") or {}
    if cfg.get("enabled") is not True:
        return ""
    value = str(cfg.get("public_url") or "").rstrip("/")
    url = urlsplit(value)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        return ""
    if any(part in (".", "..") for part in url.path.split("/")):
        return ""
    return value


def public_path() -> str:
    return urlsplit(public_url()).path.rstrip("/")


def allowed_origin(origin: str) -> bool:
    url = urlsplit(public_url())
    return bool(url.netloc and origin == f"{url.scheme}://{url.netloc}")


def service_ready() -> bool:
    """Probe only the configured local screen server; never start the desktop."""
    import json
    import urllib.request
    from hermes_cli.config import load_config_readonly
    cfg = (load_config_readonly().get("bot_desktop") or {}).get("handoff") or {}
    port = cfg.get("local_port", 8766)
    if type(port) is not int or not 1 <= port <= 65535 or not public_url():
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
            data = json.loads(response.read(1024))
        return data.get("screen_service_ready") is True and data.get("public_url") == public_url()
    except (OSError, ValueError):
        return False
