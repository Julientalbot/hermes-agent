"""The existing Hermes screen routers assembled without dashboard or RPC routes."""
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def create_screen_app():
    from gateway.screen_handoff_config import public_url, public_path
    from hermes_cli.web_routers.screen_handoff import router, _novnc_root
    from hermes_cli.web_routers.display import router as display_router
    url = public_url()
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    home = get_hermes_home()
    if not url:
        raise ValueError("Configure bot_desktop.handoff.enabled and its HTTPS public_url")

    @asynccontextmanager
    async def lifespan(app):
        from gateway.screen_handoff import ScreenHandoffStore
        ScreenHandoffStore()
        yield

    app = FastAPI(root_path=public_path(), docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.screen_only = True

    class ProfileScope:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            token = set_hermes_home_override(home)
            try:
                await self.app(scope, receive, send)
            finally:
                reset_hermes_home_override(token)

    app.add_middleware(ProfileScope)

    @app.middleware("http")
    async def private_surface(request, call_next):
        if request.url.path.endswith("/health"):
            return await call_next(request)
        if request.headers.get("host") != urlsplit(url).netloc:
            return JSONResponse({"error": "Host refused"}, status_code=403)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health")
    async def health():
        # Reachability only: no screen start, control change or model call.
        from tools.bot_desktop.runtime import status
        ready = bool(status().installed and _novnc_root())
        return JSONResponse({"screen_service_ready": ready, "public_url": url}, status_code=200 if ready else 503)

    app.include_router(router)
    app.include_router(display_router)
    return app


def start_screen_server(host: str, port: int):
    import uvicorn
    # Access URLs contain short-lived invitation/ticket values; never log them.
    uvicorn.run(create_screen_app(), host=host, port=port, access_log=False,
                proxy_headers=False, log_level="warning")
