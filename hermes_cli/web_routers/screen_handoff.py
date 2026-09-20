"""Private screen access; authorization happens in the originating messenger."""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from gateway.screen_handoff_config import allowed_origin, public_path

router = APIRouter()
_COOKIE = "hermes_screen_session"


def _store():
    from gateway.screen_handoff import ScreenHandoffStore
    return ScreenHandoffStore()


def _viewer_id(cookie):
    return "screen-" + hashlib.sha256(cookie.encode()).hexdigest()[:32]


def _error(status, message):
    return JSONResponse({"error": message}, status_code=status, headers={"Cache-Control": "no-store"})


def _mutation_allowed(request):
    return allowed_origin(request.headers.get("origin", "")) and request.headers.get("x-hermes-screen") == "1"


def _session(request, request_id, *, returned=False):
    cookie = request.cookies.get(_COOKIE, "")
    store = _store()
    row = store.any_web_session(cookie) if returned else store.web_session(cookie)
    return store, row if row and row.request_id == request_id else None, cookie


def _novnc_root():
    for root in (Path("/opt/hermes-screen/node_modules/@novnc/novnc"),
                 Path(__file__).resolve().parents[2] / "apps/desktop/node_modules/@novnc/novnc"):
        if (root / "core/rfb.js").is_file():
            return root
    return None


def _page(request_id, invite=""):
    nonce = secrets.token_urlsafe(24)
    values = json.dumps({"request": request_id, "invite": invite, "prefix": public_path()}).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Reprendre le navigateur Hermes</title>
<style>*{box-sizing:border-box}body{margin:0;background:#111827;color:#f9fafb;font:15px system-ui}
main{height:100vh;height:100dvh;display:flex;flex-direction:column;padding:12px;gap:10px}
header{display:flex;align-items:center;flex-wrap:wrap;gap:8px 16px}h1{font-size:20px;margin:0}
#status{margin:0;flex:1;min-width:220px}button{font:inherit;padding:10px 14px;border-radius:8px;border:0;background:#38bdf8;color:#082f49;cursor:pointer}
button:focus-visible{outline:3px solid white;outline-offset:2px}#code{font-size:2rem;letter-spacing:.3em;margin:0}#code:empty{display:none}
#screen{flex:1;min-height:0;min-width:0;background:#000;overflow:hidden}footer{font-size:13px;color:#cbd5e1}footer p{margin:0}
</style></head><body><main>
<header><h1>Reprendre le navigateur</h1><p id="status" role="status">L’ouverture de cette page ne prend pas le contrôle.</p>
<button id="ask">Demander l’autorisation dans la conversation privée</button>
<button id="take" hidden>Prendre la main</button><button id="give" hidden>Rendre la main et continuer</button></header>
<p id="code"></p><div id="screen" aria-label="Navigateur distant"></div>
<footer><p>Cette version s’utilise sur ordinateur. Saisissez vos accès dans le navigateur affiché ; ils ne passent pas dans la conversation.
Le navigateur fonctionne sur la VM de votre agent, sans isolation contre les processus de cette VM.</p></footer>
</main><script nonce="NONCE" type="module">
const cfg=CONFIG, base=cfg.prefix+'/screen-handoff', path=base+'/r/'+encodeURIComponent(cfg.request);
const el=id=>document.getElementById(id);let rfb=null,returned=false;
const say=text=>{el('status').textContent=text};
async function post(url){const r=await fetch(url,{method:'POST',credentials:'same-origin',headers:{'X-Hermes-Screen':'1'}});const d=await r.json();if(!r.ok)throw Error(d.error||'Accès indisponible');return d}
async function status(){try{const r=await fetch(path+'/status',{credentials:'same-origin'});const d=await r.json();
if(d.state==='authorized'||d.state==='human'){el('ask').hidden=true;el('code').textContent='';el('take').hidden=!!rfb;if(!rfb)say('Autorisation confirmée. Vous pouvez prendre la main.');}
else if(d.state==='waiting'){el('code').textContent=d.code;say('Confirmez ce même code dans votre conversation privée.');}
else if(['returned','queued','resuming','resumed','returning','needs_attention'].includes(d.state)){returned=true;if(rfb){rfb.disconnect();rfb=null}el('take').hidden=true;el('give').hidden=true;say(d.state==='needs_attention'?'Contrôle rendu. Reprise incertaine : vérifiez la conversation avant de continuer.':d.state==='returning'?'Restitution en cours.':'Contrôle rendu. Hermes va réobserver le navigateur avant de poursuivre.');}
else if(['expired','deny','superseded','unauthorized'].includes(d.state)&&!cfg.invite){if(rfb){rfb.disconnect();rfb=null}el('take').hidden=true;el('give').hidden=true;say('Accès terminé. Utilisez /screen dans votre conversation privée pour récupérer la reprise.');}
}catch{}}
el('ask').onclick=async()=>{try{const d=await post(base+'/'+encodeURIComponent(cfg.invite)+'/challenge');el('code').textContent=d.code;el('ask').hidden=true;cfg.invite='';history.replaceState(null,'',path);await status()}catch(e){say(e.message)}};
el('take').onclick=async()=>{try{const d=await post(path+'/takeover');const RFB=(await import(base+'/assets/novnc/core/rfb.js')).default;
if(rfb)rfb.disconnect();rfb=new RFB(el('screen'),(location.protocol==='https:'?'wss://':'ws://')+location.host+cfg.prefix+'/api/display/ws?display_ticket='+encodeURIComponent(d.display_ticket));
rfb.scaleViewport=true;rfb.resizeSession=false;
rfb.addEventListener('connect',()=>{el('take').hidden=true;el('give').hidden=false;say('Contrôle acquis. Connectez-vous dans ce navigateur.');});
rfb.addEventListener('disconnect',()=>{rfb=null;if(!returned){el('take').hidden=false;say('Connexion interrompue. Le contrôle reste humain. Reconnectez-vous ou utilisez /screen.');}});
}catch(e){say(e.message)}};
el('give').onclick=async()=>{try{await post(path+'/return');returned=true;if(rfb){rfb.disconnect();rfb=null}await status()}catch(e){say(e.message)}};
if(!cfg.invite)el('ask').hidden=true;await status();setInterval(status,2000);
</script></body></html>'''.replace("NONCE", nonce).replace("CONFIG", values)
    return HTMLResponse(page, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
        "X-Frame-Options": "DENY", "Content-Security-Policy": f"default-src 'self'; img-src 'self' data:; script-src 'self' 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})


@router.get("/screen-handoff/assets/novnc/{asset_path:path}")
async def asset(asset_path: str):
    root = _novnc_root()
    if not root or not asset_path.endswith(".js"):
        return Response(status_code=404)
    file = (root / asset_path).resolve()
    if root.resolve() not in file.parents or not file.is_file():
        return Response(status_code=404)
    return FileResponse(file, media_type="text/javascript")


@router.get("/screen-handoff/r/{request_id}")
async def reopen(request_id: str, request: Request):
    status = _store().access_status(request_id, request.cookies.get(_COOKIE, ""))
    return _page(request_id) if status["state"] != "unauthorized" else _error(401, "Utilisez /screen dans votre conversation privée.")


@router.get("/screen-handoff/{token}")
async def invitation(token: str):
    row = _store().by_token(token)
    return _page(row.request_id, token) if row else _error(410, "Invitation expirée ou invalide.")


@router.post("/screen-handoff/{token}/challenge")
async def challenge(token: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    value = _store().challenge(token, request.cookies.get(_COOKIE, ""))
    if not value:
        return _error(409, "Demande indisponible. Utilisez /screen pour renouveler l’accès.")
    response = JSONResponse({k: value[k] for k in ("code", "expires")}, headers={"Cache-Control": "no-store"})
    response.set_cookie(_COOKIE, value["cookie"], max_age=32*60, secure=True, httponly=True,
                        samesite="strict", path=public_path()+"/screen-handoff")
    return response


@router.get("/screen-handoff/r/{request_id}/status")
async def status(request_id: str, request: Request):
    return JSONResponse(_store().access_status(request_id, request.cookies.get(_COOKIE, "")), headers={"Cache-Control": "no-store"})


@router.post("/screen-handoff/r/{request_id}/takeover")
async def takeover(request_id: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    store, row, cookie = _session(request, request_id)
    if not row:
        return _error(401, "Autorisation expirée.")
    from gateway.drain_control import drain_requested
    if drain_requested():
        return _error(409, "Maintenance en cours. Réessayez après sa fin.")
    from tools.bot_desktop import lease, runtime
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    if not runtime.status().running:
        return _error(409, "Le bureau est arrêté. Relancez le parcours depuis la conversation.")
    viewer = _viewer_id(cookie)
    # Persist ownership before acquiring; an interrupted acquisition uses the same viewer.
    if not store.take_over(cookie, viewer):
        return _error(409, "Reprise indisponible.")
    lease.acquire(viewer, profile_key=row.profile_home, reason=row.reason)
    ticket = mint_ticket(user_id=viewer, provider="bot-desktop-handoff", extra={
        "hermes_home": row.profile_home, "viewer_id": viewer, "handoff_id": row.request_id,
        "retain_on_disconnect": True})
    return JSONResponse({"display_ticket": ticket}, headers={"Cache-Control": "no-store"})


@router.post("/screen-handoff/r/{request_id}/return")
async def return_control(request_id: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    store, row, cookie = _session(request, request_id, returned=True)
    if not row:
        return _error(401, "Autorisation expirée.")
    if row.state in {"returned", "queued", "resuming", "resumed", "needs_attention"}:
        return JSONResponse({"state": row.state})
    from tools.bot_desktop import lease
    viewer = _viewer_id(cookie)
    if row.state != "returning" and not lease.viewer_may_send_input(viewer, profile_key=row.profile_home):
        return _error(409, "Ce navigateur ne détient pas le contrôle.")
    if not store.return_to_agent(cookie, viewer):
        return _error(409, "Restitution indisponible.")
    released = lease.release(viewer, profile_key=row.profile_home)
    if released.holder != lease.AGENT:
        return _error(409, "Un autre navigateur détient le contrôle. Restitution en attente.")
    store.complete_return(request_id)
    return JSONResponse({"state": "returned"})


@router.post("/screen-handoff/r/{request_id}/revoke")
async def revoke(request_id: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    store, row, _ = _session(request, request_id, returned=True)
    if not row:
        return _error(401, "Autorisation expirée.")
    store.revoke(request_id)
    return JSONResponse({"state": "revoked"})
