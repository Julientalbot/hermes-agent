"""Private screen access; authorization happens in the originating messenger."""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from gateway.screen_handoff_config import allowed_origin, public_path, telegram_login_client_id

router = APIRouter()
_COOKIE = "hermes_screen_session"
_LOGIN_COOKIE = "hermes_screen_login"


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


def _page(request_id, invite="", *, protocol=2, login_cookie=""):
    nonce = secrets.token_urlsafe(24)
    from gateway.screen_handoff_login import login_nonce
    login = protocol == 3
    values = json.dumps({"request": request_id, "invite": invite, "prefix": public_path(),
                        "telegram": login, "clientId": telegram_login_client_id() if login else "",
                        "nonce": login_nonce(login_cookie, request_id, _store().profile_home) if login_cookie else ""}).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Connexion assistée — Hermes</title>
<style>*{box-sizing:border-box}body{margin:0;background:#111827;color:#f9fafb;font:15px system-ui}
main{height:100vh;height:100dvh;display:flex;flex-direction:column}
header{display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:6px 10px;min-height:44px}
#status{margin:0;flex:1}button{font:inherit;padding:6px 12px;border-radius:6px;border:0;background:#38bdf8;color:#082f49;cursor:pointer}
button:focus-visible{outline:3px solid white;outline-offset:2px}#code{font-size:2rem;letter-spacing:.3em;margin:0}#code:empty{display:none}
#screen{flex:1;min-height:0;min-width:0;background:#000;overflow:hidden}button:disabled{opacity:.5;cursor:wait}
</style></head><body><main>
<header><p id="status" role="status">Connectez-vous pour accéder au navigateur de votre agent.</p>
<button id="login" hidden>Continuer avec Telegram</button>
<button id="ask">Demander l’autorisation dans la conversation privée</button>
<button id="take" hidden>Prendre la main</button><button id="give" hidden>Terminé</button></header>
<p id="code"></p><div id="screen" aria-label="Navigateur distant"></div>
</main>TELEGRAM_SDK<script nonce="NONCE" type="module">
const cfg=CONFIG, base=cfg.prefix+'/screen-handoff', path=base+'/r/'+encodeURIComponent(cfg.request);
const el=id=>document.getElementById(id);let rfb=null,returned=false;
const say=text=>{el('status').textContent=text};
async function post(url,body){const r=await fetch(url,{method:'POST',credentials:'same-origin',headers:{'X-Hermes-Screen':'1','Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const d=await r.json();if(!r.ok)throw Error(d.error||'Accès indisponible');return d}
async function status(){try{const r=await fetch(path+'/status',{credentials:'same-origin'});const d=await r.json();
if(d.state==='authorized'||d.state==='human'){el('ask').hidden=true;el('login').hidden=true;el('code').textContent='';el('take').hidden=!!rfb;if(!rfb)say('Accès autorisé. Retrouvez votre navigateur.');}
else if(d.state==='waiting'){el('code').textContent=d.code;say('Confirmez ce même code dans votre conversation privée.');}
else if(['returned','queued','resuming','resumed','returning','needs_attention'].includes(d.state)){returned=true;if(rfb){rfb.disconnect();rfb=null}el('take').hidden=true;el('give').hidden=true;say(d.state==='needs_attention'?'Contrôle rendu. Reprise incertaine : vérifiez la conversation avant de continuer.':d.state==='returning'?'Restitution en cours.':'Contrôle rendu. Hermes va réobserver le navigateur avant de poursuivre.');}
else if(['expired','deny','superseded','unauthorized'].includes(d.state)&&!cfg.invite){if(rfb){rfb.disconnect();rfb=null}el('take').hidden=true;el('give').hidden=true;say('Accès expiré. Demandez à votre agent de vous renvoyer le lien dans Telegram.');}
}catch{}}
el('ask').onclick=async()=>{try{const d=await post(base+'/'+encodeURIComponent(cfg.invite)+'/challenge');el('code').textContent=d.code;el('ask').hidden=true;cfg.invite='';history.replaceState(null,'',path);await status()}catch(e){say(e.message)}};
async function connect(){try{const d=await post(path+'/takeover');const RFB=(await import(base+'/assets/novnc/core/rfb.js')).default;
if(rfb)rfb.disconnect();rfb=new RFB(el('screen'),(location.protocol==='https:'?'wss://':'ws://')+location.host+cfg.prefix+'/api/display/ws?display_ticket='+encodeURIComponent(d.display_ticket));
rfb.scaleViewport=true;rfb.resizeSession=false;
rfb.addEventListener('connect',()=>{el('take').hidden=true;el('give').hidden=false;say('Vous avez la main');});
rfb.addEventListener('disconnect',()=>{rfb=null;if(!returned){el('take').hidden=false;say('Connexion interrompue. Vous gardez la main. Reconnectez-vous ou demandez un nouveau lien dans Telegram.');}});
}catch(e){say(e.message)}};
el('take').onclick=connect;
el('login').onclick=()=>{
if(!window.Telegram?.Login){say('Connexion Telegram indisponible. Réessayez plus tard.');return}
el('login').disabled=true;
// Start the popup synchronously in the user gesture; journal the attempt in parallel.
const begun=post(base+'/'+encodeURIComponent(cfg.invite)+'/telegram/begin').then(d=>({ok:true,data:d}),e=>({ok:false,error:e}));
try{window.Telegram.Login.auth({client_id:Number(cfg.clientId),scope:['profile'],nonce:cfg.nonce,lang:'fr'},async result=>{
try{const attempt=await begun;if(!attempt.ok)throw attempt.error;
if(!result||result.error||!result.id_token)throw Error('Connexion non autorisée. Vous pouvez réessayer.');
await post(path+'/telegram/complete',{id_token:result.id_token,nonce:cfg.nonce});
cfg.invite='';history.replaceState(null,'',path);el('login').hidden=true;await connect();
}catch(e){say(e.message)}finally{el('login').disabled=false}});
}catch{el('login').disabled=false;say('Connexion Telegram indisponible. Autorisez la fenêtre de connexion puis réessayez.')}
};
el('give').onclick=async()=>{try{await post(path+'/return');returned=true;if(rfb){rfb.disconnect();rfb=null}await status()}catch(e){say(e.message)}};
if(!cfg.invite||cfg.telegram)el('ask').hidden=true;
if(cfg.telegram&&cfg.invite){el('login').hidden=false;if(!cfg.clientId){el('login').disabled=true;say('Connexion Telegram non configurée. Prévenez votre agent.')}}
await status();setInterval(status,2000);
</script></body></html>'''.replace("TELEGRAM_SDK", '<script src="https://oauth.telegram.org/js/telegram-login.js?6"></script>' if login else '').replace("NONCE", nonce).replace("CONFIG", values)
    sdk_policy = " https://oauth.telegram.org" if login else ""
    return HTMLResponse(page, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
        "X-Frame-Options": "DENY", "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
        "Content-Security-Policy": f"default-src 'self'; img-src 'self' data:; script-src 'self' 'nonce-{nonce}'{sdk_policy}; connect-src 'self'{sdk_policy}; frame-src 'self'{sdk_policy}; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})


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
    row = _store().get(request_id)
    return _page(request_id, protocol=row.protocol) if row and status["state"] != "unauthorized" else _error(401, "Demandez un nouveau lien à votre agent dans votre conversation privée.")


@router.get("/screen-handoff/{token}")
async def invitation(token: str):
    row = _store().by_token(token)
    if not row:
        return _error(410, "Invitation expirée ou invalide. Demandez un nouveau lien à votre agent.")
    cookie = secrets.token_urlsafe(32) if row.protocol == 3 else ""
    response = _page(row.request_id, token, protocol=row.protocol, login_cookie=cookie)
    if cookie:
        response.set_cookie(_LOGIN_COOKIE, cookie, max_age=10*60, secure=True, httponly=True,
                            samesite="strict", path=public_path()+"/screen-handoff")
    return response


@router.post("/screen-handoff/{token}/telegram/begin")
async def telegram_begin(token: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    if not telegram_login_client_id():
        return _error(503, "Connexion Telegram non configurée.")
    from gateway.screen_handoff_login import begin_login
    value = begin_login(_store(), token, request.cookies.get(_LOGIN_COOKIE, ""))
    return JSONResponse(value, headers={"Cache-Control": "no-store"}) if value else _error(409, "Accès expiré ou indisponible. Demandez un nouveau lien dans Telegram.")


@router.post("/screen-handoff/r/{request_id}/telegram/complete")
async def telegram_complete(request_id: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    import jwt
    from starlette.concurrency import run_in_threadpool
    from gateway.screen_handoff_login import complete_login
    cookie = request.cookies.get(_LOGIN_COOKIE, "")
    try:
        raw = await request.body()
        if len(raw) > 20000:
            return _error(413, "Réponse Telegram invalide.")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("nonce"), str) or len(data["nonce"]) > 128:
            return _error(400, "Réponse Telegram invalide.")
        accepted = await run_in_threadpool(complete_login, _store(), request_id, cookie,
                                          data["nonce"], data.get("id_token"), client_id=telegram_login_client_id())
    except (ValueError, jwt.PyJWTError) as exc:
        # Never log tokens, claims, cookies or exception text. The class is enough
        # to distinguish provider transport, signatures and claim validation.
        code = type(exc).__name__ if isinstance(exc, jwt.PyJWTError) else "InvalidLoginValue"
        logging.getLogger(__name__).warning("screen_telegram_login_rejected code=%s", code)
        return _error(401, "Connexion Telegram non validée (" + code + "). L’accès reste protégé.")
    if not accepted:
        return _error(401, "Cette autorisation ne correspond pas à cette demande et à ce navigateur.")
    response = JSONResponse({"state": "authorized"}, headers={"Cache-Control": "no-store"})
    response.set_cookie(_COOKIE, cookie, max_age=30*60, secure=True, httponly=True,
                        samesite="strict", path=public_path()+"/screen-handoff")
    response.delete_cookie(_LOGIN_COOKIE, path=public_path()+"/screen-handoff", secure=True, httponly=True, samesite="strict")
    return response


@router.post("/screen-handoff/{token}/challenge")
async def challenge(token: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    value = _store().challenge(token, request.cookies.get(_COOKIE, ""))
    if not value:
        return _error(409, "Demande indisponible. Demandez un nouveau lien dans votre conversation privée.")
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
    # Serialize acquisition against a web/Telegram return in another process.
    # If return intent won after take_over(), never reacquire its released lease.
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT state,viewer_id FROM screen_handoffs WHERE request_id=?",
                               (row.request_id,)).fetchone()
        if not current or current["state"] != "human" or current["viewer_id"] != viewer:
            return _error(409, "Le contrôle a déjà été rendu.")
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
    from gateway.screen_handoff_return import return_control as settle_return
    try:
        return JSONResponse(settle_return(store, row, _viewer_id(cookie)))
    except ValueError as exc:
        return _error(409, str(exc))


@router.post("/screen-handoff/r/{request_id}/revoke")
async def revoke(request_id: str, request: Request):
    if not _mutation_allowed(request):
        return _error(403, "Origine refusée.")
    store, row, _ = _session(request, request_id, returned=True)
    if not row:
        return _error(401, "Autorisation expirée.")
    store.revoke(request_id)
    return JSONResponse({"state": "revoked"})
