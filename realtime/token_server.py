import asyncio
import base64
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from livekit import api
from model_options import LLM_MODELS, STT_MODELS, TTS_MODELS, validate_models

load_dotenv(".env.local")
load_dotenv(".env")

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "study_config.json"

class Study:

    def __init__(self):
        self.page = "session"
        self.condition = "empathetic"
        self.session_no = 0
        self.rev = 0
        self.subscribers: set[asyncio.Queue] = set()
        self.sessions: dict[str, dict] = {}

    def snapshot(self, lease: str | None = None) -> dict:
        session = self.sessions.get(lease or "")
        return {
            "page": self.page,
            "condition": self.condition,
            "session_no": session["session_no"] if session else 0,
            "room": session["room"] if session else None,
            "busy": False,
            **(session["models"] if session else validate_models({})),
            "rev": self.rev,
        }

    async def _broadcast(self):
        self.rev += 1
        snap = self.snapshot()
        for q in list(self.subscribers):
            await q.put(snap)

    async def set_page(self, page: str):
        self.page = page
        await self._broadcast()

    async def start_session(self, config: dict):
        lease = config.pop("_lease", None)
        self.condition = "empathetic"
        models = validate_models(config)
        self.session_no += 1
        lease = lease or uuid.uuid4().hex
        payload = base64.urlsafe_b64encode(
            json.dumps(models, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        room = f"study-emp-{self.session_no}-{uuid.uuid4().hex[:6]}.{payload}"
        self.sessions[lease] = {
            "session_no": self.session_no,
            "room": room,
            "models": models,
        }
        self.page = "session"
        await self._broadcast()
        return lease

    async def end_session(self, room: str | None, lease: str | None):
        session = self.sessions.get(lease or "")
        if not session or session["room"] != room:
            return False
        del self.sessions[lease]
        await self._broadcast()
        return True

study = Study()
app = FastAPI()

@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")

@app.get("/state")
def state():
    return JSONResponse(study.snapshot())

@app.get("/models")
def models():
    return JSONResponse({"llm": LLM_MODELS, "tts": TTS_MODELS, "stt": STT_MODELS})

@app.get("/events")
async def events(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    study.subscribers.add(q)
    await q.put(study.snapshot())

    async def gen():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                if await request.is_disconnected():
                    break
        finally:
            study.subscribers.discard(q)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"})

@app.get("/token")
def token(request: Request):
    lease = request.cookies.get("study_lease")
    session = study.sessions.get(lease or "")
    if study.page != "session" or not session:
        return JSONResponse({"error": "no active session"}, status_code=409)
    room = session["room"]
    identity = f"user-{uuid.uuid4().hex[:6]}"
    jwt = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name("participant")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return JSONResponse({"url": os.environ["LIVEKIT_URL"], "token": jwt,
                         "room": room, "condition": study.condition})

@app.post("/control/page")
async def control_page(body: dict):
    page = body.get("page", "welcome")
    if page != "session":
        return JSONResponse({"error": f"unknown page {page!r}"}, status_code=400)
    await study.set_page(page)
    return study.snapshot()

@app.post("/control/session")
async def control_session(body: dict, request: Request, response: Response):
    try:
        config = {**body, "_lease": request.cookies.get("study_lease")}
        lease = await study.start_session(config)
        response.set_cookie("study_lease", lease, httponly=True, samesite="lax")
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return study.snapshot(lease)

@app.post("/control/session/end")
async def control_session_end(body: dict, request: Request):
    lease = request.cookies.get("study_lease")
    await study.end_session(body.get("room"), lease)
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
