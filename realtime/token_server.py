import asyncio
import base64
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
        self.room: str | None = None
        self.rev = 0
        self.subscribers: set[asyncio.Queue] = set()
        self.models = validate_models({})

    def snapshot(self) -> dict:
        return {
            "page": self.page,
            "condition": self.condition,
            "session_no": self.session_no,
            "room": self.room,
            **self.models,
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
        condition = config.get("condition", "empathetic")
        if condition not in {"empathetic", "control"}:
            raise ValueError("condition must be 'empathetic' or 'control'")
        self.condition = condition
        self.models = validate_models(config)
        self.session_no += 1
        payload = base64.urlsafe_b64encode(
            json.dumps(self.models, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        code = "emp" if condition == "empathetic" else "ctl"
        self.room = f"study-{code}-{self.session_no}-{uuid.uuid4().hex[:6]}.{payload}"
        self.page = "session"
        await self._broadcast()

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
def token():
    if study.page != "session" or not study.room:
        return JSONResponse({"error": "no active session"}, status_code=409)
    room = study.room
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
async def control_session(body: dict):
    try:
        await study.start_session(body)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return study.snapshot()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
