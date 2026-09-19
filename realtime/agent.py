import asyncio
import base64
import datetime
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import librosa
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent, AgentServer, AgentSession, StopResponse, inference, stt as stt_lib,
)
from livekit.plugins import silero

from emotion_engine import EmotionEngine
from prompt import (
    GREETING_CONTROL, GREETING_EMPATHETIC,
    INSTRUCTIONS_CONTROL, INSTRUCTIONS_EMPATHETIC,
)
from stt_faster_whisper import FasterWhisperSTT
from model_options import validate_models

load_dotenv(".env.local")
load_dotenv(".env")

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "log.txt"
def parse_room(name: str) -> tuple[str, int, dict, str | None]:
    parts = (name or "").split("-")
    condition = "control" if len(parts) > 1 and parts[1] == "ctl" else "empathetic"
    session_no = next((int(p) for p in parts if p.isdigit()), 0)
    try:
        encoded = name.rsplit(".", 1)[-1]
        encoded += "=" * (-len(encoded) % 4)
        room_config = json.loads(base64.urlsafe_b64decode(encoded))
        lease = room_config.pop("lease", None)
        models = validate_models(room_config)
    except (ValueError, json.JSONDecodeError, TypeError):
        models = validate_models({})
        lease = None
    return condition, session_no, models, lease

MIN_SPEECH_SEC = 0.25
MIN_SPEECH_RMS = 0.005

def _has_speech(wav: np.ndarray) -> bool:
    if wav.size < int(16000 * MIN_SPEECH_SEC):
        return False
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return rms >= MIN_SPEECH_RMS

def _frames_to_16k(frames):
    if not frames:
        return np.zeros(0, dtype=np.float32)
    combined = rtc.combine_audio_frames(frames)
    data = np.frombuffer(combined.data, dtype=np.int16).astype(np.float32) / 32768.0
    if combined.num_channels > 1:
        data = data.reshape(-1, combined.num_channels).mean(axis=1)
    if combined.sample_rate != 16000:
        data = librosa.resample(data, orig_sr=combined.sample_rate, target_sr=16000)
    return data.astype(np.float32)

def _emotion_tag(scores):
    emo = max(scores, key=scores.get)
    return f"emotion={emo} confidence={scores[emo]:.2f}"

_TAG_RE = re.compile(r"\[\s*emotion=[^\]]*\]\s*", re.IGNORECASE)

def _strip_tag(text):
    return _TAG_RE.sub("", text or "").strip()

async def _strip_tag_stream(text):
    buf = ""
    async for chunk in text:
        buf += chunk
        idx = buf.rfind("[")
        if idx != -1 and "]" not in buf[idx:]:
            out, buf = buf[:idx], buf[idx:]
        else:
            out, buf = buf, ""
        if out:
            yield _TAG_RE.sub("", out)
    if buf:
        yield _TAG_RE.sub("", buf)

def _write_log(condition, session_no, turn, user_text, scores, response):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    emo = _emotion_tag(scores) if scores else "n/a"
    line = (f"[{ts}] session {session_no} ({condition}) | turn {turn} | "
            f"user: {user_text!r} | {emo} | assistant: {_strip_tag(response)!r}\n")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)

def _release_room(room: str, lease: str | None):
    request = urllib.request.Request(
        "http://127.0.0.1:8000/control/session/end",
        data=json.dumps({"room": room, "lease": lease}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=5).close()
    except Exception:
        pass

class StudyAgent(Agent):
    def __init__(self, instructions, engine, condition, session_no):
        super().__init__(instructions=instructions)
        self.engine = engine
        self.condition = condition
        self.session_no = session_no
        self._turn_audio: list = []
        self.user_turns = 0
        self.ending = False
        self._pending = None
        self.notify = None
        self.publish = None

    async def stt_node(self, audio, model_settings):
        async def tee():
            async for frame in audio:
                self._turn_audio.append(frame)
                yield frame
        async for ev in Agent.default.stt_node(self, tee(), model_settings):
            yield ev

    async def tts_node(self, text, model_settings):
        chunks = []
        async for chunk in text:
            chunks.append(chunk)
        response = _strip_tag("".join(chunks))
        if response and self.publish is not None:
            await self.publish(
                "chat", json.dumps({"role": "assistant", "text": response})
            )

        async def response_stream():
            yield response

        async for frame in Agent.default.tts_node(self, response_stream(), model_settings):
            yield frame

    async def on_user_turn_completed(self, turn_ctx, new_message):
        text = (new_message.text_content or "").strip()
        wav = _frames_to_16k(self._turn_audio)
        self._turn_audio = []
        if not text or not _has_speech(wav):
            if self.notify is not None:
                await self.notify("no_speech")
            raise StopResponse()
        if self.publish is not None:
            await self.publish(
                "chat", json.dumps({"role": "user", "text": text})
            )
        scores = None
        if self.engine is not None:
            scores = await asyncio.to_thread(self.engine.add_user_turn, text, wav)
            new_message.content = [f"[{_emotion_tag(scores)}] {text}"]
        self.user_turns += 1
        self._pending = {"turn": self.user_turns, "user": text, "scores": scores}
        if self.publish is not None and scores:
            await self.publish("emotion", json.dumps({"label": max(scores, key=scores.get), "scores": scores}))

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    condition, session_no, models, lease = parse_room(ctx.room.name)
    engine = EmotionEngine()
    if condition == "empathetic":
        instructions, greeting = INSTRUCTIONS_EMPATHETIC, GREETING_EMPATHETIC
    else:
        instructions, greeting = INSTRUCTIONS_CONTROL, GREETING_CONTROL

    agent = StudyAgent(instructions, engine, condition, session_no)

    if models["stt_model"] == "whisper-local":
        selected_stt = stt_lib.StreamAdapter(
            stt=FasterWhisperSTT(model=os.getenv("WHISPER_MODEL", "small.en")),
            vad=silero.VAD.load(),
        )
    else:
        selected_stt = inference.STT(model=models["stt_model"], language="en")
    session = AgentSession(
        stt=selected_stt,
        llm=inference.LLM(model=models["llm_model"]),
        tts=inference.TTS(model=models["tts_model"], language="en", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection="manual",
    )

    async def end_session():
        try:
            await ctx.room.local_participant.publish_data(b"session_ended", reliable=True, topic="status")
        except Exception:
            pass
        if engine is not None:
            engine.reset()
        await ctx.room.disconnect()

    async def publish_state(state: str):
        try:
            await ctx.room.local_participant.publish_data(
                state.encode(), reliable=True, topic="agent_state")
        except Exception:
            pass

    async def publish(topic: str, msg: str):
        try:
            await ctx.room.local_participant.publish_data(
                msg.encode(), reliable=True, topic=topic)
        except Exception:
            pass

    agent.notify = lambda msg: publish("notice", msg)
    agent.publish = publish

    @session.on("conversation_item_added")
    def _on_item(ev):
        item = ev.item
        if getattr(item, "role", None) != "assistant":
            return
        text = item.text_content or ""
        if agent._pending:
            p = agent._pending
            agent._pending = None
            _write_log(agent.condition, agent.session_no, p["turn"], p["user"], p["scores"], text)

    @session.on("agent_state_changed")
    def _on_state(ev):
        state = getattr(ev, "new_state", None) or ""
        if agent.ending and state in ("listening", "idle"):
            agent.ending = False
            asyncio.create_task(end_session())
        else:
            asyncio.create_task(publish_state(state))

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        session.interrupt()
        session.clear_user_turn()
        agent._turn_audio = []
        session.room_io.set_participant(data.caller_identity)
        session.input.set_audio_enabled(True)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData):
        session.input.set_audio_enabled(False)
        await session.commit_user_turn(transcript_timeout=10.0)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("chat_message")
    async def chat_message(data: rtc.RpcInvocationData):
        message = data.payload.strip()
        if not message:
            return "empty"
        await publish("chat", json.dumps({"role": "user", "text": message}))
        scores = await asyncio.to_thread(
            agent.engine.add_user_turn, message, np.zeros(0, dtype=np.float32)
        )
        await publish("emotion", json.dumps({"label": max(scores, key=scores.get), "scores": scores}))
        await session.generate_reply(
            user_input=f"[{_emotion_tag(scores)}] {message}", input_modality="text"
        )
        return "ok"

    try:
        await session.start(agent=agent, room=ctx.room)
        session.input.set_audio_enabled(False)
        await session.generate_reply(instructions=greeting)
    finally:
        agent.engine = None
        del engine
        await asyncio.to_thread(_release_room, ctx.room.name, lease)

if __name__ == "__main__":
    agents.cli.run_app(server)
