from __future__ import annotations

import asyncio
import datetime
import os
import re
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

load_dotenv(".env.local")
load_dotenv(".env")

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "log.txt"
MAX_USER_TURNS = 7

_engine: EmotionEngine | None = None

def get_engine() -> EmotionEngine:
    global _engine
    if _engine is None:
        _engine = EmotionEngine()
    return _engine

def parse_room(name: str) -> tuple[str, int]:
    parts = (name or "").split("-")
    condition = "control" if len(parts) > 1 and parts[1] == "ctl" else "empathetic"
    session_no = next((int(p) for p in parts if p.isdigit()), 0)
    return condition, session_no

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

# Matches an emotion tag the model may have echoed back, e.g. "[emotion=sad confidence=0.70]".
_TAG_RE = re.compile(r"\[\s*emotion=[^\]]*\]\s*", re.IGNORECASE)

def _strip_tag(text):
    return _TAG_RE.sub("", text or "").strip()

async def _strip_tag_stream(text):
    """Remove any echoed emotion tag from the LLM text stream before it reaches TTS,
    holding back only an unclosed '[...]' so a tag split across chunks is caught."""
    buf = ""
    async for chunk in text:
        buf += chunk
        idx = buf.rfind("[")
        if idx != -1 and "]" not in buf[idx:]:
            out, buf = buf[:idx], buf[idx:]  # keep the open bracket until it closes
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

    async def stt_node(self, audio, model_settings):
        async def tee():
            async for frame in audio:
                self._turn_audio.append(frame)
                yield frame
        async for ev in Agent.default.stt_node(self, tee(), model_settings):
            yield ev

    async def tts_node(self, text, model_settings):
        async for frame in Agent.default.tts_node(self, _strip_tag_stream(text), model_settings):
            yield frame

    async def on_user_turn_completed(self, turn_ctx, new_message):
        text = (new_message.text_content or "").strip()
        wav = _frames_to_16k(self._turn_audio)
        self._turn_audio = []
        if not text or not _has_speech(wav):
            if self.notify is not None:
                await self.notify("no_speech")
            raise StopResponse()
        scores = None
        if self.engine is not None:
            scores = self.engine.add_user_turn(text, wav)
            new_message.content = [f"[{_emotion_tag(scores)}] {text}"]
        self.user_turns += 1
        self._pending = {"turn": self.user_turns, "user": text, "scores": scores}

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    condition, session_no = parse_room(ctx.room.name)
    if condition == "empathetic":
        engine = get_engine()
        engine.reset()
        instructions, greeting = INSTRUCTIONS_EMPATHETIC, GREETING_EMPATHETIC
    else:
        engine = None
        instructions, greeting = INSTRUCTIONS_CONTROL, GREETING_CONTROL

    agent = StudyAgent(instructions, engine, condition, session_no)

    session = AgentSession(
        stt=stt_lib.StreamAdapter(stt=FasterWhisperSTT(model=os.getenv("WHISPER_MODEL", "small.en")),
                                  vad=silero.VAD.load()),
        llm=inference.LLM(model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash")),
        tts=inference.TTS(model=os.getenv("TTS_MODEL", "cartesia/sonic-2"), language="en"),
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

    @session.on("conversation_item_added")
    def _on_item(ev):
        item = ev.item
        if getattr(item, "role", None) != "assistant" or not agent._pending:
            return
        p = agent._pending
        agent._pending = None
        _write_log(agent.condition, agent.session_no, p["turn"], p["user"], p["scores"],
                   item.text_content or "")
        asyncio.create_task(publish("progress", f"{agent.user_turns}/{MAX_USER_TURNS}"))
        if agent.user_turns >= MAX_USER_TURNS:
            agent.ending = True

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

    await session.start(agent=agent, room=ctx.room)
    session.input.set_audio_enabled(False)
    await session.generate_reply(instructions=greeting)

if __name__ == "__main__":
    agents.cli.run_app(server)
