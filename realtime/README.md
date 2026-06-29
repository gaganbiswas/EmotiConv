# Realtime emotion-aware companion (LiveKit) + human-evaluation study harness

Push-to-talk voice agent. Per user turn: faster-whisper transcribes the audio, the
trained GAT classifies the user's emotion (+confidence), that signal is injected into
the prompt, and Google Gemini 2.5 Flash (via LiveKit Inference) generates an empathetic reply spoken via LiveKit TTS.
Each session runs for 15 turns (1 greeting + 7 user + 7 agent), logs every turn to
`log.txt`, then ends and clears the in-session memory.

The harness supports a **researcher-controlled human-evaluation study**: a demographic
form, repeated 7-turn sessions, and a post-session survey, with two agent conditions
chosen per session from the terminal. See **Running the study** below.

## Conditions

- **A — empathetic** (`session a`): emotion classifier on + empathetic dialogue (the full system).
- **B — control** (`session b`): no classifier, plain non-empathetic assistant.

The condition is encoded in the LiveKit room name (`study-emp-…` / `study-ctl-…`), so the
agent worker reads it straight off `ctx.room.name`.

## Pieces

- `agent.py` — LiveKit agent (manual/push-to-talk turns, STT tee -> emotion -> Gemini 2.5 Flash LLM -> TTS, logging, turn limit, both conditions).
- `emotion_engine.py` — loads `../iemocap/model_4class/best.pt`, runs the causal GAT over the running user-turn context.
- `stt_faster_whisper.py` — faster-whisper STT plugin.
- `tts_qwen.py` — optional local Qwen3-TTS plugin (not currently wired; the agent uses LiveKit Inference TTS). Swap it back in if you want fully local TTS.
- `token_server.py` — experiment-state server: serves the participant SPA, mints LiveKit tokens, pushes page changes over SSE, exposes `/control/*` for the researcher console.
- `static/index.html` — participant SPA: welcome → demographic form → voice session → survey form → thanks, all driven by server state.
- `study_config.json` — the two Microsoft Forms URLs (demographic + survey). **Edit this before running.**
- `../main.py` — runs the agent worker + server + interactive researcher console in one terminal.

## Setup

1. A LiveKit Cloud project (gives `LIVEKIT_URL/API_KEY/API_SECRET`; LiveKit Inference
   serves both the LLM and TTS, so no separate provider key is needed). Optionally set
   `LLM_MODEL` (default `google/gemini-2.5-flash`).
2. Install deps (from repo root): `uv sync`.
3. Copy `.env.example` to `.env` and fill in the keys.

## Run (quick, single session)

```
# 1) the agent worker
python agent.py dev

# 2) the web + token server
python token_server.py
```

The server starts on the `welcome` screen; use the study console (below) or POST to
`/control/session` to begin a session.

## Running the study (one terminal)

1. Put your two Microsoft Forms URLs in `study_config.json` (`demographic_url`, `survey_url`).
   Use normal share links — they are auto-converted to embeddable (`?embed=true`) iframes.
2. From the repo root:
   ```
   uv run main.py        # or: python main.py
   ```
   This launches the agent worker + server and drops you into the `emo-study>` console.
3. Open **http://127.0.0.1:8000** on the participant's screen.
4. Drive the participant's screen from the console:

   | command                   | participant sees                                                           |
   | ------------------------- | -------------------------------------------------------------------------- |
   | `d` / `demographic`       | demographic questionnaire (iframe)                                         |
   | `a` / `session a`         | a new **empathetic** session (participant clicks _Start session_, 7 turns) |
   | `b` / `session b`         | a new **control** session                                                  |
   | `s` / `survey`            | post-session survey (iframe)                                               |
   | `t` / `thanks`            | closing thank-you screen                                                   |
   | `w` / `welcome`           | idle holding screen                                                        |
   | `state` / `help` / `quit` | status / help / stop everything                                            |

   Typical loop: `d` → (participant fills form) → `a` (or `b`) → participant does 7 turns →
   screen shows _"Session complete — please wait"_ → `s` → (participant fills survey) →
   `b` (or `a`) → … → `t`. Each `session` command starts a **fresh** session with no
   memory of previous ones; alternate/counterbalance A and B as your design requires.

## Notes / knobs

- Emotion model: IEMOCAP 4-class (`neutral, happy, angry, sad`).
  Point `EmotionEngine(ckpt_path=...)` elsewhere to use the 6-class or MELD checkpoint.
- `WHISPER_MODEL` (default `small.en`), `TTS_MODEL` (default `cartesia/sonic-2s` via LiveKit
  Inference), and `LLM_MODEL` (default `google/gemini-2.5-flash` via LiveKit Inference) are env-configurable.
- The emotion graph uses the running context of **user** turns only (speaker 0); bot turns are not fed in.
- `log.txt` is appended across sessions; only the model/dialogue memory is reset between sessions.
- Pinned to `livekit-agents ~=1.5`. Running end-to-end needs LiveKit creds + a mic;
  if a method signature differs in your installed version (e.g. `commit_user_turn`, `inference.TTS`, `agent_state_changed`),
  adjust to that version's API.
