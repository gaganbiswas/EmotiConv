from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REALTIME = ROOT / "realtime"

load_dotenv(REALTIME / ".env.local")
load_dotenv(REALTIME / ".env")

sys.path.insert(0, str(REALTIME))

import uvicorn  # noqa: E402
from token_server import app  # noqa: E402

BASE = "http://127.0.0.1:8000"

def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return json.loads(r.read())

def _wait_up() -> bool:
    for _ in range(60):
        try:
            _get("/state")
            return True
        except Exception:
            time.sleep(0.25)
    return False

def _show(state: dict):
    cond = state.get("condition")
    extra = ""
    if state["page"] == "session":
        label = "EMPATHETIC (A)" if cond == "empathetic" else "CONTROL (B)"
        extra = f"  condition={label}  (#{state['session_no']})"
    print(f"  ▶ page={state['page']}{extra}")

HELP = """
    w  | welcome        show the idle welcome screen
    d  | demographic    show the demographic questionnaire
    a  | session a      start a session with condition A (empathetic + emotion classifier)
    b  | session b      start a session with condition B (plain control agent)
    s  | survey         show the post-session survey (form A)
    bs | survey b       show the second post-session survey (form B)
    t  | thanks         show the closing thank-you screen
    state               print the current experiment state
    h  | help           show this help
    q  | quit           stop everything and exit
"""

def console():
    state = _get("/state")
    print("=" * 60)
    print("  Emotion-aware CA study  —  participant URL:  " + BASE)
    if "REPLACE_WITH" in (state.get("demographic_url") or "") or not state.get("demographic_url"):
        print("  ⚠  demographic_url not set in realtime/study_config.json")
    if "REPLACE_WITH" in (state.get("survey_url") or "") or not state.get("survey_url"):
        print("  ⚠  survey_url not set in realtime/study_config.json")
    if "REPLACE_WITH" in (state.get("survey_b_url") or "") or not state.get("survey_b_url"):
        print("  ⚠  survey_b_url not set in realtime/study_config.json")
    print("  Type 'help' for commands, 'quit' to stop.")
    print("=" * 60)
    _show(state)

    while True:
        try:
            raw = input("emo-study> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            continue
        cmd = raw.replace("session ", "").strip()
        try:
            if cmd in ("q", "quit", "exit"):
                return
            elif cmd in ("h", "help", "?"):
                print(HELP)
            elif cmd == "state":
                _show(_get("/state"))
            elif cmd in ("w", "welcome"):
                _show(_post("/control/page", {"page": "welcome"}))
            elif cmd in ("d", "demographic", "demo"):
                _show(_post("/control/page", {"page": "demographic"}))
            elif cmd in ("a", "emp", "empathetic"):
                _show(_post("/control/session", {"condition": "empathetic"}))
            elif cmd in ("b", "ctl", "control"):
                _show(_post("/control/session", {"condition": "control"}))
            elif cmd in ("s", "survey"):
                _show(_post("/control/page", {"page": "survey"}))
            elif cmd in ("bs", "survey b"):
                _show(_post("/control/page", {"page": "survey_b"}))
            elif cmd in ("t", "thanks", "thankyou"):
                _show(_post("/control/page", {"page": "thanks"}))
            else:
                print(f"  unknown command: {raw!r}  (type 'help')")
        except Exception as e:
            print(f"  ! command failed: {e}")

def main() -> None:
    agent = subprocess.Popen([sys.executable, str(REALTIME / "agent.py"), "dev"], cwd=REALTIME)

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        if not _wait_up():
            print("server failed to start", file=sys.stderr)
            return
        console()
    finally:
        server.should_exit = True
        if agent.poll() is None:
            agent.terminate()
            try:
                agent.wait(timeout=10)
            except subprocess.TimeoutExpired:
                agent.kill()

if __name__ == "__main__":
    main()
