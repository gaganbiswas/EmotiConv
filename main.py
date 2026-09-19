import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REALTIME = ROOT / "realtime"

load_dotenv(REALTIME / ".env.local")
load_dotenv(REALTIME / ".env")

sys.path.insert(0, str(REALTIME))

import uvicorn
from token_server import app

BASE = "http://127.0.0.1:8000"

def _get(path: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(BASE + path, timeout=5) as response:
        return response.read()

def _wait_up() -> bool:
    for _ in range(60):
        try:
            _get("/state")
            return True
        except Exception:
            time.sleep(0.25)
    return False

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
        print(f"Emotion-aware CA is running at {BASE}")
        while agent.poll() is None:
            time.sleep(1)
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
