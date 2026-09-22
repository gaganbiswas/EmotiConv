# EmotiConv

Run the local conversation pipeline with Python 3.11+.

## Setup

From the repository root:

```bash
uv sync
cp realtime/.env.example realtime/.env
```

Edit `realtime/.env` and add your LiveKit credentials:

```dotenv
LIVEKIT_URL=wss://YOUR_PROJECT.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

Keep the remaining model settings from `.env.example`, or change them as needed.

## Run

```bash
uv run python main.py
```

Open <http://127.0.0.1:8000> in a browser. The command starts the local web server and LiveKit agent together.

Stop the pipeline with `Ctrl+C`.

## Without uv

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
python main.py
```

_Note: The first run may download the configured Whisper and other model files._
