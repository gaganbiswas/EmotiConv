import re
from pathlib import Path

import pandas as pd

EVAL_LINE = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+"
    r"(?P<turn>\S+)\s+(?P<emotion>\S+)\s+"
    r"\[(?P<v>[\d.]+),\s*(?P<a>[\d.]+),\s*(?P<d>[\d.]+)\]"
)
TRANS_LINE = re.compile(r"^(?P<turn>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")
TURN_NAME = re.compile(r"^Ses(?P<session>\d{2})[FM]_\w+_(?P<gender>[FM])\d+$")

def parse_eval_file(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EVAL_LINE.match(line)
        if not m:
            continue
        rows.append({
            "turn": m["turn"],
            "emotion_code": m["emotion"],
            "start_time": float(m["start"]),
            "end_time": float(m["end"]),
        })
    rows.sort(key=lambda r: r["start_time"])
    return rows

def parse_transcription_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRANS_LINE.match(line)
        if m:
            out[m["turn"]] = m["text"].strip()
    return out

def actor_info(turn: str) -> tuple[str | None, int | None, str | None]:
    m = TURN_NAME.match(turn)
    if not m:
        return None, None, None
    session = int(m["session"])
    g = m["gender"]
    actor_id = f"Ses{session:02d}{g}"
    gender = "female" if g == "F" else "male"
    return actor_id, session, gender

def build_dataset(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for session_dir in sorted(root.glob("Session*")):
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        wav_root = session_dir / "sentences" / "wav"
        if not eval_dir.is_dir():
            continue
        for eval_file in sorted(eval_dir.glob("*.txt")):
            dialog = eval_file.stem
            transcripts = parse_transcription_file(trans_dir / f"{dialog}.txt")
            for turn_index, r in enumerate(parse_eval_file(eval_file)):
                turn = r["turn"]
                actor_id, session, gender = actor_info(turn)
                if actor_id is None:
                    continue
                wav_path = wav_root / dialog / f"{turn}.wav"
                rows.append({
                    "dialog_id": dialog,
                    "uttr_id": turn,
                    "actor": actor_id,
                    "session": session,
                    "gender": gender,
                    "audio_path": str(wav_path.resolve()),
                    "transcript": transcripts.get(turn, ""),
                    "emotion": r["emotion_code"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                })
    return pd.DataFrame(rows)

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = repo_root / "data" / "IEMOCAP_full_release"
    output_csv = repo_root / "iemocap" / "iemocap.csv"

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    df = build_dataset(input_dir)
    if df.empty:
        raise SystemExit(f"No utterances parsed under {input_dir}")

    df = df[df["emotion"].isin(['neu', 'hap', 'sad', 'ang', 'exc', 'fru'])].reset_index(drop=True)

    mask = df["audio_path"].map(lambda p: Path(p).exists())
    missing = (~mask).sum()
    if missing:
        print(f"Dropping {missing} row(s) with missing audio.")
    df = df[mask].reset_index(drop=True)

    df = df.sort_values(["dialog_id", "start_time"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Wrote {len(df)} rows to {output_csv}")
    print(f"Actors: {df['actor'].nunique()}  "
          f"(female={df[df.gender == 'female']['actor'].nunique()}, "
          f"male={df[df.gender == 'male']['actor'].nunique()})")
    print(f"Sessions: {sorted(df['session'].unique())}")
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())

if __name__ == "__main__":
    main()
