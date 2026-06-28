from pathlib import Path

import pandas as pd

SPLIT_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}

def hms_to_seconds(t: str) -> float:
    t = str(t).strip().replace(",", ".")
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def build_split(csv_path: Path, video_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        did = int(r["Dialogue_ID"])
        uid = int(r["Utterance_ID"])
        mp4 = video_dir / f"dia{did}_utt{uid}.mp4"
        rows.append({
            "uttr_id": f"dia{did}_utt{uid}",
            "dialog_id": f"{split}_dia{did}",
            "speaker": str(r["Speaker"]),
            "audio_path": str(mp4.resolve()),
            "transcript": str(r["Utterance"]),
            "emotion": str(r["Emotion"]).lower(),
            "start_time": hms_to_seconds(r["StartTime"]),
            "end_time": hms_to_seconds(r["EndTime"]),
            "utt_index": uid,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["dialog_id", "utt_index"]).reset_index(drop=True)
    return out

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw = repo_root / "data" / "MELD.Raw"
    out_dir = repo_root / "meld"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, subdir in SPLIT_DIRS.items():
        csv_path = raw / f"{split}_sent_emo.csv"
        video_dir = raw / subdir
        if not csv_path.exists():
            raise SystemExit(f"Missing annotation file: {csv_path}")
        df = build_split(csv_path, video_dir, split)
        missing = (~df["audio_path"].map(lambda p: Path(p).exists())).sum()
        out_csv = out_dir / f"meld_{split}.csv"
        df.to_csv(out_csv, index=False)
        print(f"{split:5s}: wrote {len(df):5d} rows -> {out_csv.name}"
              f"  (missing audio: {missing})")
        if missing:
            print(f"        extract {split}.tar.gz into {video_dir} first.")
    print("\nEmotion classes: anger disgust fear joy neutral sadness surprise")

if __name__ == "__main__":
    main()
