"""Build emodb.csv with English (machine-translated) transcripts and canonical
7-class EmoDB emotion labels.

The single CSV holds all seven EmoDB emotions; the in-corpus 4-class experiment is a
filtered subset (neutral/happy/angry/sad) selected downstream in prepare_features.py,
and the cross-corpus inference (inference.py) filters to whatever classes overlap with
the source model. Transcripts are produced with Whisper in German->English translation
mode so the (English) BERT text encoder can be reused unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from translator import Translator
from splits import RAW_TO_EMO7

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EMODB_DIR = REPO_ROOT / "data" / "emodb_2.0"


def duration_to_seconds(value: str) -> float:
    return pd.to_timedelta(value).total_seconds()


def build_dataframe(emodb_dir: Path) -> pd.DataFrame:
    emo = pd.read_csv(emodb_dir / "db.emotion.csv")
    files = pd.read_csv(emodb_dir / "db.files.csv")
    speakers = pd.read_csv(emodb_dir / "db.speaker.csv")

    df = emo.merge(files, on="file").merge(speakers, on="speaker", how="left")

    # Keep all seven EmoDB emotions and map to the canonical short names.
    df = df[df["emotion"].isin(RAW_TO_EMO7)].reset_index(drop=True)

    df["uttr_id"] = df["file"].map(lambda p: Path(p).stem)
    df["dialog_id"] = df["uttr_id"]
    df["actor"] = df["speaker"].map(lambda s: f"spk{int(s):02d}")
    df["session"] = df["speaker"].astype(int)
    df["audio_path"] = df["file"].map(lambda p: str((emodb_dir / p).resolve()))
    df["emotion"] = df["emotion"].map(RAW_TO_EMO7)
    df["start_time"] = 0.0
    df["end_time"] = df["duration"].map(duration_to_seconds)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emodb_dir", default=str(EMODB_DIR))
    p.add_argument("--out_csv", default=str(HERE / "emodb.csv"))
    p.add_argument("--model_size", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute_type", default="float16")
    args = p.parse_args()

    emodb_dir = Path(args.emodb_dir)
    if not emodb_dir.exists():
        raise SystemExit(f"EmoDB directory not found: {emodb_dir}")

    df = build_dataframe(emodb_dir)
    if df.empty:
        raise SystemExit("No utterances matched the EmoDB emotions.")

    print(f"{len(df)} utterances across {df['actor'].nunique()} speakers")
    print("Emotion distribution:\n" + df["emotion"].value_counts().to_string())
    print(f"\nTranscribing (German -> English) with Whisper {args.model_size} ...")

    translator = Translator(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    df["transcript"] = [
        translator.transcribe(path)
        for path in tqdm(df["audio_path"], desc="translate")
    ]

    columns = [
        "dialog_id", "uttr_id", "actor", "session", "gender",
        "audio_path", "transcript", "emotion", "start_time", "end_time",
    ]
    df[columns].to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
