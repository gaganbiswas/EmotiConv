from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from splits import names_for, label_map_for, speaker_of, split_for_speaker

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def extract_text_emb(text, tokenizer, model, device):
    text = "" if text is None or str(text).lower() == "nan" else str(text)
    if len(text.strip()) == 0:
        return [0.0] * 768
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    out = model(**toks)
    cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()
    return cls.tolist()


def extract_audio_features(wav_path, sr=16000, n_mels=64):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        return [0.0] * (2 * n_mels)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(S + 1e-6)
    delta = librosa.feature.delta(logmel)
    m1 = logmel.mean(axis=1); m2 = delta.mean(axis=1)
    feat = np.concatenate([m1, m2], axis=0).astype(np.float32)
    return feat.tolist()


def build(n_classes: int) -> None:
    csv_path = HERE / "emodb.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run generate_csv.py first.")
    df = pd.read_csv(csv_path)

    names = set(names_for(n_classes))
    label_map = label_map_for(n_classes)
    df = df[df["emotion"].isin(names)].reset_index(drop=True)

    feat_dir = HERE / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{n_classes}-class] extracting features on {device} for {len(df)} utterances ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"emodb{n_classes}"):
        audio_path = str(r["audio_path"])
        if not os.path.isfile(audio_path):
            continue
        uttr_id = str(r["uttr_id"])
        split = split_for_speaker(speaker_of(uttr_id))
        transcript = str(r["transcript"]) if pd.notna(r["transcript"]) else ""
        text_emb = extract_text_emb(transcript, tokenizer, bert, device)
        try:
            audio_feat = extract_audio_features(audio_path)
        except Exception:
            audio_feat = [0.0] * 128
        turn = {
            "speaker_idx": 0,
            "text_emb": text_emb,
            "audio": audio_feat,
            "label": label_map[str(r["emotion"])],
            "dialogue_key": uttr_id,
            "utterance_id": uttr_id,
            "start": float(r["start_time"]),
            "end": float(r["end_time"]),
            "audio_path": audio_path,
        }
        buckets[split].append({"dialog_id": uttr_id, "speakers": [speaker_of(uttr_id)], "turns": [turn]})

    for split in ("train", "val", "test"):
        out = feat_dir / f"emodb{n_classes}_{split}.jsonl"
        with open(out, "w", encoding="utf-8") as fout:
            for s in buckets[split]:
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{split}: {len(buckets[split])} utterances -> {out}")

    (HERE / "config.json").write_text(json.dumps({"d_audio": 128, "d_text": 768}))
    print("wrote config.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", type=int, choices=[4, 7], required=True)
    args = p.parse_args()
    build(args.classes)


if __name__ == "__main__":
    main()
