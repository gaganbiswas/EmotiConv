import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

torch.backends.cudnn.benchmark = True

EMOTION2ID = {
    "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
    "joy": 4, "disgust": 5, "anger": 6,
}

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

def rows_to_dialogs(df):
    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[str(row["dialog_id"])].append(row)
    for dkey in groups:
        groups[dkey] = sorted(groups[dkey], key=lambda r: int(r["utt_index"]))
    return groups

def build_features(csv_path, out_jsonl, tokenizer, model, device, sr=16000):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    df = pd.read_csv(csv_path)
    if "transcript" not in df.columns:
        df["transcript"] = ""

    dialogs = rows_to_dialogs(df)
    n_written = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for dialog_key, rows in tqdm(dialogs.items(), desc=os.path.basename(csv_path)):
            turns, speakers_seen, spk_counter = [], {}, 0
            for r in rows:
                audio_path = str(r["audio_path"])
                transcript = str(r["transcript"]) if pd.notna(r["transcript"]) else ""
                label = EMOTION2ID.get(str(r["emotion"]).lower(), EMOTION2ID["neutral"])

                spk_tag = str(r["speaker"])
                if spk_tag not in speakers_seen:
                    speakers_seen[spk_tag] = spk_counter; spk_counter += 1
                speaker_idx = speakers_seen[spk_tag]

                text_emb = extract_text_emb(transcript, tokenizer, model, device)
                try:
                    audio_feat = extract_audio_features(audio_path, sr=sr)
                except Exception:
                    audio_feat = [0.0] * 128

                turns.append({
                    "speaker_idx": int(speaker_idx),
                    "text_emb": text_emb,
                    "audio": audio_feat,
                    "label": int(label),
                    "dialogue_key": dialog_key,
                    "utterance_id": str(r["uttr_id"]),
                    "start": float(r["start_time"]),
                    "end": float(r["end_time"]),
                    "audio_path": audio_path,
                })

            if not turns:
                continue
            fout.write(json.dumps({
                "dialog_id": dialog_key,
                "speakers": list(speakers_seen.keys()),
                "turns": turns,
            }, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"  wrote {n_written} dialogues -> {out_jsonl}")

def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sr", type=int, default=16000)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(args.device).eval()

    for split in args.splits:
        csv_path = root / f"meld_{split}.csv"
        out_jsonl = root / "features" / f"meld_features_{split}.jsonl"
        if not csv_path.exists():
            raise SystemExit(f"Missing {csv_path}. Run create_dataset.py first.")
        build_features(csv_path, str(out_jsonl), tokenizer, model, args.device, sr=args.sr)

    with open(root / "config.json", "w") as f:
        json.dump({"d_audio": 128, "d_text": 768}, f)
    print("Config written:", root / "config.json")

if __name__ == "__main__":
    main()
