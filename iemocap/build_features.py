import argparse
import json
from pathlib import Path
import torch
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import os
from collections import defaultdict
import re

import torch
torch.backends.cudnn.benchmark = True

from transformers import AutoTokenizer, AutoModel

EMOTION2ID = {
    "neu": 0, "fru": 1, "ang": 2, "sad": 3,
    "hap": 4, "exc": 5,
}

def rows_to_dialogs_iemocap(df):
    groups = defaultdict(list)
    for _, row in df.iterrows():
        uid = str(row["uttr_id"])
        groups[dialog_key_from_uttr_id(uid)].append(row)

    def turn_index(uid: str) -> int:
        m = re.search(r"_[FM](\d{3})$", uid)
        return int(m.group(1)) if m else 0

    for dkey in groups:
        groups[dkey] = sorted(groups[dkey], key=lambda r: turn_index(str(r["uttr_id"])))
    return groups

def dialog_key_from_uttr_id(uttr_id: str) -> str:
    m = re.match(r"^(Ses\d{2}[FM]_impro\d{2})_[FM]\d{3}$", uttr_id)
    if m:
        return m.group(1)
    parts = uttr_id.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 3 else uttr_id

def speaker_from_uttr_id(uttr_id: str) -> str:
    m = re.search(r"_([FM])\d{3}$", uttr_id)
    return m.group(1) if m else "UNK"

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

def build_features(csv_path, out_jsonl, tokenizer, model, device, sr=16000):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    df = pd.read_csv(csv_path)
    required = ["uttr_id", "start_time", "end_time", "audio_path", "emotion"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Columns required but missing: {c}")
    if "transcript" not in df.columns:
        df["transcript"] = ""

    dialogs = rows_to_dialogs_iemocap(df)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for dialog_key, rows in tqdm(dialogs.items(), desc=f"Processing {os.path.basename(csv_path)}"):
            turns, speakers_seen, spk_counter = [], {}, 0
            for _, r in enumerate(rows):
                uttr_id = str(r["uttr_id"])
                start = float(r["start_time"])
                end = float(r["end_time"])
                audio_path = str(r["audio_path"])
                transcript = str(r["transcript"]) if pd.notna(r["transcript"]) else ""

                if not os.path.isfile(audio_path):
                    continue

                lbl = r["emotion"]
                if pd.api.types.is_number(lbl):
                    label = int(lbl)
                else:
                    label = EMOTION2ID.get(str(lbl).lower(), EMOTION2ID["neu"])

                spk_tag = speaker_from_uttr_id(uttr_id)
                if spk_tag not in speakers_seen:
                    speakers_seen[spk_tag] = spk_counter; spk_counter += 1
                speaker_idx = speakers_seen[spk_tag]

                text_emb = extract_text_emb(transcript, tokenizer, model, device)

                try:
                    audio_feat = extract_audio_features(r['audio_path'], sr=sr, n_mels=64)
                except Exception:
                    audio_feat = [0.0] * 128

                turns.append({
                    "speaker_idx": int(speaker_idx),
                    "text_emb": text_emb,
                    "audio": audio_feat,
                    "label": int(label),
                    "dialogue_key": dialog_key,
                    "utterance_id": uttr_id,
                    "start": float(start),
                    "end": float(end),
                    "audio_path": audio_path
                })

            if len(turns) == 0:
                continue

            sample = {
                "dialog_id": dialog_key,
                "speakers": list(speakers_seen.keys()),
                "turns": turns
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

def main():
    root = Path(__file__).resolve().parents[1]
    CSV_PATH = root / "iemocap" / "iemocap.csv"
    OUT_JSONL = root / "iemocap" / "features" / "iemocap_features.jsonl"
    OUT_DIR = root / "iemocap"

    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default=CSV_PATH, help="CSV path to iemocap dataset")
    p.add_argument("--out_jsonl", default=OUT_JSONL, help="Output JSONL path for features")
    p.add_argument("--out_dir", default=OUT_DIR, help="Output directory for config files")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sr", type=int, default=16000)

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(args.device).eval()

    build_features(args.csv_path, args.out_jsonl, tokenizer, model, args.device, sr=args.sr)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"d_audio": 128, "d_text": 768}, f)

    print("Config files written to:", args.out_dir)

if __name__ == "__main__":
    main()
