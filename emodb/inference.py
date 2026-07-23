from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from transformers import AutoTokenizer, AutoModel

from model import EmotionGAT
from confusion_matrix import plot_cm
from splits import EMO7_NAMES

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SOURCES = {
    "iemocap": {
        "ckpt": REPO / "iemocap" / "model_4class" / "best.pt",
        "names": ["neutral", "happy", "angry", "sad"],
        "to_emodb": {"neutral": "neutral", "happy": "happy", "angry": "angry", "sad": "sad"},
        "title": "IEMOCAP -> EmoDB",
    },
}


@torch.no_grad()
def _text_emb(text, tokenizer, bert, device):
    text = "" if text is None or str(text).lower() == "nan" else str(text)
    if len(text.strip()) == 0:
        return np.zeros(768, dtype=np.float32)
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    cls = bert(**toks).last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()
    return cls.astype(np.float32)


def _audio_feat(wav_path, sr=16000, n_mels=64):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        return np.zeros(2 * n_mels, dtype=np.float32)
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels) + 1e-6)
    delta = librosa.feature.delta(logmel)
    return np.concatenate([logmel.mean(axis=1), delta.mean(axis=1)]).astype(np.float32)


def extract_emodb(device):
    import pandas as pd
    csv_path = HERE / "emodb.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run generate_csv.py first.")
    df = pd.read_csv(csv_path)
    df = df[df["emotion"].isin(EMO7_NAMES)].reset_index(drop=True)

    print(f"Extracting EmoDB features on {device} for {len(df)} utterances ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    items = []
    from tqdm import tqdm
    for _, r in tqdm(df.iterrows(), total=len(df), desc="features"):
        path = str(r["audio_path"])
        if not os.path.isfile(path):
            continue
        transcript = str(r["transcript"]) if pd.notna(r["transcript"]) else ""
        try:
            audio = _audio_feat(path)
        except Exception:
            audio = np.zeros(128, dtype=np.float32)
        items.append((str(r["emotion"]), _text_emb(transcript, tokenizer, bert, device), audio))
    return items


def load_source(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, arch = ck["config"], ck.get("arch", {})
    model = EmotionGAT(
        d_audio=cfg["d_audio"], d_text=cfg["d_text"],
        n_classes=arch.get("n_classes"), d_model=arch.get("d_model", 128),
        heads=arch.get("heads", 8), n_layers=arch.get("layers", 4),
        past_window=arch.get("past_window", 0), future_window=arch.get("future_window", 0),
        ffn_mult=arch.get("ffn_mult", 2), tie_layers=arch.get("tie_layers", True),
        p=arch.get("dropout", 0.1),
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def evaluate_source(key, items, device, out_dir):
    src = SOURCES[key]
    if not Path(src["ckpt"]).exists():
        print(f"[skip {key}: checkpoint not found at {src['ckpt']}]")
        return None

    names, to_emodb = src["names"], src["to_emodb"]
    keep_idx = [i for i, nm in enumerate(names) if nm in to_emodb]
    idx_to_emodb = {i: to_emodb[names[i]] for i in keep_idx}
    overlap = [e for e in EMO7_NAMES if e in set(idx_to_emodb.values())]
    eval_id = {e: k for k, e in enumerate(overlap)}
    keep_t = torch.tensor(keep_idx, device=device)

    model = load_source(src["ckpt"], device)

    gts, preds = [], []
    for emo, text_emb, audio in items:
        if emo not in eval_id:
            continue
        text = torch.from_numpy(text_emb)[None, None].to(device)
        aud = torch.from_numpy(audio)[None, None].to(device)
        spk = torch.zeros(1, 1, dtype=torch.long, device=device)
        pad = torch.ones(1, 1, dtype=torch.bool, device=device)
        logits, *_ = model(text, aud, spk, pad)
        logit = logits[0, 0]
        best = keep_t[logit[keep_t].argmax()].item()
        preds.append(eval_id[idx_to_emodb[best]])
        gts.append(eval_id[emo])

    gts, preds = np.array(gts), np.array(preds)
    wf1 = f1_score(gts, preds, average="weighted") * 100
    mf1 = f1_score(gts, preds, average="macro") * 100
    acc = accuracy_score(gts, preds) * 100
    print(f"\n{src['title']}  ({len(overlap)}-class overlap, N={len(gts)})  "
          f"WF1 {wf1:.2f} | macroF1 {mf1:.2f} | acc {acc:.2f}\n")
    print(classification_report(gts, preds, labels=range(len(overlap)),
                                target_names=overlap, digits=3, zero_division=0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(gts, preds, labels=range(len(overlap)))
    png = out_dir / f"cm_emodb_crosscorpus_{key}.png"
    plot_cm(cm, overlap, png, f"{src['title']}  (WF1 {wf1:.1f})")
    tsv = out_dir / f"predictions_emodb_crosscorpus_{key}.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("true\tpred\n")
        for g, pr in zip(gts, preds):
            f.write(f"{overlap[g]}\t{overlap[pr]}\n")
    print(f"saved: {png}\nsaved: {tsv}")
    return {"wf1": wf1, "macro_f1": mf1, "acc": acc, "n": int(len(gts)), "classes": overlap}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(HERE / "results"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = extract_emodb(device)

    res = evaluate_source("iemocap", items, device, args.out_dir)
    if res is not None:
        print("\n=== cross-corpus summary (EmoDB) ===")
        print(f"iemocap  {len(res['classes'])}-class  N={res['n']:4d}  "
              f"WF1 {res['wf1']:.2f} | macroF1 {res['macro_f1']:.2f} | acc {res['acc']:.2f}")


if __name__ == "__main__":
    main()
