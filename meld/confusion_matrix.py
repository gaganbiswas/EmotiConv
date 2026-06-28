from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import EmotionGAT

HERE = Path(__file__).resolve().parent

CLASS_NAMES = {
    7: ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
}

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, arch = ck["config"], ck.get("arch", {})
    model = EmotionGAT(
        d_audio=cfg["d_audio"], d_text=cfg["d_text"],
        n_classes=arch.get("n_classes", 7), d_model=arch.get("d_model", 128),
        heads=arch.get("heads", 8), n_layers=arch.get("layers", 5),
        past_window=arch.get("past_window", 8), future_window=arch.get("future_window", 0),
        ffn_mult=arch.get("ffn_mult", 2), tie_layers=arch.get("tie_layers", True),
        p=arch.get("dropout", 0.1),
    ).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck

@torch.no_grad()
def predict(model, jsonl, device):
    preds, gts = [], []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)["turns"]
            text = torch.tensor(np.array([x["text_emb"] for x in t], dtype=np.float32))[None].to(device)
            audio = torch.tensor(np.array([x["audio"] for x in t], dtype=np.float32))[None].to(device)
            spk = torch.tensor(np.array([x["speaker_idx"] for x in t], dtype=np.int64))[None].to(device)
            pad = torch.ones(1, len(t), dtype=torch.bool, device=device)
            logits, *_ = model(text, audio, spk, pad)
            preds.append(logits[0].argmax(-1).cpu().numpy())
            gts.append(np.array([x["label"] for x in t]))
    return np.concatenate(gts), np.concatenate(preds)

def plot_cm(cm, names, out_png, title):
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.25 * n + 2.5, 1.1 * n + 2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j] * 100:.0f}%\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def generate(ckpt_path, test_path, out_dir, title="MELD", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ckpt_path, device)
    n_classes = model.n_classes
    names = CLASS_NAMES.get(n_classes, [str(i) for i in range(n_classes)])

    gts, preds = predict(model, test_path, device)
    wf1 = f1_score(gts, preds, average="weighted") * 100
    mf1 = f1_score(gts, preds, average="macro") * 100
    acc = accuracy_score(gts, preds) * 100
    print(f"\n{title} ({n_classes}-class)  WF1 {wf1:.2f} | macroF1 {mf1:.2f} | acc {acc:.2f}\n")
    print(classification_report(gts, preds, target_names=names, digits=3, zero_division=0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(gts, preds, labels=range(n_classes))
    png = out_dir / f"cm_meld_{n_classes}class.png"
    plot_cm(cm, names, png, f"{title} {n_classes}-class  (WF1 {wf1:.1f})")

    tsv = out_dir / f"predictions_meld_{n_classes}class.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("true\tpred\n")
        for g, pr in zip(gts, preds):
            f.write(f"{names[g]}\t{names[pr]}\n")
    print(f"saved: {png}\nsaved: {tsv}")
    return wf1, mf1, acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(HERE / "model" / "best.pt"))
    p.add_argument("--test", default=str(HERE / "features" / "meld_features_test.jsonl"))
    p.add_argument("--out_dir", default=str(HERE / "results"))
    p.add_argument("--title", default="MELD")
    args = p.parse_args()
    generate(args.ckpt, args.test, args.out_dir, args.title)

if __name__ == "__main__":
    main()
