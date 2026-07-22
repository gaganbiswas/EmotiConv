"""Train and test the EmotionGAT entirely on EmoDB (in-corpus baseline).

Fully self-contained (no imports from the IEMOCAP/MELD training scripts): the training
helpers below are the same ones used across the project, inlined here so EmoDB mirrors
the iemocap/ravdess package layout. Trains on EmoDB's own speaker-independent train
split and evaluates on its held-out test speakers, for either the 4-class subset or the
full 7-class taxonomy.

Usage:
    python train_incorpus.py --classes 4
    python train_incorpus.py --classes 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from model import EmotionGAT
from splits import names_for

HERE = Path(__file__).resolve().parent

EPOCHS = 50
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 5e-4
W_CONSISTENCY = 0.015
RECON_WEIGHT = 0.1
SEED = 42


def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class DialogDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.dialogs = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                turns = d["turns"]
                self.dialogs.append({
                    "text": np.asarray([t["text_emb"] for t in turns], dtype=np.float32),
                    "audio": np.asarray([t["audio"] for t in turns], dtype=np.float32),
                    "speaker": np.asarray([t["speaker_idx"] for t in turns], dtype=np.int64),
                    "label": np.asarray([t["label"] for t in turns], dtype=np.int64),
                })

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, i):
        return self.dialogs[i]


def collate(batch):
    B = len(batch)
    N = max(len(d["label"]) for d in batch)
    d_text = batch[0]["text"].shape[1]
    d_audio = batch[0]["audio"].shape[1]

    text = torch.zeros(B, N, d_text)
    audio = torch.zeros(B, N, d_audio)
    speaker = torch.full((B, N), -1, dtype=torch.long)
    label = torch.full((B, N), -100, dtype=torch.long)
    pad_mask = torch.zeros(B, N, dtype=torch.bool)

    for b, d in enumerate(batch):
        n = len(d["label"])
        text[b, :n] = torch.from_numpy(d["text"])
        audio[b, :n] = torch.from_numpy(d["audio"])
        speaker[b, :n] = torch.from_numpy(d["speaker"])
        label[b, :n] = torch.from_numpy(d["label"])
        pad_mask[b, :n] = True

    return text, audio, speaker, label, pad_mask


def class_weights(dataset: DialogDataset, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for d in dataset.dialogs:
        for y in d["label"]:
            counts[y] += 1
    counts = np.clip(counts, 1.0, None)
    w = np.sqrt(counts.sum() / (n_classes * counts))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def consistency_loss(fused_logits, text_logits, audio_logits, labels, pad_mask, n_classes):
    m = pad_mask.view(-1)
    fl = fused_logits.view(-1, n_classes)[m]
    tl = text_logits.view(-1, n_classes)[m]
    al = audio_logits.view(-1, n_classes)[m]
    y = labels.view(-1)[m]
    uni_ce = 0.5 * (F.cross_entropy(tl, y) + F.cross_entropy(al, y))
    lf, lt, la = (F.log_softmax(x, -1) for x in (fl, tl, al))
    pf, pt, pa = lf.exp(), lt.exp(), la.exp()
    agree = 0.25 * (F.kl_div(lt, pf, reduction="batchmean") + F.kl_div(lf, pt, reduction="batchmean")
                    + F.kl_div(la, pf, reduction="batchmean") + F.kl_div(lf, pa, reduction="batchmean"))
    return uni_ce + agree


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for text, audio, speaker, label, pad_mask in loader:
        text, audio = text.to(device), audio.to(device)
        speaker, pad_mask = speaker.to(device), pad_mask.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, *_ = model(text, audio, speaker, pad_mask)
        m = pad_mask.view(-1)
        p = logits.view(-1, logits.size(-1)).argmax(-1)[m].cpu().numpy()
        g = label.view(-1)[m.cpu()].numpy()
        preds.append(p)
        gts.append(g)
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    wf1 = f1_score(gts, preds, average="weighted") * 100
    mf1 = f1_score(gts, preds, average="macro") * 100
    acc = accuracy_score(gts, preds) * 100
    return wf1, mf1, acc


def run(n_classes: int) -> dict:
    feat_dir = HERE / "features"
    prefix = f"emodb{n_classes}"
    train_path = feat_dir / f"{prefix}_train.jsonl"
    val_path = feat_dir / f"{prefix}_val.jsonl"
    test_path = feat_dir / f"{prefix}_test.jsonl"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run prepare_features.py --classes {n_classes} first.")

    out_dir = HERE / f"model_{n_classes}class"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "config.json").read_text())
    d_audio, d_text = cfg["d_audio"], cfg["d_text"]

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DialogDataset(str(train_path))
    val_ds = DialogDataset(str(val_path))
    test_ds = DialogDataset(str(test_path))
    print(f"\n=== EmoDB {n_classes}-class ===")
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              generator=g, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = EmotionGAT(d_audio=d_audio, d_text=d_text, n_classes=n_classes).to(device)
    weights = class_weights(train_ds, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=False)

    best_wf1 = -1.0
    best_path = out_dir / "best.pt"
    saved_args = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                  "weight_decay": WEIGHT_DECAY, "w_consistency": W_CONSISTENCY,
                  "recon_weight": RECON_WEIGHT, "seed": SEED}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for text, audio, speaker, label, pad_mask in train_loader:
            text, audio = text.to(device), audio.to(device)
            speaker, label = speaker.to(device), label.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, recon_audio, target_audio, text_logits, audio_logits = \
                model(text, audio, speaker, pad_mask)
            cls_loss = criterion(logits.view(-1, n_classes), label.view(-1))
            rec_loss = model.recon_loss(recon_audio, target_audio, pad_mask)
            loss = cls_loss + RECON_WEIGHT * rec_loss
            if W_CONSISTENCY > 0:
                loss = loss + W_CONSISTENCY * consistency_loss(
                    logits, text_logits, audio_logits, label, pad_mask, n_classes)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * text.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_ds)
        val_wf1, _, _ = evaluate(model, val_loader, device)

        flag = ""
        if val_wf1 > best_wf1:
            best_wf1 = val_wf1
            t_wf1, t_mf1, t_acc = evaluate(model, test_loader, device)
            torch.save({
                "model": model.state_dict(), "args": saved_args,
                "arch": {"n_classes": model.n_classes, "d_model": model.d_model,
                         "heads": model.heads, "layers": model.n_layers,
                         "ffn_mult": model.ffn_mult, "tie_layers": model.tie_layers,
                         "past_window": model.past_window, "future_window": model.future_window,
                         "recon_weight": RECON_WEIGHT, "dropout": model.dropout},
                "config": cfg, "val_wf1": val_wf1,
                "wf1": t_wf1, "macro_f1": t_mf1, "acc": t_acc, "epoch": epoch,
            }, best_path)
            flag = f"  <-- best (val {val_wf1:5.2f} | test {t_wf1:5.2f})"

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val_wf1 {val_wf1:5.2f}{flag}")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"\nSelected by val WF1={best['val_wf1']:.2f}  ->  "
          f"TEST wf1 {best['wf1']:.2f} | macroF1 {best['macro_f1']:.2f} | acc {best['acc']:.2f}  "
          f"(epoch {best['epoch']}, saved to {best_path})")

    try:
        import confusion_matrix as cm
        cm.generate(str(best_path), str(test_path), str(HERE / "results"),
                    title=f"EmoDB in-corpus ({n_classes}-class)")
    except Exception as e:
        print(f"[confusion matrix skipped: {e}]")

    return {"wf1": best["wf1"], "macro_f1": best["macro_f1"], "acc": best["acc"],
            "epoch": best["epoch"], "val_wf1": best["val_wf1"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", type=int, choices=[4, 7], required=True)
    args = p.parse_args()
    _ = names_for(args.classes)  # validate
    run(args.classes)


if __name__ == "__main__":
    main()
