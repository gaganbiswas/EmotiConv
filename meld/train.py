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

HERE = Path(__file__).resolve().parent

def set_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def build_model(d_audio, d_text, device, d_model=None):
    kw = {} if d_model is None else {"d_model": d_model}
    return EmotionGAT(d_audio=d_audio, d_text=d_text, **kw).to(device)

class DialogDataset(Dataset):
    def __init__(self, jsonl_paths: str):
        self.dialogs = []
        for path in str(jsonl_paths).split(","):
            path = path.strip()
            if not path:
                continue
            with open(path, encoding="utf-8") as f:
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=str(HERE / "features" / "meld_features_train.jsonl"))
    p.add_argument("--val", default=str(HERE / "features" / "meld_features_dev.jsonl"))
    p.add_argument("--test", default=str(HERE / "features" / "meld_features_test.jsonl"))
    p.add_argument("--config", default=str(HERE / "config.json"))
    p.add_argument("--out_dir", default=str(HERE / "model"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--w_consistency", type=float, default=0.3)
    p.add_argument("--recon_weight", type=float, default=0.4)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(Path(args.config).read_text())
    d_audio, d_text = cfg["d_audio"], cfg["d_text"]

    train_ds = DialogDataset(args.train)
    val_ds = DialogDataset(args.val)
    test_ds = DialogDataset(args.test)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=g, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate)

    model = build_model(d_audio, d_text, device, args.d_model)

    weights = class_weights(train_ds, model.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100,
                                    label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=False)

    os.makedirs(args.out_dir, exist_ok=True)
    best_wf1 = -1.0
    best_path = Path(args.out_dir) / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for text, audio, speaker, label, pad_mask in train_loader:
            text, audio = text.to(device), audio.to(device)
            speaker, label = speaker.to(device), label.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                logits, recon_audio, target_audio, text_logits, audio_logits = \
                    model(text, audio, speaker, pad_mask)
                cls_loss = criterion(logits.view(-1, model.n_classes), label.view(-1))
                rec_loss = model.recon_loss(recon_audio, target_audio, pad_mask)
                loss = cls_loss + args.recon_weight * rec_loss
                if args.w_consistency > 0:
                    loss = loss + args.w_consistency * consistency_loss(
                        logits, text_logits, audio_logits, label, pad_mask, model.n_classes)

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
                "model": model.state_dict(),
                "args": vars(args),
                "arch": {"n_classes": model.n_classes, "d_model": model.d_model,
                         "heads": model.heads, "layers": model.n_layers,
                         "ffn_mult": model.ffn_mult, "tie_layers": model.tie_layers,
                         "past_window": model.past_window, "future_window": model.future_window,
                         "recon_weight": args.recon_weight, "dropout": model.dropout},
                "config": cfg,
                "val_wf1": val_wf1,
                "wf1": t_wf1, "macro_f1": t_mf1, "acc": t_acc, "epoch": epoch,
            }, best_path)
            flag = f"  <-- best (val {val_wf1:5.2f} | test {t_wf1:5.2f})"

        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val_wf1 {val_wf1:5.2f}{flag}")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"\nSelected by dev WF1={best['val_wf1']:.2f}  ->  "
          f"TEST wf1 {best['wf1']:.2f} | macroF1 {best['macro_f1']:.2f} | acc {best['acc']:.2f}  "
          f"(saved to {best_path})")

    try:
        import confusion_matrix
        confusion_matrix.generate(str(best_path), args.test, HERE / "results", title="MELD")
    except Exception as e:
        print(f"[confusion matrix skipped: {e}]")

if __name__ == "__main__":
    main()
