from pathlib import Path

import numpy as np
import torch
import librosa
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent.parent
from model import EmotionGAT

EMOTIONS = ["neutral", "happy", "angry", "sad"]
DEFAULT_CKPT = ROOT / "realtime" / "best.pt"

class EmotionEngine:
    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg, arch = ck["config"], ck.get("arch", {})
        self.model = EmotionGAT(
            d_audio=cfg["d_audio"], d_text=cfg["d_text"],
            n_classes=arch.get("n_classes", 4),
            d_model=arch.get("d_model", 128),
            heads=arch.get("heads", 8),
            n_layers=arch.get("layers", 4),
            past_window=arch.get("past_window", 10),
            future_window=arch.get("future_window", 0),
            ffn_mult=arch.get("ffn_mult", 2),
            tie_layers=arch.get("tie_layers", True),
            p=arch.get("dropout", 0.1),
        ).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        self.bert = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").to(self.device).eval()
        self.reset()

    def reset(self):
        self._texts = []
        self._audios = []

    @torch.no_grad()
    def _text_emb(self, text):
        text = (text or "").strip()
        if not text:
            return np.zeros(768, dtype=np.float32)
        t = self.tok(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        return self.bert(**t).last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()

    def _audio_feat(self, wav_16k):
        y = np.asarray(wav_16k, dtype=np.float32)
        if y.size == 0:
            return np.zeros(128, dtype=np.float32)
        logmel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64) + 1e-6)
        delta = librosa.feature.delta(logmel) if logmel.shape[1] >= 9 else np.zeros_like(logmel)
        return np.concatenate([logmel.mean(axis=1), delta.mean(axis=1)]).astype(np.float32)

    @torch.no_grad()
    def add_user_turn(self, text, wav_16k):
        self._texts.append(self._text_emb(text))
        self._audios.append(self._audio_feat(wav_16k))
        n = len(self._texts)
        text_t = torch.from_numpy(np.stack(self._texts).astype(np.float32))[None].to(self.device)
        audio_t = torch.from_numpy(np.stack(self._audios).astype(np.float32))[None].to(self.device)
        speaker = torch.zeros(1, n, dtype=torch.long, device=self.device)
        pad_mask = torch.ones(1, n, dtype=torch.bool, device=self.device)
        logits, *_ = self.model(text_t, audio_t, speaker, pad_mask)
        probs = torch.softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
        return {emo: float(p) for emo, p in zip(EMOTIONS, probs)}
