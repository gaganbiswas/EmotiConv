import torch
import torch.nn as nn
import torch.nn.functional as F

def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)

def _mlp(d_in, d_hidden, d_out, p=0.1):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(d_hidden, d_out),
    )

class ModalityEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, p: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = _gelu(h)
        h = self.drop(self.norm(h))
        return h

N_RELATIONS = 6
N_DIST_BUCKETS = 12

def distance_bucket(delta: torch.Tensor, n_buckets: int = N_DIST_BUCKETS) -> torch.Tensor:
    d = delta.abs()
    max_exact = n_buckets // 2
    is_small = d < max_exact
    large = max_exact + (
        torch.log(d.float().clamp_min(max_exact) / max_exact)
        / torch.log(torch.tensor(float(N_DIST_BUCKETS) / max_exact))
        * (n_buckets - max_exact)
    ).long()
    large = large.clamp_max(n_buckets - 1)
    return torch.where(is_small, d.long(), large)

def build_graph_structure(
    speaker_idx: torch.Tensor,
    pad_mask: torch.Tensor,
    past_window: int,
    future_window: int,
):
    B, N = speaker_idx.shape
    device = speaker_idx.device
    idx = torch.arange(N, device=device)
    delta = idx[None, :] - idx[:, None]

    temporal = torch.ones(N, N, dtype=torch.long, device=device)
    temporal = torch.where(delta < 0, torch.zeros_like(temporal), temporal)
    temporal = torch.where(delta > 0, torch.full_like(temporal, 2), temporal)

    same_spk = speaker_idx[:, :, None] == speaker_idx[:, None, :]
    speaker_rel = torch.where(same_spk, 0, 1)
    relation = speaker_rel * 3 + temporal[None]

    dist = distance_bucket(delta)[None].expand(B, N, N)

    valid_pair = pad_mask[:, :, None] & pad_mask[:, None, :]
    in_window = (delta >= -past_window) & (delta <= future_window)
    diag = torch.eye(N, dtype=torch.bool, device=device)
    attn_mask = (valid_pair & (in_window[None] | diag[None])) | diag[None]
    return relation, dist, attn_mask

class RelationalGATLayer(nn.Module):
    def __init__(self, d_model: int, heads: int = 8, p: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.dh = d_model // heads
        self.W = nn.Linear(d_model, d_model)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        self.rel_bias = nn.Embedding(N_RELATIONS, heads)
        self.pos_bias = nn.Embedding(N_DIST_BUCKETS, heads)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p)
        self.attn_drop = nn.Dropout(p)
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.rel_bias.weight)
        nn.init.zeros_(self.pos_bias.weight)

    def forward(self, h, relation, dist, attn_mask):
        B, N, _ = h.shape
        H, dh = self.heads, self.dh
        v = self.W(h).view(B, N, H, dh)

        s = (v * self.a_src).sum(-1)
        d = (v * self.a_dst).sum(-1)
        e = self.leaky(s[:, :, None, :] + d[:, None, :, :])
        e = e + self.rel_bias(relation) + self.pos_bias(dist)

        e = e.masked_fill(~attn_mask[..., None], float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha)
        alpha = self.attn_drop(alpha)

        ctx = torch.einsum("bijh,bjhd->bihd", alpha, v)
        ctx = ctx.reshape(B, N, H * dh)
        return self.drop(self.out(ctx))

class EmotionGAT(nn.Module):
    def __init__(
        self,
        d_audio: int,
        d_text: int,
        n_classes: int = 7,
        d_model: int = 128,
        heads: int = 8,
        n_layers: int = 5,
        past_window: int = 8,
        future_window: int = 0,
        ffn_mult: int = 2,
        tie_layers: bool = True,
        p: float = 0.1,
    ):
        super().__init__()
        self.past_window = past_window
        self.future_window = future_window
        self.d_model = d_model
        self.n_layers = n_layers
        self.tie_layers = tie_layers
        self.n_classes = n_classes
        self.heads = heads
        self.ffn_mult = ffn_mult
        self.dropout = p

        self.text_enc = ModalityEncoder(d_text, d_model, p)
        self.audio_enc = ModalityEncoder(d_audio, d_model, p)

        self.recon_head = _mlp(d_model, d_model, d_model, p)

        self.fuse = _mlp(2 * d_model, d_model, d_model, p)
        self.in_norm = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(p)

        n_blocks = 1 if tie_layers else n_layers
        self.gat_layers = nn.ModuleList(
            [RelationalGATLayer(d_model, heads, p) for _ in range(n_blocks)]
        )
        self.gat_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_blocks)]
        )
        self.ffns = nn.ModuleList(
            [_mlp(d_model, d_model * ffn_mult, d_model, p) for _ in range(n_blocks)]
        )
        self.ffn_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_blocks)]
        )

        self.classifier = _mlp(d_model, d_model, n_classes, p)
        self.text_head = _mlp(d_model, d_model, n_classes, p)
        self.audio_head = _mlp(d_model, d_model, n_classes, p)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        speaker_idx: torch.Tensor,
        pad_mask: torch.Tensor,
        use_recon_audio: bool = False,
    ):
        h_text = self.text_enc(text)
        h_audio = self.audio_enc(audio)

        text_logits = self.text_head(h_text)
        audio_logits = self.audio_head(h_audio)

        recon_audio = self.recon_head(h_text)
        if use_recon_audio:
            h_audio = recon_audio

        h = self.fuse(torch.cat([h_text, h_audio], dim=-1))
        h = self.in_drop(self.in_norm(h))

        relation, dist, attn_mask = build_graph_structure(
            speaker_idx, pad_mask, self.past_window, self.future_window
        )

        for step in range(self.n_layers):
            k = 0 if self.tie_layers else step
            h = h + self.gat_layers[k](self.gat_norms[k](h), relation, dist, attn_mask)
            h = h + self.ffns[k](self.ffn_norms[k](h))

        logits = self.classifier(h)
        return logits, recon_audio, h_audio.detach(), text_logits, audio_logits

    @staticmethod
    def recon_loss(recon_audio, target_audio, pad_mask):
        m = pad_mask.unsqueeze(-1).float()
        diff = (recon_audio - target_audio) ** 2 * m
        return diff.sum() / m.sum().clamp_min(1.0) / recon_audio.size(-1)
