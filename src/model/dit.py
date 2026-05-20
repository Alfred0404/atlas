import math
import torch
import torch.nn as nn

from .diffusion_config import DiffusionConfig


class _SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float) / (half - 1)
        )
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([x.sin(), x.cos()], dim=-1)


class _DiTBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(h)
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x


class DiT(nn.Module):
    """Conditional position-denoiser.

    Inputs: noisy positions (B, N, 3), timesteps (B,), and the bag of bricks
    (part_ids, color_ids, rot_ids) as conditioning.
    Output: noise_pred (B, N, 3).
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        d = cfg.d_model

        self.pos_embed = nn.Sequential(
            nn.Linear(3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.part_embed = nn.Embedding(cfg.n_parts, d, padding_idx=0)
        self.color_embed = nn.Embedding(cfg.n_colors, d, padding_idx=0)
        self.rot_embed = nn.Embedding(cfg.n_rots, d)

        self.time_embed = nn.Sequential(
            _SinusoidalEmbedding(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.blocks = nn.ModuleList(
            [_DiTBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )

        self.norm = nn.LayerNorm(d)
        self.head_noise = nn.Linear(d, 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for emb in (self.part_embed, self.color_embed, self.rot_embed):
            nn.init.normal_(emb.weight, std=0.02)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()
        nn.init.zeros_(self.head_noise.weight)
        nn.init.zeros_(self.head_noise.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        part_ids: torch.Tensor,
        color_ids: torch.Tensor,
        rot_ids: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x:            (B, N, 3) noisy positions
            t:            (B,) timestep indices
            part_ids:     (B, N) int64, 0=PAD/UNK
            color_ids:    (B, N) int64, 0=PAD/UNK
            rot_ids:      (B, N) int64
            padding_mask: (B, N) bool, True = padded brick
        """
        h = (
            self.pos_embed(x)
            + self.part_embed(part_ids)
            + self.color_embed(color_ids)
            + self.rot_embed(rot_ids)
        )
        h = h + self.time_embed(t).unsqueeze(1)

        for block in self.blocks:
            h = block(h, key_padding_mask=padding_mask)

        h = self.norm(h)

        return {"noise_pred": self.head_noise(h)}
