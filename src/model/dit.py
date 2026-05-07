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
    """Diffusion Transformer for LEGO set generation.

    Input:  (B, N, 3) noisy positions + (B,) timesteps
    Output: noise_pred (B, N, 3), part/color/rot logits (B, N, *)
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        d = cfg.d_model

        self.pos_embed = nn.Sequential(
            nn.Linear(3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

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
        self.head_part = nn.Linear(d, cfg.n_parts)
        self.head_color = nn.Linear(d, cfg.n_colors)
        self.head_rot = nn.Linear(d, cfg.n_rots)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init noise head so initial predictions are ~0
        nn.init.zeros_(self.head_noise.weight)
        nn.init.zeros_(self.head_noise.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x:            (B, N, 3) noisy positions
            t:            (B,) timestep indices
            padding_mask: (B, N) bool, True = padded brick (ignored in attention)
        Returns:
            dict with noise_pred, part_logits, color_logits, rot_logits
        """
        h = self.pos_embed(x)
        h = h + self.time_embed(t).unsqueeze(1)

        for block in self.blocks:
            h = block(h, key_padding_mask=padding_mask)

        h = self.norm(h)

        return {
            "noise_pred": self.head_noise(h),
            "part_logits": self.head_part(h),
            "color_logits": self.head_color(h),
            "rot_logits": self.head_rot(h),
        }
