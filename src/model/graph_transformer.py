"""Graph Transformer model for LEGO assembly generation.

Architecture
------------
At each generation step the model receives:
  - A partial assembly graph  G=(V, E)  where V are placed bricks and E are
    their stud/anti-stud connections.
  - A padded list of open ports on that partial assembly.

It outputs logits for five independent heads:
  port      — which open port to snap the next brick onto  (pointer)
  part      — which part ID to place
  color     — which color
  self_port — which port on the new part connects to the selected open port
  rot_steps — discrete rotation offset (0/1/2/3 → 0°/90°/180°/270°)

Forward pass summary
--------------------
1. Node embedding   : embed (part_id, color) + project continuous (trans, rot6d)
                      → sum → LayerNorm → (N_total, d)
2. Graph encoder    : L × TransformerConv layers with edge attributes
                      → contextual node embeddings
3. Graph summary    : global mean-pool per graph in batch → (B, d)
4. Open port encoder: for each port, concat world features + source node emb
                      → linear → (B, P, d)
5. Pointer          : scaled dot-product  (B,1,d) × (B,d,P) → (B, P) logits
                      masked to valid ports
6. Soft context     : weighted sum of port embeddings using softmax(pointer)
                      → (B, d)  — differentiable selection during training
7. Classification   : g + soft_context → four linear heads
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv, global_mean_pool

from src.model.graph_config import GraphModelConfig


class _FFN(nn.Module):
    """Position-wise feed-forward block with pre-norm."""

    def __init__(self, d: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc2(self.drop(F.gelu(self.fc1(self.norm(x)))))
        return x + self.drop(h)


class _GNNLayer(nn.Module):
    """One TransformerConv + FFN block."""

    def __init__(
        self, d: int, heads: int, edge_dim: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        assert d % heads == 0
        self.norm = nn.LayerNorm(d)
        self.conv = TransformerConv(
            in_channels=d,
            out_channels=d // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True,  # learnable skip connection inside conv
        )
        self.ffn = _FFN(d, d_ff, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Pre-norm message passing
        h = self.conv(self.norm(x), edge_index, edge_attr)
        x = x + h
        return self.ffn(x)


class GraphTransformer(nn.Module):
    """Open-port-selection graph transformer.

    Parameters
    ----------
    cfg : GraphModelConfig
    """

    def __init__(self, cfg: GraphModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ---- 1. Node input projections ------------------------------------
        self.part_emb = nn.Embedding(cfg.n_parts, d, padding_idx=0)
        self.color_emb = nn.Embedding(cfg.n_colors, d, padding_idx=0)
        self.cont_proj = nn.Linear(cfg.node_cont_dim, d, bias=False)
        self.node_norm = nn.LayerNorm(d)

        # Edge attribute projection (edge_dim expected by TransformerConv)
        self.edge_proj = nn.Linear(cfg.edge_attr_dim, d, bias=False)

        # ---- 2. Graph encoder layers ---------------------------------------
        self.gnn_layers = nn.ModuleList(
            [
                _GNNLayer(d, cfg.n_heads, d, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )

        # ---- 4. Open port encoder -----------------------------------------
        # Input per port: world_pos(3) + world_nrm(3) + port_type(1) = 7 floats
        # plus the source node embedding (d) → concat → project to d
        self.port_proj = nn.Sequential(
            nn.Linear(7 + d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.port_norm = nn.LayerNorm(d)

        # ---- 6 & 7. Classification heads -----------------------------------
        # ctx_norm normalises g + soft_context before the classifier heads.
        # Without it, ctx has uncontrolled magnitude (~25 LDU units) which makes
        # logit variance >> 1 and all heads initialise much worse than random
        # (observed: color 6.89 vs expected 4.74, self_port 5.83 vs 3.47).
        self.ctx_norm = nn.LayerNorm(d)
        self.head_part = nn.Linear(d, cfg.n_parts)
        self.head_color = nn.Linear(d, cfg.n_colors)
        self.head_self_port = nn.Linear(d, cfg.max_port_id)
        self.head_rot = nn.Linear(d, cfg.n_rot_steps)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch,  # torch_geometric Batch object
        open_ports: Tensor,  # (B, P, port_feat_dim)  float32
        open_port_mask: Tensor,  # (B, P)  bool — True = valid
        target_port_idx: Tensor | None = None,  # (B,) int64 — teacher-forced port index; None = inference
    ) -> dict[str, Tensor]:
        """
        Returns a dict with keys:
          port_logits      (B, P)
          part_logits      (B, n_parts)
          color_logits     (B, n_colors)
          self_port_logits (B, max_port_id)
          rot_logits       (B, n_rot_steps)

        During training pass ``target_port_idx`` (the ground-truth port) so the
        classification heads receive exact local context rather than a noisy
        soft-attention average over all 244 ports.  At inference leave it None
        and the top-1 predicted port is used instead (greedy hard selection).
        """
        d = self.cfg.d_model
        device = self.part_emb.weight.device

        # Normalize all tensors onto model device to avoid mixed CPU/CUDA inputs.
        part_ids = batch.part_ids.to(device)
        colors = batch.colors.to(device)
        node_x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr_in = batch.edge_attr.to(device)
        batch_index = batch.batch.to(device)
        batch_ptr = batch.ptr.to(device)
        open_ports = open_ports.to(device)
        open_port_mask = open_port_mask.to(device)

        # ---- 1. Node embeddings -------------------------------------------
        x = self.part_emb(part_ids) + self.color_emb(colors) + self.cont_proj(node_x)
        x = self.node_norm(x)  # (N_total, d)

        # Project edge attributes once
        edge_attr = self.edge_proj(edge_attr_in)  # (E_total, d)

        # ---- 2. Graph encoder ---------------------------------------------
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        # ---- 3. Graph summary token  g = mean-pool per graph --------------
        g = global_mean_pool(x, batch_index)  # (B, d)

        # ---- 4. Open port encoder -----------------------------------------
        # open_ports: (B, P, 9)
        #   [:, :, 0:3] = world_pos (normalised)
        #   [:, :, 3:6] = world_nrm
        #   [:, :, 6]   = source node seq_idx  (float)
        #   [:, :, 7]   = port_id              (float)
        #   [:, :, 8]   = port_type            (float: 1=male 2=female)

        B, P, _ = open_ports.shape

        # Gather source node embeddings for each port.
        # open_ports[:, :, 6] contains the node sequence index within each graph.
        # We need to map these to indices into the flat x tensor using batch.ptr.
        node_seq_idx = open_ports[:, :, 6].long()  # (B, P)
        ptr = batch_ptr  # (B+1,)
        global_node_idx = ptr[:-1].unsqueeze(1) + node_seq_idx  # (B, P)
        # Clamp to valid range (masked slots have seq_idx from padding zeros)
        global_node_idx = global_node_idx.clamp(max=x.size(0) - 1)
        src_node_emb = x[global_node_idx]  # (B, P, d)

        # Concat port geometric features (7 floats) + source node emb
        port_geo = open_ports[:, :, [0, 1, 2, 3, 4, 5, 8]]  # (B, P, 7)
        port_feat = torch.cat([port_geo, src_node_emb], dim=-1)  # (B, P, 7+d)
        port_emb = self.port_norm(self.port_proj(port_feat))  # (B, P, d)

        # ---- 5. Pointer (scaled dot-product attention) --------------------
        query = g.unsqueeze(1)  # (B, 1, d)
        port_logits = torch.bmm(query, port_emb.transpose(1, 2)) / math.sqrt(d)
        port_logits = port_logits.squeeze(1)  # (B, P)
        # Mask invalid (padded) ports
        port_logits = port_logits.masked_fill(~open_port_mask, float("-inf"))

        # ---- 6. Port context for classification heads ------------------------
        # Training: use the ground-truth port embedding (teacher forcing).
        #   → classification heads see exact local context, not a noisy average.
        # Inference: use the top-1 predicted port (greedy hard selection).
        if target_port_idx is not None:
            # (B,) → (B, 1, 1) index → (B, 1, d) → (B, d)
            idx = target_port_idx.clamp(min=0).view(B, 1, 1).expand(B, 1, port_emb.size(-1))
            port_context = port_emb.gather(1, idx).squeeze(1)  # (B, d)
        else:
            best_port = port_logits.argmax(dim=-1)  # (B,)
            idx = best_port.view(B, 1, 1).expand(B, 1, port_emb.size(-1))
            port_context = port_emb.gather(1, idx).squeeze(1)  # (B, d)

        # ---- 7. Classification heads --------------------------------------
        ctx = self.ctx_norm(g + port_context)  # (B, d)

        return dict(
            port_logits=port_logits,
            part_logits=self.head_part(ctx),
            color_logits=self.head_color(ctx),
            self_port_logits=self.head_self_port(ctx),
            rot_logits=self.head_rot(ctx),
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_loss(
        logits: dict[str, Tensor],
        targets: dict[str, Tensor],
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Cross-entropy loss summed over all heads.

        Parameters
        ----------
        logits:
            Output of ``forward()``.
        targets:
            Dict with keys matching logit keys (without ``_logits`` suffix):
            ``port``, ``part_id``, ``color``, ``self_port``, ``rot_steps``.
        loss_weights:
            Per-head scalar weights.  Defaults to 1.0 for all heads.

        Returns
        -------
        total_loss : scalar Tensor
        per_head   : dict of scalar Tensors for logging
        """
        if loss_weights is None:
            loss_weights = {}

        pairs = [
            ("port", "port_logits", targets["port"]),
            ("part", "part_logits", targets["part_id"]),
            ("color", "color_logits", targets["color"]),
            ("self_port", "self_port_logits", targets["self_port"]),
            ("rot", "rot_logits", targets["rot_steps"]),
        ]

        per_head: dict[str, Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(logits.values())).device)

        for name, logit_key, target in pairs:
            # Skip items where target == -1 (invalid / disconnected seed)
            valid = target >= 0
            if valid.sum() == 0:
                per_head[name] = torch.tensor(0.0)
                continue
            loss = F.cross_entropy(logits[logit_key][valid], target[valid])
            w = loss_weights.get(name, 1.0)
            total = total + w * loss
            per_head[name] = loss.detach()

        return total, per_head
