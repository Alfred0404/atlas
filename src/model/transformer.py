import torch
import torch.nn as nn

from .config import ModelConfig


class ATLASTransformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings (summed)
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.field_embedding = nn.Embedding(7, config.d_model)  # 6 fields + SOS

        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer (encoder used as decoder-only with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight

        # Pre-build the logit mask template (registered as buffer so it moves with .to())
        self._register_logit_mask(config)

    def _register_logit_mask(self, config: ModelConfig) -> None:
        """Build a (max_seq_len, vocab_size) boolean mask where True = allowed token."""
        offsets = config.offsets
        V = config.vocab_size
        S = config.max_seq_len

        # For each field index (0-5), define the allowed token range
        # Field 0 = part_id, 1 = x, 2 = z, 3 = y, 4 = rotation, 5 = color
        # Y (vertical) is generated last among positions so the model
        # picks height after knowing the full horizontal position (x, z).
        field_masks = torch.zeros(6, V, dtype=torch.bool)

        # part_id: [parts_offset, vocab_size) + EOS (2)
        field_masks[0, offsets["parts"]:V] = True
        field_masks[0, 2] = True  # EOS

        # x_bin: [positions_x, positions_y)
        field_masks[1, offsets["positions_x"]:offsets["positions_y"]] = True

        # z_bin: [positions_z, colors)
        field_masks[2, offsets["positions_z"]:offsets["colors"]] = True

        # y_bin: [positions_y, positions_z)
        field_masks[3, offsets["positions_y"]:offsets["positions_z"]] = True

        # rotation: [rotations, positions_x)
        field_masks[4, offsets["rotations"]:offsets["positions_x"]] = True

        # color: [colors, parts)
        field_masks[5, offsets["colors"]:offsets["parts"]] = True

        # Expand to (max_seq_len, vocab_size)
        # Output at position t predicts target t, where target field = t % 6
        positions = torch.arange(S)
        field_indices = positions % 6  # (S,)
        logit_mask = field_masks[field_indices]  # (S, V)

        self.register_buffer("logit_mask", logit_mask)

    def forward(self, x: torch.Tensor, mask_logits: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs
            mask_logits: if True, apply field-based logit masking (use for generation)
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        B, S = x.shape

        # Padding mask: True where padded (PAD=0)
        padding_mask = x == 0  # (B, S)

        # Causal mask (bool so it matches padding_mask dtype)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            S, device=x.device, dtype=torch.bool
        )

        # Position indices
        positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)

        # Field indices: position 0 is SOS → field 6; positions >= 1 → (i-1) % 6
        field_indices = torch.where(
            positions == 0,
            torch.tensor(6, device=x.device),
            (positions - 1) % 6,
        )  # (1, S)

        # Embeddings (summed)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        fld_emb = self.field_embedding(field_indices)
        h = self.embed_dropout(tok_emb + pos_emb + fld_emb)

        # Transformer
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Output projection
        h = self.output_norm(h)
        logits = self.output_head(h)  # (B, S, V)

        # Logit masking — only during generation to enforce structural constraints
        if mask_logits:
            mask = self.logit_mask[:S]  # (S, V)
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

        return logits
