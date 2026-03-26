import torch
import torch.nn.functional as F

from .config import ModelConfig
from ..utils.logging import setup_logging

logger = setup_logging()


class Generator:

    def __init__(self, model, config: ModelConfig, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        max_bricks: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Autoregressively generate a sequence of brick tokens.

        Args:
            max_bricks: Maximum number of bricks to generate.
            temperature: Sampling temperature (1.0 = neutral).
            top_k: If > 0, only sample from the top-k logits.

        Returns:
            1-D tensor of token IDs (including SOS, excluding padding).
        """
        SOS = 1
        EOS = 2
        max_len = max_bricks * self.config.brick_fields + 2  # +SOS +EOS

        logger.info(f"Generating sequence (max_bricks={max_bricks}, temp={temperature}, top_k={top_k})")
        tokens = torch.tensor([[SOS]], dtype=torch.long, device=self.device)

        # Collision tracking
        # Field order: [part_id, x, z, y, rotation, color]
        used_positions = set()  # {(x_tok, z_tok, y_tok), ...}
        current_brick_tokens = []
        collision_penalty = self.config.collision_penalty

        for _ in range(max_len - 1):
            logits = self.model(tokens, mask_logits=True)  # (1, S, V)
            next_logits = logits[0, -1]  # (V,)

            # Collision penalty on y field (vertical, generated last)
            field_index = (tokens.shape[1] - 1) % self.config.brick_fields
            if field_index == 3 and len(current_brick_tokens) >= 2:
                current_x = current_brick_tokens[-2]  # x token
                current_z = current_brick_tokens[-1]  # z token
                for px, pz, py in used_positions:
                    if px == current_x and pz == current_z:
                        next_logits[py] -= collision_penalty

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_values, _ = torch.topk(next_logits, top_k)
                threshold = top_values[-1]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,)

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == EOS:
                break

            # Track brick fields for collision detection
            current_brick_tokens.append(next_token.item())
            if len(current_brick_tokens) == self.config.brick_fields:
                x, z, y = current_brick_tokens[1], current_brick_tokens[2], current_brick_tokens[3]
                used_positions.add((x, z, y))
                current_brick_tokens = []

        seq = tokens.squeeze(0)
        n_tokens = seq.shape[0] - 1  # exclude SOS
        hit_eos = seq[-1].item() == EOS
        logger.info(f"Generated {n_tokens} tokens (EOS={'yes' if hit_eos else 'no'})")
        return seq

    def decode_sequence(self, tokens: torch.Tensor) -> list[dict]:
        """Decode a token sequence into a list of brick dictionaries.

        Args:
            tokens: 1-D tensor of token IDs (with SOS/EOS).

        Returns:
            List of dicts with keys: part_id, x, y, z, rotation, color (raw token IDs).
        """
        tokens = tokens.cpu().tolist()

        # Strip SOS (1) and EOS (2)
        if tokens and tokens[0] == 1:
            tokens = tokens[1:]
        if tokens and tokens[-1] == 2:
            tokens = tokens[:-1]

        fields = ["part_id", "x", "z", "y", "rotation", "color"]
        bricks = []

        for i in range(0, len(tokens) - len(fields) + 1, len(fields)):
            chunk = tokens[i : i + len(fields)]
            if len(chunk) == len(fields):
                bricks.append(dict(zip(fields, chunk)))

        return bricks
