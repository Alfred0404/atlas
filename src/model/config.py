from dataclasses import dataclass, field
from typing import Dict

from ..config import Config


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int  # set dynamically from VocabularyManager
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 1202  # 200 bricks * 6 + SOS + EOS
    dropout: float = 0.1
    brick_fields: int = 6

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 100
    grad_clip_norm: float = 1.0
    warmup_steps: int = 50
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 1

    # Generation
    max_gen_bricks: int = 100
    temperature: float = 5.0
    top_k: int = 50

    # Token range offsets (from Config.OFFSETS) — used for logit masking
    offsets: Dict[str, int] = field(default_factory=lambda: dict(Config.OFFSETS))
