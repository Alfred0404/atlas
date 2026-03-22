from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int  # set dynamically from VocabularyManager
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 2404  # 400 bricks * 6 + SOS + EOS
    dropout: float = 0.1
    brick_fields: int = 6

    # Training
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 1

    # Generation
    max_gen_bricks: int = 100
    temperature: float = 1.0
    top_k: int = 0

    # Token range offsets (from Config.OFFSETS) — used for logit masking
    offsets: Dict[str, int] = field(default_factory=lambda: {
        "special": 0,
        "rotations": 4,
        "positions_x": 28,
        "positions_y": 1028,
        "positions_z": 2028,
        "colors": 3028,
        "parts": 3128,
    })
