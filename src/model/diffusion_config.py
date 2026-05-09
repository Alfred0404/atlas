from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    # Set at build time from vocabulary
    theme: str = "City"
    max_bricks: int = 128
    n_parts: int = 0
    n_colors: int = 0
    n_rots: int = 24

    # Diffusion
    T: int = 500
    beta_schedule: str = "cosine"

    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 200
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    checkpoint_interval: int = 500

    # Loss weights
    lambda_pos: float = 1.0
    lambda_part: float = 1.0
    lambda_color: float = 0.5
    lambda_rot: float = 0.5

    # Compute discrete CE loss at all timesteps (set < T to restrict to low-noise steps only)
    discrete_t_max: int = 500
