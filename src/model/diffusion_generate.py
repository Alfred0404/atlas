import numpy as np
import torch

from .dit import DiT
from .ddpm import DDPM
from .diffusion_config import DiffusionConfig
from ..maths.rotations import generate_chiral_rotation_matrices
from ..data.parser import RawBrickData
from ..utils.logging import setup_logging

logger = setup_logging()

_ROTATION_MATRICES = generate_chiral_rotation_matrices()

# LEGO grid constants (LDU)
_GRID_XZ = 20.0
_GRID_Y = 8.0


def _snap(positions: np.ndarray) -> np.ndarray:
    p = positions.copy()
    p[:, 0] = np.round(p[:, 0] / _GRID_XZ) * _GRID_XZ
    p[:, 1] = np.round(p[:, 1] / _GRID_Y) * _GRID_Y
    p[:, 2] = np.round(p[:, 2] / _GRID_XZ) * _GRID_XZ
    return p


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sample_categorical(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Sample class indices from logits with temperature.

    temperature=0 → argmax (deterministic mode collapse)
    temperature=1 → sample from learned distribution
    temperature>1 → more uniform / diverse
    """
    if temperature <= 0.0:
        return logits.argmax(axis=-1)
    probs = _softmax(logits / temperature)
    shape = probs.shape[:-1]
    flat = probs.reshape(-1, probs.shape[-1])
    # Vectorised inverse-CDF sampling
    cumprobs = flat.cumsum(axis=-1)
    u = np.random.uniform(size=(flat.shape[0], 1))
    return (u < cumprobs).argmax(axis=-1).reshape(shape)


def generate(
    model: DiT,
    ddpm: DDPM,
    cfg: DiffusionConfig,
    vocab: dict,
    device: str,
    n_sets: int = 1,
    n_bricks: int = None,
    min_confidence: float = 0.02,
    temperature: float = 1.0,
) -> list[list[RawBrickData]]:
    """Generate n_sets LEGO sets via reverse diffusion.

    Args:
        n_bricks: target brick count per set (default: avg of training sets).
            The top-n_bricks most confident positions are kept.
        min_confidence: minimum softmax probability for a brick to be kept.

    Returns a list of RawBrickData lists (one per set).
    """
    model.eval()
    result = ddpm.sample(model, (n_sets, cfg.max_bricks, 3), device)

    positions = result["positions"].cpu().numpy() * vocab["global_scale"]
    positions = _snap(positions.reshape(-1, 3)).reshape(n_sets, cfg.max_bricks, 3)

    # Deduplicate: mark positions that share an identical grid cell as not-keep
    # Done after snap so (x,y,z) tuples are exact integers.
    _snapped_keys = [
        {tuple(positions[s, i].tolist()): i for i in range(cfg.max_bricks - 1, -1, -1)}
        for s in range(n_sets)
    ]  # last brick at a cell wins (arbitrary, keeps a consistent survivor)

    part_logits = result["part_logits"].cpu().numpy()        # (n_sets, N, n_parts)
    color_ids = _sample_categorical(                          # (n_sets, N)
        result["color_logits"].cpu().numpy(), temperature)
    rot_ids = _sample_categorical(                            # (n_sets, N)
        result["rot_logits"].cpu().numpy(), temperature)

    # Exclude index 0 (UNK/PAD) from part selection
    part_logits_no_unk = part_logits.copy()
    part_logits_no_unk[:, :, 0] = -np.inf
    part_ids = _sample_categorical(part_logits_no_unk, temperature)  # (n_sets, N), always ≥ 1
    # Confidence based on temperature-scaled probabilities (for brick filtering)
    confidence = _softmax(part_logits_no_unk / max(temperature, 1e-6)).max(axis=-1)

    inv_part = {v: k for k, v in vocab["part_vocab"].items()}
    inv_color = {v: k for k, v in vocab["color_vocab"].items()}

    all_sets = []
    for s in range(n_sets):
        conf = confidence[s]  # (N,)

        # Keep top-n_bricks by confidence, then threshold
        if n_bricks is not None:
            top_k = min(n_bricks, cfg.max_bricks)
            keep = np.zeros(cfg.max_bricks, dtype=bool)
            keep[np.argsort(conf)[-top_k:]] = True
        else:
            keep = conf > min_confidence

        kept = keep.sum()
        logger.info("Set %d: keeping %d / %d positions (min_conf=%.3f)", s, kept, cfg.max_bricks, min_confidence)

        unique_indices = set(_snapped_keys[s].values())  # one survivor per grid cell
        bricks = []
        for i in np.where(keep)[0]:
            if int(i) not in unique_indices:              # skip clipping duplicates
                continue
            pid = int(part_ids[s, i])
            brick_id = inv_part.get(pid, "3001")
            color = inv_color.get(int(color_ids[s, i]), 15)
            rot_mat = _ROTATION_MATRICES[int(rot_ids[s, i]) % len(_ROTATION_MATRICES)]

            world = np.eye(4)
            world[:3, :3] = rot_mat
            world[:3, 3] = positions[s, i]

            bricks.append(RawBrickData(brick_id=brick_id, world_matrix=world, color=color))

        all_sets.append(bricks)

    return all_sets
