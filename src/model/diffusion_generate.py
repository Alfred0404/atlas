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


def generate(
    model: DiT,
    ddpm: DDPM,
    cfg: DiffusionConfig,
    vocab: dict,
    device: str,
    bag: dict,
    n_sets: int = 1,
) -> list[list[RawBrickData]]:
    """Generate n_sets LEGO sets via reverse diffusion conditioned on a bag.

    Args:
        bag: dict with part_ids, color_ids, rot_ids, padding_mask each (max_bricks,) — or
             (n_sets, max_bricks). A 1-D bag is broadcast to n_sets.
    """
    model.eval()

    def _expand(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(n_sets, -1)
        return t.to(device)

    cond = {
        "part_ids": _expand(bag["part_ids"]),
        "color_ids": _expand(bag["color_ids"]),
        "rot_ids": _expand(bag["rot_ids"]),
        "padding_mask": _expand(bag["padding_mask"]),
    }

    positions = ddpm.sample(model, (n_sets, cfg.max_bricks, 3), device, cond=cond)
    positions = positions.cpu().numpy() * vocab["global_scale"]
    positions = _snap(positions.reshape(-1, 3)).reshape(n_sets, cfg.max_bricks, 3)

    part_ids = cond["part_ids"].cpu().numpy()
    color_ids = cond["color_ids"].cpu().numpy()
    rot_ids = cond["rot_ids"].cpu().numpy()
    padding_mask = cond["padding_mask"].cpu().numpy()

    inv_part = {v: k for k, v in vocab["part_vocab"].items()}
    inv_color = {v: k for k, v in vocab["color_vocab"].items()}

    all_sets = []
    for s in range(n_sets):
        seen = set()
        bricks = []
        for i in range(cfg.max_bricks):
            if padding_mask[s, i]:
                continue
            pid = int(part_ids[s, i])
            if pid == 0:
                continue
            key = tuple(positions[s, i].tolist())
            if key in seen:
                continue
            seen.add(key)

            brick_id = inv_part.get(pid, "3001")
            color = inv_color.get(int(color_ids[s, i]), 15)
            rot_mat = _ROTATION_MATRICES[int(rot_ids[s, i]) % len(_ROTATION_MATRICES)]

            world = np.eye(4)
            world[:3, :3] = rot_mat
            world[:3, 3] = positions[s, i]

            bricks.append(RawBrickData(brick_id=brick_id, world_matrix=world, color=color))

        logger.info("Set %d: %d bricks (after grid dedup)", s, len(bricks))
        all_sets.append(bricks)

    return all_sets
