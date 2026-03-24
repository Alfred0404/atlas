"""Data augmentation for LEGO brick sequences.

Applies geometric transforms (Y-axis rotations, X-axis mirror) and BFS
order permutations to multiply the training dataset.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .parser import RawBrickData

# --- Y-axis rotation matrices (3x3) ---
# LDraw Y points down, but Y-axis rotations only affect X and Z.

ROT_Y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
ROT_Y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
ROT_Y_270 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)

# --- Mirror matrix ---
MIRROR_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

# All geometric transforms: (suffix, 3x3 matrix)
_ROTATIONS = [
    ("_rot90", ROT_Y_90),
    ("_rot180", ROT_Y_180),
    ("_rot270", ROT_Y_270),
]

_MIRROR = ("_mirX", MIRROR_X)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    enable_rotations: bool = True
    enable_mirror: bool = True
    num_permutations: int = 1
    seed: int = 42


def apply_global_transform(
    bricks: List[RawBrickData], transform: np.ndarray
) -> List[RawBrickData]:
    """Apply a 3x3 transform to all bricks (position and rotation).

    Args:
        bricks: List of RawBrickData to transform.
        transform: 3x3 matrix (rotation or improper rotation).

    Returns:
        New list of RawBrickData with transformed world matrices.
    """
    result = []
    for brick in bricks:
        new_matrix = brick.world_matrix.copy()
        # Transform rotation (upper-left 3x3)
        new_matrix[:3, :3] = transform @ brick.world_matrix[:3, :3]
        # Transform position (translation column)
        new_matrix[:3, 3] = transform @ brick.world_matrix[:3, 3]
        result.append(RawBrickData(brick.brick_id, new_matrix, brick.color))
    return result


def generate_augmented_variants(
    bricks: List[RawBrickData], config: AugmentationConfig
) -> List[Tuple[str, List[RawBrickData]]]:
    """Generate all augmented variants of a brick list.

    Returns a list of (suffix, transformed_bricks) pairs.
    The original (identity) is always included with suffix "".
    Geometric variants need re-sorting and re-centering by the caller.

    Args:
        bricks: Original parsed and transformed bricks.
        config: Augmentation configuration.

    Returns:
        List of (suffix, bricks) tuples.
    """
    # Start with geometric transforms (identity is always included)
    geometric_variants: List[Tuple[str, List[RawBrickData]]] = [("", bricks)]

    if config.enable_rotations:
        for suffix, matrix in _ROTATIONS:
            geometric_variants.append((suffix, apply_global_transform(bricks, matrix)))

    if config.enable_mirror:
        # Mirror X alone
        mirrored = apply_global_transform(bricks, MIRROR_X)
        geometric_variants.append((_MIRROR[0], mirrored))

        # Mirror X composed with each rotation
        if config.enable_rotations:
            for rot_suffix, rot_matrix in _ROTATIONS:
                composed = MIRROR_X @ rot_matrix
                suffix = f"_mirX{rot_suffix}"
                geometric_variants.append(
                    (suffix, apply_global_transform(bricks, composed))
                )

    # Add permutation variants
    if config.num_permutations <= 1:
        return geometric_variants

    all_variants = []
    for geo_suffix, geo_bricks in geometric_variants:
        # Original ordering (permutation 1)
        all_variants.append((geo_suffix, geo_bricks))
        # Additional permutations
        for p in range(2, config.num_permutations + 1):
            all_variants.append((f"{geo_suffix}_p{p}", geo_bricks))

    return all_variants
