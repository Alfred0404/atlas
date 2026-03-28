from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CollisionBox:
    """An oriented bounding box from a .col collision file.

    Attributes:
        box_type: 9 = axis-aligned box, 8192 = oriented bounding box.
        rotation: (3,3) rotation matrix.
        center: (3,) center position in part-local LDU.
        half_extents: (3,) half-sizes along each axis.
    """

    box_type: int
    rotation: np.ndarray  # (3, 3)
    center: np.ndarray  # (3,)
    half_extents: np.ndarray  # (3,)

    def world_aabb(self, world_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the world-space axis-aligned bounding box.

        Returns (min_corner, max_corner) each of shape (3,).
        """
        R_world = world_matrix[:3, :3]
        t_world = world_matrix[:3, 3]

        world_center = R_world @ self.center + t_world
        # Composed rotation of the box in world space
        R_composed = R_world @ self.rotation
        # Half-extent contribution along each world axis
        abs_r = np.abs(R_composed) @ self.half_extents

        return world_center - abs_r, world_center + abs_r


class ColParser:
    """Parser for LDraw .col collision files."""

    def __init__(self, col_dir: str | Path | None = None) -> None:
        if col_dir is None:
            col_dir = Path(__file__).resolve().parent.parent / "collider"
        self.col_dir = Path(col_dir)

    def _resolve_path(self, part_id: str) -> Path | None:
        for name in (f"_{part_id}.col", f"{part_id}.col"):
            p = self.col_dir / name
            if p.exists():
                return p
        return None

    def parse(self, part_id: str) -> list[CollisionBox]:
        """Parse a .col file and return its collision boxes."""
        path = self._resolve_path(part_id)
        if path is None:
            return []

        boxes: list[CollisionBox] = []
        for line in path.read_text().splitlines():
            tokens = line.split()
            if len(tokens) < 17:
                continue

            box_type = int(tokens[0])
            # tokens[1] = flags (unused)
            rot = np.array([float(tokens[i]) for i in range(2, 11)]).reshape(3, 3)
            center = np.array([float(tokens[i]) for i in range(11, 14)])
            half_extents = np.array([float(tokens[i]) for i in range(14, 17)])

            boxes.append(
                CollisionBox(
                    box_type=box_type,
                    rotation=rot,
                    center=center,
                    half_extents=half_extents,
                )
            )

        return boxes
