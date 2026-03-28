from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.geometry.col_parser import CollisionBox


class SpatialHash:
    """Voxel-based spatial hash for fast collision detection.

    Each placed brick occupies a set of voxels derived from its collision boxes.
    Overlap queries return candidate brick indices that share at least one voxel.

    Attributes:
        cell_size: Side length of each cubic voxel in LDU (default 8.0,
            matching plate height — the finest standard LEGO grid dimension).
    """

    def __init__(self, cell_size: float = 8.0) -> None:
        self.cell_size = cell_size
        self._grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        self._brick_voxels: dict[int, set[tuple[int, int, int]]] = {}

    def _pos_to_voxel(self, pos: np.ndarray) -> tuple[int, int, int]:
        return (
            math.floor(pos[0] / self.cell_size),
            math.floor(pos[1] / self.cell_size),
            math.floor(pos[2] / self.cell_size),
        )

    def _boxes_to_voxels(
        self,
        boxes: list[CollisionBox],
        world_matrix: np.ndarray,
    ) -> set[tuple[int, int, int]]:
        """Enumerate all voxels covered by the world-space AABBs of *boxes*."""
        if not boxes:
            # Fallback: single voxel at the brick's position.
            pos = world_matrix[:3, 3]
            return {self._pos_to_voxel(pos)}

        voxels: set[tuple[int, int, int]] = set()
        cs = self.cell_size
        for box in boxes:
            bb_min, bb_max = box.world_aabb(world_matrix)
            ix_min = math.floor(bb_min[0] / cs)
            iy_min = math.floor(bb_min[1] / cs)
            iz_min = math.floor(bb_min[2] / cs)
            ix_max = math.floor(bb_max[0] / cs)
            iy_max = math.floor(bb_max[1] / cs)
            iz_max = math.floor(bb_max[2] / cs)
            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    for iz in range(iz_min, iz_max + 1):
                        voxels.add((ix, iy, iz))
        return voxels

    def insert(
        self,
        brick_idx: int,
        boxes: list[CollisionBox],
        world_matrix: np.ndarray,
    ) -> None:
        """Register a brick in the spatial hash."""
        voxels = self._boxes_to_voxels(boxes, world_matrix)
        self._brick_voxels[brick_idx] = voxels
        for v in voxels:
            self._grid[v].append(brick_idx)

    def check_overlap(
        self,
        boxes: list[CollisionBox],
        world_matrix: np.ndarray,
        exclude: int | None = None,
    ) -> list[int]:
        """Return brick indices that share at least one voxel with *boxes*."""
        voxels = self._boxes_to_voxels(boxes, world_matrix)
        candidates: set[int] = set()
        for v in voxels:
            for idx in self._grid.get(v, ()):
                if idx != exclude:
                    candidates.add(idx)
        return sorted(candidates)

    def remove(self, brick_idx: int) -> None:
        """Remove a brick from the spatial hash (for backtracking)."""
        voxels = self._brick_voxels.pop(brick_idx, set())
        for v in voxels:
            cell = self._grid.get(v)
            if cell is not None:
                try:
                    cell.remove(brick_idx)
                except ValueError:
                    pass
                if not cell:
                    del self._grid[v]

    def clear(self) -> None:
        """Remove all bricks."""
        self._grid.clear()
        self._brick_voxels.clear()
