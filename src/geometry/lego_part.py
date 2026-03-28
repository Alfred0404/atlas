from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from src.geometry.col_parser import ColParser, CollisionBox
from src.geometry.conn_parser import ConnParser
from src.geometry.port import Port

logger = logging.getLogger(__name__)

# Default bounding box for a 1x1 brick (20x24x20 LDU) when no geometry data.
_DEFAULT_BB_MIN = np.array([-10.0, -24.0, -10.0])
_DEFAULT_BB_MAX = np.array([10.0, 0.0, 10.0])


class LegoPart:
    """Geometric description of a single LEGO part.

    Combines connection ports (from .dat) and collision volumes (from .col)
    into one unified representation.
    """

    def __init__(
        self,
        part_id: str,
        ports: list[Port],
        collision_boxes: list[CollisionBox],
    ) -> None:
        self.part_id = part_id
        self.ports = ports
        self.collision_boxes = collision_boxes

        self.male_ports = [p for p in ports if p.port_type == "male"]
        self.female_ports = [p for p in ports if p.port_type == "female"]

        self.bounding_box = self._compute_bounding_box()
        self.y_max = float(self.bounding_box[1][1])  # physical bottom (LDraw Y+)

    def _compute_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box in local space."""
        if self.collision_boxes:
            all_min = []
            all_max = []
            identity = np.eye(4)
            for box in self.collision_boxes:
                bb_min, bb_max = box.world_aabb(identity)
                all_min.append(bb_min)
                all_max.append(bb_max)
            return np.min(all_min, axis=0), np.max(all_max, axis=0)

        if self.ports:
            positions = np.array([p.local_position for p in self.ports])
            margin = 6.0  # stud radius
            return positions.min(axis=0) - margin, positions.max(axis=0) + margin

        return _DEFAULT_BB_MIN.copy(), _DEFAULT_BB_MAX.copy()

    @classmethod
    def from_files(
        cls,
        part_id: str,
        conn_parser: ConnParser,
        col_parser: ColParser,
    ) -> LegoPart:
        """Build a LegoPart by parsing its .dat and .col files."""
        ports = conn_parser.parse(part_id)
        boxes = col_parser.parse(part_id)
        return cls(part_id=part_id, ports=ports, collision_boxes=boxes)


class PartDatabase:
    """Lazy-loading cache of LegoPart geometry.

    Parts are parsed from LDraw .dat files (for stud positions) and .col
    files (for collision boxes) on first access and cached in memory.
    """

    def __init__(
        self,
        parts_dir: str | Path | None = None,
        col_dir: str | Path | None = None,
    ) -> None:
        self._conn_parser = ConnParser(parts_dir)
        self._col_parser = ColParser(col_dir)
        self._cache: dict[str, LegoPart] = {}

    def get(self, part_id: str) -> LegoPart | None:
        """Return cached LegoPart or parse from disk.  None if no data."""
        if part_id in self._cache:
            return self._cache[part_id]

        part = LegoPart.from_files(part_id, self._conn_parser, self._col_parser)
        if part.ports or part.collision_boxes:
            self._cache[part_id] = part
            return part
        return None

    def get_or_default(self, part_id: str) -> LegoPart:
        """Return LegoPart, creating a geometry-less stub if needed."""
        cached = self.get(part_id)
        if cached is not None:
            return cached
        stub = LegoPart(part_id=part_id, ports=[], collision_boxes=[])
        self._cache[part_id] = stub
        return stub

    def has_geometry(self, part_id: str) -> bool:
        """Check whether a .dat file exists for *part_id*."""
        return self._conn_parser._resolve_dat(f"{part_id}.dat") is not None

    def save_cache(self, path: str | Path) -> None:
        """Persist the cache to a pickle file."""
        with open(path, "wb") as fh:
            pickle.dump(self._cache, fh)

    def load_cache(self, path: str | Path) -> None:
        """Restore the cache from a pickle file."""
        with open(path, "rb") as fh:
            self._cache = pickle.load(fh)
