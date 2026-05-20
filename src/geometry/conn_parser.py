from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.geometry.port import Port

# LDraw parts library root.
_DEFAULT_PARTS_DIR = Path("C:/Users/Public/Documents/LDraw/parts")

# Primitive names that represent a stud (male connection point).
_STUD_NAMES = frozenset(
    {
        # Standard open/solid studs
        "stud.dat",
        "stud2.dat",
        "stud2a.dat",
        "stud2s.dat",
        "stud2s2.dat",
        "studa.dat",
        # Logo variants — functionally identical to stud/stud2
        "stud-logo.dat",
        "stud-logo2.dat",
        "stud-logo3.dat",
        "stud-logo4.dat",
        "stud-logo5.dat",
        "stud2-logo.dat",
        "stud2-logo2.dat",
        "stud2-logo3.dat",
        "stud2-logo4.dat",
        "stud2-logo5.dat",
        # SNOT and specialty studs
        "stud10.dat",
        "stud13.dat",
        "stud15.dat",
        "stud16.dat",
        "stud16a.dat",
        "stud17.dat",
        "stud17a.dat",
        "stud20.dat",
        "studp01.dat",
    }
)

# Primitive names that represent an anti-stud / underside tube (female).
_ANTISTUD_NAMES = frozenset(
    {
        # Standard tubes
        "stud3.dat",
        "stud3a.dat",
        "stud4.dat",
        "stud4a.dat",
        "stud4h.dat",
        "stud4o.dat",
        "stud4od.dat",
        "stud4oda.dat",
        "stud4s.dat",
        "stud4s2.dat",
        # Fraction variants for curved / fan-shaped parts
        "1-16stud4.dat",
        "1-4stud4.dat",
        "1-8stud4.dat",
        "2-4stud4.dat",
        "2-4stud4a.dat",
        "2-4stud4f1w.dat",
        "3-16stud4.dat",
        "3-16stud4t4.dat",
        "3-4stud4.dat",
        "5-16stud4.dat",
        # Partial-width stud4 faces
        "stud4f1n.dat",
        "stud4f1s.dat",
        "stud4f1w.dat",
        "stud4f2n.dat",
        "stud4f2s.dat",
        "stud4f2w.dat",
        "stud4f3n.dat",
        "stud4f3s.dat",
        "stud4f4n.dat",
        "stud4f4s.dat",
        "stud4f5n.dat",
    }
)

# Max recursion depth when resolving subfiles.
_MAX_DEPTH = 4


def _parse_type1_line(line: str) -> tuple[np.ndarray, str] | None:
    """Parse an LDraw Type-1 line: ``1 color x y z a b c d e f g h i file``.

    Returns ``(matrix_4x4, filename)`` or ``None`` on failure.
    """
    parts = line.split()
    if len(parts) < 15 or parts[0] != "1":
        return None

    try:
        vals = [float(parts[i]) for i in range(2, 14)]
    except ValueError:
        return None

    x, y, z = vals[0], vals[1], vals[2]
    a, b, c = vals[3], vals[4], vals[5]
    d, e, f = vals[6], vals[7], vals[8]
    g, h, i = vals[9], vals[10], vals[11]

    matrix = np.array(
        [
            [a, b, c, x],
            [d, e, f, y],
            [g, h, i, z],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    filename = " ".join(parts[14:]).lower().replace("\\", "/")
    return matrix, filename


class ConnParser:
    """Extracts stud and anti-stud positions from LDraw .dat part files.

    Primary source: .dat files from the LDraw parts library.  The parser
    recursively resolves subfiles (``s/`` subparts) up to a limited depth,
    accumulating transformation matrices to produce stud positions in the
    part's local coordinate system.

    For each ``stud.dat`` reference found, a **male** port is created at
    the reference position.  A matching **female** port is generated at the
    same (x, z) but at the bottom Y of the part (inferred from the
    ``stud4.dat`` / ``stud3.dat`` references, or offset by 8 LDU as
    fallback).
    """

    def __init__(self, parts_dir: str | Path | None = None) -> None:
        if parts_dir is None:
            parts_dir = _DEFAULT_PARTS_DIR
        self.parts_dir = Path(parts_dir)
        # LDraw primitives live in a sibling ``p/`` directory.
        self._primitives_dir = self.parts_dir.parent / "p"

    def _resolve_dat(self, filename: str) -> Path | None:
        """Resolve an LDraw filename to a path on disk.

        Searches in ``parts/``, ``parts/s/``, and ``p/`` (primitives).
        """
        filename = filename.replace("\\", "/")
        candidates = [
            self.parts_dir / filename,
            self.parts_dir / filename.lower(),
            self._primitives_dir / filename,
            self._primitives_dir / filename.lower(),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _get_bottom_y(self, filename: str, depth: int = 0) -> float:
        """Determine the bottom Y of a part by scanning geometry vertices.

        Recursively follows part subfile references (in ``parts/`` and
        ``parts/s/``) but **not** primitives from ``p/`` — those use
        normalised unit coordinates that would give wrong results when
        scaled.

        Returns the maximum Y coordinate found, which in LDraw's Y-down
        system corresponds to the physical bottom of the part.
        """
        if depth > _MAX_DEPTH:
            return 0.0

        path = self._resolve_dat(filename)
        if path is None:
            return 0.0

        max_y = 0.0
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split()
            if not parts:
                continue

            line_type = parts[0]

            # Geometry lines (triangles, quads, optional lines) have vertices.
            if line_type in ("3", "4", "5"):
                try:
                    coords = [float(x) for x in parts[2:]]
                    for i in range(1, len(coords), 3):
                        if coords[i] > max_y:
                            max_y = coords[i]
                except (ValueError, IndexError):
                    pass
            elif line_type == "1" and len(parts) >= 15:
                ref_name = " ".join(parts[14:]).lower().replace("\\", "/")
                base_name = ref_name.split("/")[-1]
                if base_name in _STUD_NAMES or base_name in _ANTISTUD_NAMES:
                    continue
                # Only recurse into part subfiles, not primitives
                # (primitives use normalised unit coordinates).
                ref_path = self._resolve_dat(ref_name)
                if ref_path is None:
                    continue
                if self._primitives_dir in ref_path.parents:
                    continue
                parsed = _parse_type1_line(line)
                if parsed is None:
                    continue
                local_matrix, _ = parsed
                y_trans = local_matrix[1, 3]
                sub_y = self._get_bottom_y(ref_name, depth + 1)
                if sub_y > 0:
                    y_scale = np.sqrt(
                        local_matrix[1, 0] ** 2
                        + local_matrix[1, 1] ** 2
                        + local_matrix[1, 2] ** 2
                    )
                    max_y = max(max_y, y_trans + y_scale * sub_y)
                else:
                    max_y = max(max_y, y_trans)

        return max_y

    def _collect_studs(
        self,
        filename: str,
        parent_matrix: np.ndarray,
        depth: int,
    ) -> tuple[
        list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]
    ]:
        """Recursively collect stud and anti-stud positions with their normals.

        Returns ``(males, females)`` where each entry is a ``(position, normal)``
        tuple.  The normal is derived from the transformation matrix: a stud
        protrudes in its local ``-Y`` direction, so the world-space normal is
        ``-R[:,1]`` (the negated Y-column of the rotation part).  This correctly
        captures both vertical studs (normal ≈ [0,-1,0]) and SNOT studs whose
        transformation rotates the Y-axis to a horizontal direction.
        """
        if depth > _MAX_DEPTH:
            return [], []

        path = self._resolve_dat(filename)
        if path is None:
            return [], []

        males: list[tuple[np.ndarray, np.ndarray]] = []
        females: list[tuple[np.ndarray, np.ndarray]] = []

        for line in path.read_text(errors="replace").splitlines():
            parsed = _parse_type1_line(line)
            if parsed is None:
                continue

            local_matrix, ref_name = parsed
            world_matrix = parent_matrix @ local_matrix
            base_name = ref_name.split("/")[-1]
            base = base_name.lower()

            # Group primitives such as ``stug-1x4.dat`` expand into multiple
            # terminal ``stud.dat`` / ``stud3.dat`` references, so recurse into
            # them instead of treating the wrapper as a single port.
            resolved = self._resolve_dat(ref_name)
            if resolved is not None and self._primitives_dir in resolved.parents:
                if base not in _STUD_NAMES and base not in _ANTISTUD_NAMES:
                    sub_m, sub_f = self._collect_studs(
                        ref_name, world_matrix, depth + 1
                    )
                    males.extend(sub_m)
                    females.extend(sub_f)
                    continue

            if base in _STUD_NAMES:
                pos = world_matrix[:3, 3].copy()
                y_col = world_matrix[:3, 1]
                length = np.linalg.norm(y_col)
                normal = (
                    -y_col / length if length > 1e-8 else np.array([0.0, -1.0, 0.0])
                )
                males.append((pos, normal))
            elif base in _ANTISTUD_NAMES:
                pos = world_matrix[:3, 3].copy()
                y_col = world_matrix[:3, 1]
                length = np.linalg.norm(y_col)
                normal = y_col / length if length > 1e-8 else np.array([0.0, 1.0, 0.0])
                females.append((pos, normal))
            elif base_name.endswith(".dat"):
                # Recurse into subfiles (e.g. s/3001s01.dat).
                sub_m, sub_f = self._collect_studs(ref_name, world_matrix, depth + 1)
                males.extend(sub_m)
                females.extend(sub_f)

        return males, females

    def parse(self, part_id: str) -> list[Port]:
        """Parse a .dat file and return stud/anti-stud ports.

        Male ports come from detected stud primitives; their normals are derived
        from the transformation matrix so SNOT (horizontal) studs get the correct
        non-vertical normal.

        Female ports are handled differently by orientation:
        - **Vertical** males (|normal.y| > 0.7): the female is inferred at the
          same (x, z) but at the part's physical bottom Y.  stud4.dat references
          are placed inside the brick, not at the mating surface, so we use
          ``_get_bottom_y()`` for the correct Y.
        - **Horizontal** males (SNOT): no inferred female is generated.  Instead,
          anti-stud primitives with a horizontal normal are added directly as
          female ports using their actual positions.
        """
        filename = f"{part_id}.dat"
        males, raw_females = self._collect_studs(filename, np.eye(4), 0)

        if not males:
            return []

        # Deduplicate males by rounded position.
        seen_m: set[tuple[float, float, float]] = set()
        unique_males: list[tuple[np.ndarray, np.ndarray]] = []
        for pos, normal in males:
            key = (round(pos[0], 0), round(pos[1], 0), round(pos[2], 0))
            if key not in seen_m:
                seen_m.add(key)
                unique_males.append((pos, normal))

        vertical_males = [(p, n) for p, n in unique_males if abs(n[1]) > 0.7]
        horiz_males = [(p, n) for p, n in unique_males if abs(n[1]) <= 0.7]

        ports: list[Port] = []
        port_id = 0

        # --- Vertical ports (standard top/bottom connections) ---
        if vertical_males:
            bottom_y = self._get_bottom_y(filename)
            y_male = float(np.min([p[1] for p, _ in vertical_males]))
            y_female = bottom_y if bottom_y > y_male + 1.0 else y_male + 8.0

            for pos, normal in vertical_males:
                ports.append(
                    Port(
                        port_id=port_id,
                        local_position=pos,
                        normal=normal,
                        port_type="male",
                    )
                )
                port_id += 1
                ports.append(
                    Port(
                        port_id=port_id,
                        local_position=np.array([pos[0], y_female, pos[2]]),
                        normal=-normal,
                        port_type="female",
                    )
                )
                port_id += 1

        # --- Horizontal (SNOT) male ports ---
        for pos, normal in horiz_males:
            ports.append(
                Port(
                    port_id=port_id,
                    local_position=pos,
                    normal=normal,
                    port_type="male",
                )
            )
            port_id += 1

        # --- Horizontal female ports from detected anti-stud primitives ---
        # Only add horizontal anti-studs; vertical ones sit inside the brick
        # at the wrong Y and are already covered by the inferred vertical females.
        seen_f: set[tuple[float, float, float]] = set()
        for pos, normal in raw_females:
            if abs(normal[1]) <= 0.7:
                key = (round(pos[0], 0), round(pos[1], 0), round(pos[2], 0))
                if key not in seen_f:
                    seen_f.add(key)
                    ports.append(
                        Port(
                            port_id=port_id,
                            local_position=pos,
                            normal=normal,
                            port_type="female",
                        )
                    )
                    port_id += 1

        return ports
