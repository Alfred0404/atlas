from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.data.parser import RawBrickData
from src.geometry.lego_part import PartDatabase
from src.geometry.port import Port


def check_snap(
    part_db: PartDatabase,
    brick_a: RawBrickData,
    brick_b: RawBrickData,
    pos_tol: float = 2.0,
    angle_tol: float = 0.75,
) -> list[tuple[Port, Port]]:
    """Check whether two bricks are connected via stud/anti-stud mating.

    Returns a list of matched (port_a, port_b) pairs in world space.
    A match requires:
        - positions within *pos_tol* LDU (default 5.0 accounts for the
          Y offset between stud tops and anti-stud tube centres)
        - normals anti-aligned: ``dot(n_a, n_b) < -angle_tol``
    """
    lp_a = part_db.get(brick_a.brick_id)
    lp_b = part_db.get(brick_b.brick_id)
    if lp_a is None or lp_b is None:
        return []

    world_a_male = [p.transformed(brick_a.world_matrix) for p in lp_a.male_ports]
    world_a_female = [p.transformed(brick_a.world_matrix) for p in lp_a.female_ports]
    world_b_male = [p.transformed(brick_b.world_matrix) for p in lp_b.male_ports]
    world_b_female = [p.transformed(brick_b.world_matrix) for p in lp_b.female_ports]

    matches: list[tuple[Port, Port]] = []

    # Male A <-> Female B
    _match_ports(world_a_male, world_b_female, pos_tol, angle_tol, matches)
    # Female A <-> Male B
    _match_ports(world_a_female, world_b_male, pos_tol, angle_tol, matches)

    return matches


def _match_ports(
    ports_a: list[Port],
    ports_b: list[Port],
    pos_tol: float,
    angle_tol: float,
    out: list[tuple[Port, Port]],
) -> None:
    """Find all mating pairs between two port lists using KDTree."""
    if not ports_a or not ports_b:
        return

    pos_a = np.array([p.local_position for p in ports_a])
    pos_b = np.array([p.local_position for p in ports_b])

    tree_b = cKDTree(pos_b)
    for i, pa in enumerate(ports_a):
        indices = tree_b.query_ball_point(pos_a[i], r=pos_tol)
        for j in indices:
            pb = ports_b[j]
            dot = float(np.dot(pa.normal, pb.normal))
            if dot < -angle_tol:
                out.append((pa, pb))


def find_all_connections(
    part_db: PartDatabase,
    bricks: list[RawBrickData],
    proximity_threshold: float = 80.0,
    pos_tol: float = 2.0,
    angle_tol: float = 0.75,
) -> list[tuple[int, int, list[tuple[Port, Port]]]]:
    """Find all stud/anti-stud connections among a list of bricks.

    Uses a KDTree pre-filter on brick positions to avoid O(N^2) full scans.

    Returns list of ``(idx_a, idx_b, matched_ports)`` for every pair
    with at least one connection.
    """
    if len(bricks) < 2:
        return []

    positions = np.array([b.world_matrix[:3, 3] for b in bricks])
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=proximity_threshold, output_type="ndarray")

    results: list[tuple[int, int, list[tuple[Port, Port]]]] = []
    for i, j in pairs:
        matched = check_snap(part_db, bricks[i], bricks[j], pos_tol, angle_tol)
        if matched:
            results.append((int(i), int(j), matched))

    return results
