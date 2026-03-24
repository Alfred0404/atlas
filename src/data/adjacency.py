"""Adjacency-based brick sorting using BFS closest-first traversal.

Sorts bricks so that consecutive bricks in the sequence are spatially
close to each other, teaching the model to build outward from a seed.
"""

import heapq
from typing import List

import numpy as np
from scipy.spatial import KDTree

from ..config import Config
from ..maths.transforms import get_position_from_world_matrix
from ..utils.logging import setup_logging
from .parser import RawBrickData

logger = setup_logging()


def extract_positions(bricks: List[RawBrickData]) -> np.ndarray:
    """Extract (N, 3) position array from a list of RawBrickData."""
    return np.array([get_position_from_world_matrix(b.world_matrix) for b in bricks])


def _normalize_positions(positions: np.ndarray) -> np.ndarray:
    """Normalize positions by grid step sizes for isotropic distance computation."""
    grid = Config.ADJACENCY_GRID_STEPS
    scales = np.array([grid["x"], grid["y"], grid["z"]])
    return positions / scales


def build_adjacency_graph(
    positions: np.ndarray, threshold: float
) -> List[List[int]]:
    """Build an adjacency list using KDTree on normalized positions.

    Args:
        positions: (N, 3) array of brick positions in LDU.
        threshold: Max normalized Chebyshev distance for adjacency.

    Returns:
        Adjacency list where adj[i] contains indices of neighbors of brick i.
    """
    normed = _normalize_positions(positions)

    # KDTree with Chebyshev (infinity norm) metric
    tree = KDTree(normed)
    neighbors = tree.query_ball_tree(tree, r=threshold, p=np.inf)

    # Remove self-loops
    adj = []
    for i, nbrs in enumerate(neighbors):
        adj.append([j for j in nbrs if j != i])

    return adj


def sort_bricks_by_adjacency(
    bricks: List[RawBrickData],
    threshold: float = None,
    rng: np.random.Generator = None,
) -> List[RawBrickData]:
    """Sort bricks via BFS closest-first from the base of the model.

    Args:
        bricks: List of RawBrickData to sort.
        threshold: Adjacency threshold in normalized grid units.
                   Defaults to Config.ADJACENCY_THRESHOLD.
        rng: Optional random generator for shuffling neighbors before
             pushing to the heap. Produces different BFS orderings
             for data augmentation.

    Returns:
        Bricks reordered by BFS visit order.
    """
    if len(bricks) <= 1:
        return bricks

    if threshold is None:
        threshold = Config.ADJACENCY_THRESHOLD

    positions = extract_positions(bricks)
    normed = _normalize_positions(positions)
    adj = build_adjacency_graph(positions, threshold)

    # Seed: brick with max Y (LDraw Y points down, so max Y = physical base)
    # Break ties with (X, Z) for determinism
    seed = max(
        range(len(bricks)),
        key=lambda i: (positions[i][1], positions[i][0], positions[i][2]),
    )

    # BFS with min-heap sorted by normalized distance to parent
    visited = set()
    order = []
    # Heap entries: (distance, tie_breaker, brick_index)
    counter = 0
    heap = [(0.0, counter, seed)]

    while len(order) < len(bricks):
        # Drain heap to find next unvisited brick
        found = False
        while heap:
            dist, _, idx = heapq.heappop(heap)
            if idx not in visited:
                visited.add(idx)
                order.append(idx)
                found = True

                # Push unvisited neighbors
                neighbors = [nbr for nbr in adj[idx] if nbr not in visited]
                if rng is not None:
                    rng.shuffle(neighbors)
                for nbr in neighbors:
                    d = np.max(np.abs(normed[idx] - normed[nbr]))
                    counter += 1
                    heapq.heappush(heap, (d, counter, nbr))
                break

        # Disconnected component: find closest unvisited brick to any visited brick
        if not found:
            visited_positions = normed[list(visited)]
            unvisited = [i for i in range(len(bricks)) if i not in visited]
            unvisited_positions = normed[unvisited]

            # Find the unvisited brick closest to any visited brick
            tree = KDTree(visited_positions)
            dists, _ = tree.query(unvisited_positions, p=np.inf)
            closest_idx = unvisited[np.argmin(dists)]

            counter += 1
            heapq.heappush(heap, (0.0, counter, closest_idx))

    logger.debug(f"Adjacency sort: {len(bricks)} bricks, seed at index {seed}")
    return [bricks[i] for i in order]
