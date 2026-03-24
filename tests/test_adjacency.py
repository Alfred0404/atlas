"""Tests for adjacency-based brick sorting."""

import numpy as np
import pytest

from src.data.adjacency import (
    build_adjacency_graph,
    extract_positions,
    sort_bricks_by_adjacency,
)
from src.data.parser import RawBrickData


def _make_brick(x: float, y: float, z: float, brick_id="3001.dat", color=1) -> RawBrickData:
    """Helper to create a RawBrickData at a given position."""
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return RawBrickData(brick_id=brick_id, world_matrix=matrix, color=color)


class TestExtractPositions:
    def test_basic(self):
        bricks = [_make_brick(0, 0, 0), _make_brick(20, 8, 0)]
        pos = extract_positions(bricks)
        assert pos.shape == (2, 3)
        np.testing.assert_array_equal(pos[1], [20, 8, 0])


class TestBuildAdjacencyGraph:
    def test_adjacent_bricks(self):
        """Two bricks 1 stud apart (20 LDU in X) should be adjacent at threshold 1.5."""
        positions = np.array([[0, 0, 0], [20, 0, 0]], dtype=float)
        adj = build_adjacency_graph(positions, threshold=1.5)
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_distant_bricks(self):
        """Two bricks 3 studs apart should NOT be adjacent at threshold 1.5."""
        positions = np.array([[0, 0, 0], [60, 0, 0]], dtype=float)
        adj = build_adjacency_graph(positions, threshold=1.5)
        assert adj[0] == []
        assert adj[1] == []

    def test_vertical_adjacency(self):
        """Two bricks 1 plate apart (8 LDU in Y) should be adjacent."""
        positions = np.array([[0, 0, 0], [0, 8, 0]], dtype=float)
        adj = build_adjacency_graph(positions, threshold=1.5)
        assert 1 in adj[0]


class TestSortBricksByAdjacency:
    def test_l_shape_order(self):
        """L-shaped arrangement: BFS should visit in spatial order from base."""
        # Vertical column (Y increases = lower in LDraw)
        bricks = [
            _make_brick(0, 24, 0),  # base (highest Y = bottom)
            _make_brick(0, 16, 0),
            _make_brick(0, 8, 0),
            _make_brick(0, 0, 0),   # top
            _make_brick(20, 24, 0), # branch right from base
        ]
        sorted_bricks = sort_bricks_by_adjacency(bricks, threshold=1.5)

        # Should start from base (Y=24)
        pos0 = extract_positions([sorted_bricks[0]])[0]
        assert pos0[1] == 24.0, "Should start from the base (max Y)"

        # All bricks should be present (valid permutation)
        assert len(sorted_bricks) == 5

    def test_disconnected_components(self):
        """Two separate clusters should both be fully visited."""
        cluster_a = [_make_brick(0, 0, 0), _make_brick(20, 0, 0)]
        cluster_b = [_make_brick(200, 0, 0), _make_brick(220, 0, 0)]
        bricks = cluster_a + cluster_b

        sorted_bricks = sort_bricks_by_adjacency(bricks, threshold=1.5)
        assert len(sorted_bricks) == 4, "All bricks should be visited"

    def test_single_brick(self):
        bricks = [_make_brick(0, 0, 0)]
        assert sort_bricks_by_adjacency(bricks) == bricks

    def test_empty(self):
        assert sort_bricks_by_adjacency([]) == []

    def test_consecutive_distance_improvement(self):
        """Adjacency sort should produce smaller consecutive distances than Y/X/Z sort."""
        # Grid of bricks: the Y/X/Z sort would zigzag, BFS should be local
        bricks = []
        for x in range(0, 80, 20):
            for z in range(0, 80, 20):
                bricks.append(_make_brick(float(x), 0, float(z)))

        sorted_bricks = sort_bricks_by_adjacency(bricks, threshold=1.5)
        positions = extract_positions(sorted_bricks)

        # Compute mean consecutive distance
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        mean_dist = np.mean(distances)

        # For a 4x4 grid with BFS, mean consecutive distance should be ~20 LDU (1 stud)
        assert mean_dist < 40, f"Mean consecutive distance {mean_dist:.1f} should be < 40 LDU"
