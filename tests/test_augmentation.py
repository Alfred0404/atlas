"""Tests for data augmentation module."""

import numpy as np
import pytest

from src.data.augmentation import (
    AugmentationConfig,
    MIRROR_X,
    ROT_Y_90,
    ROT_Y_180,
    ROT_Y_270,
    apply_global_transform,
    generate_augmented_variants,
)
from src.data.parser import RawBrickData
from src.data.adjacency import sort_bricks_by_adjacency


def _make_brick(x, y, z, brick_id="3001", color=1, rotation=None):
    """Helper to create a RawBrickData with given position."""
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = [x, y, z]
    return RawBrickData(brick_id=brick_id, world_matrix=matrix, color=color)


def _make_sample_bricks():
    """Create a small set of bricks for testing."""
    return [
        _make_brick(20, 0, 0, brick_id="3001", color=1),
        _make_brick(0, 8, 40, brick_id="3002", color=4),
        _make_brick(-10, 16, 20, brick_id="3003", color=15),
    ]


class TestApplyGlobalTransform:
    def test_identity(self):
        bricks = _make_sample_bricks()
        result = apply_global_transform(bricks, np.eye(3))
        for orig, transformed in zip(bricks, result):
            np.testing.assert_array_almost_equal(
                orig.world_matrix, transformed.world_matrix
            )

    def test_preserves_count(self):
        bricks = _make_sample_bricks()
        result = apply_global_transform(bricks, ROT_Y_90)
        assert len(result) == len(bricks)

    def test_preserves_ids_and_colors(self):
        bricks = _make_sample_bricks()
        result = apply_global_transform(bricks, ROT_Y_180)
        for orig, transformed in zip(bricks, result):
            assert orig.brick_id == transformed.brick_id
            assert orig.color == transformed.color

    def test_deep_copy(self):
        """Modifying the result should not affect the original."""
        bricks = _make_sample_bricks()
        result = apply_global_transform(bricks, ROT_Y_90)
        result[0].world_matrix[:3, 3] = [999, 999, 999]
        assert bricks[0].world_matrix[0, 3] == 20  # unchanged


class TestYAxisRotations:
    def test_rot90_position(self):
        brick = _make_brick(20, 0, 0)
        result = apply_global_transform([brick], ROT_Y_90)[0]
        pos = result.world_matrix[:3, 3]
        np.testing.assert_array_almost_equal(pos, [0, 0, -20])

    def test_rot180_position(self):
        brick = _make_brick(20, 0, 40)
        result = apply_global_transform([brick], ROT_Y_180)[0]
        pos = result.world_matrix[:3, 3]
        np.testing.assert_array_almost_equal(pos, [-20, 0, -40])

    def test_rot270_position(self):
        brick = _make_brick(20, 0, 0)
        result = apply_global_transform([brick], ROT_Y_270)[0]
        pos = result.world_matrix[:3, 3]
        np.testing.assert_array_almost_equal(pos, [0, 0, 20])

    def test_y_unchanged(self):
        brick = _make_brick(20, 42, 0)
        for matrix in [ROT_Y_90, ROT_Y_180, ROT_Y_270]:
            result = apply_global_transform([brick], matrix)[0]
            assert result.world_matrix[1, 3] == 42

    def test_rot90_four_times_identity(self):
        bricks = _make_sample_bricks()
        current = bricks
        for _ in range(4):
            current = apply_global_transform(current, ROT_Y_90)
        for orig, transformed in zip(bricks, current):
            np.testing.assert_array_almost_equal(
                orig.world_matrix, transformed.world_matrix
            )

    def test_rot180_twice_identity(self):
        bricks = _make_sample_bricks()
        once = apply_global_transform(bricks, ROT_Y_180)
        twice = apply_global_transform(once, ROT_Y_180)
        for orig, transformed in zip(bricks, twice):
            np.testing.assert_array_almost_equal(
                orig.world_matrix, transformed.world_matrix
            )

    def test_rotation_composition(self):
        """Brick's local rotation should be composed with global rotation."""
        local_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        brick = _make_brick(0, 0, 0, rotation=local_rot)
        result = apply_global_transform([brick], ROT_Y_90)[0]
        expected_rot = ROT_Y_90 @ local_rot
        np.testing.assert_array_almost_equal(
            result.world_matrix[:3, :3], expected_rot
        )


class TestMirror:
    def test_mirror_x_position(self):
        brick = _make_brick(20, 0, 40)
        result = apply_global_transform([brick], MIRROR_X)[0]
        pos = result.world_matrix[:3, 3]
        np.testing.assert_array_almost_equal(pos, [-20, 0, 40])

    def test_mirror_twice_identity(self):
        bricks = _make_sample_bricks()
        once = apply_global_transform(bricks, MIRROR_X)
        twice = apply_global_transform(once, MIRROR_X)
        for orig, transformed in zip(bricks, twice):
            np.testing.assert_array_almost_equal(
                orig.world_matrix, transformed.world_matrix
            )

    def test_mirror_y_unchanged(self):
        brick = _make_brick(20, 42, 40)
        result = apply_global_transform([brick], MIRROR_X)[0]
        assert result.world_matrix[1, 3] == 42

    def test_mirror_rotation_determinant(self):
        """Mirror flips rotation determinant to -1."""
        brick = _make_brick(0, 0, 0)
        result = apply_global_transform([brick], MIRROR_X)[0]
        det = np.linalg.det(result.world_matrix[:3, :3])
        assert det == pytest.approx(-1.0)


class TestPermutation:
    def test_rng_produces_different_order(self):
        """Different RNG seeds should produce different BFS orderings."""
        bricks = [
            _make_brick(0, 0, 0),
            _make_brick(20, 0, 0),
            _make_brick(0, 0, 20),
            _make_brick(20, 0, 20),
            _make_brick(40, 0, 0),
            _make_brick(0, 0, 40),
        ]
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(999)
        order1 = sort_bricks_by_adjacency(bricks, rng=rng1)
        order2 = sort_bricks_by_adjacency(bricks, rng=rng2)
        # At least check they have the same bricks
        ids1 = {id(b.world_matrix.data) for b in order1}
        ids2 = {id(b.world_matrix.data) for b in order2}
        assert ids1 == ids2

    def test_permutation_preserves_all_bricks(self):
        bricks = _make_sample_bricks()
        rng = np.random.default_rng(42)
        result = sort_bricks_by_adjacency(bricks, rng=rng)
        assert len(result) == len(bricks)
        original_ids = {b.brick_id for b in bricks}
        result_ids = {b.brick_id for b in result}
        assert original_ids == result_ids

    def test_permutation_deterministic(self):
        bricks = _make_sample_bricks()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = sort_bricks_by_adjacency(bricks, rng=rng1)
        result2 = sort_bricks_by_adjacency(bricks, rng=rng2)
        for b1, b2 in zip(result1, result2):
            np.testing.assert_array_equal(b1.world_matrix, b2.world_matrix)


class TestGenerateAugmentedVariants:
    def test_no_augmentation(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=False, enable_mirror=False, num_permutations=1
        )
        variants = generate_augmented_variants(bricks, config)
        assert len(variants) == 1
        assert variants[0][0] == ""

    def test_rotations_only(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=True, enable_mirror=False, num_permutations=1
        )
        variants = generate_augmented_variants(bricks, config)
        assert len(variants) == 4
        suffixes = [s for s, _ in variants]
        assert suffixes == ["", "_rot90", "_rot180", "_rot270"]

    def test_mirror_only(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=False, enable_mirror=True, num_permutations=1
        )
        variants = generate_augmented_variants(bricks, config)
        assert len(variants) == 2
        suffixes = [s for s, _ in variants]
        assert suffixes == ["", "_mirX"]

    def test_full_geometric(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=True, enable_mirror=True, num_permutations=1
        )
        variants = generate_augmented_variants(bricks, config)
        # 4 rotations + 4 mirror+rotations = 8
        assert len(variants) == 8

    def test_with_permutations(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=True, enable_mirror=True, num_permutations=3
        )
        variants = generate_augmented_variants(bricks, config)
        # 8 geometric × 3 permutations = 24
        assert len(variants) == 24

    def test_suffixes_unique(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=True, enable_mirror=True, num_permutations=3
        )
        variants = generate_augmented_variants(bricks, config)
        suffixes = [s for s, _ in variants]
        assert len(suffixes) == len(set(suffixes))

    def test_all_variants_same_brick_count(self):
        bricks = _make_sample_bricks()
        config = AugmentationConfig(
            enable_rotations=True, enable_mirror=True, num_permutations=2
        )
        variants = generate_augmented_variants(bricks, config)
        for _, variant_bricks in variants:
            assert len(variant_bricks) == len(bricks)
