import numpy as np
import pytest

from src.geometry.col_parser import ColParser, CollisionBox


class TestCollisionBox:
    def test_world_aabb_identity(self):
        box = CollisionBox(
            box_type=9,
            rotation=np.eye(3),
            center=np.array([0.0, 2.0, 0.0]),
            half_extents=np.array([10.0, 1.0, 10.0]),
        )
        bb_min, bb_max = box.world_aabb(np.eye(4))
        np.testing.assert_allclose(bb_min, [-10.0, 1.0, -10.0])
        np.testing.assert_allclose(bb_max, [10.0, 3.0, 10.0])

    def test_world_aabb_translated(self):
        box = CollisionBox(
            box_type=9,
            rotation=np.eye(3),
            center=np.array([0.0, 0.0, 0.0]),
            half_extents=np.array([5.0, 5.0, 5.0]),
        )
        m = np.eye(4)
        m[:3, 3] = [100.0, 200.0, 300.0]
        bb_min, bb_max = box.world_aabb(m)
        np.testing.assert_allclose(bb_min, [95.0, 195.0, 295.0])
        np.testing.assert_allclose(bb_max, [105.0, 205.0, 305.0])


class TestColParser:
    def test_parse_nonexistent_returns_empty(self):
        parser = ColParser()
        boxes = parser.parse("nonexistent_part_99999")
        assert boxes == []
