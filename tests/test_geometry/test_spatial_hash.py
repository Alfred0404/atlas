import numpy as np

from src.geometry.col_parser import CollisionBox
from src.geometry.spatial_hash import SpatialHash


def _make_box(center, half_extents):
    return CollisionBox(
        box_type=9,
        rotation=np.eye(3),
        center=np.array(center, dtype=float),
        half_extents=np.array(half_extents, dtype=float),
    )


class TestSpatialHash:
    def test_insert_and_query_overlap(self):
        sh = SpatialHash(cell_size=8.0)
        box = _make_box([0, 0, 0], [10, 4, 10])
        sh.insert(0, [box], np.eye(4))

        # Same location should overlap
        overlaps = sh.check_overlap([box], np.eye(4))
        assert 0 in overlaps

    def test_no_overlap_far_away(self):
        sh = SpatialHash(cell_size=8.0)
        box = _make_box([0, 0, 0], [5, 5, 5])
        sh.insert(0, [box], np.eye(4))

        far_matrix = np.eye(4)
        far_matrix[:3, 3] = [500, 0, 0]
        overlaps = sh.check_overlap([box], far_matrix)
        assert 0 not in overlaps

    def test_remove(self):
        sh = SpatialHash(cell_size=8.0)
        box = _make_box([0, 0, 0], [5, 5, 5])
        sh.insert(0, [box], np.eye(4))
        sh.remove(0)

        overlaps = sh.check_overlap([box], np.eye(4))
        assert len(overlaps) == 0

    def test_exclude_self(self):
        sh = SpatialHash(cell_size=8.0)
        box = _make_box([0, 0, 0], [5, 5, 5])
        sh.insert(0, [box], np.eye(4))

        overlaps = sh.check_overlap([box], np.eye(4), exclude=0)
        assert 0 not in overlaps

    def test_clear(self):
        sh = SpatialHash(cell_size=8.0)
        box = _make_box([0, 0, 0], [5, 5, 5])
        sh.insert(0, [box], np.eye(4))
        sh.insert(1, [box], np.eye(4))
        sh.clear()
        assert len(sh._grid) == 0
        assert len(sh._brick_voxels) == 0

    def test_empty_boxes_fallback_to_position(self):
        sh = SpatialHash(cell_size=8.0)
        m = np.eye(4)
        m[:3, 3] = [10, 10, 10]
        sh.insert(0, [], m)
        assert 0 in sh._brick_voxels
