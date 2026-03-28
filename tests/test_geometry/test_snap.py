import numpy as np

from src.data.parser import RawBrickData
from src.geometry.lego_part import PartDatabase
from src.geometry.snap import check_snap, find_all_connections


class TestCheckSnap:
    def setup_method(self):
        self.part_db = PartDatabase()

    def test_stacked_3021_connects(self):
        """Two 2x3 plates stacked with -8 LDU Y offset should connect."""
        bottom = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)
        top_matrix = np.eye(4)
        top_matrix[1, 3] = -8.0
        top = RawBrickData(brick_id="3021", world_matrix=top_matrix, color=4)

        matches = check_snap(self.part_db, bottom, top)
        assert len(matches) > 0, "Stacked 3021 plates should have snap connections"

    def test_far_apart_no_connection(self):
        """Two bricks far apart should have no connections."""
        a = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)
        m = np.eye(4)
        m[:3, 3] = [1000.0, 0.0, 0.0]
        b = RawBrickData(brick_id="3021", world_matrix=m, color=4)

        # Far apart in XZ but proximity_threshold in find_all_connections
        # filters them. Here check_snap alone still checks Y-level which
        # might match, so we test via find_all_connections instead.
        results = find_all_connections(self.part_db, [a, b], proximity_threshold=50.0)
        assert len(results) == 0

    def test_partial_overlap_fewer_connections(self):
        """Two bricks offset by 1 stud should have fewer connections than full overlap."""
        a = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)

        full_m = np.eye(4)
        full_m[1, 3] = -8.0
        full = RawBrickData(brick_id="3021", world_matrix=full_m, color=4)

        offset_m = np.eye(4)
        offset_m[1, 3] = -8.0
        offset_m[0, 3] = 20.0  # 1 stud offset in X
        offset = RawBrickData(brick_id="3021", world_matrix=offset_m, color=4)

        full_matches = check_snap(self.part_db, a, full)
        offset_matches = check_snap(self.part_db, a, offset)
        assert len(offset_matches) < len(full_matches)


class TestFindAllConnections:
    def test_two_stacked_bricks(self):
        part_db = PartDatabase()
        bottom = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)
        top_matrix = np.eye(4)
        top_matrix[1, 3] = -8.0
        top = RawBrickData(brick_id="3021", world_matrix=top_matrix, color=4)

        results = find_all_connections(part_db, [bottom, top])
        assert len(results) > 0
        idx_a, idx_b, matched = results[0]
        assert {idx_a, idx_b} == {0, 1}
        assert len(matched) > 0
