import numpy as np

from src.data.parser import RawBrickData
from src.geometry.lego_core import LegoCore
from src.geometry.lego_part import PartDatabase


class TestLegoCore:
    def setup_method(self):
        self.part_db = PartDatabase()

    def test_place_single_brick(self):
        core = LegoCore(self.part_db)
        nid = core.place_brick("3021", 16, np.eye(4))
        assert nid is not None
        assert nid in core.nodes

    def test_place_two_stacked_bricks_creates_edges(self):
        """Stacked bricks placed via from_raw_bricks (no validation) should connect."""
        bottom = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)
        top_m = np.eye(4)
        top_m[1, 3] = -8.0
        top = RawBrickData(brick_id="3021", world_matrix=top_m, color=4)

        core = LegoCore.from_raw_bricks([bottom, top], self.part_db)
        nodes, edges = core.get_graph()
        assert len(nodes) == 2
        assert len(edges) > 0

    def test_validate_placement_no_collision(self):
        core = LegoCore(self.part_db)
        core.place_brick("3021", 16, np.eye(4))

        far = np.eye(4)
        far[:3, 3] = [200, 0, 0]
        result = core.validate_placement("3021", far)
        assert result["valid"] is True

    def test_validate_detects_collision(self):
        core = LegoCore(self.part_db)
        core.place_brick("3021", 16, np.eye(4))
        result = core.validate_placement("3021", np.eye(4))
        assert result["valid"] is False

    def test_remove_brick(self):
        core = LegoCore(self.part_db)
        nid = core.place_brick("3021", 16, np.eye(4))
        core.remove_brick(nid)
        assert nid not in core.nodes
        assert len(core.edges) == 0

    def test_from_raw_bricks(self):
        bottom = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)
        top_m = np.eye(4)
        top_m[1, 3] = -8.0
        top = RawBrickData(brick_id="3021", world_matrix=top_m, color=4)

        core = LegoCore.from_raw_bricks([bottom, top], self.part_db)
        nodes, edges = core.get_graph()
        assert len(nodes) == 2
        assert len(edges) > 0

    def test_to_raw_brick_data(self):
        core = LegoCore(self.part_db)
        core.place_brick("3021", 16, np.eye(4))
        raw = core.to_raw_brick_data()
        assert len(raw) == 1
        assert raw[0].brick_id == "3021"

    def test_adjacency_matrix(self):
        bottom = RawBrickData(brick_id="3021", world_matrix=np.eye(4), color=16)
        top_m = np.eye(4)
        top_m[1, 3] = -8.0
        top = RawBrickData(brick_id="3021", world_matrix=top_m, color=4)

        core = LegoCore.from_raw_bricks([bottom, top], self.part_db)
        adj = core.to_adjacency_matrix()
        assert adj.shape == (2, 2)
        assert adj[0, 1] == 1
        assert adj[1, 0] == 1

    def test_place_unknown_part(self):
        """Parts without geometry data should still be placeable."""
        core = LegoCore(self.part_db)
        nid = core.place_brick("unknown_part_99999", 0, np.eye(4))
        assert nid is not None
