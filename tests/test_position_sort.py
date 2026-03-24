"""Tests for deterministic positional brick ordering."""

from pathlib import Path

import numpy as np

from src.data.builder import DatasetBuilder
from src.data.parser import MPDParser, RawBrickData


def _make_brick(
    x: float, y: float, z: float, brick_id: str = "3001", color: int = 1
) -> RawBrickData:
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return RawBrickData(brick_id=brick_id, world_matrix=matrix, color=color)


def test_parser_sort_bricks_by_position_bottom_to_top_then_x_then_z(tmp_path: Path):
    parser = MPDParser(str(tmp_path / "does_not_exist.mpd"))
    parser.raw_data = [
        _make_brick(20, 8, 20, "a"),
        _make_brick(0, 16, 0, "b"),
        _make_brick(-20, 16, 10, "c"),
        _make_brick(-10, 8, -20, "d"),
    ]

    parser.sort_bricks_by_position()

    ordered = [
        (b.brick_id, b.world_matrix[0, 3], b.world_matrix[1, 3], b.world_matrix[2, 3])
        for b in parser.raw_data
    ]
    assert ordered == [
        ("c", -20.0, 16.0, 10.0),
        ("b", 0.0, 16.0, 0.0),
        ("d", -10.0, 8.0, -20.0),
        ("a", 20.0, 8.0, 20.0),
    ]


def test_builder_position_sort_is_deterministic_across_calls(tmp_path: Path):
    atlas_config_path = tmp_path / "atlas_config.json"
    builder = DatasetBuilder(str(atlas_config_path))
    bricks = [
        _make_brick(10, 0, 0, "a"),
        _make_brick(-10, 16, 0, "b"),
        _make_brick(0, 16, -10, "c"),
        _make_brick(0, 8, 10, "d"),
    ]

    order1 = [b.brick_id for b in builder._sort_bricks_by_position(bricks)]
    order2 = [b.brick_id for b in builder._sort_bricks_by_position(bricks)]

    assert order1 == order2
    assert order1 == ["b", "c", "d", "a"]
