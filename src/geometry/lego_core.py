from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.data.parser import RawBrickData
from src.geometry.lego_part import PartDatabase
from src.geometry.port import Port
from src.geometry.snap import check_snap
from src.geometry.snap import find_all_connections
from src.geometry.spatial_hash import SpatialHash

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A brick in the assembly graph."""

    node_id: int
    part_id: str
    color: int
    world_matrix: np.ndarray  # (4, 4)


@dataclass
class GraphEdge:
    """A stud/anti-stud connection between two bricks."""

    node_a: int
    node_b: int
    port_a_id: int
    port_b_id: int
    relative_rotation: np.ndarray  # (3, 3)  R_a^T @ R_b


class LegoCore:
    """Central geometric engine for LEGO assembly graph construction.

    Maintains the assembly state: a graph of placed bricks (nodes) connected
    by stud/anti-stud mating (edges), backed by a spatial hash for fast
    collision detection.

    Typical usage — **preprocessing** (build graph from existing set)::

        core = LegoCore.from_raw_bricks(raw_bricks, part_db)
        nodes, edges = core.get_graph()

    Typical usage — **generation** (validate model predictions)::

        result = core.validate_placement(part_id, world_matrix)
        if result["valid"]:
            core.place_brick(part_id, color, world_matrix)
    """

    def __init__(self, part_db: PartDatabase, cell_size: float = 8.0) -> None:
        self.part_db = part_db
        self.spatial_hash = SpatialHash(cell_size=cell_size)
        self.nodes: dict[int, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._adjacency: dict[int, list[int]] = {}
        self._next_node_id: int = 0
        # Keep RawBrickData per node for snap checking against new bricks.
        self._raw: dict[int, RawBrickData] = {}

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place_brick(
        self,
        part_id: str,
        color: int,
        world_matrix: np.ndarray,
        validate: bool = True,
    ) -> int | None:
        """Place a brick and update graph + spatial hash.

        Returns the new node_id, or ``None`` if placement is invalid
        (collision detected and *validate* is True).
        """
        lp = self.part_db.get_or_default(part_id)

        if validate:
            overlaps = self.spatial_hash.check_overlap(lp.collision_boxes, world_matrix)
            if overlaps:
                return None

        node_id = self._next_node_id
        self._next_node_id += 1

        # Insert into spatial hash.
        self.spatial_hash.insert(node_id, lp.collision_boxes, world_matrix)

        # Create graph node.
        node = GraphNode(
            node_id=node_id,
            part_id=part_id,
            color=color,
            world_matrix=world_matrix,
        )
        self.nodes[node_id] = node
        self._adjacency[node_id] = []
        self._raw[node_id] = RawBrickData(
            brick_id=part_id, world_matrix=world_matrix, color=color
        )

        # Discover connections to existing bricks.
        new_brick = self._raw[node_id]
        for other_id, other_brick in self._raw.items():
            if other_id == node_id:
                continue
            matched = check_snap(self.part_db, other_brick, new_brick)
            if not matched:
                continue

            R_a = other_brick.world_matrix[:3, :3]
            R_b = world_matrix[:3, :3]
            rel_rot = R_a.T @ R_b

            for port_a, port_b in matched:
                self.edges.append(
                    GraphEdge(
                        node_a=other_id,
                        node_b=node_id,
                        port_a_id=port_a.port_id,
                        port_b_id=port_b.port_id,
                        relative_rotation=rel_rot,
                    )
                )
            self._adjacency[other_id].append(node_id)
            self._adjacency[node_id].append(other_id)

        return node_id

    # ------------------------------------------------------------------
    # Validation (read-only)
    # ------------------------------------------------------------------

    def validate_placement(
        self,
        part_id: str,
        world_matrix: np.ndarray,
    ) -> dict:
        """Check whether a brick can be placed without modifying state.

        Returns::

            {
                "valid": bool,
                "collisions": list[int],   # overlapping node ids
                "connections": list[...],   # (other_node_id, port_a, port_b)
                "num_snaps": int,
            }
        """
        lp = self.part_db.get_or_default(part_id)
        collisions = self.spatial_hash.check_overlap(lp.collision_boxes, world_matrix)

        connections: list[tuple[int, Port, Port]] = []
        candidate = RawBrickData(brick_id=part_id, world_matrix=world_matrix, color=0)
        for other_id, other_brick in self._raw.items():
            matched = check_snap(self.part_db, other_brick, candidate)
            for pa, pb in matched:
                connections.append((other_id, pa, pb))

        return {
            "valid": len(collisions) == 0,
            "collisions": collisions,
            "connections": connections,
            "num_snaps": len(connections),
        }

    # ------------------------------------------------------------------
    # Removal (backtracking)
    # ------------------------------------------------------------------

    def remove_brick(self, node_id: int) -> None:
        """Remove a brick and all its edges from the assembly."""
        self.spatial_hash.remove(node_id)
        self.nodes.pop(node_id, None)
        self._raw.pop(node_id, None)

        # Remove edges involving this node.
        self.edges = [
            e for e in self.edges if e.node_a != node_id and e.node_b != node_id
        ]

        # Clean up adjacency.
        neighbors = self._adjacency.pop(node_id, [])
        for nb in neighbors:
            adj = self._adjacency.get(nb)
            if adj is not None:
                try:
                    adj.remove(node_id)
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_graph(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return a snapshot of the current assembly graph."""
        return list(self.nodes.values()), list(self.edges)

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return an (N, N) binary adjacency matrix."""
        ids = sorted(self.nodes.keys())
        idx_map = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        adj = np.zeros((n, n), dtype=np.int8)
        for e in self.edges:
            i, j = idx_map.get(e.node_a), idx_map.get(e.node_b)
            if i is not None and j is not None:
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    def to_raw_brick_data(self) -> list[RawBrickData]:
        """Export current assembly as a list of RawBrickData (for MPD writing)."""
        return [
            RawBrickData(
                brick_id=node.part_id,
                world_matrix=node.world_matrix,
                color=node.color,
            )
            for node in self.nodes.values()
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_raw_bricks(
        cls,
        bricks: list[RawBrickData],
        part_db: PartDatabase,
        cell_size: float = 8.0,
    ) -> LegoCore:
        """Build a LegoCore from an existing set of bricks.

        Places all bricks without collision validation (existing sets are
        assumed correct) and discovers all stud/anti-stud connections.
        """
        core = cls(part_db=part_db, cell_size=cell_size)

        # 1) Insert all nodes first.
        for brick in bricks:
            node_id = core._next_node_id
            core._next_node_id += 1

            lp = part_db.get_or_default(brick.brick_id)
            core.spatial_hash.insert(node_id, lp.collision_boxes, brick.world_matrix)

            core.nodes[node_id] = GraphNode(
                node_id=node_id,
                part_id=brick.brick_id,
                color=brick.color,
                world_matrix=brick.world_matrix,
            )
            core._adjacency[node_id] = []
            core._raw[node_id] = brick

        # 2) Discover all connections with KDTree pre-filtering.
        all_connections = find_all_connections(part_db, bricks)
        for idx_a, idx_b, matched in all_connections:
            node_a = int(idx_a)
            node_b = int(idx_b)

            raw_a = core._raw[node_a]
            raw_b = core._raw[node_b]
            rel_rot = raw_a.world_matrix[:3, :3].T @ raw_b.world_matrix[:3, :3]

            for port_a, port_b in matched:
                core.edges.append(
                    GraphEdge(
                        node_a=node_a,
                        node_b=node_b,
                        port_a_id=port_a.port_id,
                        port_b_id=port_b.port_id,
                        relative_rotation=rel_rot,
                    )
                )

            if matched:
                core._adjacency[node_a].append(node_b)
                core._adjacency[node_b].append(node_a)
        return core
