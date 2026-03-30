"""MPD → labeled assembly graph preprocessing pipeline.

For each MPD file this module produces a ``.npz`` archive that encodes the
assembly as a graph together with the supervised action sequence needed to
train the graph transformer.

Saved arrays
------------
node_part_ids     (N,)    int32   part index (shared vocabulary)
node_colors       (N,)    int32   color index (shared vocabulary)
node_translations (N, 3)  float32 world x/y/z divided by TRANSLATION_SCALE
node_rotations    (N, 6)  float32 6-D rotation (first two columns of R)
edge_index        (2, E)  int32   COO source/target (undirected: both dirs)
edge_rotations    (E, 9)  float32 relative rotation R_a^T @ R_b (flattened)
actions           (N, 4)  int32   per-step supervision:
                                  [target_node_id, target_port_id,
                                   self_port_id,   rot_steps]
                                  Row 0 (seed brick) is all -1.
build_order       (N,)    int32   build_order[step] = original node_id
num_nodes         ()      int32   N
num_edges         ()      int32   E  (directed count, i.e. 2*undirected)
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from src.data.parser import MPDParser, RawBrickData
from src.geometry.lego_core import LegoCore, GraphEdge
from src.geometry.lego_part import PartDatabase
from src.geometry.snap_math import extract_rot_steps

logger = logging.getLogger(__name__)

TRANSLATION_SCALE: float = 100.0  # LDU → ~O(1) floats
MAX_PORT_ID: int = 32  # ports beyond this index are ignored


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_6d(R: np.ndarray) -> np.ndarray:
    """6-D rotation representation: first two columns of R, flattened (6,)."""
    return R[:, :2].T.flatten()  # [r00,r10,r20, r01,r11,r21]


def _bfs_order(
    seed_node: int,
    adjacency: dict[int, list[int]],
    all_node_ids: list[int],
) -> list[int]:
    """BFS traversal from *seed_node* over the connection graph.

    Unconnected nodes (no path from seed) are appended at the end so every
    brick appears exactly once in the returned order.
    """
    visited: set[int] = set()
    order: list[int] = []
    queue: deque[int] = deque([seed_node])
    visited.add(seed_node)

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for nb in adjacency.get(nid, []):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    # Append any disconnected nodes (separate connected components).
    for nid in all_node_ids:
        if nid not in visited:
            order.append(nid)

    return order


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_mpd(
    mpd_path: str | Path,
    part_db: PartDatabase,
    part_vocab: dict[str, int],
    color_vocab: dict[int, int],
) -> dict | None:
    """Parse one MPD file and return the labeled graph data as numpy arrays.

    Returns ``None`` if the file cannot be parsed or yields fewer than 2
    connected bricks.

    Parameters
    ----------
    mpd_path:
        Path to the ``.mpd`` / ``.ldr`` source file.
    part_db:
        Shared ``PartDatabase`` instance (reused across calls for caching).
    part_vocab:
        Mapping ``part_id_str → integer index`` (from the training vocabulary).
    color_vocab:
        Mapping ``color_int → integer index``.
    """
    mpd_path = Path(mpd_path)

    # ---- 1. Parse raw bricks ------------------------------------------------
    try:
        parser = MPDParser(str(mpd_path))
        first_key = next(iter(parser._submodels))
        raw_bricks: list[RawBrickData] = parser.parse(first_key)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", mpd_path.name, exc)
        return None

    if len(raw_bricks) < 2:
        return None

    # ---- 2. Build full assembly graph (no collision validation) -------------
    core = LegoCore.from_raw_bricks(raw_bricks, part_db)
    nodes, edges = core.get_graph()

    if not nodes:
        return None

    # ---- 3. BFS build order starting from bottommost brick -----------------
    # "Bottommost" = largest Y in LDraw (Y is down).
    bottommost_id = max(
        (n.node_id for n in nodes),
        key=lambda nid: core.nodes[nid].world_matrix[1, 3],
    )
    all_node_ids = [n.node_id for n in nodes]
    build_order = _bfs_order(bottommost_id, core._adjacency, all_node_ids)

    # ---- 4. Build edge lookup by node id -------------------------------------
    # Maps node_id -> list[(neighbor_id, edge, node_is_edge_a)].
    edge_lookup_by_node: dict[int, list[tuple[int, GraphEdge, bool]]] = {}
    for e in edges:
        edge_lookup_by_node.setdefault(e.node_a, []).append((e.node_b, e, True))
        edge_lookup_by_node.setdefault(e.node_b, []).append((e.node_a, e, False))

    # ---- 5. Replay build in BFS order, collecting actions -------------------
    # placed_set[node_id] = True once the brick has been "placed"
    placed_set: set[int] = set()

    actions: list[tuple[int, int, int, int]] = (
        []
    )  # (target_nid, tgt_port, self_port, rot)

    for step, nid in enumerate(build_order):
        raw = core._raw[nid]
        lp = part_db.get(core.nodes[nid].part_id)

        if step == 0:
            # Seed brick — no parent action.
            actions.append((-1, -1, -1, -1))
            placed_set.add(nid)
            continue

        # Find an edge connecting this brick to an already-placed brick.
        parent_edge: GraphEdge | None = None
        parent_nid = -1
        nid_is_edge_a = False
        for neighbor_nid, edge, is_edge_a in edge_lookup_by_node.get(nid, []):
            if neighbor_nid in placed_set:
                parent_edge = edge
                parent_nid = neighbor_nid
                nid_is_edge_a = is_edge_a
                break

        if parent_edge is None:
            # No detected connection — new disconnected seed; skip action.
            logger.debug(
                "%s step %d: node %d has no connection to partial assembly, skipping",
                mpd_path.name,
                step,
                nid,
            )
            actions.append((-1, -1, -1, -1))
            placed_set.add(nid)
            continue

        # Determine which edge side belongs to the new node and parent node.
        if nid_is_edge_a:
            self_port_id = parent_edge.port_a_id
            parent_pid = parent_edge.port_b_id
        else:
            self_port_id = parent_edge.port_b_id
            parent_pid = parent_edge.port_a_id

        if self_port_id >= MAX_PORT_ID or parent_pid >= MAX_PORT_ID:
            actions.append((-1, -1, -1, -1))
            placed_set.add(nid)
            continue

        # Recover rot_steps from the ground-truth world matrices.
        parent_raw = core._raw[parent_nid]
        parent_lp = part_db.get(parent_nid if False else parent_raw.brick_id)
        new_lp = lp

        rot = -1
        if (
            parent_lp is not None
            and new_lp is not None
            and parent_pid < len(parent_lp.ports)
            and self_port_id < len(new_lp.ports)
        ):
            rot = extract_rot_steps(
                parent_world_matrix=parent_raw.world_matrix,
                parent_port=parent_lp.ports[parent_pid],
                new_part_port=new_lp.ports[self_port_id],
                new_world_matrix=raw.world_matrix,
            )

        actions.append((parent_nid, parent_pid, self_port_id, rot))
        placed_set.add(nid)

    # ---- 6. Encode node features --------------------------------------------
    N = len(build_order)
    node_part_ids = np.zeros(N, dtype=np.int32)
    node_colors = np.zeros(N, dtype=np.int32)
    node_translations = np.zeros((N, 3), dtype=np.float32)
    node_rotations = np.zeros((N, 6), dtype=np.float32)
    node_rotations_9d = np.zeros(
        (N, 9), dtype=np.float32
    )  # for world-matrix reconstruction

    # Port geometry: saved so the dataset doesn't need PartDatabase at load time.
    node_port_local_pos = np.zeros((N, MAX_PORT_ID, 3), dtype=np.float32)
    node_port_local_nrm = np.zeros((N, MAX_PORT_ID, 3), dtype=np.float32)
    node_port_types = np.zeros((N, MAX_PORT_ID), dtype=np.int8)  # 1=male 2=female
    node_port_count = np.zeros(N, dtype=np.int32)

    node_index_map: dict[int, int] = {}  # original node_id → sequential index

    for seq_idx, nid in enumerate(build_order):
        node_index_map[nid] = seq_idx
        node = core.nodes[nid]
        M = node.world_matrix

        node_part_ids[seq_idx] = part_vocab.get(node.part_id, 0)
        node_colors[seq_idx] = color_vocab.get(node.color, 0)
        node_translations[seq_idx] = (M[:3, 3] / TRANSLATION_SCALE).astype(np.float32)
        node_rotations[seq_idx] = _to_6d(M[:3, :3]).astype(np.float32)
        node_rotations_9d[seq_idx] = M[:3, :3].flatten().astype(np.float32)

        lp = part_db.get(node.part_id)
        if lp is not None:
            n_ports = min(len(lp.ports), MAX_PORT_ID)
            node_port_count[seq_idx] = n_ports
            for pidx in range(n_ports):
                port = lp.ports[pidx]
                node_port_local_pos[seq_idx, pidx] = port.local_position.astype(
                    np.float32
                )
                node_port_local_nrm[seq_idx, pidx] = port.normal.astype(np.float32)
                node_port_types[seq_idx, pidx] = 1 if port.port_type == "male" else 2

    # ---- 7. Encode edges (COO, both directions) -----------------------------
    edge_src, edge_dst, edge_rots = [], [], []
    for e in edges:
        i = node_index_map.get(e.node_a)
        j = node_index_map.get(e.node_b)
        if i is None or j is None:
            continue
        rot9 = e.relative_rotation.flatten().astype(np.float32)
        edge_src.append(i)
        edge_dst.append(j)
        edge_rots.append(rot9)
        edge_src.append(j)
        edge_dst.append(i)
        edge_rots.append(rot9)

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int32)
        edge_rotations = np.array(edge_rots, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_rotations = np.zeros((0, 9), dtype=np.float32)

    # ---- 8. Encode actions with sequential node indices ---------------------
    actions_arr = np.full((N, 4), -1, dtype=np.int32)
    for step, (tgt_nid, tgt_pid, self_pid, rot) in enumerate(actions):
        if tgt_nid >= 0:
            actions_arr[step] = [node_index_map[tgt_nid], tgt_pid, self_pid, rot]

    # ---- 9. Port consumption schedule ---------------------------------------
    # port_consumed_step[i, j] = the build step at which port j of node i was
    # consumed (connected).  -1 means the port was never connected and remains
    # open for the rest of the sequence.
    port_consumed_step = np.full((N, MAX_PORT_ID), -1, dtype=np.int32)
    for step in range(N):
        tgt_seq, tgt_pid, self_pid, _ = actions_arr[step]
        if tgt_seq >= 0 and 0 <= tgt_pid < MAX_PORT_ID:
            port_consumed_step[tgt_seq, tgt_pid] = step
        if 0 <= self_pid < MAX_PORT_ID:
            # self_port is consumed immediately when the brick is placed
            port_consumed_step[step, self_pid] = step

    return dict(
        node_part_ids=node_part_ids,
        node_colors=node_colors,
        node_translations=node_translations,
        node_rotations=node_rotations,
        node_rotations_9d=node_rotations_9d,
        node_port_local_pos=node_port_local_pos,
        node_port_local_nrm=node_port_local_nrm,
        node_port_types=node_port_types,
        node_port_count=node_port_count,
        port_consumed_step=port_consumed_step,
        edge_index=edge_index,
        edge_rotations=edge_rotations,
        actions=actions_arr,
        build_order=np.array(
            [node_index_map[nid] for nid in build_order], dtype=np.int32
        ),
        num_nodes=np.int32(N),
        num_edges=np.int32(len(edge_src)),
    )


def build_dataset(
    mpd_dir: str | Path,
    output_dir: str | Path,
    part_db: PartDatabase,
    part_vocab: dict[str, int],
    color_vocab: dict[int, int],
    skip_existing: bool = True,
) -> None:
    """Process all MPD files in *mpd_dir* and save `.npz` files to *output_dir*.

    Parameters
    ----------
    mpd_dir:
        Directory containing ``.mpd`` / ``.ldr`` files.
    output_dir:
        Destination directory for ``.npz`` output files.
    part_db:
        Shared ``PartDatabase`` (reused for caching — do not recreate per file).
    part_vocab / color_vocab:
        Vocabulary mappings built from the full dataset before calling this.
    skip_existing:
        If True, skip files whose ``.npz`` already exists in *output_dir*.
    """
    mpd_dir = Path(mpd_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpd_files = sorted(
        p for p in mpd_dir.iterdir() if p.suffix.lower() in {".mpd", ".ldr"}
    )

    ok = skip = fail = 0
    for mpd_path in mpd_files:
        out_path = output_dir / (mpd_path.stem + ".npz")
        if skip_existing and out_path.exists():
            skip += 1
            continue

        data = process_mpd(mpd_path, part_db, part_vocab, color_vocab)
        if data is None:
            fail += 1
            continue

        np.savez_compressed(out_path, **data)
        ok += 1
        if ok % 50 == 0:
            logger.info("Processed %d / %d files", ok, len(mpd_files))

    logger.info(
        "Done — ok=%d  skipped=%d  failed=%d  total=%d",
        ok,
        skip,
        fail,
        len(mpd_files),
    )
