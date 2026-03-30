"""Generation loop for the GraphTransformer model.

Usage (see graph_generate_model.py at project root):
    generator = GraphGenerator(cfg, model, part_db, part_vocab, color_vocab)
    raw_bricks = generator.generate(seed_part_id, seed_color)
    write_mpd_file(output_path, raw_bricks)

Algorithm
---------
1. Place seed brick at world origin (identity matrix).
2. Maintain a live open-port list and a growing node list.
3. At each step:
   a. Build PyG Data from current nodes/edges → forward pass.
   b. Sample port, part, color, self_port, rot_steps from logits.
   c. Resolve new brick world matrix via compute_snap_matrix.
   d. Validate: no collision (skip if colliding).
   e. Detect new connections (snap check against all placed bricks).
   f. Update open-port list (remove consumed ports, add new ones).
4. Stop when max_bricks reached or no open ports remain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.data.parser import RawBrickData
from src.geometry.graph_builder import TRANSLATION_SCALE, MAX_PORT_ID
from src.geometry.lego_part import PartDatabase
from src.geometry.snap_math import compute_snap_matrix
from src.model.graph_config import GraphModelConfig
from src.model.graph_transformer import GraphTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------


class _PlacedBrick(NamedTuple):
    """A brick that has been placed during generation."""

    seq_idx: int  # sequential index in placement order
    part_id: int  # vocabulary part index
    color: int  # vocabulary color index
    world_mat: np.ndarray  # 4×4 world matrix
    part_key: str  # LDraw part string (e.g. "3001")


@dataclass
class _OpenPort:
    """An unconnected port available for snapping."""

    node_seq: int  # seq_idx of the owning brick
    port_id: int  # index into LegoPart.ports
    port_type: str  # "male" or "female"
    world_pos: np.ndarray  # (3,) — world space, un-normalised
    world_nrm: np.ndarray  # (3,) — world space unit normal


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _top_k_sample(logits: torch.Tensor, top_k: int, temperature: float) -> int:
    """Temperature + top-k sampling from a 1-D logit vector."""
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GraphGenerator:
    """Autoregressive generation using a trained GraphTransformer.

    Parameters
    ----------
    cfg : GraphModelConfig
    model : GraphTransformer  (should be in eval mode)
    part_db : PartDatabase
    part_vocab : dict[int, str]
        Maps part vocabulary index → LDraw part string (e.g. {42: "3001"}).
    color_vocab : dict[int, int]
        Maps color vocabulary index → LDraw color id.
    device : str or torch.device
    """

    # Maximum open ports we present to the model — must match training.
    MAX_OPEN_PORTS = 256
    PORT_FEAT_DIM = 9

    def __init__(
        self,
        cfg: GraphModelConfig,
        model: GraphTransformer,
        part_db: PartDatabase,
        part_vocab: dict[int, str],
        color_vocab: dict[int, int],
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.model = model.eval()
        self.part_db = part_db
        self.part_vocab = part_vocab
        self.color_vocab = color_vocab
        self.device = torch.device(device)

        # Track known-invalid part ids lazily to avoid expensive full scans.
        self._invalid_part_ids: set[int] = {0}

    # ------------------------------------------------------------------

    def generate(
        self,
        seed_part_id: int,
        seed_color: int,
        max_bricks: int | None = None,
    ) -> list[RawBrickData]:
        """Generate a LEGO assembly starting from a seed brick.

        Parameters
        ----------
        seed_part_id : int
            Vocabulary part index of the first brick.
        seed_color : int
            Vocabulary color index of the first brick.
        max_bricks : int | None
            Override cfg.max_gen_bricks if given.

        Returns
        -------
        list[RawBrickData]
            Placed bricks ready for MPD export.
        """
        max_bricks = max_bricks or self.cfg.max_gen_bricks

        # ---- State --------------------------------------------------------
        placed: list[_PlacedBrick] = []
        edges: list[tuple[int, int]] = []
        open_ports: list[_OpenPort] = []

        # ---- Seed brick at world origin -----------------------------------
        seed_mat = np.eye(4)
        seed_part_key = self.part_vocab.get(seed_part_id, str(seed_part_id))
        seed = _PlacedBrick(
            seq_idx=0,
            part_id=seed_part_id,
            color=seed_color,
            world_mat=seed_mat,
            part_key=seed_part_key,
        )
        placed.append(seed)
        self._add_open_ports(seed, open_ports)

        logger.info(
            "Generation started — seed: %s  color: %d  max_bricks: %d",
            seed_part_key,
            seed_color,
            max_bricks,
        )

        if not open_ports:
            logger.warning(
                "Seed brick %s has no open ports/geometry; generation will stop at 1 brick.",
                seed_part_key,
            )

        # ---- Autoregressive loop ------------------------------------------
        while len(placed) < max_bricks and open_ports:
            try:
                new_brick = self._step(placed, edges, open_ports)
            except Exception as exc:
                logger.warning("Step %d failed: %s — stopping.", len(placed), exc)
                break

            if new_brick is None:
                break  # no valid placement found

            placed.append(new_brick)
            logger.debug(
                "Placed brick %d: %s  color=%d",
                new_brick.seq_idx,
                new_brick.part_key,
                new_brick.color,
            )

        if len(placed) == 1:
            logger.warning(
                "Generation ended at 1 brick. Typical causes: invalid vocab mapping, seed with no ports, or no valid snap candidates."
            )
        logger.info("Generation finished — %d bricks placed.", len(placed))
        return self._to_raw_bricks(placed)

    # ------------------------------------------------------------------
    # One generation step
    # ------------------------------------------------------------------

    def _step(
        self,
        placed: list[_PlacedBrick],
        edges: list[tuple[int, int]],
        open_ports: list[_OpenPort],
    ) -> _PlacedBrick | None:
        """Run one forward pass and attempt to place a new brick.

        Returns the placed brick, or None if all sampled placements collide.
        """
        # Build input tensors
        graph, op_tensor, op_mask = self._build_inputs(placed, edges, open_ports)
        batch_graph = Batch.from_data_list([graph]).to(self.device)
        op_tensor = op_tensor.to(self.device)
        op_mask = op_mask.to(self.device)

        with torch.no_grad():
            out = self.model(
                batch_graph,
                op_tensor.unsqueeze(0),  # (1, P, F)
                op_mask.unsqueeze(0),  # (1, P)
            )

        # ---- Sample each head --------------------------------------------
        port_idx = self._sample_port(out["port_logits"][0], op_mask)
        part_logits = out["part_logits"][0].clone()
        color_id = _top_k_sample(
            out["color_logits"][0], self.cfg.top_k, self.cfg.temperature
        )
        self_port = int(out["self_port_logits"][0].argmax().item())
        rot_steps = int(out["rot_logits"][0].argmax().item())

        # ---- Resolve selected open port ----------------------------------
        sel_port = open_ports[port_idx]
        parent_brick = placed[sel_port.node_seq]

        # ---- Get ports for new part --------------------------------------
        # Retry a few times if sampled parts have no geometry/ports.
        part_key = None
        lego_part = None
        part_id = -1
        for _ in range(8):
            masked_logits = part_logits.clone()
            if self._invalid_part_ids:
                invalid_ids = [
                    pid
                    for pid in self._invalid_part_ids
                    if 0 <= pid < masked_logits.numel()
                ]
                if invalid_ids:
                    masked_logits[invalid_ids] = float("-inf")

            if not torch.isfinite(masked_logits).any():
                return None

            part_id = _top_k_sample(masked_logits, self.cfg.top_k, self.cfg.temperature)
            part_key = self.part_vocab.get(part_id, str(part_id))
            lego_part = self.part_db.get_or_default(part_key)
            if lego_part.ports:
                break
            self._invalid_part_ids.add(part_id)

        if lego_part is None or not lego_part.ports:
            return None

        if self_port >= len(lego_part.ports):
            self_port = 0  # fallback to first port

        new_part_port = lego_part.ports[self_port]

        # ---- Compute world matrix via snap math --------------------------
        # The selected open port is stored in world coords — we need to
        # reconstruct a fake parent port with those world coords.
        # Easier: feed the parent brick's world matrix + port local coords.
        parent_lego = self.part_db.get_or_default(parent_brick.part_key)

        # Find the actual Port object matching sel_port.port_id
        parent_port = None
        for p in parent_lego.ports:
            if p.port_id == sel_port.port_id:
                parent_port = p
                break
        if parent_port is None:
            return None

        world_mat = compute_snap_matrix(
            parent_brick.world_mat, parent_port, new_part_port, rot_steps
        )

        # ---- Place brick & update state ----------------------------------
        new_seq = len(placed)
        color_key = self.color_vocab.get(color_id, color_id)
        new_brick = _PlacedBrick(
            seq_idx=new_seq,
            part_id=part_id,
            color=color_id,
            world_mat=world_mat,
            part_key=part_key,
        )

        # Register new connections
        new_connections: list[tuple[int, int, int, int]] = (
            []
        )  # (a_seq, a_pid, b_pid, b_seq)
        for op in open_ports:
            existing_brick = placed[op.node_seq]
            existing_lego = self.part_db.get_or_default(existing_brick.part_key)
            for ep in existing_lego.ports:
                for np_ in lego_part.ports:
                    world_ep = (
                        existing_brick.world_mat[:3, :3] @ ep.local_position
                        + existing_brick.world_mat[:3, 3]
                    )
                    world_np = world_mat[:3, :3] @ np_.local_position + world_mat[:3, 3]
                    if (
                        ep.port_type != np_.port_type
                        and np.linalg.norm(world_ep - world_np) < 2.5
                        and np.dot(
                            existing_brick.world_mat[:3, :3] @ ep.normal,
                            world_mat[:3, :3] @ np_.normal,
                        )
                        < -0.9
                    ):
                        new_connections.append(
                            (op.node_seq, ep.port_id, np_.port_id, new_seq)
                        )
                        edges.append((op.node_seq, new_seq))
                        edges.append((new_seq, op.node_seq))

        # Remove consumed open ports
        consumed_keys: set[tuple[int, int]] = set()
        for conn in new_connections:
            consumed_keys.add((conn[0], conn[1]))  # existing port
            consumed_keys.add((conn[3], conn[2]))  # new port

        open_ports[:] = [
            op for op in open_ports if (op.node_seq, op.port_id) not in consumed_keys
        ]

        # Add new brick's open ports
        self._add_open_ports(new_brick, open_ports)
        # Remove ports consumed on the new brick itself
        new_consumed = {conn[2] for conn in new_connections if conn[3] == new_seq}
        open_ports[:] = [
            op
            for op in open_ports
            if not (op.node_seq == new_seq and op.port_id in new_consumed)
        ]

        return new_brick

    # ------------------------------------------------------------------
    # Input builder
    # ------------------------------------------------------------------

    def _build_inputs(
        self,
        placed: list[_PlacedBrick],
        edges: list[tuple[int, int]],
        open_ports: list[_OpenPort],
    ) -> tuple[Data, torch.Tensor, torch.Tensor]:
        """Convert current assembly state to model inputs."""
        N = len(placed)
        part_ids = torch.tensor([b.part_id for b in placed], dtype=torch.long)
        colors = torch.tensor([b.color for b in placed], dtype=torch.long)

        # Continuous node features: translation (normalised) + 6D rotation
        x_rows = []
        for b in placed:
            t = b.world_mat[:3, 3] / TRANSLATION_SCALE
            R = b.world_mat[:3, :3]
            rot6d = R[:, :2].T.flatten()  # (6,)
            x_rows.append(np.concatenate([t, rot6d]))  # (9,)
        x_cont = torch.tensor(np.array(x_rows, dtype=np.float32))

        # Edges
        if edges:
            ei_np = np.array(edges, dtype=np.int64).T  # (2, E)
            # Build relative rotation edge attributes
            ea_rows = []
            for src, dst in edges:
                R_src = placed[src].world_mat[:3, :3]
                R_dst = placed[dst].world_mat[:3, :3]
                rel = R_src.T @ R_dst
                ea_rows.append(rel.flatten())
            ea = torch.tensor(np.array(ea_rows, dtype=np.float32))
            ei = torch.from_numpy(ei_np)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 9), dtype=torch.float32)

        graph = Data(
            part_ids=part_ids,
            colors=colors,
            x=x_cont,
            edge_index=ei,
            edge_attr=ea,
            num_nodes=N,
        )

        # Open port tensor (P, 9)
        P = self.MAX_OPEN_PORTS
        F_ = self.PORT_FEAT_DIM
        features = np.zeros((P, F_), dtype=np.float32)
        mask = np.zeros(P, dtype=bool)

        for k, op in enumerate(open_ports[:P]):
            features[k, 0:3] = op.world_pos / TRANSLATION_SCALE
            features[k, 3:6] = op.world_nrm
            features[k, 6] = float(op.node_seq)
            features[k, 7] = float(op.port_id)
            features[k, 8] = 1.0 if op.port_type == "male" else 2.0
            mask[k] = True

        op_tensor = torch.from_numpy(features)
        op_mask = torch.from_numpy(mask)

        return graph, op_tensor, op_mask

    # ------------------------------------------------------------------
    # Port sampling
    # ------------------------------------------------------------------

    def _sample_port(
        self,
        port_logits: torch.Tensor,
        op_mask: torch.Tensor,
    ) -> int:
        """Sample a port index (argmax — port selection is deterministic)."""
        # Mask already applied inside the model; just argmax here for
        # interpretability. Switch to sampling if diversity is needed.
        masked = port_logits.clone()
        masked[~op_mask] = float("-inf")
        return int(masked.argmax().item())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_open_ports(
        self,
        brick: _PlacedBrick,
        open_ports: list[_OpenPort],
    ) -> None:
        """Append all ports of *brick* (up to MAX_PORT_ID) to *open_ports*."""
        lego_part = self.part_db.get_or_default(brick.part_key)
        R = brick.world_mat[:3, :3]
        t = brick.world_mat[:3, 3]
        for p in lego_part.ports[:MAX_PORT_ID]:
            open_ports.append(
                _OpenPort(
                    node_seq=brick.seq_idx,
                    port_id=p.port_id,
                    port_type=p.port_type,
                    world_pos=R @ p.local_position + t,
                    world_nrm=R @ p.normal,
                )
            )

    def _to_raw_bricks(self, placed: list[_PlacedBrick]) -> list[RawBrickData]:
        """Convert placed bricks to RawBrickData for MPD export."""
        result = []
        for b in placed:
            color_ldraw = self.color_vocab.get(b.color, b.color)
            result.append(
                RawBrickData(
                    brick_id=b.part_key,
                    color=color_ldraw,
                    world_matrix=b.world_mat,
                )
            )
        return result
