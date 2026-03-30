"""AssemblyStepDataset — PyTorch / PyG dataset for graph transformer training.

Each item corresponds to one generation step within one LEGO assembly:
  - A partial graph of the bricks placed so far (nodes 0…t, edges among them)
  - The list of open (unconnected) ports on that partial assembly
  - Supervision targets: which port to select, what part/color/port/rotation

Loading
-------
The dataset reads ``.npz`` files produced by ``graph_builder.build_dataset()``.
One file with N bricks yields N−1 valid items (steps 1 through N−1), but only
steps where a real connection action exists (``actions[t+1, 0] >= 0``) are
included — purely disconnected seed bricks are skipped as training targets.

Batching
--------
Use ``collate_fn`` (exported below) with a standard ``DataLoader``.  It calls
``torch_geometric.data.Batch.from_data_list`` for the graph part and stacks
the remaining tensors normally.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.geometry.graph_builder import MAX_PORT_ID, TRANSLATION_SCALE

logger = logging.getLogger(__name__)

# Maximum open ports we present to the model at any step.
MAX_OPEN_PORTS: int = 256

# Open-port feature dimension:
#   world_pos (3) + world_nrm (3) + node_seq_idx (1) + port_id (1) + port_type (1) = 9
PORT_FEAT_DIM: int = 9


# ---------------------------------------------------------------------------
# Typed dict for dataset items
# ---------------------------------------------------------------------------


class AssemblyItem(TypedDict):
    graph: Data  # PyG graph with nodes 0..t
    open_ports: torch.Tensor  # (MAX_OPEN_PORTS, PORT_FEAT_DIM) float32
    open_port_mask: torch.Tensor  # (MAX_OPEN_PORTS,) bool — True = valid
    target_port_idx: torch.Tensor  # () int64
    target_part_id: torch.Tensor  # () int64
    target_color: torch.Tensor  # () int64
    target_self_port: torch.Tensor  # () int64
    target_rot_steps: torch.Tensor  # () int64


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AssemblyStepDataset(Dataset):
    """One item per valid build step across all ``.npz`` files in *graph_dir*.

    Parameters
    ----------
    graph_dir:
        Directory containing ``.npz`` files from ``graph_builder``.
    max_open_ports:
        Cap on how many open ports to present per step (padded / truncated).
    cache_size:
        Number of ``.npz`` files to keep in memory (LRU eviction).
    """

    def __init__(
        self,
        graph_dir: str | Path,
        max_open_ports: int = MAX_OPEN_PORTS,
        cache_size: int = 256,
        log_every_files: int = 500,
    ) -> None:
        self.graph_dir = Path(graph_dir)
        self.max_open_ports = max_open_ports

        self._files: list[Path] = sorted(self.graph_dir.glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(f"No .npz files found in {graph_dir}")

        logger.info(
            "Indexing graph dataset from %s (%d files)",
            self.graph_dir,
            len(self._files),
        )

        # Build index: list of (file_idx, step_t) where step_t+1 is the
        # brick being predicted and actions[step_t+1] is a valid action.
        self._index: list[tuple[int, int]] = []
        for fi, path in enumerate(self._files):
            try:
                npz = np.load(path)
                actions = npz["actions"]  # (N, 4)
                N = int(npz["num_nodes"])
                for t in range(N - 1):
                    # actions[t+1] must have a real target (not a disconnected seed)
                    if actions[t + 1, 0] >= 0 and actions[t + 1, 3] >= 0:
                        self._index.append((fi, t))
            except Exception:
                continue  # skip corrupt files silently

            if (fi + 1) % max(1, log_every_files) == 0:
                logger.info(
                    "Index progress: %d/%d files, %d samples",
                    fi + 1,
                    len(self._files),
                    len(self._index),
                )

        logger.info(
            "Index complete: %d files, %d samples",
            len(self._files),
            len(self._index),
        )

        # Bounded LRU cache for loaded npz data.
        @lru_cache(maxsize=cache_size)
        def _cached_load(idx: int) -> dict:
            npz = np.load(self._files[idx])
            return {k: npz[k] for k in npz.files}

        self._load = _cached_load

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> AssemblyItem:
        fi, t = self._index[idx]
        d = self._load(fi)

        # ---- partial graph: nodes 0..t ------------------------------------
        t1 = t + 1  # number of nodes placed (inclusive of t)
        part_ids = torch.from_numpy(d["node_part_ids"][:t1].astype(np.int64))
        colors = torch.from_numpy(d["node_colors"][:t1].astype(np.int64))
        trans = torch.from_numpy(d["node_translations"][:t1])  # (t1, 3)
        rot6d = torch.from_numpy(d["node_rotations"][:t1])  # (t1, 6)
        x_cont = torch.cat([trans, rot6d], dim=1)  # (t1, 9)

        # Filter edges to those where both endpoints are in 0..t
        ei_full = d["edge_index"]  # (2, E)
        ea_full = d["edge_rotations"]  # (E, 9)
        if ei_full.shape[1] > 0:
            mask = (ei_full[0] <= t) & (ei_full[1] <= t)
            ei = torch.from_numpy(ei_full[:, mask].astype(np.int64))
            ea = torch.from_numpy(ea_full[mask])
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 9), dtype=torch.float32)

        graph = Data(
            part_ids=part_ids,  # (t1,) int64 — for embedding lookup
            colors=colors,  # (t1,) int64 — for embedding lookup
            x=x_cont,  # (t1, 9) float32 — positional/rotational
            edge_index=ei,  # (2, E_t) int64
            edge_attr=ea,  # (E_t, 9) float32
        )

        # ---- open ports after placing nodes 0..t --------------------------
        open_ports, open_port_mask, target_port_idx = self._build_open_ports(d, t)

        # ---- targets for step t+1 -----------------------------------------
        action = d["actions"][t + 1]  # [target_node_seq, target_port, self_port, rot]
        target_part = int(d["node_part_ids"][t + 1])
        target_color = int(d["node_colors"][t + 1])

        return AssemblyItem(
            graph=graph,
            open_ports=open_ports,
            open_port_mask=open_port_mask,
            target_port_idx=torch.tensor(target_port_idx, dtype=torch.long),
            target_part_id=torch.tensor(target_part, dtype=torch.long),
            target_color=torch.tensor(target_color, dtype=torch.long),
            target_self_port=torch.tensor(int(action[2]), dtype=torch.long),
            target_rot_steps=torch.tensor(int(action[3]), dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_open_ports(
        self,
        d: dict,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build the open port tensor for step *t* (after placing nodes 0..t).

        Returns
        -------
        open_ports : (max_open_ports, PORT_FEAT_DIM) float32
        open_port_mask : (max_open_ports,) bool
        target_port_idx : int — index of the target port in the list, or -1
        """
        port_consumed = d["port_consumed_step"]  # (N, MAX_PORT_ID) int32
        port_count = d["node_port_count"]  # (N,) int32
        local_pos = d["node_port_local_pos"]  # (N, MAX_PORT_ID, 3)
        local_nrm = d["node_port_local_nrm"]  # (N, MAX_PORT_ID, 3)
        port_types = d["node_port_types"]  # (N, MAX_PORT_ID) int8
        rotations_9d = d["node_rotations_9d"]  # (N, 9)
        translations = d["node_translations"]  # (N, 3)  already normalised

        # Target port: from actions[t+1]
        action = d["actions"][t + 1]
        tgt_node_seq = int(action[0])
        tgt_port_id = int(action[1])

        features = np.zeros((self.max_open_ports, PORT_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(self.max_open_ports, dtype=bool)
        target_port_idx = -1
        k = 0

        for i in range(t + 1):  # nodes placed so far
            R = rotations_9d[i].reshape(3, 3)
            t_vec = translations[i] * TRANSLATION_SCALE  # un-normalise for world pos

            n_ports = int(port_count[i])
            for j in range(n_ports):
                if j >= MAX_PORT_ID:
                    break
                consumed = int(port_consumed[i, j])
                # Port is open at step t if it was never consumed OR consumed > t
                if consumed != -1 and consumed <= t:
                    continue

                if k >= self.max_open_ports:
                    break  # silently truncate; model sees first max_open_ports ports

                world_pos = R @ local_pos[i, j] + t_vec
                world_nrm = R @ local_nrm[i, j]

                features[k, 0:3] = world_pos / TRANSLATION_SCALE  # re-normalise
                features[k, 3:6] = world_nrm
                features[k, 6] = float(i)  # source node seq index
                features[k, 7] = float(j)  # port id
                features[k, 8] = float(port_types[i, j])  # 1=male 2=female
                mask[k] = True

                if i == tgt_node_seq and j == tgt_port_id:
                    target_port_idx = k

                k += 1

        return (
            torch.from_numpy(features),
            torch.from_numpy(mask),
            target_port_idx,
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(items: list[AssemblyItem]) -> dict:
    """Collate a list of AssemblyItems into a batched dict.

    The ``graph`` field is batched via PyG's ``Batch.from_data_list``,
    which concatenates node/edge tensors and adds a ``batch`` index vector.
    All other tensors are stacked normally.
    """
    graphs = Batch.from_data_list([item["graph"] for item in items])

    return dict(
        graph=graphs,
        open_ports=torch.stack([item["open_ports"] for item in items]),
        open_port_mask=torch.stack([item["open_port_mask"] for item in items]),
        target_port_idx=torch.stack([item["target_port_idx"] for item in items]),
        target_part_id=torch.stack([item["target_part_id"] for item in items]),
        target_color=torch.stack([item["target_color"] for item in items]),
        target_self_port=torch.stack([item["target_self_port"] for item in items]),
        target_rot_steps=torch.stack([item["target_rot_steps"] for item in items]),
    )
