# Implementation Plan — Graph Transformer for LEGO Assembly Generation

## Overview

Replace the flat-token GPT sequence model with a graph-based autoregressive model. At each generation step, the model reads the current partial assembly graph, selects an open connection port to snap to, and classifies the new brick (part, color, connecting port, rotation). Position is never predicted — it's computed exactly from the snap geometry.

---

## Phase 1 — Snap Math (`src/geometry/snap_math.py`)

Single new file. Two functions, no dependencies on PyG or torch.

**`compute_snap_matrix(parent_world_matrix, parent_port, new_part_port, rot_steps) → np.ndarray`**

Given a selected open port on an existing brick and a port on the new brick, returns the new brick's world matrix (4×4). Steps:
1. Compute world position + normal of parent port
2. Find base rotation `R_base` mapping `new_part_port.normal → -world_normal` via Rodrigues' formula
3. Apply discrete twist: `R_twist = rotation(rot_steps * 90°, around world_normal)`
4. `R_new = R_twist @ R_base`, `t_new = world_port_pos - R_new @ new_part_port.local_position`

**`extract_rot_steps(parent_world_matrix, parent_port, new_part_port, new_world_matrix) → int`**

Inverse of the above — used during preprocessing to recover which `rot_steps` (0–3) produced the ground-truth world matrix. Returns the closest integer in {0,1,2,3}.

---

## Phase 2 — Graph Preprocessor (`src/geometry/graph_builder.py`)

Converts raw MPD files into labeled training graphs. Runs once, saves `.npz` to `dataset/graph_sets/`.

**Algorithm per MPD:**
1. Parse MPD → `list[RawBrickData]`
2. Build full assembly via `LegoCore.from_raw_bricks()` → get all connections
3. Sort bricks by BFS from the bottommost brick, traversing the connection graph. This ensures the parent of each brick is almost always among the most recently processed bricks
4. Replay the build in BFS order, maintaining an **open port set**:
   - At step t, the open port set = all ports with no connection yet
   - For the brick being added at step t: find which open port it snapped to (from the precomputed full-graph connections)
   - Call `extract_rot_steps()` to get `rot_steps`
   - Record action: `(target_node_id, target_port_id, self_port_id, rot_steps)`
   - Update open port set: remove the two connected ports, add the new brick's remaining open ports
5. Skip any brick with no detected connection to the partial assembly (treat as a new disconnected seed; skip for now to keep training clean)

**Output `.npz` per set:**
```
node_part_ids     (N,)    int32   — part index in shared vocabulary
node_colors       (N,)    int32   — color index
node_translations (N, 3)  float32 — world x, y, z (LDU, normalized)
node_rotations    (N, 6)  float32 — 6D rotation (first two columns of R)
edge_index        (2, E)  int32   — COO: [source, target]
edge_rotations    (E, 9)  float32 — relative rotation R_a^T @ R_b
actions           (N, 4)  int32   — [target_node_id, target_port_id, self_port_id, rot_steps]
                                    row 0 = first brick (all -1, no parent)
build_order       (N,)    int32   — mapping: step index → original node_id
```

**Why 6D rotation, not 9D?** The 6D representation (2 columns of R) is the minimal continuous rotation representation — avoids gimbal lock and discontinuities compared to Euler/quaternion, and the third column is recoverable via cross product. Used in both nodes and edges.

**Normalization:** translations divided by a fixed scale constant (e.g. 100 LDU) to keep values ~O(1).

---

## Phase 3 — Dataset (`src/data/graph_dataset.py`)

```python
class AssemblyStepDataset(Dataset):
    """Each item = one (partial_graph, target_action) pair."""
```

Loads `.npz` files. One MPD with N bricks → N−1 training samples (one per step after the first brick). At item `i` from set `k` at step `t`:

- **Graph input**: nodes 0…t, edges among those nodes → `Data(x, edge_index, edge_attr)`
- **Open ports tensor**: `(P, 6+3)` — world position (3) + world normal (3) + source node index (1) + source port_id (1) + port_type (1), for all P currently open ports at step t
- **Target**: `(target_port_idx_in_open_list, part_id, color, self_port_id, rot_steps)` — all integers

The open port list is reconstructed at load time from the saved `actions` sequence (cheap replay).

**Collation:** PyG's `Batch.from_data_list()` handles variable-size graphs. Open port tensors are padded to `MAX_OPEN_PORTS=128` with a mask.

---

## Phase 4 — Model (`src/model/graph_transformer.py`)

```python
class GraphTransformer(nn.Module):
    def forward(self, batch, open_ports, open_port_mask) → dict[str, Tensor]
```

**Config** (`src/model/graph_config.py`, new dataclass, separate from old `ModelConfig`):
```
d_model = 256, n_heads = 8, n_layers = 6, d_ff = 1024
max_open_ports = 128, max_port_id = 32, max_parts = ~3500, max_colors = 100
dropout = 0.1
```

**Forward pass:**

```
1. Node embedding
   part_emb   = Embedding(n_parts, d_model)[node_part_ids]     # (N, d)
   color_emb  = Embedding(n_colors, d_model)[node_colors]      # (N, d)
   trans_emb  = Linear(3, d_model)[node_translations]          # (N, d)
   rot_emb    = Linear(6, d_model)[node_rotations]             # (N, d)
   x = LayerNorm(part_emb + color_emb + trans_emb + rot_emb)   # (N, d)

2. Graph encoder
   for _ in range(n_layers):
       x = TransformerConv(x, edge_index, edge_attr=edge_rotations_projected)
       x = FFN(x) + x   # pre-norm

3. Graph summary token
   g = mean_pool(x, batch_index)    # (B, d) — one vector per graph in batch

4. Open port encoder
   Each open port: world_pos (3) + world_normal (3) + source_node_emb (d) + port_type (1)
   port_emb = Linear(d+7, d)(concat(above))    # (B, MAX_PORTS, d)

5. Pointer (cross-attention)
   query = g.unsqueeze(1)                               # (B, 1, d)
   port_logits = (query @ port_emb.T) / sqrt(d)         # (B, 1, MAX_PORTS)
   port_logits[~open_port_mask] = -inf
   port_probs = softmax(port_logits)                     # (B, MAX_PORTS)
   selected_context = port_probs @ port_emb              # (B, d)  — soft selection

6. Classification heads (all conditioned on g + selected_context)
   context = g + selected_context                        # (B, d)
   part_logits      = Linear(d, n_parts)(context)        # (B, n_parts)
   color_logits     = Linear(d, n_colors)(context)       # (B, n_colors)
   self_port_logits = Linear(d, MAX_PORT_ID)(context)    # (B, MAX_PORT_ID)
   rot_logits       = Linear(d, 4)(context)              # (B, 4)
```

**Loss:**
```
L = CE(port_logits, target_port)
  + CE(part_logits, target_part)
  + CE(color_logits, target_color)
  + CE(self_port_logits, target_self_port)
  + CE(rot_logits, target_rot)
```
Equal weights initially; can tune per-head later.

---

## Phase 5 — Training (`src/model/graph_train.py`)

Mostly same structure as current `train.py`:
- AdamW, lr=3e-4, linear warmup → cosine decay
- Gradient clipping at 1.0
- Checkpoint every N epochs
- Log each head's loss separately (pointer loss, part loss, etc.) to see which heads learn

One important difference: the **pointer loss** is harder early in training (model needs to learn spatial reasoning). Consider warming up the pointer head weight in the combined loss.

---

## Phase 6 — Generation (`src/model/graph_generate.py`)

```
1. Place first brick at origin (identity matrix) — hardcoded start
2. Initialize open port set from first brick's ports
3. Loop:
   a. Build current graph → encode
   b. Forward pass → port_probs, part_probs, color_probs, self_port_probs, rot_probs
   c. Sample from each (temperature + top-k on part/color)
   d. selected_open_port = open_ports[argmax(port_probs)]
   e. new_part_port = PartDatabase.get(selected_part).ports[self_port_id]
   f. world_matrix = compute_snap_matrix(selected_open_port, new_part_port, rot_steps)
   g. Validate: no collision (spatial hash check)
   h. Place brick, update open port set
   i. Stop if max_bricks reached or open port set is empty
4. Export to .mpd
```

---

## Phase 7 — Cleanup

- Move old sequence model files to `src/model/legacy/` (don't delete, useful as reference)
- Update `requirements.txt`: add `torch torch_geometric`
- Update `TODO.md`
- Add `graph_train_model.py` and `graph_generate_model.py` at project root (mirrors current `train_model.py` / `generate_model.py`)

---

## File map

```
NEW:
  src/geometry/snap_math.py          Phase 1 — position computation
  src/geometry/graph_builder.py      Phase 2 — preprocessing pipeline
  src/data/graph_dataset.py          Phase 3 — PyTorch dataset
  src/model/graph_config.py          Phase 4 — model hyperparameters
  src/model/graph_transformer.py     Phase 4 — model
  src/model/graph_train.py           Phase 5 — training loop
  src/model/graph_generate.py        Phase 6 — generation
  graph_train_model.py               Phase 7 — entry point
  graph_generate_model.py            Phase 7 — entry point

MODIFIED:
  requirements.txt                   add torch_geometric
  TODO.md

MOVED:
  src/model/{transformer,train,generate,config}.py → src/model/legacy/
```

---

## Open questions (settled before coding)

1. **Disconnected bricks** (no snap detected): skip during training; treat first brick of each connected component as a seed placed at origin.
2. **MAX_PORT_ID cap**: 32 — parts with more ports have excess ports ignored during classification (rare edge case).
3. **Translation normalization scale**: divide by 100 LDU.
