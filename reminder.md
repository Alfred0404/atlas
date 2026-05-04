# ATLAS — Codebase Reminder

**Autoregressive Transformer Lego Assembly Synthesis** — two parallel models for generating LEGO sets:
- **Sequence model** (ATLASTransformer): flattened token sequence, GPT-style
- **Graph model** (GraphTransformer): iterative assembly graph, port-based placement

---

## Two Pipelines at a Glance

```
MPD files
  │
  ├─► [Sequence pipeline]
  │     MPDParser → DatasetBuilder → .npy sequences
  │     SequenceDataset → Trainer(ATLASTransformer) → checkpoint
  │     Generator → token seq → write_mpd_file()
  │
  └─► [Graph pipeline]
        graph_build_dataset.py → graph_vocab.pt + .npz files
        AssemblyStepDataset → GraphTrainer(GraphTransformer) → checkpoint
        GraphGenerator → assembly → write_mpd_file()
```

---

## CLI Commands

```bash
# ── Sequence model ─────────────────────────────────────────────────────────
python src/main.py                          # build tokenized .npy dataset
python -m src.model.train_model             # train ATLASTransformer
python -m src.model.generate_model          # generate a .mpd file

# ── Graph model ─────────────────────────────────────────────────────────────
python graph_build_dataset.py               # build .npz graph dataset (top-500 parts)
python graph_build_dataset.py --top-k-parts 500   # explicit (default)
python graph_train_model.py                 # train GraphTransformer
python graph_generate_model.py              # generate a .mpd file

# ── Rebuild from scratch (graph) ────────────────────────────────────────────
rm dataset/graph_vocab.pt dataset/graph_sets/*.npz
python graph_build_dataset.py --top-k-parts 500

# ── Misc ────────────────────────────────────────────────────────────────────
pytest                                      # run all tests
python -m src.visualize_graph <file.mpd>    # 2D graph
python -m src.visualize_graph <file.mpd> --3d   # 3D graph
# webapp:
python webapp/server.py                     # drag-and-drop .mpd → 3D vis
```

---

## Token Layout (sequence model)

Defined in `src/config.py` via `Config.OFFSETS`:

| Range | Meaning | Count |
|---|---|---|
| 0–3 | Special: PAD=0, SOS=1, EOS=2, UNK=3 | 4 |
| 4–27 | Rotations (24 discrete orthogonal) | 24 |
| 28–1027 | X positions (binned at 2 LDU, ±1000) | 1000 |
| 1028–2027 | Y positions | 1000 |
| 2028–3027 | Z positions | 1000 |
| 3028–3127 | Colors (100 slots) | 100 |
| 3128+ | Parts (dynamic, grows with vocabulary) | N |

Each brick = **6 consecutive tokens**: `[part_id, x_bin, y_bin, z_bin, rotation, color]`

> ⚠️ `Config.OFFSETS["parts"]` is hardcoded at 3128 but the true runtime value depends on how many colors are in the vocab. Always use `vocab_manager.get_offsets()`.

---

## Root Entry Points

| Script | Purpose |
|---|---|
| `src/main.py` | Build tokenized .npy dataset (sequence pipeline) |
| `graph_build_dataset.py` | Build .npz graph dataset (Pass 1: vocab with top-k filter → Pass 1.5: PartDB cache → Pass 2: parallel .npz) |
| `train_model.py` | Train ATLASTransformer |
| `generate_model.py` | Generate with ATLASTransformer |
| `graph_train_model.py` | Train GraphTransformer |
| `graph_generate_model.py` | Generate with GraphTransformer |

---

## src/config.py

`Config` — static class with spatial bounds (`l_min=-1000`, `l_max=1000`, `step=2`), token offsets, and rotation count. All offsets computed from constants here; import elsewhere.

---

## src/core/

### vocabulary.py — `VocabularyManager`
Manages `atlas_config.json`: parts, colors, rotations.
```python
VocabularyManager(atlas_config_path)
  .add_part(part_id) → int         # in-memory only; call save() after
  .add_color(color_id) → int
  .add_parts(ids) / .add_colors(ids)
  .get_part_index(id) / .get_color_index(id)
  .get_offsets() → dict            # runtime offsets (use this, not Config.OFFSETS["parts"])
  .save()                          # writes JSON + recalculates offsets
```
Generates 24 chiral rotation matrices on-the-fly (never stored).

### tokenizer.py — `AtlasTokenizer`
Encodes/decodes brick attributes to/from token IDs.
```python
AtlasTokenizer()
  .load_vocabulary(config_path)
  .position_to_bin_id(pos, axis) → int
  .bin_id_to_position(bin_id, axis) → float
  .brick_id_to_token(brick_id) → int
  .color_id_to_token(color_id) → int
  .rotation_matrix_to_token(R) → int    # finds closest of 24 rotations
  # batch versions: batch_positions_to_bin_ids, batch_rotation_matrices_to_tokens, ...
```

---

## src/data/

### parser.py — `MPDParser`
Parses .mpd/.ldr, flattens hierarchical submodels to world-space bricks.
```python
RawBrickData(brick_id: str, world_matrix: np.ndarray(4,4), color: int)

MPDParser(mpd_file_path)
  ._submodels: dict[str, list[str]]   # submodel name → raw lines
  .parse(model_name) → list[RawBrickData]   # flattens recursively
  .flatten(model_name, parent_matrix) → list[RawBrickData]
```

### builder.py — `DatasetBuilder`
Two-pass pipeline: collect vocab → tokenize. Handles augmentation and sorting.
```python
DatasetBuilder(atlas_config_path, augmentation_config=None)
  .process_dataset(max_files=None)   # writes .npy to tokenized_sets/
```
Sort order: Y → X → Z (bottom-to-top, deterministic).

### sequence_dataset.py — `SequenceDataset`
PyTorch Dataset for .npy token sequences.
```python
SequenceDataset(npy_dir, max_bricks=200, max_files=None)
  # each item: 1-D tensor [SOS, t1, t2, ..., EOS], padded with PAD
collate_fn(batch) → (inputs, targets, lengths)   # pads batch, shifts by 1
```

### graph_dataset.py — `AssemblyStepDataset`
PyTorch Dataset for graph assembly steps (one step = one training item).
```python
AssemblyStepDataset(graph_dir, max_open_ports=256, cache_size=128)
  # each item (AssemblyItem TypedDict):
  #   graph: PyG Data (node feats, edge_index, edge_attrs)
  #   open_ports: (256, 9) float — world pos/nrm + node_idx + port_id + type
  #   open_port_mask: (256,) bool
  #   target_port_idx, target_part_id, target_color, target_self_port, target_rot_steps
```

### augmentation.py
```python
AugmentationConfig(enable_rotations, enable_mirror, num_permutations, seed)
generate_augmented_variants(bricks, config) → list[(suffix, bricks)]
apply_global_transform(bricks, R_3x3) → list[RawBrickData]
```
Produces up to 8× copies: 4 Y-rotations × mirror-X + seeded permutations.

### adjacency.py
BFS sorting for assembly order (used in sequence pipeline).
```python
sort_bricks_by_adjacency(bricks, threshold=40.0, rng=None) → list[RawBrickData]
```

---

## src/model/

### config.py — `ModelConfig` (sequence)
```python
@dataclass ModelConfig:
  vocab_size: int          # set from VocabularyManager at runtime
  d_model=256, n_heads=8, n_layers=6, d_ff=1024
  max_seq_len=1202         # 200 bricks × 6 + SOS + EOS
  batch_size=8, learning_rate=3e-4, eos_weight=1.0
  max_epochs=100, warmup_steps=350, checkpoint_interval=100
  max_gen_bricks=400, temperature=0.8, top_k=0
  offsets: dict            # copied from Config.OFFSETS
```

### transformer.py — `ATLASTransformer`
Decoder-only transformer with field-aware logit masking.
```python
ATLASTransformer(config: ModelConfig)
  .forward(x, mask_logits=True) → logits (B, S, V)
```
Embeddings: token + absolute position + intra-brick field (7 fields: SOS + 6/brick).
Masking: per-field valid token ranges enforced during both training and inference.
Init: `std=0.02` for all embeddings (prevents loss stuck at ~15).

### train.py — `Trainer`
```python
Trainer(model, train_loader, config, val_loader=None, device="cuda")
  .train()                 # saves checkpoint every checkpoint_interval steps
  .load_checkpoint(path)
```
AdamW, linear warmup → cosine annealing, grad clip at 1.0.

### generate.py — `Generator`
```python
Generator(model, config, device)
  .generate(max_bricks, temperature, top_k) → token tensor
```
Collision penalty: duplicate (x, z) positions are penalized in logits.

### graph_config.py — `GraphModelConfig`
```python
@dataclass GraphModelConfig:
  n_parts=501              # 500 top-freq parts + UNK at index 0
  n_colors=128
  d_model=256, n_heads=8, n_layers=6, d_ff=1024
  max_open_ports=256, max_port_id=32, n_rot_steps=4
  node_cont_dim=9, edge_attr_dim=9, port_feat_dim=9
  batch_size=32, learning_rate=3e-4, max_epochs=100
  max_gen_bricks=400, temperature=0.8, top_k=50
```

### graph_transformer.py — `GraphTransformer`
Graph encoder → 4 classification heads.
```python
GraphTransformer(cfg: GraphModelConfig)
  .forward(nodes, edges, edge_attrs, graph_batch, open_ports, open_port_mask,
           target_port_idx=None) → dict:
    # port_logits (B, P), part_logits (B, n_parts),
    # color_logits (B, n_colors), self_port_logits (B, max_port_id),
    # rot_logits (B, n_rot_steps)
```
Key fixes:
- `ctx_norm` (LayerNorm) on `g + port_context` before classifier heads — prevents loss stuck at ~30
- Teacher-forced port context during training (`target_port_idx`) → inference uses top-1 predicted port

Node features: part embedding + color embedding + continuous projection (translation×6D-rotation).
Edge features: relative rotation matrix (9-D).

### graph_train.py — `GraphTrainer`
```python
GraphTrainer(cfg, model, train_dataset, val_dataset, device, loss_weights=None)
  .train()
```
Multi-head cross-entropy loss (port, part, color, self_port, rotation).

### graph_generate.py — `GraphGenerator`
```python
GraphGenerator(cfg, model, part_db, part_vocab, color_vocab, device)
  .generate(seed_part_id, seed_color, max_bricks) → list[RawBrickData]
```
Maintains list of open ports; samples new brick for each available port until max_bricks or no ports remain.

---

## src/geometry/

### lego_core.py — `LegoCore`
Central assembly state machine. Builds the graph G=(V, E) with spatial collision detection.
```python
GraphNode(node_id, part_id, color, world_matrix)
GraphEdge(node_a, node_b, port_a_id, port_b_id, relative_rotation)

LegoCore(part_db, cell_size=8.0)
  .place_brick(part_id, color, world_matrix, validate=True) → node_id | None
  .remove_brick(node_id)
  .validate_placement(part_id, world_matrix) → bool   # collision check
  .get_graph() → (nodes, edges)
  .from_raw_bricks(raw_bricks, part_db) → LegoCore   # classmethod
```

### lego_part.py — `LegoPart` / `PartDatabase`
```python
LegoPart(part_id, ports: list[Port], collision_boxes)
  .male_ports / .female_ports   # filtered by type
  .bounding_box, .y_max
  .from_files(part_id, conn_parser, col_parser) → LegoPart   # classmethod

PartDatabase(parts_dir, col_dir)
  .get(part_id) → LegoPart | None
  .get_or_default(part_id) → LegoPart  # fallback to minimal geometry
  .load_cache(path) / .save_cache(path)
```

### conn_parser.py — `ConnParser`
Recursively parses LDraw `.dat` files to extract Port objects (studs + anti-studs).
```python
ConnParser(parts_dir="C:/Users/Public/Documents/LDraw/parts")
  .parse(part_id) → list[Port]
```
Reads from `C:/Users/Public/Documents/LDraw/parts/` with primitives in `../p/`.

### port.py — `Port`
```python
@frozen_dataclass Port:
  port_id: int
  local_position: np.ndarray(3,)
  normal: np.ndarray(3,)
  port_type: str   # "male" | "female"
  radius: float = 6.0
  .transformed(world_matrix) → Port   # apply 4×4 transform
```

### snap.py
```python
check_snap(part_db, brick_a, brick_b, pos_tol=2.0, angle_tol=0.75) → list[(port_a, port_b)]
find_all_connections(part_db, bricks, proximity_threshold=80.0) → list[(idx_a, idx_b, ports)]
```
Uses KDTree for port matching; brick-level KDTree pre-filter.

### snap_math.py
```python
compute_snap_matrix(parent_world_mat, parent_port, new_part_port, rot_steps) → np.ndarray(4,4)
extract_rot_steps(R_rel) → int   # 0/1/2/3 → 0°/90°/180°/270°
```

### spatial_hash.py — `SpatialHash`
O(1) voxel-grid collision detection (8 LDU cells).
```python
SpatialHash(cell_size=8.0)
  .insert(brick_idx, boxes, world_matrix)
  .check_overlap(boxes, world_matrix, exclude=set()) → set[int]
  .remove(brick_idx)
```

### graph_builder.py — `process_mpd()`
MPD → labeled .npz for training. Returns dict of numpy arrays.
```python
process_mpd(mpd_path, part_db, part_vocab, color_vocab) → dict | None
# output arrays:
#   node_part_ids (N,), node_colors (N,),
#   node_translations (N,3) scaled by TRANSLATION_SCALE=100,
#   node_rotations (N,6) 6-D repr,
#   edge_index (2,E), edge_rotations (E,9),
#   actions (N,4): [target_node, target_port, self_port, rot_steps],
#   build_order (N,)
```
BFS build order from seed (most-connected) node. Parts not in `part_vocab` → index 0 (UNK).

---

## src/maths/

### rotations.py
```python
generate_chiral_rotation_matrices() → list[np.ndarray(3,3)]   # 24 matrices
find_closest_rotation_matrix(R, references) → int             # index by Frobenius norm
```

---

## src/file_io/

### mpd_writer.py
```python
write_mpd_file(output_path, raw_data: list[RawBrickData], model_name="ATLAS_generated")
```
Writes LDraw Type-1 lines (color x y z a b c d e f g h i part.dat).

### scraper.py / omr_scraper.py
Download LEGO set files from OMR (Open Model Repository). `omr_scraper.py` scrapes the OMR index; `scraper.py` handles individual file downloads.

---

## webapp/

Flask + Three.js. Drop a .mpd → see 3D assembly graph with ports, connections, and brick info panel.

```
POST /api/parse   → returns {stats, nodes, edges} JSON
GET  /            → index.html
```
Run: `python webapp/server.py`

---

## Key Data Files

| File | Contents |
|---|---|
| `atlas_config.json` | Vocab state: special/rotation/part/color mappings + runtime offsets. Rebuilt by `src/main.py`. |
| `dataset/graph_vocab.pt` | `{part_vocab, color_vocab, part_freq, top_k_parts}` — rebuilt by `graph_build_dataset.py` |
| `dataset/graph_sets/*.npz` | One per MPD: node/edge tensors + BFS action supervision |
| `tokenized_sets/*.npy` | One per MPD: flat token sequences (int32) |
| `checkpoints/` | Sequence model checkpoints |
| `checkpoints/graph/` | Graph model checkpoints |
| `dataset/part_db_cache.pkl` | Pickled PartDatabase (pre-warmed geometry for all vocab parts) |
| `dataset/technic_blacklist.txt` | Set numbers to exclude (Technic, Bionicle, Hero Factory) |

---

## Known Footguns

1. **Token offset mismatch**: `Config.OFFSETS["parts"]` is hardcoded at 3128 but shifts if colors are added. Always use `vocab_manager.get_offsets()` at runtime. If colors change after tokenization, all `.npy` files are invalid — regenerate.

2. **VocabularyManager.save() is explicit**: `add_part()`/`add_color()` only update RAM. Call `.save()` before anything reads `atlas_config.json`.

3. **Logit masking must be on during training** (`mask_logits=True`): without it, the model spreads probability over ~3500 tokens; at generation time the mask collapses this to always the same output regardless of temperature.

4. **Graph vocab top-k filter**: the current `graph_vocab.pt` was built with `--top-k-parts 500`. If you rebuild with a different k, delete both `graph_vocab.pt` and all `.npz` files so parts indices stay consistent.

5. **Embedding init**: ATLASTransformer uses `std=0.02` (not PyTorch default `N(0,1)`). Default init causes logit std ≈ 16 and loss stuck at ~60+.
