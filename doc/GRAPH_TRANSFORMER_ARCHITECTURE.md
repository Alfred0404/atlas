# Graph Transformer Architecture — ATLAS v2

## Vision

**Current (Token-based):**

```
MPD → Flat Sequence [brick1_tokens, brick2_tokens, ...] → Decoder-Only Transformer → flat .mpd
```

**Target (Graph-based):**

```
                              ┌─────────────────────────────────────┐
                              │  Current Graph State + Visual Repr. │
                              │  (nodes=bricks, edges=connections)  │
                              └────────────┬──────────────────────────┘
                                           │
                                    ┌──────▼────────┐
                                    │  Placement    │
                                    │  Enumerator   │
                                    │ (SAT+Physics) │
                                    └──────┬────────┘
                                           │
                     ┌─────────────────────▼──────────────────────┐
                     │ List of Placement Solutions               │
                     │ [{brick_id, position, rotation, ...}, ...] │
                     └─────────────────┬──────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │  Graph          │
                              │  Transformer    │ ← Neural Network (replaces decoder-only)
                              │  (GNN/GAT)      │
                              └────────┬────────┘
                                       │
                   ┌───────────────────▼───────────────────┐
                   │ (brick_id, placement) distribution    │
                   │ Next brick to place + where           │
                   └───────────────────┬───────────────────┘
                                       │
                              ┌────────▼────────────┐
                              │  Sample + Validate  │
                              │  Update Graph       │
                              │  Repeat             │
                              └─────────────────────┘
```

---

## Phase 1: Data Structures (Foundation)

### 1.1 — `src/graph/types.py`

Define core entities:

```python
@dataclass
class ConnectorInfo:
    """Represents a single connector on a brick (stud, hole, etc)."""
    element_id: int              # Unique within brick
    position: np.ndarray         # [x, y, z] in brick coords (3,)
    rotation_matrix: np.ndarray  # 3x3 orthonormal matrix
    normal: np.ndarray           # [0, ±1, 0] etc; points outward (3,)
    connector_type: str          # "stud", "hole", "universal", etc

@dataclass
class BrickNode:
    """Represents a placed brick in the model."""
    node_id: int                 # Unique index
    part_id: str                 # "3001", "3002", etc
    color: int                   # 16 (gray), 4 (red), etc
    world_matrix: np.ndarray     # 4x4 transform [R|t]
    connectors: list[ConnectorInfo]  # Parsed from .conn file

@dataclass
class Connection:
    """Edge between two bricks."""
    node_a: int                  # node_id of brick A
    node_b: int                  # node_id of brick B
    connector_a: ConnectorInfo
    connector_b: ConnectorInfo
    # Semantics: stud_A plugs into hole_B (or vice versa)
    connection_type: str         # "stud_to_hole", "stud_to_stud", etc
    contact_metrics: dict        # { "touching": 0.02, "mating": 0.95, ... }

class LDUGraph:
    """Full model representation as a graph."""
    def __init__(self):
        self.nodes: dict[int, BrickNode] = {}      # node_id → BrickNode
        self.edges: list[Connection] = []
        self.next_node_id: int = 0

    def add_brick(self, brick: BrickNode) -> int:
        """Returns node_id."""
        bid = self.next_node_id
        self.nodes[bid] = brick
        self.next_node_id += 1
        return bid

    def add_connection(self, conn: Connection):
        self.edges.append(conn)

    def to_adjacency_matrix(self) -> np.ndarray:
        """Returns n_nodes × n_nodes sparse adjacency."""

    def node_embeddings_precompute(self) -> dict[int, np.ndarray]:
        """Returns {node_id: [visual_features]}."""
        # e.g., [x, y, z, color, part_embedding, num_connectors_free, ...]
```

---

## Phase 2: Connectivity Parser (Extract .conn Data)

### 2.1 — `src/graph/conn_parser.py`

Wrap the existing `test_conn_collision_stack.py` logic:

```python
class ConnParser:
    """Parses .conn files; returns list of ConnectorInfo."""

    @staticmethod
    def load_connectors(part_id: str, repo_root: Path) -> list[ConnectorInfo]:
        """
        Load .conn file for part_id.
        Returns list of ConnectorInfo in brick-local coordinates.
        """
        # (reuse parse_conn_elements from test_conn_collision_stack.py)

    @staticmethod
    def get_world_connectors(brick_node: BrickNode) -> list[ConnectorInfo]:
        """
        Transform connectors from brick-local → world coords using world_matrix.
        Returns list with position and rotation_matrix in world space.
        """
```

### 2.2 — `src/graph/collision_checker.py`

Wrap collision/contact detection:

```python
class CollisionChecker:
    """
    Uses .col files and SAT/AABB to validate brick placements.
    """

    @staticmethod
    def load_collision_boxes(part_id: str, repo_root: Path) -> list[CollisionBox]:
        """Parse .col file; returns collision geometries in brick-local coords."""
        # (parse ASCII .col format)

    @staticmethod
    def test_placement_validity(
        candidate_brick: BrickNode,
        occupied_space: list[BrickNode]  # existing bricks
    ) -> bool:
        """Returns True if no overlaps; uses SAT."""

    @staticmethod
    def evaluate_contacts(brick_a: BrickNode, brick_b: BrickNode) -> dict:
        """
        Computes touching/mating/colliding metrics across connectors.
        Returns { "mating": count, "touching": count, "colliding": count }.
        """
```

---

## Phase 3: Placement Enumerator (Generate Valid Placements)

### 3.1 — `src/graph/placement_generator.py`

This is **KEY** — converts graph state → list of valid placements.

```python
class PlacementSolution:
    """Represents one valid placement for a brick."""
    part_id: str
    color: int  # or None = any color
    world_matrix: np.ndarray
    attachment_node: int       # Which existing brick it attaches to (or -1 for free)
    attachment_connector_a: ConnectorInfo
    attachment_connector_b: ConnectorInfo
    # For model input: encode (part_id, position, rotation) as tokens/embeddings

class PlacementEnumerator:
    """
    Given current graph state, enumerate all valid placements.
    This is what the model receives as **input context**.
    """

    @staticmethod
    def enumerate_attachments(
        graph: LDUGraph,
        candidate_part_id: str,
        collision_checker: CollisionChecker,
        conn_parser: ConnParser
    ) -> list[PlacementSolution]:
        """
        For each free connector on existing bricks:
          - Try each mating pair (stud→hole, etc)
          - Compute placement offset
          - Validate no collisions
          - Collect valid placements

        Returns sorted list of PlacementSolution.
        """
        solutions = []

        # (1) Get all free connectors across existing nodes
        free_connectors = []
        for node_id, brick in graph.nodes.items():
            for i, conn in enumerate(brick.connectors):
                if not any(e.connector_a == conn or e.connector_b == conn
                           for e in graph.edges):
                    free_connectors.append((node_id, i, conn))

        # (2) Load candidate part's connectors
        candidate_conns = conn_parser.load_connectors(candidate_part_id, Path('.'))

        # (3) Try each pairing
        for attach_node_id, attach_conn_idx, attach_conn_world in free_connectors:
            for cand_local_conn in candidate_conns:
                # Try mating: cand_local_conn.normal == -attach_conn_world.normal
                if dot(cand_local_conn.normal, attach_conn_world.normal) < -0.75:
                    # Compute offset to align connectors
                    offset = attach_conn_world.position - cand_local_conn.position
                    candidate_world_matrix = np.eye(4)
                    candidate_world_matrix[:3, 3] = offset

                    # Validate no collisions
                    cand_brick = BrickNode(
                        node_id=-1,  # temp
                        part_id=candidate_part_id,
                        color=find_best_color(candidate_part_id),  # or sample
                        world_matrix=candidate_world_matrix,
                        connectors=transform_connectors(candidate_conns, candidate_world_matrix)
                    )

                    if collision_checker.test_placement_validity(cand_brick, list(graph.nodes.values())):
                        solutions.append(PlacementSolution(
                            part_id=candidate_part_id,
                            world_matrix=candidate_world_matrix,
                            attachment_node=attach_node_id,
                            attachment_connector_a=attach_conn_world,
                            attachment_connector_b=cand_local_conn
                        ))

        # (4) Optional: add "free placements" (no attachment) in limited region
        # ...

        return solutions[:K]  # Return top K valid placements
```

---

## Phase 4: Graph Transformer Model

### 4.1 — `src/model/graph_transformer.py`

Replace `transformer.py` with a GNN-based architecture:

```python
class GraphTransformer(nn.Module):
    """
    Input:
      - Current graph state (nodes + edges)
      - List of placement candidates (K × placement_dim)

    Output:
      - Logits over placement_id (which placement to choose)
      - Optionally: brick_id distribution, color distribution
    """

    def __init__(self, config: GraphModelConfig):
        # GNN layers (GAT, GCN, or message-passing)
        # Placement embedding layers
        # Decision head
        pass

    def forward(
        self,
        graph: LDUGraph,                          # Current model
        placements: list[PlacementSolution],      # Enumeration output
        device: str = "cpu"
    ) -> dict:
        """
        Returns:
          {
            "placement_logits": [K],               # logits for which placement
            "brick_id_logits": [vocab_size],       # (optional auxiliary head)
          }
        """
        # (1) Encode graph nodes
        # node_embeddings = GATLayer(graph.nodes, graph.edges)

        # (2) Encode each placement candidate in context of graph
        # placement_encodings = [self.encode_placement(p, node_embeddings) for p in placements]

        # (3) Score placements
        # placement_logits = self.placement_head(torch.stack(placement_encodings))

        # return { "placement_logits": placement_logits, ... }
```

### 4.2 — Config: `src/model/graph_config.py`

```python
@dataclass
class GraphModelConfig:
    embedding_dim: int = 256
    gnn_layers: int = 4
    num_heads: int = 8

    # Architecture choice
    gnn_type: str = "gat"  # "gat", "gcn", "message_passing"

    # Training
    learning_rate: float = 3e-4
    warmup_steps: int = 500
```

---

## Phase 5: Training Pipeline

### 5.1 — `src/model/graph_train.py`

```python
class GraphTrainer:
    """
    Analogous to train_model.py but for graph-based generation.
    """

    def __init__(self, model: GraphTransformer, config: GraphModelConfig):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def train_epoch(self, dataset: GraphDataset):
        """
        For each training sample (target graph):
          - Start with empty graph
          - For each brick in target (in order):
            - Enumerate placements for next brick type
            - Compute loss: KL(model_output, target_placement)
            - Backprop
        """
        total_loss = 0
        for graph_target in dataset:
            loss = self.train_single_graph(graph_target)
            total_loss += loss
        return total_loss / len(dataset)

    def train_single_graph(self, graph_target: LDUGraph):
        """Unfold graph sequentially."""
        graph_current = LDUGraph()  # Start empty

        for node_id_target in sorted(graph_target.nodes.keys()):
            brick_target = graph_target.nodes[node_id_target]

            # Enumerate valid placements for this brick
            placements = PlacementEnumerator.enumerate_attachments(
                graph_current,
                brick_target.part_id,
                self.collision_checker,
                self.conn_parser
            )

            if not placements:
                # Brick doesn't fit → skip or penalize
                continue

            # Forward: model predicts which placement
            output = self.model(graph_current, placements)
            placement_logits = output["placement_logits"]

            # Label: which placement matches target_brick.world_matrix?
            target_placement_idx = find_best_matching_placement(
                brick_target.world_matrix,
                placements
            )

            # Loss
            loss = F.cross_entropy(placement_logits, torch.tensor(target_placement_idx))

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update graph state for next iteration
            graph_current.add_brick(brick_target)

        return loss
```

### 5.2 — Dataset: `src/data/graph_dataset.py`

```python
class GraphDataset(Dataset):
    """Load target .mpd files, convert to LDUGraph objects."""

    def __init__(self, mpd_paths: list[Path]):
        self.graphs = []
        for mpd_path in mpd_paths:
            raw_bricks = MPDParser.parse(mpd_path)
            graph = self.convert_to_graph(raw_bricks)
            self.graphs.append(graph)

    def convert_to_graph(self, raw_bricks: list[RawBrickData]) -> LDUGraph:
        """
        For each RawBrickData:
          - Create BrickNode
          - Load connectors from .conn file
          - Try to match with previously added bricks → add Connection edges
        Returns LDUGraph.
        """
        graph = LDUGraph()
        for raw in raw_bricks:
            brick = BrickNode(
                node_id=-1,  # Will be assigned by add_brick
                part_id=raw.brick_id,
                color=raw.color,
                world_matrix=raw.world_matrix,
                connectors=self.conn_parser.load_connectors(raw.brick_id)
            )
            node_id = graph.add_brick(brick)

            # Match with existing bricks
            for existing_id in list(graph.nodes.keys())[:-1]:
                existing_brick = graph.nodes[existing_id]
                metrics = CollisionChecker.evaluate_contacts(existing_brick, brick)
                if metrics.get("mating", 0) > 0:
                    # Add edge (simplified; could be more sophisticated)
                    conn = Connection(
                        node_a=existing_id,
                        node_b=node_id,
                        connection_type="mating",
                        contact_metrics=metrics
                    )
                    graph.add_connection(conn)

        return graph
```

---

## Phase 6: Generation Pipeline

### 6.1 — `src/model/graph_generate.py`

```python
class GraphGenerator:
    """
    Autoregressively build a model graph.
    """

    def __init__(self, model: GraphTransformer, config):
        self.model = model
        self.placement_enumerator = PlacementEnumerator()
        # ...

    def generate(
        self,
        max_bricks: int = 50,
        temperature: float = 0.8,
        top_k: int = 10
    ) -> LDUGraph:
        """
        Iteratively:
          1. Enumerate all valid placements for all candidate parts
          2. Model scores them
          3. Sample one
          4. Add to graph
          5. Repeat
        """
        graph = LDUGraph()

        for step in range(max_bricks):
            # (1) Enumerate placements for all candidate parts
            all_placements = []
            for part_id in self.get_candidate_parts():
                placements = self.placement_enumerator.enumerate_attachments(
                    graph, part_id, self.collision_checker, self.conn_parser
                )
                all_placements.extend(placements)

            if not all_placements:
                break  # No more valid placements

            # (2) Model scores
            output = self.model(graph, all_placements, device=self.device)
            logits = output["placement_logits"]  # [n_placements]

            # (3) Sample with temperature + top-k
            probs = F.softmax(logits / temperature, dim=0)
            top_k_indices = torch.topk(probs, min(top_k, len(all_placements)))[1]
            sampled_idx = top_k_indices[torch.multinomial(probs[top_k_indices], 1)].item()

            # (4) Add to graph
            placement = all_placements[sampled_idx]
            brick = BrickNode(
                node_id=-1,
                part_id=placement.part_id,
                color=placement.color,
                world_matrix=placement.world_matrix,
                connectors=transform_connectors(
                    self.conn_parser.load_connectors(placement.part_id),
                    placement.world_matrix
                )
            )
            graph.add_brick(brick)

        return graph
```

### 6.2 — Export: `src/file_io/graph_to_mpd.py`

```python
def graph_to_mpd(graph: LDUGraph, output_path: Path):
    """Convert LDUGraph back to RawBrickData list, write .mpd."""
    raw_bricks = [
        RawBrickData(
            brick_id=brick.part_id,
            world_matrix=brick.world_matrix,
            color=brick.color
        )
        for brick in graph.nodes.values()
    ]
    write_mpd_file(str(output_path), raw_bricks, model_name="graph_generated")
```

---

## Phase 7: Development Roadmap

### Milestone 1: Data Structures + Parser

- [ ] Implement `src/graph/types.py` → `BrickNode`, `Connection`, `LDUGraph`
- [ ] Wrap `.conn` parser → `src/graph/conn_parser.py`
- [ ] Wrap collision logic → `src/graph/collision_checker.py`
- [ ] **Test**: Parse one .mpd file → convert to LDUGraph + verify consistency

### Milestone 2: Placement Enumerator

- [ ] Implement `src/graph/placement_generator.py` → `PlacementEnumerator`
- [ ] Test enumeration on a known part (e.g., 24085 with 3 studs)
- [ ] **Test**: Generate 10 valid placements for a candidate brick given existing graph

### Milestone 3: Model Architecture

- [ ] Define `src/model/graph_config.py`
- [ ] Implement `src/model/graph_transformer.py` (start simple: GCN or basic MLP)
- [ ] **Test**: Forward pass on dummy graph + placements

### Milestone 4: Training

- [ ] Implement `src/data/graph_dataset.py` → convert .mpd → LDUGraph
- [ ] Implement `src/model/graph_train.py`
- [ ] **Test**: Train on 10 small .mpd files, verify loss decreases

### Milestone 5: Generation

- [ ] Implement `src/model/graph_generate.py`
- [ ] Implement `src/file_io/graph_to_mpd.py`
- [ ] **Test**: Generate a small model (5–10 bricks), verify .mpd is valid

### Milestone 6: Integration & Evaluation

- [ ] Integrate into main pipeline (can coexist with old token-based)
- [ ] Benchmark vs. old decoder-only model
- [ ] Visualize generated graphs + connectivity

---

## Directory Structure (Proposed)

```
src/
├── graph/                          [NEW]
│   ├── __init__.py
│   ├── types.py                   # BrickNode, Connection, LDUGraph
│   ├── conn_parser.py             # Wrap .conn parsing
│   ├── collision_checker.py       # Wrap .col + SAT
│   ├── placement_generator.py     # PlacementEnumerator
│   └── utils.py                   # Helpers (transform, validation)
│
├── model/
│   ├── graph_config.py            [NEW]
│   ├── graph_transformer.py       [NEW]
│   ├── graph_train.py             [NEW]
│   ├── graph_generate.py          [NEW]
│   ├── transformer.py             [existing, keep for now]
│   ├── train.py                   [existing]
│   └── generate.py                [existing]
│
├── data/
│   ├── graph_dataset.py           [NEW]
│   ├── sequence_dataset.py        [existing]
│   └── ...
│
├── file_io/
│   ├── graph_to_mpd.py            [NEW]
│   ├── mpd_writer.py              [existing]
│   └── ...
│
├── core/
│   └── ...                         [existing, unchanged for now]
│
└── ...

tests/
├── test_graph_types.py            [NEW]
├── test_placement_generator.py    [NEW]
├── test_graph_model.py            [NEW]
└── ...
```

---

## Key Design Principles

1. **Modularity**: Each component (parser, enumerator, model) is independent and testable.
2. **Reusability**: Wrap existing collision/connectivity logic; don't rewrite.
3. **Incremental**: Start with simple GCN; upgrade to GAT/Transformer later.
4. **Interpretability**: Placement enumerator is deterministic → easy to debug.
5. **Backward Compatibility**: Keep old token-based pipeline; run both in parallel initially.

---

## Next Steps

1. **Review this architecture** — Does it match your vision?
2. **Start Milestone 1** — Create branch, implement data structures
3. **Build enumerator** — Most critical component; gets everything else right
4. **Iterate on model** — Once placements are working, model training becomes straightforward
