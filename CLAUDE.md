# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ATLAS (Autoregressive Transformer Lego Assembly Synthesis) trains a GPT-style decoder-only transformer to generate novel LEGO models. It implements a full ML pipeline: raw LDraw .mpd files → tokenized sequences → transformer training → generation → .mpd export.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# GPU support (optional, CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Process dataset (parse .mpd files → tokenized .npy sequences)
python src/main.py

# Train the model
python -m src.model.train_model

# Generate a new LEGO set
python -m src.model.generate_model

# Run all tests
pytest

# Run a single test file
pytest tests/test_vocabulary.py

# Run a single test
pytest tests/test_vocabulary.py::TestVocabularyManager::test_add_part

# Visualize assembly graph of a LEGO set
python -m src.visualize_graph dataset/mpd_files/165-1.mpd
python -m src.visualize_graph dataset/mpd_files/165-1.mpd --3d
```

## Architecture

### Data Pipeline

```
dataset/mpd_files/*.mpd → MPDParser → DatasetBuilder → Tokenizer → tokenized_sets/*.npy
```

1. **MPDParser** (`src/data/parser.py`) reads LDraw .mpd files, flattens hierarchical submodels into world-space brick placements
2. **DatasetBuilder** (`src/data/builder.py`) normalizes bricks (center at origin, deterministic sort by Y→X→Z)
3. **Tokenizer** (`src/core/tokenizer.py`) converts each brick into a fixed **6-token sequence**: `[part_id, x_bin, y_bin, z_bin, rotation, color]`
4. **VocabularyManager** (`src/core/vocabulary.py`) manages dynamic part/color mappings, persisted in `atlas_config.json`

### Token Layout

Token ranges are defined by offsets in `src/config.py`:
- Special tokens (PAD=0, SOS=1, EOS=2, UNK=3) at indices 0–3
- Rotations at 4–27 (24 discrete orthogonal orientations)
- Positions at 28–3027 (binned at 2 LDU precision, range ±1000)
- Colors at 3028–3127
- Parts at 3128+

### Model

**ATLASTransformer** (`src/model/transformer.py`): ~7M parameter decoder-only transformer.
- Embeddings = token embedding + absolute position embedding + intra-brick field embedding (7 fields: SOS + 6 per brick)
- 6 layers, 8 heads, 256-dim, 1024-dim FFN, pre-norm (`norm_first=True`)
- Causal attention mask; **field-based logit masking** during both training and inference constrains outputs to valid token ranges per field position

**Trainer** (`src/model/train.py`): AdamW (lr=3e-4), linear warmup (500 steps) → cosine annealing, gradient clipping at 1.0. Logit masking is enabled during training (`mask_logits=True`).

**Generator** (`src/model/generate.py`): Temperature (0.8) + top-k (50) sampling with field-aware logit masking. Outputs .mpd files via `src/file_io/mpd_writer.py`.

### Geometry Engine (LegoCore)

Converts .mpd files into assembly graphs `G=(V, E)` where V=bricks and E=stud/anti-stud connections.

```
.mpd → MPDParser → list[RawBrickData] → LegoCore.from_raw_bricks() → G=(V, E)
```

- **ConnParser** (`src/geometry/conn_parser.py`) parses LDraw `.dat` files recursively to extract stud (male) and anti-stud (female) positions. Uses `_get_bottom_y()` to determine actual part height from geometry vertices (not from stud4.dat reference position).
- **Port** (`src/geometry/port.py`) — frozen dataclass representing a connection point (position, normal, male/female type).
- **LegoPart** / **PartDatabase** (`src/geometry/lego_part.py`) — combines ports + collision boxes per part, with lazy-loading cache.
- **snap.py** (`src/geometry/snap.py`) — `check_snap()` matches male/female ports between two bricks (KDTree, pos_tol=2.0 LDU, normal anti-alignment). `find_all_connections()` pre-filters with brick-level KDTree (proximity_threshold=80 LDU).
- **SpatialHash** (`src/geometry/spatial_hash.py`) — voxel grid (8 LDU cells) for O(1) collision detection.
- **LegoCore** (`src/geometry/lego_core.py`) — main engine: `place_brick()`, `remove_brick()`, `validate_placement()`, `from_raw_bricks()`, `get_graph()`.
- **Visualizer** (`src/visualize_graph.py`) — 2D (spring + spatial top-down) and 3D graph plots, colored by node degree.

LDraw `.dat` files are read from `C:/Users/Public/Documents/LDraw/parts/` with primitives in `../p/`.

### Key Directories

- `dataset/mpd_files/` — raw LDraw .mpd source files
- `tokenized_sets/` — processed .npy tensor files
- `checkpoints/` — saved model checkpoints
- `generated_sets/` — generated .mpd outputs
- `atlas_config.json` — vocabulary state (parts, colors, offsets); rebuilt by `src/main.py`

## Known Issues / TODOs

- No train/validation split or early stopping yet
- Position ranges and precision are hardcoded constants in `src/config.py`
- `Config.OFFSETS["parts"]` in `src/config.py` is hardcoded at 3128 but the real value is dynamic (depends on number of colors in vocabulary). Always use `vocab_manager.get_offsets()` at runtime, not the static constant.

## Workflow

- **Always update `TODO.md`** after any codebase modification: mark completed items as done with the date, add new TODOs discovered during implementation, and remove obsolete entries.

## Past Mistakes — Do Not Repeat

- **Token offset mismatch**: The `parts` offset in `atlas_config.json` is dynamic (shifts when colors are added). The `.npy` files are tokenized with a specific offset. If the vocabulary changes after tokenization, all `.npy` files become invalid and must be regenerated. Always build the complete vocabulary (all colors, then all parts) BEFORE tokenizing any file. This is why `DatasetBuilder.process_dataset()` uses a 2-pass pipeline.
- **Logit masking must be enabled during training** (`mask_logits=True`): Without it, the model distributes probability across the entire vocabulary (~3500+ tokens) instead of just the valid tokens for each field. At generation time, the mask then collapses this broad distribution to a handful of tokens, resulting in always generating the same output regardless of temperature. Training with masking focuses learning on valid tokens per field.
- **VocabularyManager.save() must be called explicitly**: `add_part()`/`add_color()` only modify in-memory state. If you need the JSON on disk (e.g., before `tokenizer.load_vocabulary()`), call `vocab_manager.save()` first.

when a task is complete, check it in the todo.md, and if it wasnt in the todo list, add it and check it

---

Voici le rapport complet de simplification du projet ATLAS, agrégé à partir des 3 analyses.
                                                                                                     ---
  Rapport de Simplification — Projet ATLAS
  Priorité Haute                                                                                                                                                                                        #: 1                                                                                               Catégorie: Performance                                                                             Problème: Pas de KV-cache en génération — le modèle re-calcule l'attention sur toute la séquence   à                                                                                                     chaque token (O(n²) au lieu de O(n))                                                            Fichiers: src/model/generate.py:49                                                                 ────────────────────────────────────────                                                           #: 2
  Catégorie: Performance
  Problème: Le masque causal est recalculé à chaque forward() au lieu d'être pré-calculé comme
    buffer
  Fichiers: src/model/transformer.py:96
  ────────────────────────────────────────
  #: 3
  Catégorie: Performance
  Problème: SequenceDataset.__getitem__ fait un np.load() depuis le disque à chaque accès — pas de
    cache mémoire
  Fichiers: src/data/sequence_dataset.py:47
  ────────────────────────────────────────
  #: 4
  Catégorie: Duplication
  Problème: 3 implémentations de "trouver la rotation la plus proche" — rotations.py, vocabulary.py

    et tokenizer.py
  Fichiers: src/maths/rotations.py, src/core/vocabulary.py:254, src/core/tokenizer.py:125
  ────────────────────────────────────────
  #: 5
  Catégorie: Abstraction
  Problème: DatasetBuilder accède au membre privé parser._submodels
  Fichiers: src/data/builder.py:196,200
  ────────────────────────────────────────
  #: 6
  Catégorie: Abstraction
  Problème: train_model.py modifie directement dataset.npy_files (encapsulation cassée)
  Fichiers: train_model.py:52
  ────────────────────────────────────────
  #: 7
  Catégorie: Bug
  Problème: mpd_writer.py a un sys.path.insert qui s'exécute inconditionnellement + un bloc
  __main__
    cassé (appelle des méthodes inexistantes)
  Fichiers: src/file_io/mpd_writer.py:14,77

  Priorité Moyenne

  ┌─────┬─────────────┬───────────────────────────────────────┬────────────────────────────────┐
  │  #  │  Catégorie  │               Problème                │            Fichiers            │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 8   │ Duplication │ download_file() dupliqué entre        │ src/file_io/scraper.py:24,     │
  │     │             │ scraper.py et omr_scraper.py          │ src/file_io/omr_scraper.py:97  │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ sort_bricks_by_position() dupliqué    │ src/data/parser.py:157,        │
  │ 9   │ Duplication │ entre MPDParser et DatasetBuilder     │ src/data/builder.py:82         │
  │     │             │ (même clé de tri)                     │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ Extraction de numéro de set par regex │ src/data/builder.py:69,        │
  │ 10  │ Duplication │  dupliquée (_extract_set_number vs    │ src/file_io/omr_scraper.py:35  │
  │     │             │ get_existing_set_numbers)             │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ SOS=1, EOS=2, PAD=0, UNK=3 utilisés   │ sequence_dataset.py:62,        │
  │ 11  │ Magic       │ en dur partout au lieu de constantes  │ generate.py:36,                │
  │     │ numbers     │ centralisées                          │ transformer.py:57,             │
  │     │             │                                       │ vocabulary.py:240              │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ _recalculate_offsets() appelé à       │                                │
  │ 12  │ Performance │ chaque add_part()/add_color()         │ src/core/vocabulary.py:164     │
  │     │             │ individuel — N recalculs inutiles en  │                                │
  │     │             │ boucle                                │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ line_to_vector() re-parse une ligne   │                                │
  │ 13  │ Performance │ déjà parsée par flatten() — double    │ src/data/parser.py:118         │
  │     │             │ split + numpy inutile                 │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 14  │ Performance │ Double normalisation des positions    │ src/data/adjacency.py:83-85    │
  │     │             │ dans sort_bricks_by_adjacency         │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ generate_chiral_rotation_matrices()   │                                │
  │ 15  │ Performance │ régénère les 24 matrices à chaque     │ src/core/vocabulary.py:267     │
  │     │             │ appel au lieu de les cacher           │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ Double dict.get() dans                │                                │
  │ 16  │ Performance │ batch_brick_ids_to_tokens /           │ src/core/tokenizer.py:199      │
  │     │             │ batch_color_ids_to_tokens             │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 17  │ Dead code   │ _to_tensor() est un no-op mais        │ src/data/builder.py:362        │
  │     │             │ toujours appelé                       │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ collect_brick_ids(),                  │                                │
  │ 18  │ Dead code   │ collect_brick_colors(),               │ src/data/builder.py:255-308    │
  │     │             │ _update_vocabulary() jamais appelés   │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ generate_model.py recharge le         │                                │
  │ 19  │ Dead code   │ checkpoint manuellement au lieu       │ generate_model.py:78-84        │
  │     │             │ d'utiliser Trainer.load_checkpoint()  │                                │
  └─────┴─────────────┴───────────────────────────────────────┴────────────────────────────────┘

  Priorité Basse

  ┌─────┬─────────────┬───────────────────────────────────────┬────────────────────────────────┐
  │  #  │  Catégorie  │               Problème                │            Fichiers            │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 20  │ Performance │ optimizer.zero_grad() sans            │ src/model/train.py:136         │
  │     │             │ set_to_none=True                      │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 21  │ Performance │ Smoothing O(n²) pour les courbes de   │ src/model/train.py:230         │
  │     │             │ loss                                  │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ self.lines stocke toutes les lignes   │                                │
  │ 22  │ Performance │ du fichier MPD mais n'est jamais      │ src/data/parser.py:38          │
  │     │             │ utilisé                               │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 23  │ Qualité     │ Commentaires step-by-step en français │ src/data/builder.py:224-251    │
  │     │             │  dans center_around_origin()          │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 24  │ Dead code   │ Variable module-level mpd_file_path   │ src/data/parser.py:16,144      │
  │     │             │ et code commenté dans parser.py       │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │ 25  │ Dead code   │ Branche if submodel_lines is None     │ src/data/parser.py:83          │
  │     │             │ inatteignable (.get() retourne [])    │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ ModelConfig mélange config            │                                │
  │ 26  │ Design      │ architecture, training et génération  │ src/model/config.py            │
  │     │             │ dans un seul dataclass plat           │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ batch_brick_ids_to_tokens et          │                                │
  │ 27  │ Duplication │ batch_color_ids_to_tokens             │ src/core/tokenizer.py:185-221  │
  │     │             │ quasi-identiques — factorisable       │                                │
  ├─────┼─────────────┼───────────────────────────────────────┼────────────────────────────────┤
  │     │             │ Pattern device = "cuda" if            │ train_model.py:16,             │
  │ 28  │ Duplication │ torch.cuda.is_available() else "cpu"  │ generate_model.py:69           │
  │     │             │ dupliqué                              │                                │
  └─────┴─────────────┴───────────────────────────────────────┴────────────────────────────────┘

  ---
  Top 3 des gains estimés

  1. KV-cache en génération (#1) — 10-50x plus rapide pour la génération
  2. Préchargement dataset en RAM (#3) — 2-10x plus rapide par epoch
  3. Cache du masque causal (#2) — 5-15% sur le training