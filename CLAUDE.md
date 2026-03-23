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
python train_model.py

# Generate a new LEGO set
python generate_model.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_vocabulary.py

# Run a single test
pytest tests/test_vocabulary.py::TestVocabularyManager::test_add_part
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
- Causal attention mask; **field-based logit masking** at inference constrains outputs to valid token ranges per field position

**Trainer** (`src/model/train.py`): AdamW (lr=3e-4), linear warmup (500 steps) → cosine annealing, gradient clipping at 1.0.

**Generator** (`src/model/generate.py`): Temperature (0.8) + top-k (50) sampling with field-aware logit masking. Outputs .mpd files via `src/file_io/mpd_writer.py`.

### Key Directories

- `dataset/mpd_files/` — raw LDraw .mpd source files
- `tokenized_sets/` — processed .npy tensor files
- `checkpoints/` — saved model checkpoints
- `generated_sets/` — generated .mpd outputs
- `atlas_config.json` — vocabulary state (parts, colors, offsets); rebuilt by `src/main.py`

## Known Issues / TODOs

- Tokenizer and VocabularyManager re-read `atlas_config.json` on every brick; needs in-memory caching
- No train/validation split or early stopping yet
- Position ranges and precision are hardcoded constants in `src/config.py`
