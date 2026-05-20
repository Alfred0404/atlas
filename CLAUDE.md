# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ATLAS (Autoregressive Transformer Lego Assembly Synthesis) — now exploring a **diffusion architecture** (branch `test-diffusion`). Each LEGO set is represented as a fixed-size matrix `(N, 6)` of bricks `[x, y, z, rot_id, brick_id, color_id]`. A DiT (Diffusion Transformer) learns to denoise random positions back into coherent LEGO assemblies.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# GPU support (optional, CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Step 1 — Build per-theme dataset (parse MPD → (N,6) tensors + vocab)
python diffusion_build_dataset.py --theme City

# Step 2 — Train the diffusion model
python diffusion_train_model.py --theme City

# Step 3 — Generate new LEGO sets
python diffusion_generate_model.py --theme City --checkpoint checkpoints/diffusion/City/latest.pt

# Organize MPD files by theme (Rebrickable API)
python src/group_by_theme.py --merge

# Run all tests
pytest
```

## Architecture

### Data Pipeline

```
dataset/mpd_files/<theme>/*.mpd → MPDParser → (N, 6) tensors → dataset/diffusion_sets/<theme>/*.pt
```

1. **MPDParser** (`src/data/parser.py`) reads LDraw .mpd files, flattens hierarchical submodels into world-space brick placements, sorts Y↑→X→Z
2. **diffusion_build_dataset.py** builds per-theme vocabulary (part/color → int, starting at 1; 0 = UNK/PAD), computes N (95th percentile brick count), normalizes positions (center + divide by global_scale), pads sets to length N

### Representation

Each brick: `[x, y, z, rot_id, brick_id, color_id]`
- `x, y, z`: continuous float32, normalized (divided by `global_scale` from vocab)
- `rot_id`: 0–23, index into the 24 octahedral rotation matrices
- `brick_id`, `color_id`: 1-indexed integers (0 = UNK/PAD, ignored in loss)

Padding: sets shorter than N are zero-padded; `padding_mask` (bool tensor) marks padded bricks.

### Model

**DiT** (`src/model/dit.py`): conditional position-denoiser. The bag of bricks (part/color/rot ids) is given as input; the model only learns the spatial arrangement.
- Position embedding: MLP(3 → d_model)
- Bag embeddings: nn.Embedding for part_id, color_id, rot_id (summed with pos_embed)
- Timestep embedding: sinusoidal → MLP(d → d)
- K transformer blocks (pre-norm self-attention with padding mask, GELU FFN)
- 1 output head: noise_pred (3)

**DDPM** (`src/model/ddpm.py`): cosine noise schedule, T=500 steps.
- Forward: `q_sample(x0, t)` → `(x_t, noise)`
- Reverse: `p_sample(model, x_t, t, cond)` → `x_{t-1}`, `sample(..., cond)` → positions

**Training** (`src/model/diffusion_train.py`): AdamW (lr=3e-4) + linear warmup + cosine annealing.
- Single MSE loss on predicted noise, non-padded bricks only
- Grad clipping at 1.0

**Generation** (`src/model/diffusion_generate.py`): bag (parts/colors/rots) supplied as conditioning, full reverse diffusion → snap to LEGO grid (20 LDU x/z, 8 LDU y) → write MPD. The bag comes from a random training set or `--template <set.pt>`.

### Key Directories

- `dataset/mpd_files/<theme>/` — raw LDraw .mpd source files, organized by theme
- `dataset/diffusion_sets/<theme>/` — per-set .pt tensors + `vocab_<theme>.pt`
- `checkpoints/diffusion/<theme>/` — model checkpoints
- `generated_sets/` — generated .mpd outputs

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
- **Theme-constrained dataset beats a large mixed dataset**: Training on all LEGO themes simultaneously forces the model to fit several near-independent distributions at once (Star Wars, LOTR, City, classic sets share almost no parts, colors, or structural patterns). This inflates entropy on the `part_id` head artificially — the model has to predict across 500 parts when a single theme only uses ~100 consistently. Filtering to one coherent theme (e.g. City) reduces the effective `part_id` random baseline from log(500)≈6.2 nats to ~log(100)≈4.6 nats before the model learns anything. The loss in raw example count is not a problem: the 8x geometric augmentation (rotations + mirrors) still applies, and coherent examples are worth far more than diverse-but-unrelated ones. A model that generates plausible City sets is a much better milestone than one that weakly fits everything.

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