# Todo

## High Priority

- [ ] Finir les corrections du code simplifier (claude.md)
- [x] Sort/classify sets by theme using Rebrickable API — `src/group_by_theme.py` moves MPD files into `dataset/mpd_files/<theme>/`; `graph_build_dataset.py --theme` and `graph_train_model.py --theme` route the full pipeline per-theme (2026-05-04)
- [x] Migrate flat `dataset/graph_sets/*.npz` into per-theme subdirectories — `migrate_graph_sets.py` uses MPD dir structure as source of truth (2026-05-05)
- [ ] Dédupliquer le dataset (ex: `10129 UCS Snowspeeder.npy` et `10129 - Ultimate Collector's Rebel Snowspeeder.npy`)

[.conn files documentation](https://forums.ldraw.org/thread-28521-post-59984.html#pid59984)
[.col files documentation](https://forums.ldraw.org/thread-28518.html)

<details>
    <summary style="font-style:bold">Done</summary>

- [x] Larger training dataset + train/val split with early stopping (graph_build_dataset.py, 2026-03-30)
- [x] GraphTransformer pre-training fixes: auto-resume from latest.pt, intra-epoch checkpoint every 500 steps, early stopping (patience=10) (2026-05-04)
- [x] Analyser la distribution des positions dans le dataset tokenisé (confirmer le biais vers le centre)
- [x] 🔴 hardcoded magic numbers — offsets now computed from constants ([config.py](src/config.py), 2026-03-23)
- [x] Augmenter le poids du token EOS dans la loss (le modèle n'apprend pas à s'arrêter)
- [x] Fix EOS collapse (eos_weight 3.0→1.0) and embedding init (N(0,1)→std=0.02) to fix loss stuck at ~15 and model generating only 1 brick (2026-03-31)
- [x] Save checkpoint every 100 steps (2026-03-31)
- [x] Fix GraphTransformer loss stuck at 15: add ctx_norm (LayerNorm on g+soft_context) — initial loss 30.11→27.21, uncontrolled ctx magnitude caused all classifier heads to initialise worse than random (2026-03-31)
- [x] Add teacher-forced port context: pass target_port_idx to forward() during training so classification heads get exact local context instead of noisy soft-attention average — initial loss 27.21→25.36 (2026-03-31)
- [x] Rebuild graph dataset with top-500 parts vocabulary (currently 6288 parts floors total loss at +8.75 nats; top-500 covers 92.8% of bricks, reduces floor to 6.21) — graph_build_dataset.py --top-k-parts 500, 2026-05-04)
- [x] Pénalité de collision dans la loss (briques générées aux mêmes positions)
- [x] Data augmentation (rotation globale 90°/180°/270°, mirroring X, permutation seedée) — `src/data/augmentation.py`, 8x multiplicateur géométrique (2026-03-24)
- [x] Ajouter des docstrings à toutes les fonctions du projet (2026-03-26)
- [x] Switch back to position sorting (bottom-to-top deterministic Y->X->Z) and replace BFS-based permutations with seeded shuffle in dataset build pipeline (2026-03-24)
- [x] Blacklist Technic/Bionicle/Hero Factory sets from dataset — `dataset/technic_blacklist.txt` + filtering in `builder.py` (2026-03-24)
- [x] update the readme by removing all the unnecessary methods listing
- [x] implement logit masking during inference and training (2026-03-23)
- [x] scrap more files (OMR scraper implemented in [omr_scraper.py](src/file_io/omr_scraper.py))
- [x] Add rotation matrix similarity mapping to map rotation to ids ([tokenizer.py](src/core/tokenizer.py), 2026-01-27)
- [x] 🟡 Change the way of centering ([builder.py](src/data/builder.py))
- [x] 🟡 add test coverage ([tests/](tests/))
- [x] 🟢 tests files in prod environnment
- [x] 🔴 json i/o optimisation — pipeline 2 passes: vocab collecté puis figé avant tokenisation, tokenizer cache le vocab en mémoire, `add_part()`/`add_color()` en mémoire + `save()` explicite (2026-03-23)
</details>

## Encoder

<details>
    <summary style="font-style:bold">Done</summary>
- [x] "Flatten" again, to get all the bricks within one vector $[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]$, and treat a lego set as a sequence, where the transformer predicts the next token based on all the previous ones ([builder.py](src/data/builder.py))
- [x] Replace true values by their corresponding idx in the vocab (tokenize the dataset) (2026-01-20)
</details>

## Decoder

<details>
    <summary style="font-style:bold">Done</summary>
- [x] 🟡 Group tokens by blocs of 6 (group them by bricks) ([generate.py](src/model/generate.py))
- [x] 🟡 De-binning (from bin idx to real position) ([tokenizer.py](src/core/tokenizer.py))
- [x] 🟡 Reconstruct an output `.mpd` file, ready to be displayed in LDView ([generate_model.py](generate_model.py))
</details>

## Model

- [ ] Theme-conditioned generation (add theme token to sequences)
- [ ] Grid snapping for generated positions

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Decoder-only transformer with field embeddings ([transformer.py](src/model/transformer.py), 2026-03-22)
- [x] Training loop with AdamW + cosine LR ([train.py](src/model/train.py), 2026-03-22)
- [x] Generation pipeline with temperature + top-k sampling ([generate.py](src/model/generate.py), 2026-03-22)
- [x] MPD export from generated tokens ([generate_model.py](generate_model.py), 2026-03-22)
- [x] Test collision/connectivity via `.conn` with generated 2-brick stacked `.mpd` ([test_conn_collision_stack.py](test_conn_collision_stack.py), 2026-03-27)
</details>

## Web App

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Web app (Flask + Three.js) for interactive 3D assembly graph visualization — drag & drop .mpd, shows bricks at real 3D positions, connections, port spheres, info panel ([webapp/](webapp/), 2026-03-29)
- [x] Fix MPDParser.flatten() case-insensitive extension matching (.DAT/.LDR uppercase variants) (2026-03-29)
- [x] Expand ConnParser stud primitive lists (logo variants, fraction stud4 variants, missing studs); use actual detected anti-stud positions instead of inferring female ports from male Y (2026-03-29)
- [x] Fix SNOT/horizontal stud detection: derive port normals from transformation matrix Y-column instead of hardcoding [0,±1,0]; add horizontal female ports from detected antistud primitives (2026-03-30)
</details>

## Graph Transformer / LegoCore

- [ ] Implement SAT (Separating Axis Theorem) for precise OBB-OBB collision detection (generation only, not blocking)
- [x] Optimize graph dataset pass 2 preprocessing (worker-shared vocab, faster connection build, lighter action replay) (2026-03-30)
- [x] Improve graph training observability: startup/index progress logs, step loss logs, and latest checkpoint each epoch (2026-03-30)
- [x] Fix graph generation stopping at one brick: load real vocab mapping, fix CPU/CUDA batch-device mismatch, validate/fallback seed part, and lazily filter invalid sampled parts (2026-03-30)
- [x] snap_math.py — compute_snap_matrix + extract_rot_steps (2026-03-30)
- [x] graph_builder.py — MPD → labeled .npz preprocessing pipeline (2026-03-30)
- [x] graph_dataset.py — AssemblyStepDataset (PyTorch + PyG) (2026-03-30)
- [x] graph_config.py + graph_transformer.py — model (2026-03-30)
- [x] graph_train.py — training loop (2026-03-30)
- [x] graph_generate.py — generation loop (2026-03-30)
- See doc/graph_transformer_plan.md for full design

<details>
    <summary style="font-style:bold">Done</summary>
- [x] LegoCore geometric engine — Port, ConnParser, ColParser, LegoPart, PartDatabase, SpatialHash, snap checking, LegoCore with graph construction ([src/geometry/](src/geometry/), 2026-03-28)
- [x] Fix ConnParser to use actual part bottom Y from .dat geometry instead of stud4.dat reference position — bricks (height 24) now connect correctly, not just plates (height 8). 165-1.mpd: 55 → 214 connections (2026-03-28)
- [x] Switch from .conn binary files to LDraw .dat parsing for stud/anti-stud extraction — universal coverage for all parts with .dat files (2026-03-28)
- [x] Fix MPDParser filename normalization and recursive primitive expansion in ConnParser so `stug-*` group wrappers expand to their constituent studs; `3010` now resolves from 4 to 8 ports and regains graph connections (2026-05-20)
- [x] Confirm tiles like `3070b` legitimately have no studs/ports and therefore no graph connections (2026-05-20)
</details>

## Project Structure

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Reorganize the src folder (2026-01-15)
- [x] Add rotations to `atlas_config.json` (2026-01-20)
- [x] add error handling (2026-01-15)
- [x] verify docstring (2026-01-15)
- [x] duplicate code for logging setup (2026-01-15)
- [x] missing `__init__.py` files (2026-01-15)
</details>

## Data Preprocessing

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Map x, y and z positions to bins idx (2 LDU is a pretty good compromise) (2026-01-18)
- [x] remove quaternion conversion (2026-01-22)
- [x] Add conversion from rotation matrix to quaternions (with $q_w \geq 0$) (2026-01-18)
- [x] First put all sets around $(0, 0, 0)$ ([builder.py](src/data/builder.py), 2026-01-18)
- [x] Split responsibility between DatasetBuilder and MPDParser (2026-01-16)
- [x] Optimize centering around origin (2026-01-18)
- [x] Finish the MPDParser and the DatasetBuilder (2026-01-16)
</details>

## Vocabulary Management

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Refactor the vocabulary creation and vocab.json, to not store every position (they can be computed when needed) ([vocabulary.py](src/core/vocabulary.py), 2026-01-20)
- [x] Create special tokens ([SOS], [EOS], [PAD], etc.) and map them to idx ([atlas_config.json](atlas_config.json), 2026-01-20)
- [x] Map all known brick ids to vocab idx (2026-01-20)
- [x] Map all known colors ids to vocab idx (2026-01-20)
- [x] Maps idx to the closest most frequent rotations (0, 90, 180, etc. on all 3 axes xyz) (2026-01-20)
</details>

## Model Architecture

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Start to think about the actual model architecture (2026-01-12)
- [x] Sort all the bricks in a deterministic way (so the model learn 'syntax') (2026-01-18)
</details>

### Research & References

### Transformer Architecture

Predict the next brick based on all the previous ones

#### Related Projects

- [Point GPT](https://github.com/CGuangyan-BIT/PointGPT)
- [Set Transformer](https://arxiv.org/pdf/1810.00825)
- [Lego GPT](https://avalovelace1.github.io/BrickGPT/)
