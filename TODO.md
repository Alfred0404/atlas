# Todo

## High Priority

- [ ] Switch back to position sorting, because proximity sorting with bfs does not reflect well the relation bettween bricks, and the model always generate the same brick positions.
- [x] Blacklist Technic/Bionicle/Hero Factory sets from dataset — `dataset/technic_blacklist.txt` + filtering in `builder.py` (2026-03-24)
- [ ] Sort/classify sets by theme using Rebrickable API (map set numbers to themes, enable theme-filtered or theme-conditioned training)
- [x] 🔴 hardcoded magic numbers — offsets now computed from constants ([config.py](src/config.py), 2026-03-23)
- [ ] Larger training dataset + train/val split with early stopping
- [x] Data augmentation (rotation globale 90°/180°/270°, mirroring X, permutation d'ordre BFS) — `src/data/augmentation.py`, 8x multiplicateur géométrique (2026-03-24)
- [ ] Dédupliquer le dataset (ex: `10129 UCS Snowspeeder.npy` et `10129 - Ultimate Collector's Rebel Snowspeeder.npy`)
- [ ] 🟢 Check how other point transformers handle positions

<details>
    <summary style="font-style:bold">Done</summary>

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
