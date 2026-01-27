# Todo

## High Priority

### In Progress

### Not Done

- [ ] 🔴 json i/o optimisation ([builder.py](src/data/builder.py), [vocabulary.py](src/core/vocabulary.py))
- [ ] 🔴 hardcoded magic numbers ([config.py](src/config.py))
- [ ] 🟡 Change the way of centering ([builder.py](src/data/builder.py))
- [ ] 🟡 add test coverage ([tests/](tests/))
- [ ] 🟢 tests files in prod environnment
- [ ] 🟢 Check how other point transformers handle positions

### Done

- [x] add rotation matrix similarity mapping to map rotation to ids ([tokenizer.py](src/core/tokenizer.py), 2026-01-27)

## Encoder

### In Progress

- [ ] "Flatten" again, to get all the bricks within one vector $[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]$, and treat a lego set as a sequence, where the transformer predicts the next token based on all the previous ones ([builder.py](src/data/builder.py))

### Not Done

### Done

- [x] Replace true values by their corresponding idx in the vocab (tokenize the dataset) (2026-01-20)

## Decoder

### In Progress

### Not Done

- [ ] 🟡 Group tokens by blocs of 6 (group them by bricks)
- [ ] 🟡 De-binning (from bin idx to real position) ([tokenizer.py](src/core/tokenizer.py))
- [ ] 🟡 Snapping positions to the real grid
- [ ] 🟡 Reconstruct an output `.mpd` file, ready to be displayed in LDView ([mpd_writer.py](src/file_io/mpd_writer.py))

### Done

## Project Structure

### In Progress

### Not Done

### Done

- [x] Reorganize the src folder (2026-01-15)
- [x] Add rotations to `atlas_config.json` (2026-01-20)
- [x] add error handling (2026-01-15)
- [x] verify docstring (2026-01-15)
- [x] duplicate code for logging setup (2026-01-15)
- [x] missing `__init__.py` files (2026-01-15)

## Data Preprocessing

### In Progress

### Not Done

### Done

- [x] Map x, y and z positions to bins idx (2 LDU is a pretty good compromise) (2026-01-18)
- [x] remove quaternion conversion (2026-01-22)
- [x] Add conversion from rotation matrix to quaternions (with $q_w \geq 0$) (2026-01-18)
- [x] First put all sets around $(0, 0, 0)$ ([builder.py](src/data/builder.py), 2026-01-18)
- [x] Split responsibility between DatasetBuilder and MPDParser (2026-01-16)
- [x] Optimize centering around origin (2026-01-18)
- [x] Finish the MPDParser and the DatasetBuilder (2026-01-16)

## Vocabulary Management

### In Progress

### Not Done

### Done

- [x] Refactor the vocabulary creation and vocab.json, to not store every position (they can be computed when needed) ([vocabulary.py](src/core/vocabulary.py), 2026-01-20)
- [x] Create special tokens ([SOS], [EOS], [PAD], etc.) and map them to idx ([atlas_config.json](atlas_config.json), 2026-01-20)
- [x] Map all known brick ids to vocab idx (2026-01-20)
- [x] Map all known colors ids to vocab idx (2026-01-20)
- [x] Maps idx to the closest most frequent rotations (0, 90, 180, etc. on all 3 axes xyz) (2026-01-20)

## Model Architecture

### In Progress

### Not Done

### Done

- [x] Start to think about the actual model architecture (2026-01-12)
- [x] Sort all the bricks in a deterministic way (so the model learn 'syntax') (2026-01-18)

### Research & References

### Transformer Architecture

Predict the next brick based on all the previous ones

#### Related Projects

- [Point GPT](https://github.com/CGuangyan-BIT/PointGPT)
- [Set Transformer](https://arxiv.org/pdf/1810.00825)
- [Lego GPT](https://avalovelace1.github.io/BrickGPT/)

## Reference Data

### Vocabulary Structure

The vocab is: tokens first (0->3), then all the known brick_ids, then all y bins (512), then all the x bins (512), then all the z bins (512), then all the rotation idx (24), then all the known colors

### 24 Rotation Quaternions (w x y z)

#### Identity

- `1 0 0 0` (no rotation)

#### 180° Rotations

- `0 1 0 0` (180° on all axes)
- `0 0 1 0`
- `0 0 0 1`

#### 90° & 270° Rotations (single axis)

- `0.7 0.7 0 0` (90° & 270° around x)
- `0.7 -0.7 0 0`
- `0.7 0 0.7 0` (90° & 270° around z)
- `0.7 0 -0.7 0`
- `0.7 0 0 0.7` (90° & 270° around y)
- `0.7 0 0 -0.7`

#### Combined Rotations

- `0 0.7 0.7 0`
- `0 0.7 -0.7 0`
- `0 0.7 0 0.7`
- `0 0.7 0 -0.7`
- `0 0 0.7 0.7`
- `0 0 0.7 -0.7`

#### Triple Axis Rotations

- `0.5 0.5 0.5 0.5`
- `0.5 0.5 0.5 -0.5`
- `0.5 0.5 -0.5 0.5`
- `0.5 0.5 -0.5 -0.5`
- `0.5 -0.5 0.5 0.5`
- `0.5 -0.5 0.5 -0.5`
- `0.5 -0.5 -0.5 0.5`
- `0.5 -0.5 -0.5 -0.5`

---

## Notes (Archived)

<!-- Normalization approach (not currently used):
- [] add normalisation to the bricks positions
$$Position_{centered} = Position - Barycenter_{set}$$
$$Position_{norm} = \frac{Position_{centered}}{max(distance_{all\_ sets})}$$
-->
