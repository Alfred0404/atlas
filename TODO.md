# TODO

## High Priority

### Project Structure

- [ ] Reorganize the src folder
- [ ] Change the way of centering
- [x] Add rotations to `atlas_config.json`

### Data Processing

- [x] Map x, y and z positions to bins idx (2 LDU is a pretty good compromise)

## to refactor

- to_quaternion() method just calls another function unnecessarily (delete the method)
- i think we can bypass the quaternion conversion, because it add complexity for no reason
- add error handling
- json i/o optimisation
- hardcoded magic numbers
- add test coverage
- tests files in prod environnment

## Tokenizer Development

### Encoder

- [x] Replace true values by their corresponding idx in the vocab (tokenize the dataset)
- [ ] "Flatten" again, to get all the bricks within one vector $[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]$, and treat a lego set as a sequence, where the transformer predicts the next token based on all the previous ones
- [ ] For tokenization, compare the actual quaternion to the closest one in the list of all 24 quaternions

### Decoder

- [ ] Group tokens by blocs of 6 (group them by bricks)
- [ ] De-binning (from bin idx to real position)
- [ ] Snapping positions to the real grid
- [ ] Reconstruct an output `.mpd` file, ready to be displayed in LDView

### Position Handling

- [ ] Check how other point transformers handle positions

## Completed Tasks

### Data Preprocessing

- [x] Add conversion from rotation matrix to quaternions (with $q_w \geq 0$)
- [x] First put all sets around $(0, 0, 0)$
- [x] Split responsibility between DatasetBuilder and MPDParser
- [x] Optimize centering around origin
- [x] Finish the MPDParser and the DatasetBuilder

### Vocabulary Management

- [x] Refactor the vocabulary creation and vocab.json, to not store every position (they can be computed when needed)
- [x] Create special tokens ([SOS], [EOS], [PAD], etc.) and map them to idx
- [x] Map all known brick ids to vocab idx
- [x] Map all known colors ids to vocab idx
- [x] Maps idx to the closest most frequent rotations (0, 90, 180, etc. on all 3 axes xyz)

### Model Architecture

- [x] Start to think about the actual model architecture
- [x] Sort all the bricks in a deterministic way (so the model learn 'syntax')

## Research & References

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

## To refactor

- [x] verify docstring
- [x] duplicate code for logging setup
- [x] missing `__init__.py` files

## Notes (Archived)

<!-- Normalization approach (not currently used):
- [] add normalisation to the bricks positions
$$Position_{centered} = Position - Barycenter_{set}$$
$$Position_{norm} = \frac{Position_{centered}}{max(distance_{all\_ sets})}$$
-->
