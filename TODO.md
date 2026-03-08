# Todo

## High Priority

- [ ] update the readme by removing all the unnecessary methods listing
- [ ] implement logit masking during inference. since we know that a brick vector is always the same, we can check the next token position in the array, and mask all the other tokens so the softmax is only applied on the right token range
- [ ] scrap more files https://www.eurobricks.com/forum/forums/topic/48285-key-topic-official-lego-sets-made-in-ldraw/#comment-849693, https://library.ldraw.org/omr/sets
- [ ] 🔴 json i/o optimisation ([builder.py](src/data/builder.py), [vocabulary.py](src/core/vocabulary.py))
- [ ] 🔴 hardcoded magic numbers ([config.py](src/config.py))
- [ ] 🟢 Check how other point transformers handle positions

<details>
    <summary style="font-style:bold">Done</summary>
- [x]Add rotation matrix similarity mapping to map rotation to ids ([tokenizer.py](src/core/tokenizer.py), 2026-01-27)
- [x] 🟡 Change the way of centering ([builder.py](src/data/builder.py))
- [x] 🟡 add test coverage ([tests/](tests/))
- [x] 🟢 tests files in prod environnment
</details>

## Encoder

- [x] "Flatten" again, to get all the bricks within one vector $[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]$, and treat a lego set as a sequence, where the transformer predicts the next token based on all the previous ones ([builder.py](src/data/builder.py))

<details>
    <summary style="font-style:bold">Done</summary>
- [x] Replace true values by their corresponding idx in the vocab (tokenize the dataset) (2026-01-20)
</details>

## Decoder

- [ ] 🟡 Group tokens by blocs of 6 (group them by bricks)
- [ ] 🟡 De-binning (from bin idx to real position) ([tokenizer.py](src/core/tokenizer.py))
- [ ] 🟡 Snapping positions to the real grid
- [ ] 🟡 Reconstruct an output `.mpd` file, ready to be displayed in LDView ([mpd_writer.py](src/file_io/mpd_writer.py))

<details>
    <summary style="font-style:bold">Done</summary>
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
