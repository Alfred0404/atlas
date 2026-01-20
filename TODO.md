# TODO

- [x] add convertion from rotation matrix to quaternions _(with $q_w \geq 0$)_
- [x] first put all sets around $(0, 0, 0)$
<!-- - [] add normalisation to the bricks positions
$$Position_{centered} = Position - Barycenter_{set}$$
$$Position_{norm} = \frac{Position_{centered}}{max(distance_{all\_ sets})}$$ -->

- [x] split responsibility between DatasetBuilder and MPDParser
- [x] optimize centering around origin
- [x] finish the MPDParser and the DatasetBuilder
- [x] start to think about the actual model architecture
- [] optimize flatten method
- [] reorganize the src folder
- [] refactor the vocabulary creation and vocab.json, to not store every position (they can be computed when needed)

## Transformer architecture (predict the next brick based on all the previous ones)

- [point gpt](https://github.com/CGuangyan-BIT/PointGPT)
- [Set Transformer](https://arxiv.org/pdf/1810.00825)
- [lego gpt](https://avalovelace1.github.io/BrickGPT/)

- [x] sort all the bricks in a deterministic way (so the model learn 'syntax')

- fix the vocab
  - [x] create special tokens ([SOS], [EOS], [PAD], etc.) and map them to idx
  - [x] map all known brick ids to vocab idx
  - [x] map all known colors ids to vocab idx
  - [x] maps idx to the closest most frequent rotations (0, 90, 180, etc. on all 3 axes xyz)

  - [] map x, y and z positions to bins idx (2 LDU is a pretty good compromise)
    - precision = 2 # in LDU
    - n_bins = farthest brick # 2 LDU
    - bins = [i for i in range(n_bin, precision)]

  So the vocab is : tokens first (0->3), then all the known brick_ids, then all y bins (512), then all the x bins (512), then all the z bins (512), then all the rotation idx (24), then all the known colors

## Tokenizer

- [] check how other point transformers handle positions

### Encoder

  - replace true values by their corresponding idx in the vocab *(tokenize the dataset)*
  - "flatten" again, to get all the bricks within one vector $[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]$, and treat a lego set as a sequence, where the transformer predicts the next token based on all the previous ones
  - [] for tokenization, compare the actual quaternion to the closest one in the list of all 24 quaternions

### Decoder

  - group tokens by blocs of 6 *(group them by bricks)*
  - de-binning (from bin idx to real position)
  - snapping positions to the real grid
  - reconstruct an output `.mpd` file, ready to be displayed in LDView

---

w x y z

1 0 0 0 (no rotation)

0 1 0 0 (180° on all axes)
0 0 1 0
0 0 0 1

0.7 0.7 0 0 (90 & 270 around x)
0.7 -0.7 0 0
0.7 0 0.7 0 (90 & 270 around z)
0.7 0 -0.7 0
0.7 0 0 0.7 (90 & 270 around y)
0.7 0 0 -0.7

0 0.7 0.7 0
0 0.7 -0.7 0
0 0.7 0 0.7
0 0.7 0 -0.7
0 0 0.7 0.7
0 0 0.7 -0.7

0.5 0.5 0.5 0.5
0.5 0.5 0.5 -0.5
0.5 0.5 -0.5 0.5
0.5 0.5 -0.5 -0.5
0.5 -0.5 0.5 0.5
0.5 -0.5 0.5 -0.5
0.5 -0.5 -0.5 0.5
0.5 -0.5 -0.5 -0.5