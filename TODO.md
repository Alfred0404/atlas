# TODO

- [x] add convertion from rotation matrix to quaternions *(with $q_w \geq 0$)*
- [x] first put all sets around $(0, 0, 0)$
- [] add normalisation to the bricks positions
$$Position_{centered} = Position - Barycenter_{set}$$
$$Position_{norm} = \frac{Position_{centered}}{max(distance_{all\_ sets})}$$

- maybe pass the wanted scale during inference to help the model *(later)*
- create `metadata.json` to store:
    - [x] the scale so the model is able to multiply the normalised generated positions to their actual full size position
    - the brick mapping, the "vocabulary" of the model, with top-k bricks (`index_to_id` and `id_to_index`)
    - color mapping


- [x] split responsibility between dataset builder and mpd parser
- [x] optimize centering around origin
- [x] finish the parser and the dataset builder
- [] optimize flatten method

- [x] start to think about the actual model architecture

- transformer architecture (predict the next brick based on all the previous ones)
    - [point gpt](https://github.com/CGuangyan-BIT/PointGPT)
    - [Set Transformer](https://arxiv.org/pdf/1810.00825)

    - [x] sort all the bricks in a deterministic way (so the model learn 'syntax')
    - fix the vocab
        - create special tokens ([SOS], [EOS], [PAD], etc.) and map them to idx
        - maps idx to the most frequent rotations (0, 90, 180, etc. on all 3 axes xyz)
        - map all known brick ids to vocab idx
        - map all known colors ids to vocab idx
        - map x, y and z positions to bins idx

- Tokenizer
    - Encoder
        - replace true values by their corresponding idx in the vocab
        - "flatten" again, to get all the bricks within one vector $[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]$
    - Decoder
        - group tokens by blocs of 6 (group them by bricks)
        - de-binning (from bin idx to real position)
        - snapping positions to the real grid
        - reconstruct an output `.mpd` file, ready to be displayed in LDView