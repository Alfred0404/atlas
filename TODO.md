# TODO

- [x] add convertion from rotation matrix to quaternions *(with $q_w \geq 0$)*
- [] add normalisation to the bricks positions
    - [x] first put all sets around $(0, 0, 0)$
    - then divide all positions by the farthest brick of the dataset
$$Position_{centered} = Position - Barycenter_{set}$$
$$Position_{norm} = \frac{Position_{centered}}{max(distance_{all\_ sets})}$$

- maybe pass the wanted scale during inference to help the model *(later)*
- create `metadata.json` to store:
    - [x] the scale so the model is able to multiply the normalised generated positions to their actual full size position
    - the brick mapping, the "vocabulary" of the model, with top-k bricks (`index_to_id` and `id_to_index`)
    - color mapping

- start to think about the actual model architecture