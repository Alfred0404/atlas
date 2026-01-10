# ATLAS _(Autoregressive Transformer Lego Assembly Synthesis)_

This project aim to generate coherent and mecanically plausible Lego sets using an autoregressive transformer architecture, predicting the next brick based on the previous ones.

## Data representation

A set is a group of bricks, which can be represented as tokens

### Brick state vector

Each brick $i$ is a vector :

$$X_i = [ID_{part}, x, y, z, q_w, q_x, q_y, q_z, color]$$

| Component          | Nature     | Processing                                              |
| :----------------- | :--------- | :------------------------------------------------------ |
| **ID_part**        | Discrete   | Mapped to unique index in vocabulary (4 to N_bricks+3)  |
| **x, y, z**        | Continuous | Normalized coordinates $[-1, 1]$                        |
| **qw, qx, qy, qz** | Continuous | **Unit quaternions** (with constraint $q_w \ge 0$)      |
| **color**          | Discrete   | Mapped to unique index in vocabulary (after all bricks) |

**Global Structure** : $(N_{max}, 9)$ Vector, where line order has no importance _(invariance by permutation)_

## Model Architecture

The architecture is based on a backbone able to capture local and global geometric relations.

![LORD_Architecture](/public/LORD_Architecture.png)

### Key Components

**MPDParser**

- parse a given `.mpd` file to get a flat list of bricks variables _(RawDataBricks named tuple)_

**DatasetBuilder**

- use MPDParser on every `.mpd` file of the dataset _(from a given directory)_ and apply some basic transformations, such as rotation matrix to quaternions conversion, and centering around origin
- create a unified token vocabulary with:
  - **Special tokens** at indices 0-3: `[SOS]`, `[EOS]`, `[PAD]`, `[UNK]`
  - **All brick IDs** starting at index 4 (sorted alphabetically)
  - **All colors** following the bricks (prefixed with `color_`, sorted numerically)
- save the vocabulary to `vocab.json` as a single flat dictionary with unique indices for all entries

### Parsing

Lego sets are stored in `.mpd` files, as recursive scene graphs. A set is described as multiple subsets, also described as sub-subsets, and so on to simple bricks.

So in order to get clean data from a `.mpd` file, we need to flatten it to get the position and orientation of each brick composing the set.

1. **Hierarchy Flattening:**
   Recursive resolution. Sub-sets are integrated by applying parent transformation matrices to obtain absolute world coordinates.
2. **Quaternion Conversion:**
   The $3 \times 3$ rotation matrices from the source file _(type 1 lines)_ are converted to quaternions to avoid numerical instabilities and to reduce vector dimensionality.
3. **Centering:**
   Point cloud centering around the origin (barycenter) to stabilize training.
4. **Vocabulary Building:**
   - Collect all unique brick IDs and colors from all `.mpd` files in the dataset
   - Build a unified vocabulary with unique indices:
     - Indices 0-3: Special tokens (`[SOS]`, `[EOS]`, `[PAD]`, `[UNK]`)
     - Indices 4+: All brick IDs (sorted alphabetically)
     - Following indices: All colors prefixed with `color_` (sorted numerically)
   - Save to `vocab.json` as a single flat dictionary

The goal of this part is to go from a line like this:

`1 71 0 0 0 1 0 0 0 1 0 0 0 1 3028.dat`

To a vector in this form:

`[brick_idx, x, y, z, q_w, q_x, q_y, q_z, color_idx]`

Where `brick_idx` is the vocabulary index for brick ID `3028`, and `color_idx` is the vocabulary index for `color_71`.

## Research & Learning Directions

### 1. Algorithmic Geometry & LDraw Standard

- **Recursive Transformation Computation:**
  Master the accumulation of transformation matrices (World Matrices) to "flatten" `.mpd` files that nest sub-models (`FILE ... .ldr`).
- **Quaternion Algebra:**
  Study the conversion of 3×3 rotation matrices from source files to unit quaternions and handling "double coverage" (enforce qw ≥ 0).
- **LDraw Units (LDU):**
  Understand the specific coordinate system (Y pointing down) and discrete grid for future snapping post-processing.

### 2. Vocabulary & Token Representation

- **Unified Vocabulary Design:**
  Understanding how to build a flat, collision-free vocabulary that maps both discrete elements (brick IDs, colors) and continuous elements (positions, rotations) to unique indices.
- **Token Embeddings:**
  How to represent discrete tokens (bricks, colors) as learnable embeddings so the model can understand structural relationships between similar parts.
- **Special Tokens:**
  Usage of `[SOS]`, `[EOS]`, `[PAD]`, and `[UNK]` tokens for sequence modeling and handling unknown elements.

### 3. Deep Learning Architectures for Point Clouds

- **Point Transformers:**
  Study the attention mechanism applied to unordered point sets (permutation invariance).
- **Transformers:**
  Understanding autoregressive generation and how to predict sequences of bricks.

### 4. Loss Functions & Geometric Optimization

- **Rotation Loss:**
  Study geodesic distance on the quaternion sphere (different from simple MSE).
- **Repulsive Potential Loss:**
  Implement cost functions of type 1/d² between part centers to penalize massive collisions during training.
- **Chamfer Distance:**
  Metric for comparing two point clouds to evaluate reconstruction quality.

### 5. Data Engineering & LEGO Semantics

- **Unified Vocabulary Management:**
  Techniques for building and maintaining a single vocabulary that handles all discrete tokens (bricks and colors) with unique, non-overlapping indices.
- **Category Embeddings:**
  How to train or use dense vectors to represent parts so the model understands that a "Plate 1x2" is structurally close to a "Plate 1x4".
- **Top-K Filtering & Long Tail:**
  Vocabulary reduction techniques to keep only statistically significant parts and ensure model convergence.

# Sources

[Bricks list](https://library.ldraw.org/parts/list)
[LDraw Colors](https://www.ldraw.org/article/547.html)
[Quaternions and rotation matrix](https://www.johndcook.com/blog/2025/05/07/quaternions-and-rotation-matrices/)
[PointFlow (3D point cloud generation)](https://www.guandaoyang.com/PointFlow/)
[OpeneAI - Point-E](https://github.com/openai/point-e/tree/main)
[DDPM Video](https://www.youtube.com/watch?v=EhndHhIvWWw&list=WL&index=17)
[Quaternion explanations](https://www.reddit.com/r/gamedev/comments/ffo8gg/quaternions_basics_for_3d_rotation_pt_1/)
[Quaternions sandbox](https://eater.net/quaternions/)
[Scipy.as_quat](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
[Point cloud search on hf](https://huggingface.co/models?search=point%20cloud)
