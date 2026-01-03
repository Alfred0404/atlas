# LORD *(Lego Oriented Recursive Diffusion)*

This project aim to generate coherent and mecanically plausible Lego sets with a conditionned diffusion model, by treating sets as semantic points clouds.

## Data representation

A set is a group of bricks, which can be represented as tokens

### Brick state vector

Each brick $i$ is a vector :

$$X_i = [ID_{part}, x, y, z, q_w, q_x, q_y, q_z, color]$$

| Component          | Nature      | Processing                                              |
| :----------------- | :---------- | :------------------------------------------------------ |
| **ID_part**        | Discrete    | **Pre-trained embedding** *(via Rebrickable catalog)*  |
| **x, y, z**        | Continuous  | Normalized coordinates $[-1, 1]$                        |
| **qw, qx, qy, qz** | Continuous  | **Unit quaternions** (with constraint $q_w \ge 0$) |
| **color**          | Discrete    | Color embedding                                         |

**Global Structure** : $(N_{max}, 9)$ Vector, where line order has no importance *(invariance by permutation)*


## Model Architecture

The architecture is based on a backbone able to capture local and global geometric relations.

![LORD_Architecture](/public/LORD_Architecture.png)

### Key Components

1. **3D Relative Positional Encoding (RPE)**
    Essential for injecting the notion of proximity. Attention is weighted by relative distance, simulating the probability of mechanical connectivity.
2. **Input Segmentor**
    Router separating IDs (for embeddings) from spatial coordinates (for linear projections).
3. **Mixed-Type Diffusion Head**
    Gaussian Diffusion for geometry and Discrete Diffusion (D3PM) for part types.


### Parsing

Lego sets are store in `.mpd` files, as recursive scene graphs. A set is describe as multiple subsets, also described as sub-subsets, and so on to simple bricks.

So in order to get clean data from a `.mpd` file, we need to flatten it to get the position and orientation of each bricks composing the set.

1. **Hierarchy Flattening:**
    Recursive resolution. Sub-sets are integrated by applying parent transformation matrices to obtain absolute world coordinates.
2. **Quaternion Conversion:**
    The $3 \times 3$ rotation matrices from the source file *(type 1 lines)* are converted to quaternions to avoid numerical instabilities and to big vectors.
3. **Filtering:**
    Limitation to **Top-K** *(e.g., 500)* most frequent parts to ensure classification convergence (`part_id`).
4. **Normalization:**
    Point cloud centering to stabilize training.

The goal of this part is to go from a line like this :

`1 71 0 0 0 1 0 0 0 1 0 0 0 1 3028.dat`

To a vector in this form

`[3028, 0, 0, 0, q_w, q_x, q_y, q_z, 71]`


## Research & Learning Directions

### 1. Algorithmic Geometry & LDraw Standard

- **Recursive Transformation Computation:**
    Master the accumulation of transformation matrices (World Matrices) to "flatten" `.mpd` files that nest sub-models (`FILE ... .ldr`).
- **Quaternion Algebra:**
    Study the conversion of 3×3 rotation matrices from source files to unit quaternions and handling "double coverage" (enforce qw ≥ 0).
- **LDraw Units (LDU):**
    Understand the specific coordinate system (Y pointing down) and discrete grid for future snapping post-processing.

### 2. Deep Learning Architectures for Point Clouds

- **Point Transformers:**
    Study the attention mechanism applied to unordered point sets (permutation invariance).
- **Relative Positional Encoding (RPE):**
    Learn to inject relative distance between parts into attention layers so the model "feels" physical proximity.
- **Graph Neural Networks (GNN):**
    Research dynamic graph construction (k-NN) to model immediate neighborhood relationships between bricks.

### 3. Generative Models: Hybrid Diffusion

- **Gaussian Diffusion (DDPM/DDIM):**
    Standard for generating continuous variables in the vector *(positions x, y, z and quaternions)*.
- **Discrete Diffusion (D3PM):**
    Research on multinomial diffusion models to handle categorical variables like part IDs (e.g., `3028.dat`, `3002.dat`) and colors present in the dataset.
- **Classifier-Free Guidance:**
    Technique to condition generation by a prompt or category *(e.g., "Star Wars")*.

### 4. Loss Functions & Geometric Optimization

- **Rotation Loss:**
    Study geodesic distance on the quaternion sphere (different from simple MSE).
- **Repulsive Potential Loss:**
    Implement cost functions of type 1/d² between part centers to penalize massive collisions during training.
- **Chamfer Distance:**
    Metric for comparing two point clouds to evaluate reconstruction quality.

### 5. Data Engineering & LEGO Semantics

- **Category Embeddings:**
    How to train or use dense vectors to represent parts so the model understands that a "Plate 1x2" is structurally close to a "Plate 1x4".
- **Top-K Filtering & Long Tail:**
    Vocabulary reduction techniques to keep only statistically significant parts and ensure model convergence.





# Sources

https://www.johndcook.com/blog/2025/05/07/quaternions-and-rotation-matrices/
https://www.guandaoyang.com/PointFlow/
https://arxiv.org/pdf/1906.12320
https://github.com/openai/point-e/tree/main
https://www.youtube.com/watch?v=EhndHhIvWWw&list=WL&index=17
https://www.reddit.com/r/gamedev/comments/ffo8gg/quaternions_basics_for_3d_rotation_pt_1/
https://eater.net/quaternions/
https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.spatial.transform.Rotation.as_quat.html