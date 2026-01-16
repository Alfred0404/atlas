<br />
<div align="center">
  <a href="https://github.com/alfred0404/atlas">
    <img src="public/logo_atlas.png" alt="Logo" width="300">
  </a>

  <h3 align="center" style="font-weight: bold">ATLAS</h3>

  <p align="center" style="font-style: italic">
    Autoregressive Transformer Lego Assembly Synthesis
    <br />
  </p>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## Overview

This project aims to generate coherent and mechanically plausible LEGO sets using an autoregressive transformer architecture, predicting the next brick based on the previous ones.

ATLAS is a data processing pipeline designed to prepare LEGO set data for deep learning models. The project parses `.mpd` files (LEGO MPD/Part format), extracts brick information, and converts it into a structured representation suitable for transformer-based generative models.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/alfred0404/atlas.git
   cd atlas
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

You can verify the installation by running the parser on a test file:

```bash
python src/DatasetBuilder.py
```

## Data Representation

A set is a group of bricks, which can be represented as tokens.

### Brick State Vector

Each brick $i$ is represented as a 9-dimensional vector:

$$X_i = [ID_{brick}, color, x, y, z, q_w, q_x, q_y, q_z]$$

| Component          | Nature     | Processing                                                    |
| :----------------- | :--------- | :------------------------------------------------------------ |
| **ID_part**        | Discrete   | Mapped to unique index in vocabulary (after rotations)        |
| **x, y, z**        | Continuous | Centered coordinates relative to set barycenter               |
| **qw, qx, qy, qz** | Continuous | **Unit quaternions** (with constraint $q_w \ge 0$)            |
| **color**          | Discrete   | Mapped to unique index in vocabulary (prefixed with `color_`) |

## Project Structure

```
ATLAS/
├── src/
│   ├── config.py                           # Configuration settings
│   ├── MPDParser.py                        # MPD file parser
│   ├── DatasetBuilder.py                   # Dataset processing pipeline
│   ├── utils.py                            # Matrix manipulation utilities
│   ├── rotation_matrix_to_quaternion.py    # Rotation conversion utilities
│   ├── write_mpd_file.py                   # MPD file writer
│   └── formating/
│       └── customFormatter.py              # Custom logging formatter
├── mpd_files/
│   ├── dataset/                            # Input MPD files
│   └── generated/                          # Generated output files
├── processed_sets/                         # Processed numpy arrays
├── vocab.json                              # Unified vocabulary
└── README.md
```

## Key Components

### MPDParser

The `MPDParser` class handles parsing of `.mpd` files (LDraw format):

- **Recursive Flattening:** Resolves nested submodels by recursively applying transformation matrices to obtain absolute world coordinates for each brick
- **Scene Graph Resolution:** Navigates the hierarchical structure of LEGO models (sets → subsets → bricks)
- **Transformation Accumulation:** Accumulates 4×4 transformation matrices throughout the hierarchy
- **Output:** Returns a flat list of `RawBrickData` (brick ID, 4×4 world matrix, color)

**Key Features:**

- Handles nested `FILE` directives in MPD format
- Applies parent transformations to child elements
- Supports all LDraw line types (type 1 for parts/submodels)

### DatasetBuilder

The `DatasetBuilder` class processes multiple MPD files and builds a unified dataset:

1. **Parsing:** Uses `MPDParser` to extract brick data from each `.mpd` file
2. **Centering:** Centers each model around its barycenter for training stability
3. **Quaternion Conversion:** Converts rotation matrices to unit quaternions with $q_w \ge 0$ for consistency
4. **Brick Sorting:** Sorts bricks by position for deterministic ordering
5. **Vocabulary Building:** Creates a unified vocabulary across all models

**Data Pipeline:**

```
Raw MPD File → RawBrickData → BrickDataQuat → ProcessedBrickData → Tensor
```

### Vocabulary Structure

The unified vocabulary is stored in `vocab.json` with the following structure:

1. **Indices 0-3:** Special tokens

   - `[SOS]` (0): Start of sequence
   - `[EOS]` (1): End of sequence
   - `[PAD]` (2): Padding token
   - `[UNK]` (3): Unknown token

2. **Indices 4-27:** 24 chiral octahedral rotations

   - Pre-computed quaternions representing all valid LEGO brick rotations (90° increments)
   - Stored as `rotation_{w,x,y,z}` keys

3. **Indices 28+:** Brick IDs

   - All unique brick part IDs from the dataset (sorted alphabetically)
   - Example: `3001`, `3003`, `3004`, etc.

4. **Following indices:** Colors
   - All unique LEGO colors prefixed with `color_`
   - Example: `color_0`, `color_14`, `color_71`, etc.

### Transformation Pipeline

Starting from an LDraw line:

```
1 71 0 0 0 1 0 0 0 1 0 0 0 1 3028.dat
```

Where:

- `1`: Line type (part/submodel reference)
- `71`: Color code
- `0 0 0`: Position (x, y, z)
- `1 0 0 0 1 0 0 0 1`: 3×3 rotation matrix (identity in this case)
- `3028.dat`: Brick part file

**Processing Steps:**

1. **MPDParser** extracts:

   - Brick ID: `3028`
   - 4×4 World Matrix with position and rotation
   - Color: `71`

2. **DatasetBuilder** transforms:

   - Extracts position from world matrix
   - Extracts rotation matrix using SVD decomposition (handles scaling)
   - Converts rotation matrix to quaternion
   - Centers position relative to barycenter
   - Maps brick ID and color to vocabulary indices

3. **Final Output:**
   ```python
   [brick_idx, x, y, z, q_w, q_x, q_y, q_z, color_idx]
   ```

### Utility Functions

**utils.py:**

- `get_rotation_matrix_from_world_matrix()`: Extracts pure rotation using SVD decomposition
- `get_position_from_world_matrix()`: Extracts translation vector

**rotation_matrix_to_quaternion.py:**

- `rotation_matrix_to_quaternion()`: Converts 3×3 rotation to quaternion (enforces $q_w \ge 0$)
- `quat_to_rotation_matrix()`: Inverse conversion
- `generate_quat_chiral_rotations()`: Generates 24 valid LEGO brick rotations

**write_mpd_file.py:**

- `write_mpd_file()`: Exports processed data back to MPD format for visualization

## Usage

### Processing Dataset

```python
from src.DatasetBuilder import DatasetBuilder
from src.config import VOCAB_PATH

# Initialize and process all MPD files
builder = DatasetBuilder(VOCAB_PATH)
builder.process_dataset()
```

This will:

- Process all `.mpd` files in `mpd_files/dataset/`
- Build unified vocabulary
- Save vocabulary to `vocab.json`

### Parsing Single File

```python
from src.MPDParser import MPDParser
import numpy as np

parser = MPDParser("path/to/file.mpd")
first_model = list(parser._submodels.keys())[0]
flat_data = parser.flatten(first_model, np.eye(4))
```

### Writing MPD File

```python
from src.write_mpd_file import write_mpd_file

write_mpd_file(
    output_path="output.mpd",
    raw_data=flat_data,
    model_name="generated_model"
)
```

## Configuration

Edit `src/config.py` to configure paths:

```python
LOGGING_LEVEL = logging.DEBUG
RAW_DATASET_DIR = "./mpd_files/dataset"
PARSED_DATASET_DIR = "./processed_sets/"
VOCAB_PATH = "./vocab.json"
```

## Dataset

The project includes 50+ Star Wars LEGO sets in `.mpd` format, ranging from small polybags to Ultimate Collector's Series models:

- 10179 - Ultimate Collector's Millennium Falcon
- 10240 - UCS Red Five X-wing Starfighter
- Various minifig-scale ships and vehicles

## Future Directions

### Tokenization & Sequence Modeling

- Implement binning for continuous positions (512 bins per axis)
- Map rotations to nearest discrete rotation from 24 chiral options
- Flatten brick sequences: `[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]`
- Treat LEGO sets as sequences for autoregressive prediction

### Model Architecture

- Transformer architecture for autoregressive brick prediction
- Study Point Transformers for permutation invariance
- Explore Set Transformer architectures
- References: [PointGPT](https://github.com/CGuangyan-BIT/PointGPT), [Set Transformer](https://arxiv.org/pdf/1810.00825)

### Post-Processing

- Decoder to reconstruct from token sequences
- Position snapping to LDraw grid
- Output validation for physical plausibility

## Technical Notes

### LDraw Coordinate System

- Y-axis points downward (unlike standard 3D conventions)
- Units are in LDU (LDraw Units): 1 LDU ≈ 0.4mm
- Grid-based positioning for brick connections

### Quaternion Conventions

- Format: [w, x, y, z] (scalar-first)
- Normalized: $|q| = 1$
- Positive w convention: $q_w \ge 0$ (resolves double coverage)
- Negating quaternion represents same rotation: $q \equiv -q$

### Rotation Handling

- SVD decomposition ensures pure rotation extraction (no scaling/shearing)
- Determinant check prevents reflection matrices (det = +1 enforced)
- 24 chiral octahedral rotations cover all valid LEGO orientations

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
[Quaternions and spatial rotations](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)

<p align="center">
	<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/footers/gray0_ctp_on_line.svg?sanitize=true" />
</p>
