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

$$X_i = [ID_{brick}, color, x, y, z, a, b, c, d, e, f, g, h, i]$$

| Component          | Nature     | Processing                                                    |
| :----------------- | :--------- | :------------------------------------------------------------ |
| **ID_part**        | Discrete   | Mapped to unique index in vocabulary (after rotations)        |
| **x, y, z**        | Continuous | Centered coordinates relative to set barycenter               |
| **a -> i** | Continuous | rotation matrix            |
| **color**          | Discrete   | Mapped to unique index in vocabulary (prefixed with `color_`) |

## Project Structure

```
ATLAS/
├── src/
│   ├── config                .py           # Configuration class (constants)
│                ├── MPDParser.py           # MPD file parser
│   ├── DatasetBuilder.py                   # Dataset processing pipeline
│   ├── VocabularyManager.py                # Vocabulary management for atlas_config.json
│   ├──       utils.py                      # Matrix manipula           tion utilities
│   ├── rotation_matrix_to_quaternion.py    # Rotation conversion utilities
│   ├── write_mpd_file.py                   # MPD file                 writer
│   ├── scrap_mpd_files.py                  # mpd file scraper (only works on seymouria.pl website)
│   └── formating/
│                └── customFormatter.py     # Custom logging                formatter
├── mpd_files/
│   ├── dataset/                            # Input MPD files
│   │   ├── LDraw_sets/                     # Input MPD files
│   │   └── seymouria_ldraw_official_sets/  # Input MPD files
│   └── generated/                          # Generated output files
├──           tests/                        # Test suite
│   ├── test_vocabulary.py                  # Unit tests for VocabularyManager
│   └── test_integration.py                 # Integration tests for DatasetBuilder
├── public/                                 # all public resources (mostly images)
├── processed_sets/                         # Processed numpy arrays
├── atlas_config.json                       # Unified vocabulary and configuration
├── requirements.txt
├── .gitignore
├── TODO.md
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
2. **Centering:** Centers each model for training stability
4. **Brick Sorting:** Sorts bricks by position for deterministic ordering
5. **Vocabulary Management:** Uses `VocabularyManager` to incrementally update vocabulary as new parts and colors are encountered

**Data Pipeline:**

```
Raw MPD File → RawBrickData → ProcessedBrickData → Tensor
```

### VocabularyManager

The `VocabularyManager` class provides a clean interface for managing parts, colors, and rotations in `atlas_config.json`:

- **Separation of Concerns:** Dedicated class handling only vocabulary operations
- **On-the-fly Rotations:** 24 chiral rotations calculated dynamically, not stored in memory
- **Incremental Updates:** Parts and colors added as encountered during dataset processing
- **Type Safety:** All methods include type hints and comprehensive docstrings
- **Automatic Persistence:** Configuration saved automatically when vocabulary is updated

**Key Methods:**

- `add_part(part_id)` / `add_parts(part_ids)` - Add parts to vocabulary
- `add_color(color)` / `add_colors(color_ids)` - Add colors to vocabulary
- `get_part_index(part_id)` - Retrieve part index (returns UNK if not found)
- `get_color_index(color)` - Retrieve color index (returns UNK if not found)
- `get_rotation_index(quaternion)` - Calculate rotation index from quaternion
- `get_special_token_index(token)` - Get special token index (PAD, SOS, EOS, UNK)

### AtlasTokenizer

The `AtlasTokenizer` class handles the conversion between brick attributes and token IDs for the transformer model:

- **Tokenization (Encoding):** Converts brick attributes (position, rotation, color, part ID) into discrete token IDs
- **Detokenization (Decoding):** Reconstructs brick attributes from token IDs
- **Binning System:** Maps continuous position values to discrete bins for tokenization
- **Rotation Matching:** Finds the closest chiral rotation matrix from the vocabulary

**Key Methods:**

- `position_to_bin_id(position, axis)` - Convert continuous position to discrete bin ID
- `bin_id_to_position(bin_id, axis)` - Convert bin ID back to continuous position
- `brick_id_to_token(brick_id)` - Convert brick part ID to token ID
- `color_id_to_token(color_id)` - Convert color ID to token ID
- `rotation_matrix_to_token(rotation_matrix)` - Convert 3×3 rotation matrix to closest chiral rotation token ID

**Position Binning:**

Continuous positions are discretized using a binning strategy:

$$\text{bin\_id} = \left\lfloor \frac{\text{position} - \text{MIN\_POSITION}}{\text{PRECISION}} \right\rfloor + \text{OFFSET}$$

This allows the model to work with discrete tokens while maintaining spatial precision. The reverse operation reconstructs the approximate position:

$$\text{position} = (\text{bin\_id} - \text{OFFSET}) \times \text{PRECISION} + \text{MIN\_POSITION}$$

### Vocabulary Structure

The vocabulary is managed by `VocabularyManager` and stored in `atlas_config.json`. The structure includes:

1. **Special Tokens (indices 0-3):** PAD, SOS, EOS, UNK
2. **Rotations (indices 4-27, 24 chiral rotations):** Calculated on-the-fly from quaternions
3. **Colors (starting after rotations):** Dynamically added as encountered
4. **Parts (starting after colors):** Dynamically added as encountered

The configuration file structure:

```json
{
  "version": "1.0",
  "spatial": {
    "l_min": -1000,
    "l_max": 1000,
    "step": 2,
    "num_bins": 1000
  },
  "offsets": {
    "special": 0,
    "rotations": 4,
    "colors": 28,
    "parts": <calculated_dynamically>
  },
  "vocabulary": {
    "special": {
      "PAD": 0,
      "SOS": 1,
      "EOS": 2,
      "UNK": 3
    },
    "rotations": {
      "[w,x,y,z]": index
    },
    "parts": {
      "part_id.dat": index
    },
    "colors": {
      "color_code": index
    }
  },
  "vocab_size": <total_tokens>
}
```

**Token Allocation:**

1. **Special Tokens (0-3):**
   - `PAD` (0): Padding token
   - `SOS` (1): Start of sequence
   - `EOS` (2): End of sequence
   - `UNK` (3): Unknown token

2. **Rotations (4-27):** 24 chiral octahedral rotations
   - Quaternions representing all valid LEGO brick rotations (90° increments)
   - Keys format: `[w,x,y,z]` with 6 decimal precision

3. **Colors (28+):** LEGO color codes
   - Dynamically added as new colors are encountered
   - Keys are string representations of color IDs

4. **Parts (after colors):** Brick part IDs
   - Dynamically added as new parts are encountered
   - Example: `3001.dat`, `3003.dat`, `3004.dat`

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
from src.config import Config

# Initialize and process all MPD files
builder = DatasetBuilder(Config.ATLAS_CONFIG_PATH)
builder.process_dataset()
```

This will:

- Process all `.mpd` files in `mpd_files/dataset/`
- Build and update vocabulary incrementally
- Save vocabulary to `atlas_config.json`

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

## Testing

The project uses **pytest** for comprehensive test coverage across all components.

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_rotations.py
pytest tests/test_vocabulary.py
pytest tests/test_integration.py

# Run specific test function
pytest tests/test_vocabulary.py::test_add_part

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Test Structure

The test suite is organized into three main files:

#### 1. **test_rotations.py** - Rotation System Tests

Tests the 24 chiral rotation matrices generation and matching:

- `test_chiral_matrices_generation()` - Validates generation of 24 unique rotation matrices
- `test_closest_rotation_matrix()` - Tests rotation matching algorithm with exact and perturbed matrices

#### 2. **test_vocabulary.py** - Vocabulary Manager Unit Tests

Comprehensive tests for vocabulary management:

- `test_add_part()` - Adding parts and handling duplicates
- `test_add_color()` - Adding colors and handling duplicates
- `test_add_multiple_parts()` - Batch part additions
- `test_get_part_index()` - Retrieving part indices with UNK fallback
- `test_special_tokens()` - Special token indices (PAD, SOS, EOS, UNK)
- `test_rotation_index()` - Rotation matrix to index mapping
- `test_vocabulary_size()` - Vocabulary size calculations

#### 3. **test_integration.py** - DatasetBuilder Integration Tests

End-to-end workflow validation:

- `test_dataset_builder_initialization()` - Builder initialization
- `test_vocabulary_manager_initialization()` - Vocabulary setup validation
- `test_manual_vocabulary_updates()` - Adding parts and colors
- `test_brick_data_collection()` - Extracting unique brick IDs and colors
- `test_id_mapping()` - Tokenization and ID mapping
- `test_rotation_index_calculation()` - Rotation processing
- `test_config_file_structure()` - Configuration file validation

### Test Fixtures

Tests use pytest fixtures for clean, isolated test environments:

- `tmp_path` - Temporary directories for config files (auto-cleanup)
- `test_config_path` - Isolated vocabulary configuration
- `vocab_manager` - Pre-configured VocabularyManager instance
- `dataset_builder` - Pre-configured DatasetBuilder instance

### Continuous Testing

The test suite ensures:

- **Idempotency:** Tests can run multiple times with consistent results
- **Isolation:** Each test runs independently without side effects
- **Cleanup:** Temporary files automatically removed after tests
- **Coverage:** All critical paths and edge cases covered

### Example Test Output

```bash
$ pytest -v
================================ test session starts ================================
tests/test_rotations.py::test_chiral_matrices_generation PASSED              [ 11%]
tests/test_rotations.py::test_closest_rotation_matrix PASSED                 [ 22%]
tests/test_vocabulary.py::test_add_part PASSED                               [ 33%]
tests/test_vocabulary.py::test_add_color PASSED                              [ 44%]
tests/test_vocabulary.py::test_add_multiple_parts PASSED                     [ 55%]
tests/test_vocabulary.py::test_get_part_index PASSED                         [ 66%]
tests/test_vocabulary.py::test_special_tokens PASSED                         [ 77%]
tests/test_vocabulary.py::test_rotation_index PASSED                         [ 88%]
tests/test_vocabulary.py::test_vocabulary_size PASSED                        [100%]
================================ 9 passed in 0.42s ==================================
```

## Configuration

Edit `src/config.py` to configure paths:

```python
LOGGING_LEVEL = logging.DEBUG
RAW_DATASET_DIR = "./mpd_files/dataset"
PARSED_DATASET_DIR = "./processed_sets/"
ATLAS_CONFIG_PATH = "./atlas_config.json"
```

## Dataset

The current dataset is composed of the LDraw base models *(sorted by theme)*, and 1000+ official lego sets from [seymouria.pl](https://www.seymouria.pl/Download/official-lego-sets-ldr.php), downloaded using the `scrap_mpd_files.py` file. All the files are either `.mpd` or `.ldr` files for now.

## Future Directions

### Tokenization & Sequence Modeling

- Implement binning for continuous positions (1000 bins per axis)
- Map rotations to nearest discrete rotation from 24 chiral options
- Flatten brick sequences: `[ID_1, X_1, Y_1, Z_1, ROT_1, COLOR_1, ID_2, ...]`
- Treat LEGO sets as sequences for autoregressive prediction

### Model Architecture

- Transformer architecture for autoregressive brick prediction
- Study Point Transformers for permutation invariance
- Explore Set Transformer architectures
- References: [PointGPT](https://github.com/CGuangyan-BIT/PointGPT), [Set Transformer](https://arxiv.org/pdf/1810.00825), [Lego GPT](https://arxiv.org/pdf/2505.05469)

### Post-Processing

- Decoder to reconstruct from token sequences
- Position snapping to LDraw grid
- Output validation for physical plausibility

## Technical Notes

### LDraw Coordinate System

- Y-axis points downward (unlike standard 3D conventions)
- Units are in LDU (LDraw Units): 1 LDU ≈ 0.4mm
- Grid-based positioning for brick connections

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
[Official lego sets ldr](https://www.seymouria.pl/Download/official-lego-sets-ldr.php)

<p align="center">
	<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/footers/gray0_ctp_on_line.svg?sanitize=true" />
</p>


<!-- LINKS & IMAGES -->
<!-- Contributors -->

[contributors-shield]: https://img.shields.io/github/contributors/alfred0404/lightseek-ocr.svg?style=for-the-badge
[contributors-url]: https://github.com/alfred0404/lightseek-ocr/graphs/contributors

<!-- Forks -->

[forks-shield]: https://img.shields.io/github/forks/alfred0404/lightseek-ocr.svg?style=for-the-badge
[forks-url]: https://github.com/alfred0404/lightseek-ocr/network/members

<!-- Stars -->

[stars-shield]: https://img.shields.io/github/stars/alfred0404/lightseek-ocr.svg?style=for-the-badge
[stars-url]: https://github.com/alfred0404/lightseek-ocr/stargazers

<!-- Issues -->

[issues-shield]: https://img.shields.io/github/issues/alfred0404/lightseek-ocr.svg?style=for-the-badge
[issues-url]: https://github.com/alfred0404/lightseek-ocr/issues

<!-- License -->

[license-shield]: https://img.shields.io/github/license/alfred0404/lightseek-ocr.svg?style=for-the-badge
[license-url]: https://github.com/alfred0404/lightseek-ocr/blob/master/LICENSE.txt

<!-- Linkedin -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alfred-de-vulpian