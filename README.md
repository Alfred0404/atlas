<div align="center">
  <a href="https://github.com/alfred0404/atlas">
    <img src="public/logo_atlas.png" alt="ATLAS logo" width="300">
  </a>

  <h1>ATLAS</h1>
  <p><em>Autoregressive Transformer Lego Assembly Synthesis</em></p>
</div>

## Overview

ATLAS is a LEGO dataset preprocessing pipeline for autoregressive sequence models.

The current repository focuses on turning LDraw `.mpd` and `.ldr` files into tokenized sequences that can later be used to train a transformer-like model. In practice, it:

- parses hierarchical LEGO models from MPD files
- flattens submodels into world-space brick placements
- recenters and sorts bricks into a deterministic order
- builds a vocabulary for parts, colors, and rotations
- tokenizes each brick into a fixed 6-token representation
- saves processed sets as `.npy` files

This repository does not yet contain a model implementation, training loop, inference pipeline, or generation workflow. The preprocessing side is the part that is currently implemented.

## Current State

- preprocessing pipeline is implemented and working
- vocabulary is stored in `atlas_config.json`
- tokenized datasets are already being generated in `tokenized_sets/`
- pytest coverage exists for rotations, vocabulary, and builder integration
- `src/model/` is still empty
- `generated_sets/` is still empty

## Dataset Representation

Each brick is converted into a fixed 6-token row:

```text
[part_token, x_bin, y_bin, z_bin, rotation_token, color_token]
```

At training time, a full set is flattened into a 1D sequence and wrapped with special tokens:

```text
[SOS, brick_1..., brick_2..., ..., EOS]
```

Positions are discretized into bins using the spatial settings defined in `src/config.py` and mirrored in `atlas_config.json`.

## Project Layout

```text
ATLAS/
├── atlas_config.json           # Active vocabulary and spatial configuration
├── dataset/
│   └── mpd_files/              # Raw .mpd / .ldr LEGO files
├── generated_sets/             # Reserved for generated outputs (currently empty)
├── public/                     # Images and public assets
├── src/
│   ├── config.py               # Global constants and paths
│   ├── main.py                 # Entry point for dataset processing
│   ├── core/
│   │   ├── tokenizer.py        # Brick-to-token conversion
│   │   └── vocabulary.py       # Vocabulary loading and updates
│   ├── data/
│   │   ├── builder.py          # End-to-end dataset processing pipeline
│   │   ├── parser.py           # MPD/LDraw parser and flattening logic
│   │   └── sequence_dataset.py # Torch dataset wrapper over tokenized .npy files
│   ├── file_io/
│   │   ├── mpd_writer.py       # MPD export utility
│   │   └── scraper.py          # Dataset download helper
│   ├── maths/
│   │   ├── rotations.py        # Discrete rotation generation and matching
│   │   └── transforms.py       # Position and rotation extraction helpers
│   ├── model/                  # Model code placeholder (currently empty)
│   └── utils/
│       └── logging.py          # Logging setup
├── tests/
│   ├── test_integration.py
│   ├── test_rotations.py
│   └── test_vocabulary.py
├── tokenized_sets/             # Saved tokenized outputs
├── requirements.txt
├── TODO.md
└── README.md
```

## Processing Flow

```text
Raw MPD/LDR
  -> MPDParser
  -> flattened RawBrickData
  -> centering + sorting
  -> vocabulary update
  -> tokenization
  -> .npy tensor per set
```

The main processing path is currently driven by `DatasetBuilder` in `src/data/builder.py`.

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/alfred0404/atlas.git
cd atlas
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers the main preprocessing pipeline and tests.

Some optional utilities currently rely on extra packages that are not yet listed there:

- `torch` for `src/data/sequence_dataset.py`
- `requests` and `beautifulsoup4` for `src/file_io/scraper.py`

## Usage

Run the dataset builder:

```bash
python src/main.py
```

At the moment, `src/main.py` processes up to 10 `.mpd` files by default.

Outputs are written to:

- `tokenized_sets/` for processed numpy arrays
- `atlas_config.json` for vocabulary/config updates

## Testing

Run the full test suite:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_integration.py
pytest tests/test_rotations.py
pytest tests/test_vocabulary.py
```

The current tests cover:

- rotation generation and matching
- vocabulary creation and updates
- dataset builder integration
- regression protection against tensor state leaking across files

## Data Sources

The raw dataset in this repository is based on LDraw-compatible LEGO files, including many official set files downloaded from:

- https://www.seymouria.pl/Download/official-lego-sets-ldr.php

Useful references:

- https://library.ldraw.org/parts/list
- https://www.ldraw.org/article/547.html

## Roadmap

The preprocessing foundation exists, but the project is not at the modeling stage yet. The main remaining areas are:

- decoder / detokenization path
- MPD reconstruction from generated sequences
- transformer architecture and training loop
- inference-time token masking
- more dataset scraping and cleanup

See `TODO.md` for the working backlog.
