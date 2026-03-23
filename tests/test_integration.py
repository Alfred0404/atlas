"""Integration tests for DatasetBuilder with VocabularyManager.

Tests the integration between DatasetBuilder and VocabularyManager,
including vocabulary updates and brick data processing.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.builder import DatasetBuilder
from src.data.parser import RawBrickData
import src.data.builder as builder_module
from src.config import Config
from src.core.vocabulary import VocabularyManager


@pytest.fixture
def test_config_path(tmp_path: Path):
    """Provide a temporary config file path.
    Yields:
        str: Path to the temporary config file.
    """

    config_path = tmp_path / "test_integration_config.json"
    yield str(config_path)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def dataset_builder(test_config_path: str) -> DatasetBuilder:
    """Create a DatasetBuilder instance for testing.
    Args:
        test_config_path (str): Path to the temporary config file.
    Returns:
        DatasetBuilder: An instance of DatasetBuilder.
    """
    return DatasetBuilder(test_config_path)


def test_dataset_builder_initialization(dataset_builder: DatasetBuilder):
    """Test DatasetBuilder initialization.
    Args:
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    assert (
        dataset_builder.vocab_manager is not None
    ), "VocabularyManager should be initialized"

    vocab_size = dataset_builder.vocab_manager.get_vocab_size()
    assert vocab_size > 0, "Vocabulary should have initial size"


def test_vocabulary_manager_initialization(dataset_builder: DatasetBuilder):
    """Test VocabularyManager initialization within DatasetBuilder.
    Args:
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    vocab_size = dataset_builder.vocab_manager.get_vocab_size()
    parts_count = dataset_builder.vocab_manager.get_parts_count()
    colors_count = dataset_builder.vocab_manager.get_colors_count()

    assert (
        vocab_size >= 28
    ), "Should have at least special tokens and rotations (4 + 24)"
    assert parts_count >= 0, "Parts count should be non-negative"
    assert colors_count >= 0, "Colors count should be non-negative"


def test_manual_vocabulary_updates(dataset_builder: DatasetBuilder):
    """Test manual vocabulary updates
    Args:
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    test_parts = {"3001.dat", "3002.dat", "3003.dat"}
    test_colors = {1, 4, 14}

    initial_parts = dataset_builder.vocab_manager.get_parts_count()
    initial_colors = dataset_builder.vocab_manager.get_colors_count()

    dataset_builder._update_vocabulary(test_parts, test_colors)

    new_parts_count = dataset_builder.vocab_manager.get_parts_count()
    new_colors_count = dataset_builder.vocab_manager.get_colors_count()

    assert new_parts_count == initial_parts + 3, "Should add 3 parts"
    assert new_colors_count == initial_colors + 3, "Should add 3 colors"


def test_brick_data_collection(dataset_builder: DatasetBuilder):
    """Test collecting brick IDs and colors from data.
    Args:
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    # Create test data using RawBrickData with world matrices
    identity_matrix = np.eye(4)
    identity_matrix[:3, 3] = [0.0, 0.0, 0.0]  # position at origin

    second_matrix = np.eye(4)
    second_matrix[:3, 3] = [20.0, 0.0, 0.0]  # position at x=20

    dataset_builder.raw_data = [
        RawBrickData(
            brick_id="3001.dat",
            world_matrix=identity_matrix,
            color=1,
        ),
        RawBrickData(
            brick_id="3002.dat",
            world_matrix=second_matrix,
            color=4,
        ),
    ]

    brick_ids = dataset_builder.collect_brick_ids()
    colors = dataset_builder.collect_brick_colors()

    assert len(brick_ids) == 2, "Should collect 2 unique brick IDs"
    assert "3001.dat" in brick_ids and "3002.dat" in brick_ids
    assert len(colors) == 2, "Should collect 2 unique colors"
    assert 1 in colors and 4 in colors


def test_id_mapping(dataset_builder: DatasetBuilder):
    """Test ID mapping functionality.
    Args:
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    # Setup test data with RawBrickData
    identity_matrix = np.eye(4)
    identity_matrix[:3, 3] = [0.0, 0.0, 0.0]

    dataset_builder.raw_data = [
        RawBrickData(
            brick_id="3001.dat",
            world_matrix=identity_matrix,
            color=1,
        ),
    ]

    # Update vocabulary with test data and load into tokenizer
    brick_ids = dataset_builder.collect_brick_ids()
    colors = dataset_builder.collect_brick_colors()
    dataset_builder._update_vocabulary(brick_ids, colors)
    dataset_builder.vocab_manager.save()
    dataset_builder.tokenizer.load_vocabulary(dataset_builder.vocab_manager.config_path)

    # Test tokenization (which does the ID mapping)
    dataset_builder.tokenize_brick_data()
    processed_data = dataset_builder.tokenized_data

    assert len(processed_data) == 1, "Should process 1 brick"

    brick = processed_data[0]
    assert hasattr(brick, "brick_idx"), "Processed brick should have brick_idx"
    assert hasattr(brick, "color_idx"), "Processed brick should have color_idx"
    assert brick.brick_idx is not None, "brick_idx should not be None"
    assert brick.color_idx is not None, "color_idx should not be None"


def test_rotation_index_calculation(dataset_builder: DatasetBuilder):
    """Test rotation index calculation.
    Args:
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    # Test with a rotation matrix instead of quaternion
    identity_rotation = np.eye(3)
    rot_idx = dataset_builder.vocab_manager.get_rotation_index(identity_rotation)

    assert 0 <= rot_idx < 24, "Rotation index should be between 0 and 23"


def test_config_file_structure(test_config_path: str, dataset_builder: DatasetBuilder):
    """Test that the config file has the correct structure.
    Args:
        test_config_path (str): Path to the temporary config file.
        dataset_builder (DatasetBuilder): DatasetBuilder instance.
    """

    # Config file is already saved during initialization
    dataset_builder.vocab_manager.add_part("test.dat")
    dataset_builder.vocab_manager.save()

    with open(test_config_path, "r") as f:
        config_data = json.load(f)

    # Check required top-level keys
    assert "version" in config_data, "Config should have version"
    assert "vocabulary" in config_data, "Config should have vocabulary"
    assert "offsets" in config_data, "Config should have offsets"

    # Check vocabulary structure
    vocab = config_data["vocabulary"]
    assert "special" in vocab, "Vocabulary should have special tokens"
    assert "rotations" in vocab, "Vocabulary should have rotations"
    assert "parts" in vocab, "Vocabulary should have parts"
    assert "colors" in vocab, "Vocabulary should have colors"

    # Verify special tokens
    special = vocab["special"]
    assert "PAD" in special and special["PAD"] == 0
    assert "SOS" in special and special["SOS"] == 1
    assert "EOS" in special and special["EOS"] == 2
    assert "UNK" in special and special["UNK"] == 3


def test_process_single_file_resets_tensor_state(
    tmp_path: Path, test_config_path: str, monkeypatch: pytest.MonkeyPatch
):
    """Ensure each processed file saves only its own tensor rows."""

    output_dir = tmp_path / "tokenized_sets"
    output_dir.mkdir()

    original_config_path = Config.ATLAS_CONFIG_PATH
    original_output_dir = Config.TOKENIZED_DATASET_DIR

    monkeypatch.setattr(Config, "ATLAS_CONFIG_PATH", test_config_path)
    monkeypatch.setattr(Config, "TOKENIZED_DATASET_DIR", str(output_dir))

    identity_matrix = np.eye(4)
    first_shifted_matrix = np.eye(4)
    first_shifted_matrix[:3, 3] = [20.0, 0.0, 0.0]
    second_matrix = np.eye(4)
    second_matrix[:3, 3] = [40.0, 0.0, 0.0]

    raw_data_by_file = {
        "first.mpd": [
            RawBrickData("3001.dat", identity_matrix.copy(), 1),
            RawBrickData("3002.dat", first_shifted_matrix.copy(), 4),
        ],
        "second.mpd": [
            RawBrickData("3003.dat", second_matrix.copy(), 14),
        ],
    }

    class FakeMPDParser:
        def __init__(self, mpd_file_path: str):
            self.file_name = Path(mpd_file_path).name
            self._submodels = {"main.ldr": ["dummy"]}
            self.raw_data = []

        def flatten(self, model_name: str, parent_matrix: np.ndarray):
            self.raw_data = [
                RawBrickData(brick.brick_id, brick.world_matrix.copy(), brick.color)
                for brick in raw_data_by_file[self.file_name]
            ]
            return self.raw_data

        def sort_bricks_by_position(self):
            self.raw_data.sort(
                key=lambda brick: (
                    brick.world_matrix[1, 3],
                    brick.world_matrix[0, 3],
                    brick.world_matrix[2, 3],
                )
            )
            return self.raw_data

    monkeypatch.setattr(builder_module, "MPDParser", FakeMPDParser)

    dataset_builder = DatasetBuilder(test_config_path)

    first_file = tmp_path / "first.mpd"
    second_file = tmp_path / "second.mpd"
    files = [first_file, second_file]

    # Pass 1: collect vocabulary
    for f in files:
        raw_data = dataset_builder._parse_and_transform(f)
        if raw_data:
            dataset_builder.all_brick_ids.update(b.brick_id for b in raw_data)
            dataset_builder.all_colors.update(b.color for b in raw_data)

    dataset_builder.vocab_manager.add_colors(dataset_builder.all_colors)
    dataset_builder.vocab_manager.add_parts(dataset_builder.all_brick_ids)
    dataset_builder.vocab_manager.save()
    dataset_builder.tokenizer.load_vocabulary(test_config_path)

    # Pass 2: tokenize and save
    for f in files:
        raw_data = dataset_builder._parse_and_transform(f)
        if raw_data:
            dataset_builder.raw_data = raw_data
            dataset_builder.tokenize_brick_data()
            dataset_builder._to_tensor()
            dataset_builder.save_dataset(str(output_dir / f"{f.stem}.npy"))

    first_tensor = np.load(output_dir / "first.npy")
    second_tensor = np.load(output_dir / "second.npy")

    assert first_tensor.shape[0] == 2, "First file should save exactly 2 bricks"
    assert second_tensor.shape[0] == 1, "Second file should save exactly 1 brick"
    assert (
        second_tensor.shape[0] != first_tensor.shape[0] + 1
    ), "Second file should not accumulate bricks from the first file"

    monkeypatch.setattr(Config, "ATLAS_CONFIG_PATH", original_config_path)
    monkeypatch.setattr(Config, "TOKENIZED_DATASET_DIR", original_output_dir)
