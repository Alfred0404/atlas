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

    # Update vocabulary with test data
    brick_ids = dataset_builder.collect_brick_ids()
    colors = dataset_builder.collect_brick_colors()
    dataset_builder._update_vocabulary(brick_ids, colors)

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
    # Add a part to trigger a save
    dataset_builder.vocab_manager.add_part("test.dat")

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
