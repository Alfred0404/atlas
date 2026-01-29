"""Test suite for the VocabularyManager.

Tests vocabulary creation, part/color management, and special tokens.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vocabulary import VocabularyManager


@pytest.fixture
def test_config_path(tmp_path: Path) -> str:
    """Provide a temporary config file path.
    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    Returns:
        str: Path to the temporary config file.
    """
    return str(tmp_path / "test_atlas_config.json")


@pytest.fixture
def vocab_manager(test_config_path: str) -> VocabularyManager:
    """Create a VocabularyManager instance for testing.
    Args:
        test_config_path (str): Path to the temporary config file.
    Returns:
        VocabularyManager: An instance of VocabularyManager.
    """
    return VocabularyManager(test_config_path)


def test_add_part(vocab_manager: VocabularyManager):
    """Test adding parts to vocabulary.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """
    part1_idx = vocab_manager.add_part("3001.dat")
    part2_idx = vocab_manager.add_part("3002.dat")
    part3_idx = vocab_manager.add_part("3001.dat")  # Should return same index

    assert part1_idx == part3_idx, "Duplicate parts should have same index"
    assert part1_idx != part2_idx, "Different parts should have different indices"


def test_add_color(vocab_manager: VocabularyManager):
    """Test adding colors to vocabulary.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """

    color1_idx = vocab_manager.add_color(1)
    color2_idx = vocab_manager.add_color(4)
    color3_idx = vocab_manager.add_color(1)  # Should return same index

    assert color1_idx == color3_idx, "Duplicate colors should have same index"
    assert color1_idx != color2_idx, "Different colors should have different indices"


def test_add_multiple_parts(vocab_manager: VocabularyManager):
    """Test adding multiple parts at once.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """

    parts_to_add = {"3003.dat", "3004.dat", "3005.dat"}
    result = vocab_manager.add_parts(parts_to_add)

    assert len(result) == 3, "Should add 3 parts"
    for part in parts_to_add:
        assert part in result, f"Part {part} should be in result"


def test_get_part_index(vocab_manager: VocabularyManager):
    """Test retrieving part indices.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """

    vocab_manager.add_part("3001.dat")

    idx = vocab_manager.get_part_index("3001.dat")
    assert idx is not None, "Should return valid index for known part"

    unknown_idx = vocab_manager.get_part_index("unknown.dat")
    assert unknown_idx == 3, "Unknown parts should return UNK token (index 3)"


def test_special_tokens(vocab_manager: VocabularyManager):
    """Test special token indices.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """

    pad_idx = vocab_manager.get_special_token_index("PAD")
    sos_idx = vocab_manager.get_special_token_index("SOS")
    eos_idx = vocab_manager.get_special_token_index("EOS")
    unk_idx = vocab_manager.get_special_token_index("UNK")

    assert pad_idx == 0, "PAD token should be index 0"
    assert sos_idx == 1, "SOS token should be index 1"
    assert eos_idx == 2, "EOS token should be index 2"
    assert unk_idx == 3, "UNK token should be index 3"


def test_rotation_index(vocab_manager: VocabularyManager):
    """Test rotation index calculation.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """

    identity_matrix = np.eye(3)  # Identity rotation matrix
    rot_idx = vocab_manager.get_rotation_index(identity_matrix)

    assert 0 <= rot_idx < 24, "Rotation index should be between 0 and 23"


def test_vocabulary_size(vocab_manager: VocabularyManager):
    """Test vocabulary size calculation.
    Args:
        vocab_manager (VocabularyManager): VocabularyManager instance.
    """

    # Get initial size (should be Config.OFFSETS["parts"] which is 3028)
    initial_size = vocab_manager.get_vocab_size()

    # Add some parts and colors
    vocab_manager.add_part("3001.dat")
    vocab_manager.add_part("3002.dat")
    vocab_manager.add_color(1)
    vocab_manager.add_color(4)

    vocab_size = vocab_manager.get_vocab_size()
    parts_count = vocab_manager.get_parts_count()
    colors_count = vocab_manager.get_colors_count()

    # Vocab size increases by the number of parts + colors added
    expected_size = initial_size + parts_count + colors_count

    assert parts_count == 2, "Should have 2 parts"
    assert colors_count == 2, "Should have 2 colors"
    assert (
        vocab_size == expected_size
    ), f"Vocabulary size {vocab_size} should match expected {expected_size}"
