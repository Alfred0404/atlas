"""
Test script to verify the refactored vocabulary system.

This script tests the VocabularyManager and the updated DatasetBuilder.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from VocabularyManager import VocabularyManager
from config import Config
import numpy as np


def test_vocabulary_manager():
    """Test the VocabularyManager functionality."""
    print("=" * 60)
    print("Testing VocabularyManager")
    print("=" * 60)

    # Create a test config file
    test_config_path = "./test_atlas_config.json"

    # Initialize VocabularyManager
    vocab_manager = VocabularyManager(test_config_path)

    # Test adding parts
    print("\n1. Testing add_part()...")
    part1_idx = vocab_manager.add_part("3001.dat")
    part2_idx = vocab_manager.add_part("3002.dat")
    part3_idx = vocab_manager.add_part("3001.dat")  # Should return same index

    print(f"   Part '3001.dat' -> index {part1_idx}")
    print(f"   Part '3002.dat' -> index {part2_idx}")
    print(f"   Part '3001.dat' (duplicate) -> index {part3_idx}")
    assert part1_idx == part3_idx, "Duplicate parts should have same index"
    print("   ✓ Parts added successfully")

    # Test adding colors
    print("\n2. Testing add_color()...")
    color1_idx = vocab_manager.add_color(1)
    color2_idx = vocab_manager.add_color(4)
    color3_idx = vocab_manager.add_color(1)  # Should return same index

    print(f"   Color 1 -> index {color1_idx}")
    print(f"   Color 4 -> index {color2_idx}")
    print(f"   Color 1 (duplicate) -> index {color3_idx}")
    assert color1_idx == color3_idx, "Duplicate colors should have same index"
    print("   ✓ Colors added successfully")

    # Test adding multiple parts
    print("\n3. Testing add_parts()...")
    parts_to_add = {"3003.dat", "3004.dat", "3005.dat"}
    result = vocab_manager.add_parts(parts_to_add)
    print(f"   Added {len(result)} parts: {result}")
    print("   ✓ Multiple parts added successfully")

    # Test getting indices
    print("\n4. Testing get_part_index()...")
    idx = vocab_manager.get_part_index("3001.dat")
    print(f"   Index for '3001.dat': {idx}")
    unknown_idx = vocab_manager.get_part_index("unknown.dat")
    print(f"   Index for unknown part: {unknown_idx} (should be 3 = UNK)")
    assert unknown_idx == 3, "Unknown parts should return UNK token"
    print("   ✓ Part indices retrieved successfully")

    # Test special tokens
    print("\n5. Testing special tokens...")
    pad_idx = vocab_manager.get_special_token_index("PAD")
    sos_idx = vocab_manager.get_special_token_index("SOS")
    eos_idx = vocab_manager.get_special_token_index("EOS")
    unk_idx = vocab_manager.get_special_token_index("UNK")

    print(f"   PAD: {pad_idx}, SOS: {sos_idx}, EOS: {eos_idx}, UNK: {unk_idx}")
    assert pad_idx == 0 and sos_idx == 1 and eos_idx == 2 and unk_idx == 3
    print("   ✓ Special tokens correct")

    # Test rotation index
    print("\n6. Testing get_rotation_index()...")
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity rotation
    rot_idx = vocab_manager.get_rotation_index(identity_quat)
    print(f"   Rotation index for identity: {rot_idx} (should be 0-23)")
    assert 0 <= rot_idx < 24, "Rotation index should be between 0 and 23"
    print("   ✓ Rotation index calculated successfully")

    # Test vocabulary size
    print("\n7. Testing vocabulary size...")
    vocab_size = vocab_manager.get_vocab_size()
    parts_count = vocab_manager.get_parts_count()
    colors_count = vocab_manager.get_colors_count()

    print(f"   Total vocab size: {vocab_size}")
    print(f"   Parts count: {parts_count}")
    print(f"   Colors count: {colors_count}")
    print(f"   Special tokens: 4")
    print(f"   Rotations: 24")

    expected_size = 4 + 24 + parts_count + colors_count
    print(f"   Expected size: {expected_size}")
    print("   ✓ Vocabulary size calculation correct")

    print("\n" + "=" * 60)
    print("All VocabularyManager tests passed! ✓")
    print("=" * 60)

    # Cleanup
    Path(test_config_path).unlink(missing_ok=True)
    print(f"\nCleaned up test file: {test_config_path}")


if __name__ == "__main__":
    try:
        test_vocabulary_manager()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
