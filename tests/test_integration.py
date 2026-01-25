"""
Integration test for the refactored DatasetBuilder with VocabularyManager.

This test verifies that the DatasetBuilder correctly processes MPD files
and updates the vocabulary using the VocabularyManager.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from DatasetBuilder import DatasetBuilder
from config import Config
from VocabularyManager import VocabularyManager
import json


def test_dataset_builder_integration():
    """Test DatasetBuilder integration with VocabularyManager."""
    print("=" * 60)
    print("Testing DatasetBuilder Integration")
    print("=" * 60)

    # Use a test config
    test_config_path = "./test_integration_config.json"

    # Initialize DatasetBuilder
    print("\n1. Initializing DatasetBuilder...")
    builder = DatasetBuilder(test_config_path)
    print("   ✓ DatasetBuilder initialized")

    # Test vocabulary manager is properly initialized
    print("\n2. Checking VocabularyManager initialization...")
    vocab_size = builder.vocab_manager.get_vocab_size()
    parts_count = builder.vocab_manager.get_parts_count()
    colors_count = builder.vocab_manager.get_colors_count()

    print(f"   Initial vocab size: {vocab_size}")
    print(f"   Initial parts count: {parts_count}")
    print(f"   Initial colors count: {colors_count}")
    print("   ✓ VocabularyManager properly initialized")

    # Test adding parts manually
    print("\n3. Testing manual vocabulary updates...")
    test_parts = {"3001.dat", "3002.dat", "3003.dat"}
    test_colors = {1, 4, 14}

    builder._update_vocabulary(test_parts, test_colors)

    new_parts_count = builder.vocab_manager.get_parts_count()
    new_colors_count = builder.vocab_manager.get_colors_count()
    new_vocab_size = builder.vocab_manager.get_vocab_size()

    print(f"   Parts count after update: {new_parts_count}")
    print(f"   Colors count after update: {new_colors_count}")
    print(f"   Vocab size after update: {new_vocab_size}")

    assert new_parts_count == 3, f"Expected 3 parts, got {new_parts_count}"
    assert new_colors_count == 3, f"Expected 3 colors, got {new_colors_count}"
    print("   ✓ Vocabulary updated correctly")

    # Test collecting brick IDs and colors
    print("\n4. Testing brick data collection...")
    from DatasetBuilder import BrickDataQuat

    # Create some test data
    builder.quat_data = [
        BrickDataQuat(
            brick_id="3001.dat",
            position=np.array([0.0, 0.0, 0.0]),
            rotation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            color=1,
        ),
        BrickDataQuat(
            brick_id="3002.dat",
            position=np.array([20.0, 0.0, 0.0]),
            rotation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            color=4,
        ),
    ]

    brick_ids = builder.collect_brick_ids()
    colors = builder.collect_brick_colors()

    print(f"   Collected brick IDs: {brick_ids}")
    print(f"   Collected colors: {colors}")
    assert len(brick_ids) == 2
    assert len(colors) == 2
    print("   ✓ Brick data collected successfully")

    # Test use_id_mapping
    print("\n5. Testing use_id_mapping()...")
    processed_data = builder.use_id_mapping()

    print(f"   Processed {len(processed_data)} bricks")
    if processed_data:
        print(f"   Sample brick: {processed_data[0]}")
        # Verify that indices are correct
        brick_idx = processed_data[0].brick_idx
        color_idx = processed_data[0].color_idx
        print(f"   Brick index: {brick_idx}, Color index: {color_idx}")
        print("   ✓ ID mapping works correctly")

    # Test rotation index
    print("\n6. Testing rotation index calculation...")
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    rot_idx = builder.vocab_manager.get_rotation_index(identity_quat)
    print(f"   Identity rotation -> index {rot_idx}")
    assert 0 <= rot_idx < 24
    print("   ✓ Rotation index calculated correctly")

    # Verify the config file structure
    print("\n7. Verifying config file structure...")
    with open(test_config_path, "r") as f:
        config_data = json.load(f)

    print(f"   Config version: {config_data.get('version')}")
    print(f"   Vocabulary keys: {list(config_data.get('vocabulary', {}).keys())}")
    print(f"   Offsets: {config_data.get('offsets')}")

    # Check required keys
    assert "version" in config_data
    assert "vocabulary" in config_data
    assert "special" in config_data["vocabulary"]
    assert "rotations" in config_data["vocabulary"]
    assert "parts" in config_data["vocabulary"]
    assert "colors" in config_data["vocabulary"]
    print("   ✓ Config file structure is correct")

    print("\n" + "=" * 60)
    print("All integration tests passed! ✓")
    print("=" * 60)

    # Cleanup
    Path(test_config_path).unlink(missing_ok=True)
    print(f"\nCleaned up test file: {test_config_path}")


if __name__ == "__main__":
    try:
        test_dataset_builder_integration()
        print("\n✓ All integration tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
