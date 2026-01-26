"""Test that all reorganized imports work correctly."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

print("Testing imports after reorganization...")
print("=" * 60)

try:
    print("\n1. Testing core imports...")
    from core import AtlasTokenizer, VocabularyManager

    print("   ✓ core.AtlasTokenizer")
    print("   ✓ core.VocabularyManager")

    print("\n2. Testing data imports...")
    from data import (
        MPDParser,
        DatasetBuilder,
        RawBrickData,
        BrickDataQuat,
        ProcessedBrickData,
    )

    print("   ✓ data.MPDParser")
    print("   ✓ data.DatasetBuilder")
    print("   ✓ data.RawBrickData, BrickDataQuat, ProcessedBrickData")

    print("\n3. Testing maths imports...")
    from maths import (
        rotation_matrix_to_quaternion,
        quat_to_rotation_matrix,
        generate_quat_chiral_rotations,
        get_rotation_matrix_from_world_matrix,
        get_position_from_world_matrix,
    )

    print("   ✓ maths.rotation_matrix_to_quaternion")
    print("   ✓ maths.quat_to_rotation_matrix")
    print("   ✓ maths.generate_quat_chiral_rotations")
    print("   ✓ maths.get_rotation_matrix_from_world_matrix")
    print("   ✓ maths.get_position_from_world_matrix")

    print("\n4. Testing file_io imports...")
    from file_io import write_mpd_file

    print("   ✓ file_io.write_mpd_file")

    print("\n5. Testing utils imports...")
    from utils import setup_logging

    print("   ✓ utils.setup_logging")

    print("\n6. Testing config import...")
    from config import Config

    print("   ✓ config.Config")

    print("\n" + "=" * 60)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
