import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.maths.rotations import (
    generate_chiral_rotation_matrices,
    find_closest_rotation_matrix,
)
import numpy as np


def test_chiral_matrices_generation():
    """Validate the generation of chiral rotation matrices."""
    matrices = generate_chiral_rotation_matrices()
    assert (
        len(matrices) == 24
    ), f"Expected 24 unique rotation matrices, got {len(matrices)}"


def test_closest_rotation_matrix():
    """Validate the closest rotation matrix identification."""
    matrices = generate_chiral_rotation_matrices()
    matrices_array = np.array(matrices)  # Convert list to numpy array

    # Test known matrix
    test_matrix = matrices_array[5]
    closest_index = find_closest_rotation_matrix(test_matrix, matrices_array)
    assert closest_index == 5, f"Expected index 5, got {closest_index}"

    # Test a slightly perturbed matrix
    perturbed_matrix = test_matrix + np.random.normal(0, 0.01, size=test_matrix.shape)
    closest_index_perturbed = find_closest_rotation_matrix(
        perturbed_matrix, matrices_array
    )
    assert (
        closest_index_perturbed == 5
    ), f"Expected index 5 for perturbed matrix, got {closest_index_perturbed}"
