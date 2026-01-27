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
    matrix_a = np.array([[3, 5, 2], [4, 1, 3], [2, 1, 1]])
    matrix_c = np.array([[2, 6, 1], [3, 1, 0], [2, 1, 0]])
    input_matrix = np.array([[2, 6, 1], [3, 0, 0], [3, 1, 1]])

    # we know after doing the calculation manually that matrix_c is closer to input_matrix than matrix_a

    # Test known matrix
    closest_index = find_closest_rotation_matrix(input_matrix, reference_matrices=[matrix_a, matrix_c])
    assert closest_index == 1, f"Expected index 1, got {closest_index}"

    # Test a slightly perturbed matrix
    perturbed_matrix = input_matrix + np.random.normal(0, 0.01, size=input_matrix.shape)
    closest_index_perturbed = find_closest_rotation_matrix(
        perturbed_matrix, reference_matrices=[matrix_a, matrix_c]
    )
    assert (
        closest_index_perturbed == 1
    ), f"Expected index 1 for perturbed matrix, got {closest_index_perturbed}"
