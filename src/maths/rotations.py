import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from ..utils.logging import setup_logging

logger = setup_logging()


def generate_chiral_rotation_matrices() -> list[np.ndarray]:
    """
    Generate the 24 orthogonal rotations of the octahedral group.
    Output format: Rotation matrices of shape (3, 3).
    """

    unique_matrices = []
    angles = [0, 90, 180, 270]

    # matrix generation
    for x in angles:
        for y in angles:
            for z in angles:
                mat = R.from_euler("xyz", [x, y, z], degrees=True).as_matrix()

                is_new = True
                for existing_mat in unique_matrices:
                    if np.allclose(mat, existing_mat, atol=1e-5):
                        is_new = False
                        break

                if is_new:
                    unique_matrices.append(np.round(mat, decimals=6))

    # sort for consistency in the vocabulary
    unique_matrices.sort(key=lambda x: tuple(x.flatten()))

    logger.debug(f"Generated {len(unique_matrices)} unique rotation matrices")

    return unique_matrices


def find_closest_rotation_matrix(
    input_matrix: np.ndarray, reference_matrices: list[np.ndarray]
) -> int:
    """Find the index of the closest rotation matrix from a list of reference matrices, using Frobenius norm.
    Args:
        input_matrix (np.ndarray): The input rotation matrix of shape (3, 3).
        reference_matrices (list[np.ndarray]): List of reference rotation matrices of shape (3, 3).
    Returns:
        int: Index of the closest rotation matrix in the reference list.
    """
    reference_matrices = np.array(reference_matrices)
    differences = reference_matrices - input_matrix
    distances = np.sum(np.square(differences), axis=(1, 2))

    return int(np.argmin(distances))


if __name__ == "__main__":
    rotations = generate_chiral_rotation_matrices()
    logger.info("Generated Rotations (Rotation Matrices):")

    for i, mat in enumerate(rotations):
        logger.info(f"Rotation {i}:\n{mat}")
