import numpy as np


def get_rotation_matrix_from_world_matrix(
    world_matrix: np.ndarray,
) -> np.ndarray:
    """Extract the rotation matrix from a world matrix.

    Args:
        world_matrix (np.ndarray): A 4x4 world transformation matrix.
    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    return world_matrix[:3, :3]


def get_position_from_world_matrix(
    world_matrix: np.ndarray,
) -> np.ndarray:
    """Extract the position vector from a world matrix.

    Args:
        world_matrix (np.ndarray): A 4x4 world transformation matrix.
    Returns:
        np.ndarray: A 3-element position vector.
    """
    return world_matrix[:3, 3]
