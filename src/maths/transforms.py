"""Matrix transformation utilities for world matrices."""

import numpy as np


def get_rotation_matrix_from_world_matrix(
    world_matrix: np.ndarray,
) -> np.ndarray:
    """Extract the rotation matrix from a world matrix.

    This function properly handles transformation matrices that contain scaling
    by using SVD decomposition to extract only the rotation component.

    Args:
        world_matrix (np.ndarray): A 4x4 world transformation matrix.
    Returns:
        np.ndarray: A 3x3 rotation matrix (orthonormal, det=1).
    """
    # Extract the 3x3 upper-left submatrix (rotation + scale + shear)
    transform_3x3 = world_matrix[:3, :3]

    # Use SVD to extract pure rotation from the transformation matrix
    # M = U * Σ * V^T, where the rotation is R = U * V^T
    U, _, Vt = np.linalg.svd(transform_3x3)

    # Reconstruct the rotation matrix
    rotation_matrix = U @ Vt

    # Ensure proper rotation (det = +1, not -1 which would be a reflection)
    if np.linalg.det(rotation_matrix) < 0:
        # Flip the sign of the last column of U to fix reflection
        U[:, -1] *= -1
        rotation_matrix = U @ Vt

    return rotation_matrix


def batch_get_rotation_matrices(world_matrices: np.ndarray) -> np.ndarray:
    """Extract rotation matrices from a batch of world matrices using SVD.

    Args:
        world_matrices: Array of shape (N, 4, 4).
    Returns:
        Array of shape (N, 3, 3) with pure rotation matrices.
    """
    transforms = world_matrices[:, :3, :3]
    U, _, Vt = np.linalg.svd(transforms)
    rotations = U @ Vt

    # Fix reflections (det < 0) by flipping last column of U
    dets = np.linalg.det(rotations)
    mask = dets < 0
    if np.any(mask):
        U[mask, :, -1] *= -1
        rotations[mask] = U[mask] @ Vt[mask]

    return rotations


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
