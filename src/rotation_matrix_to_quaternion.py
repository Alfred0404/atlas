import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
from formating.customFormatter import CustomFormatter
from config import LOGGING_LEVEL


# Set up logging
# --------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

# create console handler with CustomFormatter
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
# --------------------------------


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.
    Args:
        rotation_matrix (np.ndarray): Rotation matrix of shape (3, 3).
    Returns:
        np.ndarray: Quaternion of shape (4,) in the format [w, x, y, z] where w is the scalar part.
    """
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat(
        scalar_first=True
    )  # [w, x, y, z] with w the scalar part

    # prefer positive w for consistency to help the learning process, as q and -q represent the same rotation
    if quaternion[0] < 0:
        quaternion = -quaternion

    return quaternion


def quat_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.
    Args:
        quaternion (np.ndarray): Quaternion of shape (4,) in the format [w, x, y, z] where w is the scalar part.
    Returns:
        np.ndarray: Rotation matrix of shape (3, 3).
    """
    rotation = R.from_quat(quaternion[1:])  # scipy expects [x, y, z, w]
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix


def generate_quat_chiral_rotations() -> list[np.ndarray]:
    """
    Generate the 24 orthogonal rotations of the octahedral group.
    Output format: Quaternion [w, x, y, z] (scalar_first=True).
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
                    unique_matrices.append(mat)

    # quat conversion
    final_quats = []
    for mat in unique_matrices:
        q = R.from_matrix(mat).as_quat(scalar_first=True)

        # prefer positive first non-zero component for consistency
        nonzero_indices = np.where(np.abs(q) > 1e-5)[0]
        if len(nonzero_indices) > 0:
            first_nonzero_val = q[nonzero_indices[0]]
            if first_nonzero_val < 0:
                q = -q

        final_quats.append(np.round(q, decimals=6))

    # sort for consistency in the vocabulary
    final_quats.sort(key=lambda x: tuple(x))

    logger.debug(f"Nombre de rotations uniques détectées : {len(final_quats)}")
    for i, q in enumerate(final_quats):
        logger.debug(f"ID {i:02d} | Quat [w, x, y, z]: {q.tolist()}")

    return final_quats


if __name__ == "__main__":
    rotations = generate_quat_chiral_rotations()
    rotation_strings = [",".join(map(str, rot)) for rot in rotations]
    logger.info("Generated Rotations (Quaternions [w, x, y, z]):")
    
    for rot_str in rotation_strings:
        logger.info(rot_str)