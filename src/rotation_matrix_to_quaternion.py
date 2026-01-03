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


if __name__ == "__main__":

    # example
    r = np.array(
        [[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )  # world rotation matrix 3x3
    quat = rotation_matrix_to_quaternion(r)

    logger.debug("Rotation matrix :")
    logger.debug(r)
    logger.debug("Quaternion (w, x, y, z) :")
    logger.debug(quat)