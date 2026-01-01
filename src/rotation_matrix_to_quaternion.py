# snippet from https://www.johndcook.com/blog/2025/05/07/quaternions-and-rotation-matrices/
import numpy as np
from scipy.stats import special_ortho_group, norm


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.
    Args:
        R (np.ndarray): A 3x3 rotation matrix.
    Returns:
        np.ndarray: A quaternion represented as a 4-element array [q0, q1, q2, q3].
    """
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # Calculate quaternion components
    q0 = 0.5 * np.sqrt(1 + r11 + r22 + r33)
    q1 = 0.5 * np.sqrt(1 + r11 - r22 - r33) * np.sign(r32 - r23)
    q2 = 0.5 * np.sqrt(1 - r11 + r22 - r33) * np.sign(r13 - r31)
    q3 = 0.5 * np.sqrt(1 - r11 - r22 + r33) * np.sign(r21 - r12)

    return np.array([q0, q1, q2, q3])


def main():
    # Test conversion from rotation matrix to quaternion
    rotation_matrix = np.array([[1.0, 0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    q = rotation_matrix_to_quaternion(rotation_matrix)
    print(q)  # Expected output: [1. 0. 0. 0]


if __name__ == "__main__":
    main()
