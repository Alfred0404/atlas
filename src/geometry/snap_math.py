from __future__ import annotations

import numpy as np

from src.geometry.port import Port


def _rodrigues(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector *src* to unit vector *dst*.

    Uses Rodrigues' rotation formula.  Handles the degenerate cases where
    *src* and *dst* are parallel or anti-parallel.
    """
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    c = float(np.dot(src, dst))

    if c > 1.0 - 1e-8:
        return np.eye(3)

    if c < -1.0 + 1e-8:
        # 180° rotation — find any perpendicular axis.
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    v = np.cross(src, dst)
    s = np.linalg.norm(v)
    vx = np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for *angle* radians around *axis* (unit vector)."""
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
    ])


def compute_snap_matrix(
    parent_world_matrix: np.ndarray,
    parent_port: Port,
    new_part_port: Port,
    rot_steps: int,
) -> np.ndarray:
    """Return the world matrix of a new brick snapped onto an open port.

    The new brick is oriented so that *new_part_port* mates with
    *parent_port*: the ports meet at the same world position and their
    normals are anti-aligned.  *rot_steps* (0–3) selects a 90° rotation
    increment around the connection normal, resolving the remaining
    rotational degree of freedom.

    Parameters
    ----------
    parent_world_matrix:
        4×4 world matrix of the existing (parent) brick.
    parent_port:
        The open port on the parent brick (local coordinates).
    new_part_port:
        The port on the new brick that will mate with *parent_port*
        (local coordinates of the new brick).
    rot_steps:
        Integer in {0, 1, 2, 3}.  Each step adds 90° of rotation around
        the connection normal axis.

    Returns
    -------
    np.ndarray
        4×4 world matrix for the new brick.
    """
    R_p = parent_world_matrix[:3, :3]
    t_p = parent_world_matrix[:3, 3]

    # World-space position and outward normal of the parent port.
    world_pos    = R_p @ parent_port.local_position + t_p
    world_normal = R_p @ parent_port.normal
    world_normal = world_normal / np.linalg.norm(world_normal)

    # The new port must point in the opposite direction.
    target_normal = -world_normal

    # Base rotation: maps new_part_port.normal → target_normal.
    n_new = new_part_port.normal / np.linalg.norm(new_part_port.normal)
    R_base = _rodrigues(n_new, target_normal)

    # Discrete twist around the connection normal.
    angle = rot_steps * (np.pi / 2.0)
    R_twist = _axis_angle_rotation(target_normal, angle)

    R_new = R_twist @ R_base

    # Translation: place new port at world_pos.
    t_new = world_pos - R_new @ new_part_port.local_position

    M = np.eye(4)
    M[:3, :3] = R_new
    M[:3, 3]  = t_new
    return M


def extract_rot_steps(
    parent_world_matrix: np.ndarray,
    parent_port: Port,
    new_part_port: Port,
    new_world_matrix: np.ndarray,
) -> int:
    """Recover the *rot_steps* that would reproduce *new_world_matrix*.

    Used during preprocessing: given two connected bricks from ground-truth
    data, determine which of the four discrete rotations was applied.

    Returns the closest integer in {0, 1, 2, 3}.
    """
    R_p = parent_world_matrix[:3, :3]

    world_normal = R_p @ parent_port.normal
    world_normal = world_normal / np.linalg.norm(world_normal)
    target_normal = -world_normal

    n_new = new_part_port.normal / np.linalg.norm(new_part_port.normal)
    R_base = _rodrigues(n_new, target_normal)

    # R_new = R_twist @ R_base  →  R_twist = R_new @ R_base^T
    R_new   = new_world_matrix[:3, :3]
    R_twist = R_new @ R_base.T

    # Extract the rotation angle around target_normal.
    # For a rotation by θ around axis n̂:
    #   tr(R) = 1 + 2cos(θ)  →  cos(θ) = (tr(R) - 1) / 2
    cos_theta = np.clip((np.trace(R_twist) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))  # in [0, π]

    # Determine sign of rotation via the axis.
    # The skew-symmetric part of R gives sin(θ)·n̂.
    skew = (R_twist - R_twist.T) / 2.0
    axis_raw = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if np.linalg.norm(axis_raw) > 1e-8:
        axis = axis_raw / np.linalg.norm(axis_raw)
        if np.dot(axis, target_normal) < 0:
            theta = -theta  # rotation is in the opposite direction

    # Map to [0, 2π) and find closest multiple of π/2.
    theta_mod = theta % (2.0 * np.pi)
    steps_float = theta_mod / (np.pi / 2.0)
    return int(round(steps_float)) % 4
