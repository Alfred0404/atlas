import numpy as np
import pytest

from src.geometry.port import Port
from src.geometry.snap_math import compute_snap_matrix, extract_rot_steps


def _port(pos, normal, port_type="male", port_id=0):
    return Port(
        port_id=port_id,
        local_position=np.array(pos, dtype=float),
        normal=np.array(normal, dtype=float),
        port_type=port_type,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rotation_matrix(R: np.ndarray) -> bool:
    """True if R is a valid rotation matrix (orthonormal, det=+1)."""
    return (
        np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        and abs(np.linalg.det(R) - 1.0) < 1e-6
    )


# ---------------------------------------------------------------------------
# compute_snap_matrix
# ---------------------------------------------------------------------------

class TestComputeSnapMatrix:

    def test_returns_4x4(self):
        parent = np.eye(4)
        pp = _port([0, 0, 0], [0, -1, 0], "male")
        fp = _port([0, 0, 0], [0,  1, 0], "female")
        M = compute_snap_matrix(parent, pp, fp, rot_steps=0)
        assert M.shape == (4, 4)

    def test_result_is_rigid(self):
        """World matrix must be a valid rigid-body transform."""
        parent = np.eye(4)
        pp = _port([10, 0, 10], [0, -1, 0], "male")
        np_ = _port([0, 8, 0], [0, 1, 0], "female")
        M = compute_snap_matrix(parent, pp, np_, rot_steps=0)
        assert _is_rotation_matrix(M[:3, :3])
        assert M[3, 3] == pytest.approx(1.0)
        assert np.allclose(M[3, :3], 0.0)

    def test_ports_meet_at_same_world_position(self):
        """After snapping, both ports must be at the same world location."""
        parent = np.eye(4)
        parent_port = _port([0, 0, 0], [0, -1, 0], "male")
        new_port    = _port([0, 8, 0], [0,  1, 0], "female")

        M_new = compute_snap_matrix(parent, parent_port, new_port, rot_steps=0)

        world_parent_pos = parent[:3, :3] @ parent_port.local_position + parent[:3, 3]
        world_new_pos    = M_new[:3, :3]  @ new_port.local_position    + M_new[:3, 3]
        np.testing.assert_allclose(world_parent_pos, world_new_pos, atol=1e-6)

    def test_normals_are_anti_aligned(self):
        """The two mating port normals must point in opposite directions."""
        parent = np.eye(4)
        parent_port = _port([0, 0, 0], [0, -1, 0], "male")
        new_port    = _port([0, 8, 0], [0,  1, 0], "female")

        M_new = compute_snap_matrix(parent, parent_port, new_port, rot_steps=0)

        world_parent_normal = parent[:3, :3] @ parent_port.normal
        world_new_normal    = M_new[:3, :3]  @ new_port.normal
        dot = np.dot(world_parent_normal / np.linalg.norm(world_parent_normal),
                     world_new_normal    / np.linalg.norm(world_new_normal))
        assert dot == pytest.approx(-1.0, abs=1e-6)

    def test_rot_steps_rotate_around_normal(self):
        """Each additional rot_step adds exactly 90° around the connection normal."""
        parent = np.eye(4)
        parent_port = _port([0, 0, 0], [0, -1, 0], "male")
        new_port    = _port([0, 0, 0], [0,  1, 0], "female")

        matrices = [
            compute_snap_matrix(parent, parent_port, new_port, rot_steps=s)
            for s in range(4)
        ]

        # Each pair of consecutive matrices should differ by a 90° rotation
        # around the Y axis (the connection normal in this case).
        for i in range(4):
            R0 = matrices[i][:3, :3]
            R1 = matrices[(i + 1) % 4][:3, :3]
            R_diff = R1 @ R0.T
            cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            assert angle_deg == pytest.approx(90.0, abs=1e-4)

    def test_four_rot_steps_is_identity(self):
        """rot_steps=0 and rot_steps=4 (≡ 0) produce the same matrix."""
        parent = np.eye(4)
        parent_port = _port([0,  0, 0], [0, -1, 0], "male")
        new_port    = _port([0, 24, 0], [0,  1, 0], "female")

        M0 = compute_snap_matrix(parent, parent_port, new_port, rot_steps=0)
        M4 = compute_snap_matrix(parent, parent_port, new_port, rot_steps=4 % 4)
        np.testing.assert_allclose(M0, M4, atol=1e-6)

    def test_snot_connection_horizontal(self):
        """A side-facing (SNOT) stud should snap correctly."""
        parent = np.eye(4)
        # Stud faces +X
        parent_port = _port([20, 12, 0], [1, 0, 0], "male")
        # Hole faces -X
        new_port    = _port([ 0, 12, 0], [-1, 0, 0], "female")

        M_new = compute_snap_matrix(parent, parent_port, new_port, rot_steps=0)

        world_pp = parent[:3, :3] @ parent_port.local_position + parent[:3, 3]
        world_np = M_new[:3, :3]  @ new_port.local_position    + M_new[:3, 3]
        np.testing.assert_allclose(world_pp, world_np, atol=1e-6)

        world_pn = parent[:3, :3] @ parent_port.normal
        world_nn = M_new[:3, :3]  @ new_port.normal
        dot = np.dot(world_pn / np.linalg.norm(world_pn),
                     world_nn / np.linalg.norm(world_nn))
        assert dot == pytest.approx(-1.0, abs=1e-6)

    def test_translated_parent(self):
        """Parent at arbitrary world position — ports still meet."""
        parent = np.eye(4)
        parent[:3, 3] = [100.0, -48.0, 60.0]

        parent_port = _port([10, 0, 10], [0, -1, 0], "male")
        new_port    = _port([ 0, 8,  0], [0,  1, 0], "female")

        M_new = compute_snap_matrix(parent, parent_port, new_port, rot_steps=1)

        world_pp = parent[:3, :3] @ parent_port.local_position + parent[:3, 3]
        world_np = M_new[:3, :3]  @ new_port.local_position    + M_new[:3, 3]
        np.testing.assert_allclose(world_pp, world_np, atol=1e-6)


# ---------------------------------------------------------------------------
# extract_rot_steps
# ---------------------------------------------------------------------------

class TestExtractRotSteps:

    def _roundtrip(self, parent, parent_port, new_port, expected_steps):
        M_new = compute_snap_matrix(parent, parent_port, new_port, expected_steps)
        recovered = extract_rot_steps(parent, parent_port, new_port, M_new)
        assert recovered == expected_steps

    def test_all_steps_vertical(self):
        parent = np.eye(4)
        pp = _port([0,  0, 0], [0, -1, 0], "male")
        np_ = _port([0, 24, 0], [0,  1, 0], "female")
        for steps in range(4):
            self._roundtrip(parent, pp, np_, steps)

    def test_all_steps_horizontal(self):
        parent = np.eye(4)
        pp  = _port([20, 12, 0], [ 1, 0, 0], "male")
        np_ = _port([ 0, 12, 0], [-1, 0, 0], "female")
        for steps in range(4):
            self._roundtrip(parent, pp, np_, steps)

    def test_with_rotated_parent(self):
        """Parent brick itself is rotated — roundtrip should still work."""
        angle = np.pi / 4
        R = np.array([
            [ np.cos(angle), 0, np.sin(angle)],
            [ 0,             1, 0            ],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        parent = np.eye(4)
        parent[:3, :3] = R
        parent[:3,  3] = [10.0, -24.0, 5.0]

        pp  = _port([0,  0, 0], [0, -1, 0], "male")
        np_ = _port([0, 24, 0], [0,  1, 0], "female")
        for steps in range(4):
            self._roundtrip(parent, pp, np_, steps)

    def test_with_offset_ports(self):
        parent = np.eye(4)
        pp  = _port([10, -24, -10], [0, -1, 0], "male")
        np_ = _port([-5,   8,   5], [0,  1, 0], "female")
        for steps in range(4):
            self._roundtrip(parent, pp, np_, steps)
