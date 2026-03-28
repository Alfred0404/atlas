import numpy as np
import pytest

from src.geometry.port import Port


class TestPort:
    def test_equality(self):
        p1 = Port(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), "male")
        p2 = Port(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), "male")
        assert p1 == p2

    def test_inequality_different_type(self):
        p1 = Port(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), "male")
        p2 = Port(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "female")
        assert p1 != p2

    def test_hash_by_id_and_type(self):
        p1 = Port(0, np.array([1.0, 2.0, 3.0]), np.array([0.0, -1.0, 0.0]), "male")
        p2 = Port(0, np.array([9.0, 8.0, 7.0]), np.array([0.0, -1.0, 0.0]), "male")
        # Same port_id and type -> same hash
        assert hash(p1) == hash(p2)

    def test_transformed_identity(self):
        p = Port(0, np.array([10.0, -2.0, 5.0]), np.array([0.0, -1.0, 0.0]), "male")
        t = p.transformed(np.eye(4))
        np.testing.assert_allclose(t.local_position, p.local_position)
        np.testing.assert_allclose(t.normal, p.normal)

    def test_transformed_translation(self):
        p = Port(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), "male")
        m = np.eye(4)
        m[:3, 3] = [10.0, 20.0, 30.0]
        t = p.transformed(m)
        np.testing.assert_allclose(t.local_position, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(t.normal, [0.0, -1.0, 0.0])

    def test_transformed_rotation_90y(self):
        p = Port(0, np.array([10.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), "male")
        m = np.eye(4)
        # 90 degrees around Y axis
        m[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        t = p.transformed(m)
        np.testing.assert_allclose(t.local_position, [0.0, 0.0, -10.0], atol=1e-10)
        np.testing.assert_allclose(t.normal, [0.0, -1.0, 0.0], atol=1e-10)

    def test_frozen(self):
        p = Port(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), "male")
        with pytest.raises(AttributeError):
            p.port_id = 5
