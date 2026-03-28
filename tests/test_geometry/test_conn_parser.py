import numpy as np

from src.geometry.conn_parser import ConnParser
from src.geometry.port import Port


class TestConnParser:
    def test_parse_3021_has_studs(self):
        parser = ConnParser()
        ports = parser.parse("3021")
        # 3021 = 2x3 plate: 6 studs (male) + 6 anti-studs (female) = 12
        assert len(ports) > 0
        assert all(isinstance(p, Port) for p in ports)

    def test_parse_3021_has_male_and_female(self):
        parser = ConnParser()
        ports = parser.parse("3021")
        types = {p.port_type for p in ports}
        assert "male" in types
        assert "female" in types

    def test_parse_3021_stud_count(self):
        """3021 is a 2x3 plate: should have 6 male + 6 female = 12 ports."""
        parser = ConnParser()
        ports = parser.parse("3021")
        male = [p for p in ports if p.port_type == "male"]
        female = [p for p in ports if p.port_type == "female"]
        assert len(male) == 6
        assert len(female) == 6

    def test_parse_3021_stud_spacing(self):
        """Studs should be on a 20 LDU grid."""
        parser = ConnParser()
        ports = parser.parse("3021")
        male = [p for p in ports if p.port_type == "male"]
        xs = sorted(set(p.local_position[0] for p in male))
        # 3 X positions for a 2x3 plate, spaced 20 LDU
        assert len(xs) == 3
        for i in range(1, len(xs)):
            assert abs(xs[i] - xs[i - 1] - 20.0) < 0.1

    def test_parse_nonexistent_returns_empty(self):
        parser = ConnParser()
        ports = parser.parse("nonexistent_99999")
        assert ports == []

    def test_port_normals(self):
        parser = ConnParser()
        ports = parser.parse("3021")
        for p in ports:
            if p.port_type == "male":
                np.testing.assert_allclose(p.normal, [0, -1, 0])
            else:
                np.testing.assert_allclose(p.normal, [0, 1, 0])

    def test_unique_port_ids(self):
        parser = ConnParser()
        ports = parser.parse("3021")
        ids = [p.port_id for p in ports]
        assert len(ids) == len(set(ids))

    def test_subfile_resolution(self):
        """Parts that use subfiles (s/*.dat) should resolve studs recursively."""
        parser = ConnParser()
        # 3001 (2x4 brick) uses s/3001s01.dat which contains stud.dat refs
        ports = parser.parse("3001")
        male = [p for p in ports if p.port_type == "male"]
        assert len(male) == 8  # 4x2 studs
