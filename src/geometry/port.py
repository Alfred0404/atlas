from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Port:
    """A connection port on a LEGO part (stud or anti-stud).

    Attributes:
        port_id: Unique identifier within the part.
        local_position: (3,) position in part-local LDU coordinates.
        normal: (3,) outward-facing unit vector.
            Male (stud): [0, -1, 0] in LDraw convention (points up physically).
            Female (anti-stud): [0, 1, 0] (points down physically).
        port_type: ``"male"`` or ``"female"``.
        radius: Stud radius in LDU (default 6.0 for standard studs).
    """

    port_id: int
    local_position: np.ndarray
    normal: np.ndarray
    port_type: str  # "male" or "female"
    radius: float = 6.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Port):
            return NotImplemented
        return (
            self.port_id == other.port_id
            and self.port_type == other.port_type
            and np.allclose(self.local_position, other.local_position, atol=1e-6)
            and np.allclose(self.normal, other.normal, atol=1e-6)
        )

    def __hash__(self) -> int:
        return hash((self.port_id, self.port_type))

    def transformed(self, world_matrix: np.ndarray) -> Port:
        """Return a new Port with position and normal in world space."""
        R = world_matrix[:3, :3]
        t = world_matrix[:3, 3]
        world_pos = R @ self.local_position + t
        world_normal = R @ self.normal
        norm = np.linalg.norm(world_normal)
        if norm > 1e-8:
            world_normal = world_normal / norm
        return Port(
            port_id=self.port_id,
            local_position=world_pos,
            normal=world_normal,
            port_type=self.port_type,
            radius=self.radius,
        )
