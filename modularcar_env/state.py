from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleState:
    """Mutable container for vehicle pose/velocity related values."""

    position: np.ndarray
    velocity: np.ndarray
    accel: np.ndarray
    heading: float
    steer: float
    long_speed: float
    command_accel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    engine_accel: float = 0.0
    longitudinal_accel: float = 0.0

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def as_vector(self) -> np.ndarray:
        """Return [px, py, vx, vy, ax, ay] for compatibility with old API."""
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.velocity[0],
                self.velocity[1],
                self.accel[0],
                self.accel[1],
            ],
            dtype=np.float32,
        )
