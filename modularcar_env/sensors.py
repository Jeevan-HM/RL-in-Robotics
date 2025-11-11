from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .config import EnvConfig


class RangeSensor:
    """Simple lidar-style ray caster."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    def cast(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        heading: float,
        obstacles: Sequence[Tuple[float, float, float]],
    ) -> np.ndarray:
        n = int(max(1, self.cfg.num_rays))
        fov = np.deg2rad(self.cfg.fov_deg)
        max_r = float(self.cfg.max_range)

        if self.cfg.sensor_frame == "velocity":
            speed = float(np.hypot(vel[0], vel[1]))
            if speed > 1e-6:
                base_ang = float(np.arctan2(vel[1], vel[0]))
            else:
                base_ang = float(heading)
        else:
            base_ang = 0.0

        if n == 1:
            angles = np.array([base_ang], dtype=np.float32)
        else:
            angles = base_ang + np.linspace(-fov / 2, fov / 2, n, dtype=np.float32)

        dists = np.full((n,), max_r, dtype=np.float32)
        for i, ang in enumerate(angles):
            direction = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            t_min = max_r
            for (cx, cy, r) in obstacles:
                m = pos - np.array([cx, cy], dtype=np.float32)
                b = 2.0 * float(np.dot(direction, m))
                c = float(np.dot(m, m) - r * r)
                disc = b * b - 4.0 * c
                if disc < 0.0:
                    continue
                sqrt_disc = float(np.sqrt(disc))
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                for t in (t1, t2):
                    if 0.0 <= t <= t_min:
                        t_min = t
            dists[i] = min(t_min, max_r)
        return dists
