from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .config import EnvConfig


@dataclass
class ObstacleField:
    """Sampler + helpers for circular obstacles."""

    cfg: EnvConfig
    world_w: float
    world_h: float
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)

    def regenerate(self, rng: np.random.Generator) -> List[Tuple[float, float, float]]:
        if self.cfg.obstacle_generator is not None:
            self.obstacles = self.cfg.obstacle_generator(rng, self.cfg)
            return self.obstacles

        obs: List[Tuple[float, float, float]] = []
        tries = 0
        while len(obs) < self.cfg.n_obstacles and tries < 2000:
            tries += 1
            r = float(rng.uniform(*self.cfg.obstacle_radius_range))
            x = float(rng.uniform(-self.world_w / 2 + r, self.world_w / 2 - r))
            y = float(rng.uniform(-self.world_h / 2 + r, self.world_h / 2 - r))

            if np.hypot(x - self.cfg.start_pos[0], y - self.cfg.start_pos[1]) < (r + self.cfg.min_obs_goal_margin):
                continue
            if np.hypot(x - self.cfg.goal_pos[0], y - self.cfg.goal_pos[1]) < (r + self.cfg.min_obs_goal_margin):
                continue

            ok = True
            for (ox, oy, or_) in obs:
                if np.hypot(x - ox, y - oy) < (r + or_ + 0.2):
                    ok = False
                    break
            if ok:
                obs.append((x, y, r))

        self.obstacles = obs
        return self.obstacles

    def collides(self, point: np.ndarray) -> bool:
        for (ox, oy, r) in self.obstacles:
            if np.hypot(point[0] - ox, point[1] - oy) <= r:
                return True
        return False

    def min_clearances(self, px: float, py: float) -> Tuple[float, float]:
        half_w = self.world_w / 2.0
        half_h = self.world_h / 2.0
        wall_clearance = min(px + half_w, half_w - px, py + half_h, half_h - py)

        obstacle_clearance = max(self.world_w, self.world_h)
        for (ox, oy, r) in self.obstacles:
            obstacle_clearance = min(obstacle_clearance, float(np.hypot(px - ox, py - oy) - r))
        return float(wall_clearance), float(obstacle_clearance)

    def relative_features(self, px: float, py: float, limit: int) -> List[float]:
        feats: List[float] = []
        for (ox, oy, r) in self.obstacles[:limit]:
            feats += [ox - px, oy - py, r]
        while len(feats) < 3 * limit:
            feats += [0.0, 0.0, 0.0]
        return feats
