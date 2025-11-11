from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from modularcar_env import ModularCar2DEnv

__all__ = ["CLFConfig", "GoalCLF"]


@dataclass
class CLFConfig:
    """Hyper-parameters for the discrete-time CLF-based navigation reward."""

    beta0: float = 0.85  # decay factor from Eq. (9); must be in (0, 1)


class GoalCLF:
    """Computes the CLF l(s) = ||p - p_goal|| used for Stage-2 navigation rewards."""

    def __init__(self, env: ModularCar2DEnv, cfg: Optional[CLFConfig] = None):
        self.env = env
        self.cfg = cfg or CLFConfig()
        if not (0.0 < self.cfg.beta0 < 1.0):
            raise ValueError("CLFConfig.beta0 must be within (0, 1)")

    def lyapunov(self, obs: np.ndarray) -> float:
        """Return l(s) = ||p - p_goal|| computed from the observation."""
        assert obs.ndim == 1, "Observation must be 1-D."
        px, py = float(obs[0]), float(obs[1])
        gx, gy = self.env.cfg.goal_pos
        return float(math.hypot(px - gx, py - gy))

    def reward(self, obs: np.ndarray, next_obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Return Stage-2 navigation reward r2 along with diagnostics."""
        l_curr = self.lyapunov(obs)
        l_next = self.lyapunov(next_obs)
        term = l_next + (self.cfg.beta0 - 1.0) * l_curr
        reward = -float(max(term, 0.0))
        return reward, {"l_curr": l_curr, "l_next": l_next, "term": term}
