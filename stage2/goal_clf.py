from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

__all__ = ["CLFConfig", "GoalCLF"]


@dataclass
class CLFConfig:
    """Hyper-parameters for the discrete-time CLF-based navigation reward."""

    beta0: float = 0.85  # decay factor from Eq. (9); must be in (0, 1)


class GoalCLF:
    """Computes the CLF l(s) = ||p - p_goal|| used for Stage-2 navigation rewards."""

    def __init__(self, env: gym.Env, cfg: Optional[CLFConfig] = None):
        self.env = env
        self.cfg = cfg or CLFConfig()
        if not (0.0 < self.cfg.beta0 < 1.0):
            raise ValueError("CLFConfig.beta0 must be within (0, 1)")

    def lyapunov(self, obs: np.ndarray) -> float:
        """Return l(s) = distance to goal computed from the observation."""
        assert obs.ndim == 1, "Observation must be 1-D."
        px, py = float(obs[0]), float(obs[1])
        
        # Try to get goal from environment
        if hasattr(self.env, 'goal_position'):
            gx, gy = self.env.goal_position
        elif hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'goal_pos'):
            gx, gy = self.env.cfg.goal_pos
        elif hasattr(self.env.unwrapped, 'goal_position'):
            gx, gy = self.env.unwrapped.goal_position
        else:
            # Default fallback
            gx, gy = 48.0, 48.0
            
        return float(math.hypot(px - gx, py - gy))

    def reward(self, obs: np.ndarray, next_obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Return Stage-2 navigation reward r2 along with diagnostics."""
        l_curr = self.lyapunov(obs)
        l_next = self.lyapunov(next_obs)
        term = l_next + (self.cfg.beta0 - 1.0) * l_curr
        reward = -float(max(term, 0.0))
        return reward, {"l_curr": l_curr, "l_next": l_next, "term": term}
