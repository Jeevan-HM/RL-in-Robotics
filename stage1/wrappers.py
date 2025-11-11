from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np

from modularcar_env import ModularCar2DEnv

from .cbf import SafetyCBF

__all__ = ["SafetyRewardWrapper"]


class SafetyRewardWrapper(gym.Wrapper):
    """Wraps env to provide Stage-1 safety reward r1 = exp(min(δh, 0))."""

    def __init__(self, env: ModularCar2DEnv, cbf: SafetyCBF):
        super().__init__(env)
        self.cbf = cbf
        self.alpha0 = cbf.cfg.alpha0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_h: Optional[float] = None
        self.cfg = env.cfg  # expose wrapped env config

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs.copy()
        self._prev_h, diag = self.cbf.h(obs)
        info = dict(info)
        info.update(
            {
                "cbf_h": float(self._prev_h),
                "cbf_d_clear": float(diag["d_clear"]),
                "cbf_d_ray": float(diag["d_ray"]),
                "safety_delta_h": None,
                "safety_r1": None,
            }
        )
        return obs, info

    def step(self, action):
        obs_next, _, terminated, truncated, info = self.env.step(action)
        h_next, diag = self.cbf.h(obs_next)
        delta_h = float(min(h_next + (self.alpha0 - 1.0) * self._prev_h, 0.0))
        r1 = float(math.exp(delta_h))

        info = dict(info)
        info.update(
            {
                "cbf_h_prev": float(self._prev_h),
                "cbf_h_next": float(h_next),
                "cbf_d_clear": float(diag["d_clear"]),
                "cbf_d_ray": float(diag["d_ray"]),
                "safety_delta_h": float(delta_h),
                "safety_r1": float(r1),
            }
        )

        self._prev_h = h_next
        self._prev_obs = obs_next.copy()
        return obs_next, r1, terminated, truncated, info
