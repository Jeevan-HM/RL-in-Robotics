from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from modularcar_env import ModularCar2DEnv

from .config import CBFConfig

__all__ = ["SafetyCBF"]


class SafetyCBF:
    """Computes the discrete-time Control Barrier Function h(s) from env observations."""

    def __init__(self, env: ModularCar2DEnv, cfg: Optional[CBFConfig] = None):
        self.env = env
        self.cfg: CBFConfig = cfg or CBFConfig()
        self._idx = self._compute_index_map()

    def _compute_index_map(self) -> Dict[str, Any]:
        cfg = self.env.cfg
        i = 0
        i += 4  # px, py, vx, vy
        if cfg.include_applied_accel:
            i += 2
        i += 2  # goal dx, dy
        speed_idx = None
        if cfg.include_speed:
            speed_idx = i
            i += 1
        wall_idx = obstacle_idx = None
        if cfg.include_clearances:
            wall_idx = i
            obstacle_idx = i + 1
            i += 2
        if (self.env._model == "car") and cfg.include_heading:
            i += 1
        if (self.env._model == "car") and cfg.include_steering_angle:
            i += 1
        obstacle_feats_start = i
        i += 3 * cfg.n_obstacles
        lidar_start = lidar_end = None
        if cfg.sensor_in_obs:
            lidar_start = i
            lidar_end = i + cfg.num_rays
            i = lidar_end
        return {
            "speed_idx": speed_idx,
            "wall_clear_idx": wall_idx,
            "obs_clear_idx": obstacle_idx,
            "obs_triplets_start": obstacle_feats_start,
            "lidar_start": lidar_start,
            "lidar_end": lidar_end,
            "obs_dim": i,
        }

    def h(self, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Return h(s) and diagnostics."""
        assert obs.ndim == 1, "obs must be 1D array"
        lid_s, lid_e = self._idx["lidar_start"], self._idx["lidar_end"]
        wall_i, obs_i = self._idx["wall_clear_idx"], self._idx["obs_clear_idx"]

        if lid_s is not None and lid_e is not None:
            d_ray = float(np.min(obs[lid_s:lid_e]))
        else:
            d_ray = float("inf")

        comps = [d_ray]
        if wall_i is not None:
            comps.append(float(obs[wall_i]))
        if obs_i is not None:
            comps.append(float(obs[obs_i]))
        d_clear = float(np.min(comps)) if len(comps) else d_ray

        d_safe = self.cfg.d_safe_car if self.env._model == "car" else self.cfg.d_safe_point
        h_val = float(self.cfg.alpha_cbf * (d_clear - d_safe))
        return h_val, {"d_ray": d_ray, "d_clear": d_clear, "d_safe": d_safe}
