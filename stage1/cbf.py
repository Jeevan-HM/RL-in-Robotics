from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .config import CBFConfig

__all__ = ["SafetyCBF"]


class SafetyCBF:
    """Computes the discrete-time Control Barrier Function h(s) from env observations.

    IMPROVED: Adds goal-aware safety constraints that relax the safety buffer
    when approaching the goal, preventing the robot from stopping too early.
    """

    def __init__(self, env: gym.Env, cfg: Optional[CBFConfig] = None):
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

    def _get_distance_to_goal(self, obs: np.ndarray) -> float:
        """Extract distance to goal from observation."""
        # Position is in first 2 elements
        px, py = float(obs[0]), float(obs[1])

        # Try to get goal position from environment
        if hasattr(self.env, "goal_position"):
            gx, gy = self.env.goal_position
        elif hasattr(self.env, "cfg") and hasattr(self.env.cfg, "goal_pos"):
            gx, gy = self.env.cfg.goal_pos
        elif hasattr(self.env.unwrapped, "goal_position"):
            gx, gy = self.env.unwrapped.goal_position
        elif hasattr(self.env.unwrapped, "goal_pos"):
            gx, gy = self.env.unwrapped.goal_pos
        else:
            # Default fallback - assume far from goal
            return float("inf")

        return float(np.hypot(px - gx, py - gy))

    def h(self, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Return h(s) and diagnostics.

        IMPROVED: Implements goal-aware safety that relaxes constraints near the goal.
        This prevents the robot from stopping before reaching the goal due to
        overly conservative safety constraints.
        """
        assert obs.ndim == 1, "obs must be 1D array"
        lid_s, lid_e = self._idx["lidar_start"], self._idx["lidar_end"]
        wall_i, obs_i = self._idx["wall_clear_idx"], self._idx["obs_clear_idx"]

        # Get clearance distances
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

        # Get base safety distance
        d_safe_base = (
            self.cfg.d_safe_car if self.env._model == "car" else self.cfg.d_safe_point
        )

        # IMPROVEMENT: Apply goal-aware safety relaxation
        distance_to_goal = self._get_distance_to_goal(obs)

        if distance_to_goal < self.cfg.goal_proximity_threshold:
            # Gradually relax safety as we approach the goal
            # Linear interpolation from full safety to relaxed safety
            proximity_factor = distance_to_goal / self.cfg.goal_proximity_threshold
            relaxation = (
                self.cfg.goal_safety_relaxation
                + (1 - self.cfg.goal_safety_relaxation) * proximity_factor
            )
            d_safe = d_safe_base * relaxation
        else:
            d_safe = d_safe_base

        # Compute CBF value
        h_val = float(self.cfg.alpha_cbf * (d_clear * d_clear - d_safe * d_safe))

        return h_val, {
            "d_ray": d_ray,
            "d_clear": d_clear,
            "d_safe": d_safe,
            "d_safe_base": d_safe_base,
            "distance_to_goal": distance_to_goal,
        }
