"""
ModularCar2DEnv — a lightweight, modular RL environment for a point-mass car
or an optional bicycle-model vehicle with throttle/steering controls.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import matplotlib.pyplot as plt

    _HAS_PLT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PLT = False

from .config import EnvConfig
from .models import BaseVehicleModel, BicycleVehicleModel, PointVehicleModel
from .obstacles import ObstacleField
from .sensors import RangeSensor
from .state import VehicleState


class ModularCar2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig = EnvConfig()):
        super().__init__()
        self.cfg = config
        self.dt = self.cfg.dt
        self.world_w, self.world_h = self.cfg.world_size
        self.x_min, self.x_max, self.y_min, self.y_max = self._world_bounds()
        self._model = self.cfg.vehicle_model.lower()
        if self._model not in {"point", "car"}:
            raise ValueError(f"Unsupported vehicle_model '{self.cfg.vehicle_model}'")

        self._vehicle_model: BaseVehicleModel = (
            PointVehicleModel(self.cfg) if self._model == "point" else BicycleVehicleModel(self.cfg)
        )
        self.action_space = self._vehicle_model.action_space()
        obs_dim = self._compute_obs_dim()
        high = np.full((obs_dim,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._np_random: np.random.Generator = np.random.default_rng()
        self._state: Optional[VehicleState] = None
        self.state: Optional[np.ndarray] = None  # legacy compatibility
        self._obstacle_field = ObstacleField(
            self.cfg,
            self.world_w,
            self.world_h,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
        )
        self.obstacles: List[Tuple[float, float, float]] = self._obstacle_field.obstacles
        self._sensor = RangeSensor(self.cfg)
        self.steps = 0

        # Rendering
        self._fig = None
        self._ax = None
        self._trail: List[Tuple[float, float]] = []

    def seed(self, seed: Optional[int] = None):
        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

        self._refresh_obstacles()
        self._state = self._vehicle_model.reset_state()
        self.state = self._state.as_vector()
        self.steps = 0
        self._trail = []

        obs = self._get_obs()
        gx, gy = self.cfg.goal_pos
        px, py = self._state.position
        dist = float(np.hypot(px - gx, py - gy))
        info = self._build_info(dist, terminated_reason=None)
        return obs, info

    def step(self, action: np.ndarray):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        self._state = self._vehicle_model.step(self._state, action, self.dt)
        if self.cfg.solid_walls:
            self._enforce_wall_bounds()
        hit_obstacle = False
        if self.cfg.solid_obstacles:
            hit_obstacle = self._resolve_obstacle_penetration()
        self.state = self._state.as_vector()

        if self.cfg.render_agent_trail:
            self._trail.append((float(self._state.position[0]), float(self._state.position[1])))

        position = self._state.position
        hit_obstacle = self._obstacle_field.collides(position) or hit_obstacle
        done_collision = hit_obstacle and self.cfg.obstacle_collisions_terminate
        done_oob = self.cfg.out_of_bounds_terminates and (not self._within_bounds(position))
        done_goal = self._goal_reached(position)
        terminated = bool(done_collision or done_goal or done_oob)
        truncated = bool(self.steps >= self.cfg.max_steps and not terminated)

        gx, gy = self.cfg.goal_pos
        dist = float(np.hypot(position[0] - gx, position[1] - gy))
        control_cost = float(np.linalg.norm(self._state.accel))
        reward = (
            self.cfg.step_penalty
            - self.cfg.distance_scale * dist
            - self.cfg.control_penalty * control_cost
        )
        if hit_obstacle:
            reward += self.cfg.r_collision
        if done_goal:
            reward += self.cfg.r_goal

        reason = "collision" if done_collision else ("goal" if done_goal else ("oob" if done_oob else None))
        info = self._build_info(dist, terminated_reason=reason)
        return self._get_obs(), float(reward), terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _enforce_wall_bounds(self):
        if self._state is None:
            return
        px = float(self._state.position[0])
        py = float(self._state.position[1])
        clamped_x = float(np.clip(px, self.x_min, self.x_max))
        clamped_y = float(np.clip(py, self.y_min, self.y_max))
        if (clamped_x == px) and (clamped_y == py):
            return
        self._state.position[0] = clamped_x
        self._state.position[1] = clamped_y

    def _resolve_obstacle_penetration(self) -> bool:
        if self._state is None or not self.obstacles:
            return False
        collided = False
        for _ in range(5):
            adjusted = False
            px = float(self._state.position[0])
            py = float(self._state.position[1])
            for (ox, oy, radius) in self.obstacles:
                dx = px - ox
                dy = py - oy
                dist = float(np.hypot(dx, dy))
                if dist < radius:
                    collided = True
                    if dist <= 1e-6:
                        normal = np.array([1.0, 0.0], dtype=np.float32)
                    else:
                        normal = np.array([dx / dist, dy / dist], dtype=np.float32)
                    contact = np.array([ox, oy], dtype=np.float32) + normal * radius
                    self._state.position[:] = contact
                    vel = self._state.velocity
                    vn = float(vel[0] * normal[0] + vel[1] * normal[1])
                    if vn < 0.0:
                        self._state.velocity[:] = vel - vn * normal
                    px = float(contact[0])
                    py = float(contact[1])
                    adjusted = True
            if not adjusted:
                break
        return collided

    def _world_bounds(self) -> Tuple[float, float, float, float]:
        if self.cfg.world_origin is not None:
            x_min = float(self.cfg.world_origin[0])
            y_min = float(self.cfg.world_origin[1])
            return x_min, x_min + self.world_w, y_min, y_min + self.world_h
        half_w = self.world_w / 2.0
        half_h = self.world_h / 2.0
        return -half_w, half_w, -half_h, half_h

    def _compute_obs_dim(self) -> int:
        base_dim = 4  # px, py, vx, vy
        if self.cfg.include_applied_accel:
            base_dim += 2
        base_dim += 2  # goal dx, dy
        if self.cfg.include_speed:
            base_dim += 1
        if self.cfg.include_clearances:
            base_dim += 2
        if self._model == "car" and self.cfg.include_heading:
            base_dim += 1
        if self._model == "car" and self.cfg.include_steering_angle:
            base_dim += 1

        obs_dim = base_dim + 3 * self.cfg.n_obstacles
        if self.cfg.sensor_in_obs:
            obs_dim += self.cfg.num_rays
        return obs_dim

    def _refresh_obstacles(self):
        self.obstacles = self._obstacle_field.regenerate(self._np_random)

    def _within_bounds(self, pos: np.ndarray) -> bool:
        return (self.x_min <= pos[0] <= self.x_max) and (self.y_min <= pos[1] <= self.y_max)

    def _goal_reached(self, pos: np.ndarray) -> bool:
        gx, gy = self.cfg.goal_pos
        return np.hypot(pos[0] - gx, pos[1] - gy) <= self.cfg.goal_radius

    def _build_info(self, distance: float, terminated_reason: Optional[str]) -> dict:
        if self._state is None:
            raise RuntimeError("State is undefined. Call reset first.")

        if self._model == "car":
            speed_val = self._state.long_speed if self.cfg.allow_reverse else abs(self._state.long_speed)
        else:
            speed_val = self._state.speed

        wall_clearance, obstacle_clearance = self._obstacle_field.min_clearances(
            float(self._state.position[0]),
            float(self._state.position[1]),
        )

        info = {
            "distance_to_goal": distance,
            "speed": float(speed_val),
            "applied_accel": self._state.accel.copy(),
            "wall_clearance": wall_clearance,
            "obstacle_clearance": obstacle_clearance,
            "terminated_reason": terminated_reason,
        }
        if self._model == "car":
            info["heading"] = float(self._state.heading)
            info["steering_angle"] = float(self._state.steer)
            info["longitudinal_speed"] = float(self._state.long_speed)
            info["longitudinal_accel"] = float(self._state.longitudinal_accel)
        if self.cfg.include_sensor_in_info and self.cfg.sensor_in_obs:
            info["lidar"] = self._sensor.cast(
                self._state.position,
                self._state.velocity,
                self._state.heading,
                self.obstacles,
            )
        return info

    def _get_obs(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("State is undefined. Call reset first.")

        px, py = self._state.position
        vx, vy = self._state.velocity
        ax, ay = self._state.accel
        gx, gy = self.cfg.goal_pos
        base = [px, py, vx, vy]

        if self.cfg.include_applied_accel:
            base += [ax, ay]
        base += [gx - px, gy - py]

        if self.cfg.include_speed:
            if self._model == "car":
                speed_val = self._state.long_speed if self.cfg.allow_reverse else abs(self._state.long_speed)
                base += [float(speed_val)]
            else:
                base += [self._state.speed]

        if self.cfg.include_clearances:
            wall_clearance, obstacle_clearance = self._obstacle_field.min_clearances(float(px), float(py))
            base += [wall_clearance, obstacle_clearance]

        if self._model == "car":
            if self.cfg.include_heading:
                base += [float(self._state.heading)]
            if self.cfg.include_steering_angle:
                base += [float(self._state.steer)]

        obs_feats = self._obstacle_field.relative_features(float(px), float(py), self.cfg.n_obstacles)

        sensor: List[float] = []
        if self.cfg.sensor_in_obs:
            sensor = self._sensor.cast(
                self._state.position,
                self._state.velocity,
                self._state.heading,
                self.obstacles,
            ).tolist()

        return np.asarray(base + obs_feats + sensor, dtype=np.float32)

    # --------------------------------------------------------------------- #
    # Rendering
    # --------------------------------------------------------------------- #
    def render(self, mode: str = "human"):
        if not _HAS_PLT:
            raise RuntimeError("matplotlib is required for rendering")
        if self._state is None:
            raise RuntimeError("Call reset() before render().")

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.set_aspect("equal", "box")
            self._ax.set_xlim(self.x_min, self.x_max)
            self._ax.set_ylim(self.y_min, self.y_max)
            self._ax.set_title("ModularCar2DEnv")

        self._ax.clear()
        self._ax.set_aspect("equal", "box")
        self._ax.set_xlim(self.x_min, self.x_max)
        self._ax.set_ylim(self.y_min, self.y_max)
        self._ax.grid(True, alpha=0.3)

        for (ox, oy, r) in self.obstacles:
            circ = plt.Circle((ox, oy), r, color="red", alpha=0.4)
            self._ax.add_patch(circ)

        sx, sy = self.cfg.start_pos
        gx, gy = self.cfg.goal_pos
        self._ax.scatter([sx], [sy], marker="s", s=60, label="start")
        goal = plt.Circle((gx, gy), self.cfg.goal_radius, color="green", alpha=0.4)
        self._ax.add_patch(goal)

        px, py = self._state.position
        vx, vy = self._state.velocity
        self._ax.scatter([px], [py], s=50, label="agent")
        self._ax.arrow(px, py, vx * 0.2, vy * 0.2, head_width=0.3, length_includes_head=True, alpha=0.6)

        if self._model == "car":
            heading_vec = np.array([np.cos(self._state.heading), np.sin(self._state.heading)], dtype=np.float32)
            self._ax.arrow(
                px,
                py,
                heading_vec[0] * 1.2,
                heading_vec[1] * 1.2,
                head_width=0.35,
                length_includes_head=True,
                alpha=0.8,
                color="blue",
                label="heading",
            )

        if self.cfg.render_agent_trail and len(self._trail) > 1:
            xs, ys = zip(*self._trail)
            self._ax.plot(xs, ys, alpha=0.5)

        self._ax.legend(loc="upper left")
        self._fig.canvas.draw()
        if mode == "rgb_array":
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img
        elif mode == "human":
            plt.pause(0.001)
        else:
            raise NotImplementedError(f"Unsupported render mode {mode}")

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None
