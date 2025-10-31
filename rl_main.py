"""
ModularCar2DEnv — a lightweight, modular RL environment for a point-mass car
or an optional bicycle-model vehicle with throttle/steering controls.

Features
- 2D continuous space with physics: mass, engine lag (takes time to accel/decel), viscous drag.
- Continuous action space: desired acceleration (point-mass) or normalized throttle/steer rate (car model) with engine lag.
- Dynamic obstacles: circles randomized each episode OR supplied by a user-defined generator callback.
- Goal-reaching task with configurable reward shaping.
- Gymnasium-compatible API (reset, step, render, seed/np_random).
- Clear config dataclass so you can tweak everything without touching logic.
- Simple Matplotlib renderer (optional).

Dependencies
- Python 3.9+
- gymnasium
- numpy
- matplotlib (only if you call render())

Install
    pip install gymnasium numpy matplotlib

Quickstart
    import gymnasium as gym
    import numpy as np
    from ModularCar2DEnv import ModularCar2DEnv, EnvConfig

    cfg = EnvConfig()
    env = ModularCar2DEnv(cfg)
    obs, info = env.reset(seed=42)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # env.render()  # optional, slower

    env.close()

Notes
- `EnvConfig(vehicle_model="point")` (default) matches the original point-mass dynamics.
- Switch to `EnvConfig(vehicle_model="car")` for throttle/brake + steering using a bicycle model; tweak wheelbase, steering limits, etc. via config fields.
- You can plug this into SB3 by registering it with Gymnasium's registry or by constructing env=ModularCar2DEnv(EnvConfig()).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


@dataclass
class EnvConfig:
    # World
    world_size: Tuple[float, float] = (20.0, 20.0)  # width, height (meters)
    dt: float = 0.05  # seconds per step
    max_steps: int = 600

    # Vehicle physics
    mass: float = 1200.0  # kg
    max_accel: float = 4.0  # m/s^2, magnitude cap of commanded accel
    engine_tau: float = 0.3  # s, time constant for accel response (larger = slower response)
    drag_coeff: float = 0.15  # linear velocity drag (kg/s effectively)
    speed_limit: float = 20.0  # m/s, for observation/action clipping (not a hard sim limit)

    # Task
    start_pos: Tuple[float, float] = (-8.0, -8.0)
    start_vel: Tuple[float, float] = (0.0, 0.0)
    start_heading: float = 0.0  # radians
    start_speed: Optional[float] = None  # overrides start_vel magnitude when set
    start_steering: float = 0.0  # radians

    # Vehicle model
    vehicle_model: str = "point"  # "point" or "car"
    allow_reverse: bool = False
    wheelbase: float = 2.7  # meters, front-to-rear axle distance
    max_throttle: float = 3.5  # m/s^2 (longitudinal accel)
    max_brake: float = 6.0  # m/s^2 (positive number, applied as decel)
    max_steer: float = 0.6  # radians (~34 degrees)
    max_steer_rate: float = 2.0  # radians per second
    goal_pos: Tuple[float, float] = (8.0, 8.0)
    goal_radius: float = 0.75

    # Obstacles
    n_obstacles: int = 6
    obstacle_radius_range: Tuple[float, float] = (0.6, 1.8)
    min_obs_goal_margin: float = 1.0  # keep obstacles away from start/goal by this margin
    obstacle_generator: Optional[Callable[[np.random.Generator, "EnvConfig"], List[Tuple[float, float, float]]]] = None

    # Rewards
    r_goal: float = 200.0
    r_collision: float = -150.0
    step_penalty: float = -0.1
    control_penalty: float = 0.02  # penalize accel magnitude
    distance_scale: float = 0.5  # shaping: -distance_to_goal * scale

    # Termination
    out_of_bounds_terminates: bool = True

    # Observation layout
    # obs = [px, py, vx, vy, ax_applied_x, ax_applied_y, goal_dx, goal_dy, speed]
    #      + (heading, steering_angle if vehicle_model=="car")
    #      + K * [obs_i_dx, obs_i_dy, obs_i_r] padded to n_obstacles
    include_applied_accel: bool = True
    include_speed: bool = True
    include_heading: bool = True   # appended when vehicle_model == "car"
    include_steering_angle: bool = True  # appended when vehicle_model == "car"

    # Range sensor (lidar-like)
    sensor_in_obs: bool = True
    num_rays: int = 36               # number of beams
    fov_deg: float = 360.0           # field of view
    max_range: float = 10.0          # max beam length (meters)
    sensor_frame: str = "global"     # "global" or "velocity"-aligned frame
    include_sensor_in_info: bool = False

    # Rendering
    render_agent_trail: bool = True


class ModularCar2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig = EnvConfig()):
        super().__init__()
        self.cfg = config
        self.dt = self.cfg.dt
        self.world_w, self.world_h = self.cfg.world_size
        self._model = self.cfg.vehicle_model.lower()
        if self._model not in {"point", "car"}:
            raise ValueError(f"Unsupported vehicle_model '{self.cfg.vehicle_model}'")

        if self._model == "point":
            high_act = np.array([self.cfg.max_accel, self.cfg.max_accel], dtype=np.float32)
            self.action_space = spaces.Box(low=-high_act, high=high_act, dtype=np.float32)
        else:
            high_act = np.ones(2, dtype=np.float32)
            self.action_space = spaces.Box(low=-high_act, high=high_act, dtype=np.float32)

        # Observation bound heuristics
        k = self.cfg.n_obstacles
        base_dim = 4  # px, py, vx, vy
        if self.cfg.include_applied_accel:
            base_dim += 2
        base_dim += 2  # goal dx, dy
        if self.cfg.include_speed:
            base_dim += 1
        if self._model == "car" and self.cfg.include_heading:
            base_dim += 1
        if self._model == "car" and self.cfg.include_steering_angle:
            base_dim += 1
        # each obstacle contributes (dx, dy, r)
        obs_dim = base_dim + 3 * k
        if self.cfg.sensor_in_obs:
            obs_dim += self.cfg.num_rays

        high = np.full((obs_dim,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._np_random: np.random.Generator = np.random.default_rng()
        self.state = None  # (px, py, vx, vy, ax, ay)
        self._applied_accel = np.zeros(2, dtype=np.float32)
        self._applied_long = 0.0  # longitudinal accel for car model
        self._heading = float(self.cfg.start_heading)
        self._steer = float(np.clip(self.cfg.start_steering, -self.cfg.max_steer, self.cfg.max_steer))
        self._speed = 0.0
        self._long_speed = 0.0
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.steps = 0

        # Rendering
        self._fig = None
        self._ax = None
        self._trail = []

    def seed(self, seed: Optional[int] = None):
        self.reset(seed=seed)

    def _sample_obstacles(self) -> List[Tuple[float, float, float]]:
        if self.cfg.obstacle_generator is not None:
            return self.cfg.obstacle_generator(self._np_random, self.cfg)

        obs = []
        tries = 0
        while len(obs) < self.cfg.n_obstacles and tries < 2000:
            tries += 1
            r = float(self._np_random.uniform(*self.cfg.obstacle_radius_range))
            x = float(self._np_random.uniform(-self.world_w/2 + r, self.world_w/2 - r))
            y = float(self._np_random.uniform(-self.world_h/2 + r, self.world_h/2 - r))
            # Keep clear of start and goal
            if np.hypot(x - self.cfg.start_pos[0], y - self.cfg.start_pos[1]) < (r + self.cfg.min_obs_goal_margin):
                continue
            if np.hypot(x - self.cfg.goal_pos[0], y - self.cfg.goal_pos[1]) < (r + self.cfg.min_obs_goal_margin):
                continue
            # Avoid heavy overlap with existing obstacles
            ok = True
            for (ox, oy, or_) in obs:
                if np.hypot(x - ox, y - oy) < (r + or_ + 0.2):
                    ok = False
                    break
            if ok:
                obs.append((x, y, r))
        return obs

    def _within_bounds(self, p: np.ndarray) -> bool:
        return (-self.world_w/2 <= p[0] <= self.world_w/2) and (-self.world_h/2 <= p[1] <= self.world_h/2)

    def _collides(self, p: np.ndarray) -> bool:
        for (ox, oy, r) in self.obstacles:
            if np.hypot(p[0]-ox, p[1]-oy) <= r:
                return True
        return False

    def _goal_reached(self, p: np.ndarray) -> bool:
        gx, gy = self.cfg.goal_pos
        return np.hypot(p[0]-gx, p[1]-gy) <= self.cfg.goal_radius

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            # reseed from entropy to avoid shared RNG when vectorized
            self._np_random = np.random.default_rng()

        self.obstacles = self._sample_obstacles()
        px, py = self.cfg.start_pos
        self._applied_accel = np.zeros(2, dtype=np.float32)
        self._applied_long = 0.0

        if self._model == "car":
            heading = float(self.cfg.start_heading)
            steer = float(np.clip(self.cfg.start_steering, -self.cfg.max_steer, self.cfg.max_steer))
            if self.cfg.start_speed is not None:
                long_speed = float(self.cfg.start_speed)
            else:
                vx0, vy0 = self.cfg.start_vel
                if np.hypot(vx0, vy0) > 1e-6:
                    heading = float(np.arctan2(vy0, vx0))
                long_speed = float(np.cos(heading) * vx0 + np.sin(heading) * vy0)
            vx = float(long_speed * np.cos(heading))
            vy = float(long_speed * np.sin(heading))
            self._heading = heading
            self._steer = steer
            self._long_speed = long_speed
            self._speed = float(abs(long_speed))
        else:
            vx, vy = self.cfg.start_vel
            self._speed = float(np.hypot(vx, vy))
            if self._speed > 1e-6:
                self._heading = float(np.arctan2(vy, vx))
            else:
                self._heading = float(self.cfg.start_heading)
            self._steer = 0.0
            self._long_speed = float(np.cos(self._heading) * vx + np.sin(self._heading) * vy)

        self.state = np.array([px, py, vx, vy, 0.0, 0.0], dtype=np.float32)
        self.steps = 0
        self._trail = []
        obs = self._get_obs()

        gx, gy = self.cfg.goal_pos
        dist = float(np.hypot(px - gx, py - gy))
        speed = float(np.hypot(vx, vy))
        info = {
            "distance_to_goal": dist,
            "speed": speed,
            "applied_accel": self._applied_accel.copy(),
            "terminated_reason": None,
        }
        if self._model == "car":
            info["heading"] = self._heading
            info["steering_angle"] = self._steer
            info["longitudinal_speed"] = float(self._long_speed)
        if self.cfg.include_sensor_in_info and self.cfg.sensor_in_obs:
            pos = np.array([px, py], dtype=np.float32)
            vel = np.array([vx, vy], dtype=np.float32)
            info["lidar"] = self._compute_lidar(pos, vel)
        return obs, info

    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        longitudinal_accel = None

        if self._model == "car":
            throttle_cmd = float(np.clip(action[0], -1.0, 1.0))
            steer_cmd = float(np.clip(action[1], -1.0, 1.0))

            if throttle_cmd >= 0.0:
                desired_long = throttle_cmd * self.cfg.max_throttle
            else:
                desired_long = throttle_cmd * self.cfg.max_brake  # negative when braking

            tau = max(self.cfg.engine_tau, 1e-6)
            self._applied_long = self._applied_long + (self.dt / tau) * (desired_long - self._applied_long)

            drag_coeff = self.cfg.drag_coeff / max(self.cfg.mass, 1e-6)
            long_speed_prev = self._long_speed
            a_long = self._applied_long - drag_coeff * long_speed_prev

            long_speed_new = long_speed_prev + a_long * self.dt
            if not self.cfg.allow_reverse:
                long_speed_new = max(0.0, long_speed_new)
            else:
                long_speed_new = float(np.clip(long_speed_new, -self.cfg.speed_limit, self.cfg.speed_limit))
            long_speed_mid = 0.5 * (long_speed_prev + long_speed_new)

            steer_rate = steer_cmd * self.cfg.max_steer_rate
            steer_new = float(np.clip(self._steer + steer_rate * self.dt, -self.cfg.max_steer, self.cfg.max_steer))

            wheelbase = max(self.cfg.wheelbase, 1e-6)
            heading_rate = long_speed_mid * np.tan(steer_new) / wheelbase if abs(long_speed_mid) > 1e-6 else 0.0
            heading_new = self._heading + heading_rate * self.dt
            heading_new = float(np.arctan2(np.sin(heading_new), np.cos(heading_new)))  # wrap to [-pi, pi]
            heading_mid = self._heading + 0.5 * (heading_new - self._heading)

            px, py, _, _, _, _ = self.state
            direction_mid = np.array([np.cos(heading_mid), np.sin(heading_mid)], dtype=np.float32)
            p_prev = np.array([px, py], dtype=np.float32)
            displacement = direction_mid * (long_speed_mid * self.dt)
            p_new = p_prev + displacement

            direction_new = np.array([np.cos(heading_new), np.sin(heading_new)], dtype=np.float32)
            v_new = direction_new * long_speed_new
            applied_vec = direction_new * a_long
            longitudinal_accel = float(a_long)

            self.state = np.array(
                [p_new[0], p_new[1], v_new[0], v_new[1], applied_vec[0], applied_vec[1]],
                dtype=np.float32,
            )
            self._applied_accel = applied_vec.astype(np.float32)
            self._heading = heading_new
            self._steer = steer_new
            self._long_speed = float(long_speed_new)
            self._speed = float(abs(long_speed_new))
        else:
            # sanitize action
            a_cmd = np.clip(action, -self.cfg.max_accel, self.cfg.max_accel)

            # engine lag: a_applied <- a_applied + (dt/tau) * (a_cmd - a_applied)
            tau = max(self.cfg.engine_tau, 1e-6)
            self._applied_accel = self._applied_accel + (self.dt / tau) * (a_cmd - self._applied_accel)

            # physics integration (semi-implicit Euler):
            # F = m * a_applied - drag * v
            px, py, vx, vy, _, _ = self.state
            v = np.array([vx, vy], dtype=np.float32)
            drag = - (self.cfg.drag_coeff / max(self.cfg.mass, 1e-6)) * v
            a = self._applied_accel + drag

            v_new = v + a * self.dt
            p_new = np.array([px, py], dtype=np.float32) + v_new * self.dt

            self.state = np.array([p_new[0], p_new[1], v_new[0], v_new[1], a[0], a[1]], dtype=np.float32)
            self._speed = float(np.linalg.norm(v_new))
            if self._speed > 1e-6:
                self._heading = float(np.arctan2(v_new[1], v_new[0]))
            self._steer = 0.0
            self._long_speed = float(self._speed)

        if self.cfg.render_agent_trail:
            self._trail.append((self.state[0], self.state[1]))

        # rewards and termination
        p = np.array([self.state[0], self.state[1]], dtype=np.float32)
        done_collision = self._collides(p)
        done_oob = self.cfg.out_of_bounds_terminates and (not self._within_bounds(p))
        done_goal = self._goal_reached(p)
        terminated = bool(done_collision or done_goal or done_oob)
        truncated = bool(self.steps >= self.cfg.max_steps and not terminated)

        # reward shaping
        gx, gy = self.cfg.goal_pos
        dist = float(np.hypot(p[0]-gx, p[1]-gy))
        r = self.cfg.step_penalty
        r += - self.cfg.distance_scale * dist
        r += - self.cfg.control_penalty * float(np.linalg.norm(self._applied_accel))
        if done_collision:
            r += self.cfg.r_collision
        if done_goal:
            r += self.cfg.r_goal

        info = {
            "distance_to_goal": dist,
            "speed": float(self._speed),
            "applied_accel": self._applied_accel.copy(),
            "terminated_reason": "collision" if done_collision else ("goal" if done_goal else ("oob" if done_oob else None)),
        }
        if self._model == "car":
            info["heading"] = self._heading
            info["steering_angle"] = self._steer
            info["longitudinal_speed"] = float(self._long_speed)
            if longitudinal_accel is not None:
                info["longitudinal_accel"] = longitudinal_accel

        return self._get_obs(), float(r), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        px, py, vx, vy, ax, ay = self.state
        gx, gy = self.cfg.goal_pos
        base = [px, py, vx, vy]
        if self.cfg.include_applied_accel:
            base += [ax, ay]
        base += [gx - px, gy - py]
        if self.cfg.include_speed:
            if self._model == "car":
                speed_val = self._long_speed if self.cfg.allow_reverse else self._speed
                base += [float(speed_val)]
            else:
                base += [float(np.hypot(vx, vy))]
        if self._model == "car":
            if self.cfg.include_heading:
                base += [float(self._heading)]
            if self.cfg.include_steering_angle:
                base += [float(self._steer)]

        # Flatten obstacle info as relative vectors and radii, padded to n_obstacles
        obs_feats: List[float] = []
        for (ox, oy, r) in self.obstacles[: self.cfg.n_obstacles]:
            obs_feats += [ox - px, oy - py, r]
        # pad
        while len(obs_feats) < 3 * self.cfg.n_obstacles:
            obs_feats += [0.0, 0.0, 0.0]

        sensor = []
        if self.cfg.sensor_in_obs:
            sensor = self._compute_lidar(np.array([px, py], dtype=np.float32), np.array([vx, vy], dtype=np.float32)).tolist()
        return np.asarray(base + obs_feats + sensor, dtype=np.float32)

    # --- Lidar / range sensor helpers ---
    def _compute_lidar(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Ray-cast distances to circular obstacles within max_range.
        Returns an array [d0..d_{num_rays-1}] in meters.
        Frame:
          - 'global': 0 deg points along +x, increases CCW
          - 'velocity': 0 deg aligns with current velocity direction (if speed<1e-6, falls back to global)
        """
        n = int(max(1, self.cfg.num_rays))
        fov = np.deg2rad(self.cfg.fov_deg)
        max_r = float(self.cfg.max_range)

        # starting angle
        if self.cfg.sensor_frame == "velocity":
            speed = float(np.hypot(vel[0], vel[1]))
            if speed > 1e-6:
                base_ang = float(np.arctan2(vel[1], vel[0]))
            elif self._model == "car":
                base_ang = float(self._heading)
            else:
                base_ang = 0.0
        else:
            base_ang = 0.0

        if n == 1:
            angles = np.array([base_ang], dtype=np.float32)
        else:
            angles = base_ang + np.linspace(-fov/2, fov/2, n, dtype=np.float32)

        dists = np.full((n,), max_r, dtype=np.float32)
        # Ray-circle intersection per beam
        for i, ang in enumerate(angles):
            d = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)  # unit dir
            t_min = max_r
            for (cx, cy, r) in self.obstacles:
                m = pos - np.array([cx, cy], dtype=np.float32)
                b = 2.0 * float(np.dot(d, m))
                c = float(np.dot(m, m) - r*r)
                disc = b*b - 4.0*c
                if disc < 0.0:
                    continue
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / 2.0
                t2 = (-b + sqrt_disc) / 2.0
                for t in (t1, t2):
                    if 0.0 <= t <= t_min:
                        t_min = t
            dists[i] = min(t_min, max_r)
        return dists

    def render(self, mode: str = "human"):
        if not _HAS_PLT:
            raise RuntimeError("matplotlib is required for rendering")

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._ax.set_aspect('equal', 'box')
            self._ax.set_xlim(-self.world_w/2, self.world_w/2)
            self._ax.set_ylim(-self.world_h/2, self.world_h/2)
            self._ax.set_title("ModularCar2DEnv")

        self._ax.clear()
        self._ax.set_aspect('equal', 'box')
        self._ax.set_xlim(-self.world_w/2, self.world_w/2)
        self._ax.set_ylim(-self.world_h/2, self.world_h/2)
        self._ax.grid(True, alpha=0.3)

        # draw obstacles
        for (ox, oy, r) in self.obstacles:
            circ = plt.Circle((ox, oy), r, color='red', alpha=0.4)
            self._ax.add_patch(circ)

        # draw start and goal
        sx, sy = self.cfg.start_pos
        gx, gy = self.cfg.goal_pos
        self._ax.scatter([sx], [sy], marker='s', s=60, label='start')
        goal = plt.Circle((gx, gy), self.cfg.goal_radius, color='green', alpha=0.4)
        self._ax.add_patch(goal)

        # draw agent
        px, py, vx, vy, _, _ = self.state
        self._ax.scatter([px], [py], s=50, label='agent')
        self._ax.arrow(px, py, vx*0.2, vy*0.2, head_width=0.3, length_includes_head=True, alpha=0.6)
        if self._model == "car":
            heading_vec = np.array([np.cos(self._heading), np.sin(self._heading)], dtype=np.float32)
            self._ax.arrow(
                px,
                py,
                heading_vec[0] * 1.2,
                heading_vec[1] * 1.2,
                head_width=0.35,
                length_includes_head=True,
                alpha=0.8,
                color='blue',
                label='heading',
            )

        if self.cfg.render_agent_trail and len(self._trail) > 1:
            xs, ys = zip(*self._trail)
            self._ax.plot(xs, ys, alpha=0.5)

        self._ax.legend(loc='upper left')
        self._fig.canvas.draw()
        if mode == "rgb_array":
            # Convert canvas to image array
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img
        elif mode == "human":
            plt.pause(0.001)
        else:
            raise NotImplementedError

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None


# -------------------- Customization examples --------------------
# 1) Custom obstacle generator with moving obstacles or curriculum
#
# def my_obstacle_gen(rng: np.random.Generator, cfg: EnvConfig):
#     obs = []
#     for i in range(cfg.n_obstacles):
#         r = float(rng.uniform(*cfg.obstacle_radius_range))
#         x = float(rng.uniform(-cfg.world_size[0]/2 + r, cfg.world_size[0]/2 - r))
#         y = float(rng.uniform(-cfg.world_size[1]/2 + r, cfg.world_size[1]/2 - r))
#         obs.append((x, y, r))
#     return obs
#
# env = ModularCar2DEnv(EnvConfig(obstacle_generator=my_obstacle_gen))
#
# 2) Need tighter car controls? Try tweaking EnvConfig(vehicle_model="car",
#       max_throttle=..., max_brake=..., max_steer=..., max_steer_rate=..., wheelbase=...).
