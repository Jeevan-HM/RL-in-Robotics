"""
Goal-Oriented Car Navigation Environment with Realistic Physics
============================================================

Features:
- Realistic car dynamics (bicycle model with actual vehicle parameters)
- Goal-oriented navigation with waypoints
- Enhanced visualization with car sprite, trails, and environment details
- Safety-focused design with CBF integration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class CarPhysicsParams:
    """Physically realistic car parameters based on typical sedan/compact car."""

    # Vehicle dimensions (meters)
    length: float = 4.5  # Toyota Camry: ~4.9m, Honda Civic: ~4.6m
    width: float = 1.8  # Typical sedan width
    wheelbase: float = 2.7  # Distance between front and rear axles

    # Mass and inertia
    mass: float = 1500.0  # kg (typical sedan)
    moment_inertia: float = 2500.0  # kg⋅m² (approximate for sedan)

    # Tire and friction parameters
    tire_friction: float = 0.8  # Coefficient of friction (dry asphalt)
    rolling_resistance: float = 0.015  # Rolling resistance coefficient

    # Engine and drivetrain
    max_engine_force: float = 5000.0  # N (reasonable for ~150 HP car)
    max_brake_force: float = 8000.0  # N (modern braking systems)
    max_steering_angle: float = 35.0 * np.pi / 180  # radians (~35 degrees)
    max_steering_rate: float = 45.0 * np.pi / 180  # rad/s (~45 deg/s)

    # Aerodynamics
    drag_coefficient: float = 0.3  # Cd (modern sedan)
    frontal_area: float = 2.2  # m² (typical sedan)
    air_density: float = 1.225  # kg/m³ (sea level)

    # Speed limits
    max_speed: float = 80.0  # m/s (~108 km/h, ~67 mph)
    max_reverse_speed: float = 5.0  # m/s

    # Sensor parameters
    lidar_range: float = 15.0  # meters
    num_lidar_rays: int = 32  # Number of LIDAR beams
    lidar_fov: float = 270.0 * np.pi / 180  # Field of view (radians)

    # Safety parameters
    safety_radius: float = 2.5  # Minimum safe distance (meters)
    collision_radius: float = 2.0  # Collision detection radius


@dataclass
class NavigationConfig:
    """Configuration for the enhanced environment."""

    # World settings
    world_width: float = 100.0  # meters
    world_height: float = 100.0  # meters

    # Start and goal
    start_pos: Tuple[float, float] = (10.0, 10.0)
    start_heading: float = 0.0  # radians
    goal_pos: Tuple[float, float] = (90.0, 90.0)
    goal_radius: float = 3.0  # Goal reached when within this radius

    # Waypoints (optional intermediate goals)
    waypoints: List[Tuple[float, float]] = field(default_factory=list)

    # Obstacles
    n_static_obstacles: int = 8
    obstacle_min_radius: float = 2.0
    obstacle_max_radius: float = 5.0

    # Dynamic obstacles (moving cars/pedestrians)
    n_dynamic_obstacles: int = 3
    dynamic_obstacle_max_speed: float = 5.0  # m/s

    # Dynamic obstacles (moving cars/pedestrians)
    n_dynamic_obstacles: int = 3
    dynamic_obstacle_max_speed: float = 5.0  # m/s

    # Walls
    use_walls: bool = True

    # CBF compatibility attributes (required by stage1/cbf.py)
    include_applied_accel: bool = False
    include_speed: bool = False
    include_clearances: bool = False
    include_heading: bool = False
    include_steering_angle: bool = False
    sensor_in_obs: bool = True  # We have LIDAR
    num_rays: int = 32  # Same as car_params.num_lidar_rays
    n_obstacles: int = 8  # Will be set to n_static_obstacles

    # Walls
    use_walls: bool = True
    wall_thickness: float = 1.0

    # Episode settings
    dt: float = 0.2  # Simulation timestep (seconds)
    max_steps: int = 2000

    # Reward shaping
    goal_reward: float = 1000.0
    collision_penalty: float = -500.0
    time_penalty: float = -0.1  # Small penalty per step
    progress_reward_scale: float = 1.0  # Reward for getting closer to goal

    # Visualization
    render_fps: int = 30
    render_trail_length: int = 100  # Number of past positions to show
    show_lidar: bool = True
    show_safety_zones: bool = True


class Obstacle:
    """Represents a static or dynamic obstacle in the environment."""

    def __init__(
        self, pos: np.ndarray, radius: float, velocity: Optional[np.ndarray] = None
    ):
        self.pos = pos.copy()
        self.radius = radius
        self.velocity = velocity if velocity is not None else np.zeros(2)
        self.is_dynamic = np.linalg.norm(self.velocity) > 0.01

    def update(self, dt: float, bounds: Tuple[float, float]):
        """Update position for dynamic obstacles."""
        if self.is_dynamic:
            self.pos += self.velocity * dt

            # Bounce off walls
            if self.pos[0] <= self.radius or self.pos[0] >= bounds[0] - self.radius:
                self.velocity[0] *= -1
            if self.pos[1] <= self.radius or self.pos[1] >= bounds[1] - self.radius:
                self.velocity[1] *= -1

            # Keep within bounds
            self.pos[0] = np.clip(self.pos[0], self.radius, bounds[0] - self.radius)
            self.pos[1] = np.clip(self.pos[1], self.radius, bounds[1] - self.radius)


class GoalOrientedCarEnv(gym.Env):
    """
    Enhanced car navigation environment with realistic physics and visualization.

    State Space:
        - Position (x, y)
        - Velocity (vx, vy)
        - Heading (theta)
        - Angular velocity (omega)
        - Steering angle (delta)
        - Goal relative position (dx, dy)
        - LIDAR readings (32 rays)
        - Current waypoint index (if using waypoints)

    Action Space:
        - Throttle: [-1, 1] (negative for braking)
        - Steering rate: [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        car_params: Optional[CarPhysicsParams] = None,
        env_config: Optional[NavigationConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.car_params = car_params or CarPhysicsParams()
        self.cfg = env_config or NavigationConfig()
        self.render_mode = render_mode

        # CBF compatibility
        self._model = "car"
        self.cfg.n_obstacles = self.cfg.n_static_obstacles
        self.cfg.num_rays = self.car_params.num_lidar_rays

        # State variables
        self.pos = np.zeros(2)
        self.velocity = np.zeros(2)  # [vx, vy]
        self.heading = 0.0  # theta (radians)
        self.angular_velocity = 0.0  # omega (rad/s)
        self.steering_angle = 0.0  # delta (radians)
        self.speed = 0.0  # Scalar speed

        # Goal and waypoints
        self.goal_pos = np.array(self.cfg.goal_pos)
        self.waypoints = [np.array(wp) for wp in self.cfg.waypoints]
        self.current_waypoint_idx = 0

        # Obstacles
        self.obstacles: List[Obstacle] = []

        # Episode tracking
        self.steps = 0
        self.trajectory = []

        # Rendering
        self.screen = None
        self.clock = None

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        # Observation: [px, py, vx, vy, theta, omega, delta, goal_dx, goal_dy,
        #               waypoint_idx, lidar_readings...]
        obs_dim = 10 + self.car_params.num_lidar_rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reset car state
        self.pos = np.array(self.cfg.start_pos, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = self.cfg.start_heading
        self.angular_velocity = 0.0
        self.steering_angle = 0.0
        self.speed = 0.0

        # Reset waypoint tracking
        self.current_waypoint_idx = 0

        # Generate obstacles
        self._generate_obstacles()

        # Reset episode tracking
        self.steps = 0
        self.trajectory = [self.pos.copy()]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray):
        throttle = np.clip(action[0], -1.0, 1.0)
        steering_rate = np.clip(action[1], -1.0, 1.0)

        # Update car dynamics
        self._update_dynamics(throttle, steering_rate)

        # Update dynamic obstacles
        for obs in self.obstacles:
            obs.update(self.cfg.dt, (self.cfg.world_width, self.cfg.world_height))

        # Track trajectory
        self.trajectory.append(self.pos.copy())
        if len(self.trajectory) > self.cfg.render_trail_length:
            self.trajectory.pop(0)

        self.steps += 1

        # Check termination conditions
        terminated, reward, reason = self._check_termination()
        truncated = self.steps >= self.cfg.max_steps

        # Add progress reward
        if not terminated:
            reward += self._compute_progress_reward()

        obs = self._get_observation()
        info = self._get_info()
        info["terminated_reason"] = reason

        return obs, reward, terminated, truncated, info

    def _update_dynamics(self, throttle: float, steering_rate: float):
        """Update car state using realistic bicycle model."""
        dt = self.cfg.dt
        cp = self.car_params

        # Update steering angle
        delta_steering = steering_rate * cp.max_steering_rate * dt
        self.steering_angle = np.clip(
            self.steering_angle + delta_steering,
            -cp.max_steering_angle,
            cp.max_steering_angle,
        )

        # Compute forces
        if throttle >= 0:
            drive_force = throttle * cp.max_engine_force
        else:
            drive_force = throttle * cp.max_brake_force

        # Current speed (scalar)
        self.speed = np.linalg.norm(self.velocity)

        # Aerodynamic drag: F_drag = 0.5 * rho * Cd * A * v²
        drag_force = (
            0.5 * cp.air_density * cp.drag_coefficient * cp.frontal_area * self.speed**2
        )

        # Rolling resistance: F_roll = Crr * m * g
        roll_force = cp.rolling_resistance * cp.mass * 9.81

        # Net force in heading direction
        net_force = drive_force - drag_force - roll_force

        # Lateral force for turning (simplified)
        # In a proper bicycle model, this would involve slip angles
        if abs(self.steering_angle) > 1e-6 and self.speed > 0.1:
            # Compute turning radius
            turn_radius = cp.wheelbase / np.tan(abs(self.steering_angle))
            # Centripetal acceleration
            centripetal_accel = self.speed**2 / turn_radius
            # Required lateral force
            lateral_force = cp.mass * centripetal_accel
            # Limit by tire friction
            max_lateral = cp.tire_friction * cp.mass * 9.81
            lateral_force = np.clip(lateral_force, 0, max_lateral)
        else:
            lateral_force = 0.0

        # Update heading (yaw) rate using bicycle model
        if self.speed > 0.1:  # Avoid singularity at zero speed
            self.angular_velocity = (self.speed / cp.wheelbase) * np.tan(
                self.steering_angle
            )
        else:
            self.angular_velocity = 0.0

        # Update heading
        self.heading += self.angular_velocity * dt
        self.heading = self._normalize_angle(self.heading)

        # Update velocity in world frame
        # Longitudinal acceleration
        accel = net_force / cp.mass

        # Update speed (scalar)
        self.speed += accel * dt
        self.speed = np.clip(self.speed, -cp.max_reverse_speed, cp.max_speed)

        # Convert to velocity components in world frame
        self.velocity[0] = self.speed * np.cos(self.heading)
        self.velocity[1] = self.speed * np.sin(self.heading)

        # Update position
        self.pos += self.velocity * dt

        # Enforce world boundaries
        self.pos[0] = np.clip(
            self.pos[0], cp.length / 2, self.cfg.world_width - cp.length / 2
        )
        self.pos[1] = np.clip(
            self.pos[1], cp.width / 2, self.cfg.world_height - cp.width / 2
        )

    def _get_lidar_readings(self) -> np.ndarray:
        """Compute LIDAR distance measurements."""
        cp = self.car_params
        readings = np.full(cp.num_lidar_rays, cp.lidar_range, dtype=np.float32)

        # Compute ray angles relative to car heading
        start_angle = self.heading - cp.lidar_fov / 2
        angle_step = (
            cp.lidar_fov / (cp.num_lidar_rays - 1) if cp.num_lidar_rays > 1 else 0
        )

        for i in range(cp.num_lidar_rays):
            angle = start_angle + i * angle_step
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            min_dist = cp.lidar_range

            # Check obstacles
            for obs in self.obstacles:
                dist = self._ray_circle_intersection(
                    self.pos, ray_dir, obs.pos, obs.radius
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Check walls
            if self.cfg.use_walls:
                # Check all four walls
                wall_dist = self._ray_wall_intersection(self.pos, ray_dir)
                if wall_dist < min_dist:
                    min_dist = wall_dist

            readings[i] = min_dist

        return readings

    def _ray_circle_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        circle_pos: np.ndarray,
        circle_radius: float,
    ) -> Optional[float]:
        """Compute ray-circle intersection distance."""
        to_circle = circle_pos - ray_origin
        proj = np.dot(to_circle, ray_dir)

        if proj < 0:  # Circle is behind ray
            return None

        # Distance from circle center to ray
        perp_dist = np.linalg.norm(to_circle - proj * ray_dir)

        if perp_dist > circle_radius:
            return None

        # Distance along ray to intersection
        chord_half = np.sqrt(circle_radius**2 - perp_dist**2)
        dist = proj - chord_half

        return dist if dist >= 0 else None

    def _ray_wall_intersection(
        self, ray_origin: np.ndarray, ray_dir: np.ndarray
    ) -> float:
        """Compute minimum distance to any wall."""
        min_dist = self.car_params.lidar_range

        # Left wall (x = 0)
        if abs(ray_dir[0]) > 1e-6:
            t = -ray_origin[0] / ray_dir[0]
            if t > 0:
                min_dist = min(min_dist, t)

        # Right wall (x = world_width)
        if abs(ray_dir[0]) > 1e-6:
            t = (self.cfg.world_width - ray_origin[0]) / ray_dir[0]
            if t > 0:
                min_dist = min(min_dist, t)

        # Bottom wall (y = 0)
        if abs(ray_dir[1]) > 1e-6:
            t = -ray_origin[1] / ray_dir[1]
            if t > 0:
                min_dist = min(min_dist, t)

        # Top wall (y = world_height)
        if abs(ray_dir[1]) > 1e-6:
            t = (self.cfg.world_height - ray_origin[1]) / ray_dir[1]
            if t > 0:
                min_dist = min(min_dist, t)

        return min_dist

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Current target (next waypoint or final goal)
        if self.current_waypoint_idx < len(self.waypoints):
            target = self.waypoints[self.current_waypoint_idx]
        else:
            target = self.goal_pos

        goal_rel = target - self.pos

        lidar = self._get_lidar_readings()

        obs = np.concatenate(
            [
                self.pos,  # [0:2] position
                self.velocity,  # [2:4] velocity
                [self.heading],  # [4] heading
                [self.angular_velocity],  # [5] angular velocity
                [self.steering_angle],  # [6] steering angle
                goal_rel,  # [7:9] relative goal position
                [float(self.current_waypoint_idx)],  # [9] waypoint index
                lidar,  # [10:] LIDAR readings
            ]
        ).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        """Get additional information."""
        # Compute clearances
        wall_clearance = self._compute_wall_clearance()
        obstacle_clearance = self._compute_obstacle_clearance()

        return {
            "position": self.pos.copy(),
            "heading": self.heading,
            "speed": self.speed,
            "steering_angle": self.steering_angle,
            "wall_clearance": wall_clearance,
            "obstacle_clearance": obstacle_clearance,
            "waypoint_idx": self.current_waypoint_idx,
            "distance_to_goal": np.linalg.norm(self.goal_pos - self.pos),
        }

    def _compute_wall_clearance(self) -> float:
        """Minimum distance to any wall."""
        return min(
            self.pos[0],
            self.cfg.world_width - self.pos[0],
            self.pos[1],
            self.cfg.world_height - self.pos[1],
        )

    def _compute_obstacle_clearance(self) -> float:
        """Minimum distance to any obstacle."""
        if not self.obstacles:
            return float("inf")

        min_dist = float("inf")
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs.pos) - obs.radius
            min_dist = min(min_dist, dist)

        return min_dist

    def _check_termination(self) -> Tuple[bool, float, str]:
        """Check if episode should terminate and compute reward."""
        # Check goal reached
        current_target = (
            self.waypoints[self.current_waypoint_idx]
            if self.current_waypoint_idx < len(self.waypoints)
            else self.goal_pos
        )

        dist_to_target = np.linalg.norm(self.pos - current_target)

        if dist_to_target < self.cfg.goal_radius:
            if self.current_waypoint_idx < len(self.waypoints):
                # Reached waypoint, move to next
                self.current_waypoint_idx += 1
                return False, 50.0, ""  # Waypoint bonus
            else:
                # Reached final goal!
                return True, self.cfg.goal_reward, "goal"

        # Check collisions with obstacles
        for obs in self.obstacles:
            dist = np.linalg.norm(self.pos - obs.pos)
            if dist < (obs.radius + self.car_params.collision_radius):
                return True, self.cfg.collision_penalty, "collision"

        # Check wall collision
        if self.cfg.use_walls:
            margin = self.car_params.length / 2
            if (
                self.pos[0] < margin
                or self.pos[0] > self.cfg.world_width - margin
                or self.pos[1] < margin
                or self.pos[1] > self.cfg.world_height - margin
            ):
                return True, self.cfg.collision_penalty, "wall_collision"

        # No termination
        return False, self.cfg.time_penalty, ""

    def _compute_progress_reward(self) -> float:
        """Reward for making progress toward goal."""
        current_target = (
            self.waypoints[self.current_waypoint_idx]
            if self.current_waypoint_idx < len(self.waypoints)
            else self.goal_pos
        )

        dist = np.linalg.norm(self.pos - current_target)

        # Reward inversely proportional to distance
        # This encourages moving toward the goal
        max_dist = np.linalg.norm([self.cfg.world_width, self.cfg.world_height])
        progress = 1.0 - (dist / max_dist)

        return self.cfg.progress_reward_scale * progress

    def _generate_obstacles(self):
        """Generate random obstacles in the environment."""
        self.obstacles = []
        rng = np.random.RandomState(self.np_random.integers(0, 2**31))

        # Static obstacles
        for _ in range(self.cfg.n_static_obstacles):
            while True:
                pos = rng.uniform(
                    [10, 10], [self.cfg.world_width - 10, self.cfg.world_height - 10]
                )
                radius = rng.uniform(
                    self.cfg.obstacle_min_radius, self.cfg.obstacle_max_radius
                )

                # Check if too close to start or goal
                if (
                    np.linalg.norm(pos - self.pos) > radius + 5
                    and np.linalg.norm(pos - self.goal_pos) > radius + 5
                ):
                    self.obstacles.append(Obstacle(pos, radius))
                    break

        # Dynamic obstacles
        for _ in range(self.cfg.n_dynamic_obstacles):
            while True:
                pos = rng.uniform(
                    [15, 15], [self.cfg.world_width - 15, self.cfg.world_height - 15]
                )
                radius = rng.uniform(1.5, 3.0)

                # Random velocity
                angle = rng.uniform(0, 2 * np.pi)
                speed = rng.uniform(1.0, self.cfg.dynamic_obstacle_max_speed)
                velocity = speed * np.array([np.cos(angle), np.sin(angle)])

                if (
                    np.linalg.norm(pos - self.pos) > radius + 8
                    and np.linalg.norm(pos - self.goal_pos) > radius + 8
                ):
                    self.obstacles.append(Obstacle(pos, radius, velocity))
                    break

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        return self._render_matplotlib()

    def _render_matplotlib(self):
        """Render using matplotlib."""
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        if self.screen is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.screen = True

        self.ax.clear()
        self.ax.set_xlim(0, self.cfg.world_width)
        self.ax.set_ylim(0, self.cfg.world_height)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(
            f"Realistic Car Navigation - Step {self.steps}",
            fontsize=14,
            fontweight="bold",
        )

        # Draw walls
        if self.cfg.use_walls:
            wall_rect = patches.Rectangle(
                (0, 0),
                self.cfg.world_width,
                self.cfg.world_height,
                linewidth=3,
                edgecolor="black",
                facecolor="lightgray",
                alpha=0.1,
            )
            self.ax.add_patch(wall_rect)

        # Draw obstacles
        for obs in self.obstacles:
            color = "orange" if obs.is_dynamic else "gray"
            alpha = 0.7 if obs.is_dynamic else 0.5
            circle = patches.Circle(obs.pos, obs.radius, color=color, alpha=alpha)
            self.ax.add_patch(circle)

            # Draw velocity arrow for dynamic obstacles
            if obs.is_dynamic:
                self.ax.arrow(
                    obs.pos[0],
                    obs.pos[1],
                    obs.velocity[0],
                    obs.velocity[1],
                    head_width=0.5,
                    head_length=0.5,
                    fc="red",
                    ec="red",
                    alpha=0.6,
                )

        # Draw waypoints
        for i, wp in enumerate(self.waypoints):
            marker = "o" if i < self.current_waypoint_idx else "s"
            color = "green" if i < self.current_waypoint_idx else "blue"
            self.ax.plot(
                wp[0],
                wp[1],
                marker=marker,
                markersize=12,
                color=color,
                markeredgecolor="black",
                markeredgewidth=2,
            )
            self.ax.text(
                wp[0],
                wp[1] + 2,
                f"WP{i + 1}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        # Draw goal
        goal_circle = patches.Circle(
            self.goal_pos,
            self.cfg.goal_radius,
            color="green",
            alpha=0.3,
            linewidth=2,
            edgecolor="darkgreen",
        )
        self.ax.add_patch(goal_circle)
        self.ax.plot(
            self.goal_pos[0],
            self.goal_pos[1],
            "*",
            markersize=20,
            color="gold",
            markeredgecolor="darkgreen",
            markeredgewidth=2,
        )

        # Draw trajectory trail
        if len(self.trajectory) > 1:
            trail = np.array(self.trajectory)
            self.ax.plot(trail[:, 0], trail[:, 1], "b-", alpha=0.3, linewidth=1)

        # Draw car
        self._draw_car(self.ax)

        # Draw LIDAR
        if self.cfg.show_lidar:
            self._draw_lidar(self.ax)

        # Draw safety zone
        if self.cfg.show_safety_zones:
            safety_circle = patches.Circle(
                self.pos,
                self.car_params.safety_radius,
                color="yellow",
                alpha=0.1,
                linewidth=1,
                edgecolor="orange",
                linestyle="--",
            )
            self.ax.add_patch(safety_circle)

        # Info text
        info_text = (
            f"Speed: {self.speed:.1f} m/s ({self.speed * 3.6:.1f} km/h)\n"
            f"Heading: {np.degrees(self.heading):.1f}°\n"
            f"Steering: {np.degrees(self.steering_angle):.1f}°\n"
            f"Distance to goal: {np.linalg.norm(self.goal_pos - self.pos):.1f} m"
        )
        self.ax.text(
            0.02,
            0.98,
            info_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.draw()
        plt.pause(0.001)

    def _draw_car(self, ax):
        """Draw car as a rectangle with heading indicator."""
        import matplotlib.patches as patches
        import matplotlib.transforms as transforms

        cp = self.car_params

        # Car body rectangle (centered at origin, will be transformed)
        car_rect = patches.Rectangle(
            (-cp.length / 2, -cp.width / 2),
            cp.length,
            cp.width,
            linewidth=2,
            edgecolor="darkblue",
            facecolor="blue",
            alpha=0.8,
        )

        # Apply rotation and translation
        t = (
            transforms.Affine2D().rotate(self.heading).translate(*self.pos)
            + ax.transData
        )
        car_rect.set_transform(t)
        ax.add_patch(car_rect)

        # Front indicator (triangle at front of car)
        front_offset = cp.length / 2
        front_pos = self.pos + front_offset * np.array(
            [np.cos(self.heading), np.sin(self.heading)]
        )
        ax.plot(
            front_pos[0],
            front_pos[1],
            "^",
            markersize=10,
            color="red",
            transform=transforms.Affine2D()
            .rotate(self.heading - np.pi / 2)
            .translate(*front_pos)
            + ax.transData,
        )

        # Steering indicator (wheels)
        wheel_offset = cp.wheelbase * 0.7
        left_wheel = self.pos + wheel_offset * np.array(
            [
                np.cos(self.heading + self.steering_angle),
                np.sin(self.heading + self.steering_angle),
            ]
        )
        ax.plot(
            [self.pos[0], left_wheel[0]],
            [self.pos[1], left_wheel[1]],
            "r-",
            linewidth=2,
            alpha=0.6,
        )

    def _draw_lidar(self, ax):
        """Draw LIDAR rays."""
        cp = self.car_params
        readings = self._get_lidar_readings()

        start_angle = self.heading - cp.lidar_fov / 2
        angle_step = (
            cp.lidar_fov / (cp.num_lidar_rays - 1) if cp.num_lidar_rays > 1 else 0
        )

        for i, dist in enumerate(readings):
            angle = start_angle + i * angle_step
            end_pos = self.pos + dist * np.array([np.cos(angle), np.sin(angle)])

            # Color based on distance (red = close, green = far)
            color_val = dist / cp.lidar_range
            color = (1 - color_val, color_val, 0)

            ax.plot(
                [self.pos[0], end_pos[0]],
                [self.pos[1], end_pos[1]],
                color=color,
                alpha=0.3,
                linewidth=0.5,
            )
            ax.plot(end_pos[0], end_pos[1], "o", markersize=2, color=color, alpha=0.5)

    def close(self):
        """Clean up rendering resources."""
        if self.screen is not None:
            import matplotlib.pyplot as plt

            plt.close("all")
            self.screen = None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
