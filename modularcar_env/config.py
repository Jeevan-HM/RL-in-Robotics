from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class EnvConfig:
    """All knobs for the ModularCar2DEnv."""

    # World
    world_size: Tuple[float, float] = (50.0, 50.0)  # width, height (meters)
    world_origin: Optional[Tuple[float, float]] = None  # if None, frame is centered; else (xmin, ymin)
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
    r_goal: float = 500.0
    r_collision: float = -150.0
    step_penalty: float = -0.1
    control_penalty: float = 0.02  # penalize accel magnitude
    distance_scale: float = 0.5  # shaping: -distance_to_goal * scale

    # Termination
    out_of_bounds_terminates: bool = True
    solid_walls: bool = False  # clamp agent inside walls instead of terminating when enabled
    obstacle_collisions_terminate: bool = True
    solid_obstacles: bool = False  # keep agent outside obstacle discs when enabled

    # Observation layout
    include_applied_accel: bool = True
    include_speed: bool = True
    include_clearances: bool = True  # append wall/obstacle clearance scalars
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
