"""
Integration module for Enhanced Realistic Car Environment with CAC Framework
=============================================================================

This module adapts the RealisticCarEnv to work seamlessly with the existing
Stage 1 and Stage 2 training pipelines.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from realistic_car_env import GoalOrientedCarEnv, CarPhysicsParams, NavigationConfig
from device_config import DeviceConfig, DeviceType

class CarNavigationEnv(gym.Wrapper):
    """
    Wrapper to make GoalOrientedCarEnv compatible with the CAC training framework.
    
    This ensures the observation space and info dict match what the existing
    Stage1/Stage2 code expects.
    """
    
    def __init__(
        self,
        car_params: Optional[CarPhysicsParams] = None,
        env_config: Optional[NavigationConfig] = None,
        render_mode: Optional[str] = None,
    ):
        env = GoalOrientedCarEnv(car_params, env_config, render_mode)
        super().__init__(env)
        
        # Store configs for CBF/CLF compatibility
        self.cfg = env.cfg
        self._model = "car"  # For CBF to know we're using car model
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Ensure info has required fields for CAC
        info = self._augment_info(info)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Augment info with CAC-required fields
        info = self._augment_info(info)
        
        return obs, reward, terminated, truncated, info
    
    def _augment_info(self, info: dict) -> dict:
        """Add fields expected by CAC framework."""
        # Ensure clearances are present
        if "wall_clearance" not in info:
            info["wall_clearance"] = self.env._compute_wall_clearance()
        
        if "obstacle_clearance" not in info:
            info["obstacle_clearance"] = self.env._compute_obstacle_clearance()
        
        # Add position info if not present
        if "position" not in info:
            info["position"] = self.env.pos.copy()
        
        return info


def create_navigation_env(
    use_waypoints: bool = True,
    n_waypoints: int = 3,
    difficulty: str = "medium",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> CarNavigationEnv:
    """
    Factory function to create enhanced environment with preset configurations.
    
    Args:
        use_waypoints: Whether to use intermediate waypoints
        n_waypoints: Number of waypoints (if use_waypoints=True)
        difficulty: "easy", "medium", or "hard"
        render_mode: "human" or "rgb_array" or None
        seed: Random seed for reproducibility
    
    Returns:
        CarNavigationEnv ready for CAC training
    """
    
    # Realistic car parameters (Toyota Camry-like sedan)
    car_params = CarPhysicsParams()
    
    # Configure environment based on difficulty
    if difficulty == "easy":
        env_config = NavigationConfig(
            world_width=80.0,
            world_height=80.0,
            start_pos=(10.0, 10.0),
            goal_pos=(70.0, 70.0),
            n_static_obstacles=4,
            n_dynamic_obstacles=1,
            max_steps=1500,
        )
    elif difficulty == "medium":
        env_config = NavigationConfig(
            world_width=100.0,
            world_height=100.0,
            start_pos=(10.0, 10.0),
            goal_pos=(90.0, 90.0),
            n_static_obstacles=8,
            n_dynamic_obstacles=3,
            max_steps=2000,
        )
    else:  # hard
        env_config = NavigationConfig(
            world_width=120.0,
            world_height=120.0,
            start_pos=(10.0, 10.0),
            goal_pos=(110.0, 110.0),
            n_static_obstacles=12,
            n_dynamic_obstacles=5,
            max_steps=2500,
        )
    
    # Add waypoints if requested
    if use_waypoints and n_waypoints > 0:
        start = np.array(env_config.start_pos)
        goal = np.array(env_config.goal_pos)
        
        # Create evenly spaced waypoints along a path
        waypoints = []
        for i in range(1, n_waypoints + 1):
            alpha = i / (n_waypoints + 1)
            # Add some randomness to make it more interesting
            offset = np.random.randn(2) * 5.0 if seed is None else np.zeros(2)
            wp = start + alpha * (goal - start) + offset
            waypoints.append(tuple(wp))
        
        env_config.waypoints = waypoints
    
    env = CarNavigationEnv(car_params, env_config, render_mode)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


# Preset configurations for different scenarios

def create_city_navigation_env(render_mode: Optional[str] = None) -> CarNavigationEnv:
    """City-like environment with many obstacles (buildings, parked cars)."""
    car_params = CarPhysicsParams(
        max_speed=15.0,  # 54 km/h (urban speed limit)
        max_steering_angle=40.0 * np.pi / 180,
    )
    
    env_config = NavigationConfig(
        world_width=100.0,
        world_height=100.0,
        start_pos=(15.0, 15.0),
        goal_pos=(85.0, 85.0),
        waypoints=[(35.0, 30.0), (50.0, 60.0), (70.0, 75.0)],  # Street path
        n_static_obstacles=15,  # Buildings, parked cars
        n_dynamic_obstacles=5,   # Moving traffic
        obstacle_min_radius=2.0,
        obstacle_max_radius=4.0,
        goal_radius=4.0,
        time_penalty=-0.2,
        progress_reward_scale=2.0,
    )
    
    return CarNavigationEnv(car_params, env_config, render_mode)


def create_highway_env(render_mode: Optional[str] = None) -> CarNavigationEnv:
    """Highway scenario with high-speed navigation and lane changes."""
    car_params = CarPhysicsParams(
        max_speed=35.0,  # 126 km/h (highway speed)
        max_engine_force=6000.0,
    )
    
    env_config = NavigationConfig(
        world_width=150.0,
        world_height=80.0,  # Elongated for highway
        start_pos=(10.0, 40.0),
        goal_pos=(140.0, 40.0),
        waypoints=[(50.0, 35.0), (100.0, 45.0)],  # Lane changes
        n_static_obstacles=5,  # Road barriers
        n_dynamic_obstacles=8,  # Other vehicles
        dynamic_obstacle_max_speed=25.0,
        obstacle_min_radius=1.8,
        obstacle_max_radius=2.2,
        goal_radius=5.0,
    )
    
    return CarNavigationEnv(car_params, env_config, render_mode)


def create_parking_env(render_mode: Optional[str] = None) -> CarNavigationEnv:
    """Tight parking scenario requiring precise control."""
    car_params = CarPhysicsParams(
        max_speed=5.0,  # 18 km/h (slow for parking)
        max_steering_angle=45.0 * np.pi / 180,  # Full lock
        max_steering_rate=30.0 * np.pi / 180,
    )
    
    env_config = NavigationConfig(
        world_width=60.0,
        world_height=60.0,
        start_pos=(10.0, 30.0),
        start_heading=0.0,
        goal_pos=(50.0, 30.0),
        goal_radius=2.0,  # Tight tolerance
        n_static_obstacles=10,  # Parked cars
        n_dynamic_obstacles=1,
        obstacle_min_radius=1.8,
        obstacle_max_radius=2.0,
        max_steps=1000,
        collision_penalty=-1000.0,  # High penalty for precision task
    )
    
    return CarNavigationEnv(car_params, env_config, render_mode)


def create_offroad_env(render_mode: Optional[str] = None) -> CarNavigationEnv:
    """Off-road scenario with rough terrain simulation."""
    car_params = CarPhysicsParams(
        max_speed=20.0,
        rolling_resistance=0.05,  # Higher for rough terrain
        tire_friction=0.6,  # Lower grip
        max_steering_angle=40.0 * np.pi / 180,
    )
    
    env_config = NavigationConfig(
        world_width=120.0,
        world_height=120.0,
        start_pos=(20.0, 20.0),
        goal_pos=(100.0, 100.0),
        waypoints=[(40.0, 50.0), (70.0, 60.0), (85.0, 90.0)],
        n_static_obstacles=12,  # Rocks, trees
        n_dynamic_obstacles=2,  # Wildlife
        obstacle_min_radius=1.5,
        obstacle_max_radius=6.0,
        dynamic_obstacle_max_speed=3.0,
        use_walls=False,  # Open terrain
    )
    
    return CarNavigationEnv(car_params, env_config, render_mode)
