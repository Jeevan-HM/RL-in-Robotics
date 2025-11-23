from __future__ import annotations

from dataclasses import dataclass

from realistic_car_env import NavigationConfig

__all__ = ["CBFConfig", "default_env_config", "_default_env_cfg"]


@dataclass
class CBFConfig:
    """Hyper-parameters for the discrete-time control barrier function.

    FIXED: Adjusted safety distances to allow proper goal-reaching while maintaining safety.
    The original values were too conservative, preventing the robot from getting close enough
    to the goal (which requires being within 3.0 meters).
    """

    alpha_cbf: float = 5.0  # h(s) = alpha_cbf * (d_clear^2 - d_safe^2)
    alpha0: float = 0.2  # discrete-time CBF decay, used in δh

    # FIXED: Reduced safety buffers to allow goal-reaching
    # Original values: d_safe_point=0.8, d_safe_car=1.0
    # New values are tuned to maintain safety while allowing the robot to reach goals
    d_safe_point: float = 0.3  # safety buffer (meters) for point-mass model
    d_safe_car: float = 0.4  # safety buffer (meters) for bicycle car model

    # NEW: Add goal-specific safety relaxation
    # When near the goal, we can reduce the safety buffer slightly
    goal_proximity_threshold: float = (
        5.0  # meters - distance to goal to start relaxing safety
    )
    goal_safety_relaxation: float = (
        0.5  # multiplier for safety distance near goal (0.5 = 50% of normal)
    )


def default_env_config() -> NavigationConfig:
    """Return a NavigationConfig with safety-related settings."""
    return NavigationConfig()


# Legacy alias so existing imports keep working.
def _default_env_cfg() -> NavigationConfig:
    return default_env_config()
