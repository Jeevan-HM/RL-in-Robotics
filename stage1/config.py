from __future__ import annotations

from dataclasses import dataclass

from modularcar_env import EnvConfig

__all__ = ["CBFConfig", "default_env_config", "_default_env_cfg"]


@dataclass
class CBFConfig:
    """Hyper-parameters for the discrete-time control barrier function."""

    alpha_cbf: float = 5.0         # h(s) = alpha_cbf * (d_clear - d_safe)
    alpha0: float = 0.2            # discrete-time CBF decay, used in δh
    d_safe_point: float = 0.8      # safety buffer (meters) for point-mass model
    d_safe_car: float = 1.0        # safety buffer (meters) for bicycle car model


def default_env_config() -> EnvConfig:
    """Return an EnvConfig with safety-related signals enabled."""
    cfg = EnvConfig()
    cfg.sensor_in_obs = True
    cfg.include_clearances = True
    return cfg


# Legacy alias so existing imports keep working.
def _default_env_cfg() -> EnvConfig:
    return default_env_config()
