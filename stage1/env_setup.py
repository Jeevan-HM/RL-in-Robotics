from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym

from realistic_car_env import NavigationConfig, GoalOrientedCarEnv
from scenarios import CarNavigationEnv

from .cbf import SafetyCBF
from .config import CBFConfig
from .wrappers import SafetyRewardWrapper

__all__ = ["make_stage1_env"]


def make_stage1_env(
    env_cfg: Optional[NavigationConfig] = None,
    cbf_cfg: Optional[CBFConfig] = None,
    seed: Optional[int] = None,
) -> Tuple[gym.Env, SafetyCBF]:
    env = CarNavigationEnv(env_config=env_cfg or NavigationConfig())
    if seed is not None:
        env.reset(seed=seed)
    cbf = SafetyCBF(env.unwrapped, cbf_cfg)
    wrapped = SafetyRewardWrapper(env, cbf)
    return wrapped, cbf
