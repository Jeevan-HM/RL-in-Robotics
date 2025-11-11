from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym

from modularcar_env import EnvConfig, ModularCar2DEnv

from .cbf import SafetyCBF
from .config import CBFConfig
from .wrappers import SafetyRewardWrapper

__all__ = ["make_stage1_env"]


def make_stage1_env(
    env_cfg: Optional[EnvConfig] = None,
    cbf_cfg: Optional[CBFConfig] = None,
    seed: Optional[int] = None,
) -> Tuple[gym.Env, SafetyCBF]:
    env = ModularCar2DEnv(env_cfg or EnvConfig())
    if seed is not None:
        env.reset(seed=seed)
    cbf = SafetyCBF(env, cbf_cfg)
    wrapped = SafetyRewardWrapper(env, cbf)
    return wrapped, cbf
