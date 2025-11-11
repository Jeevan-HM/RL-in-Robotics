"""Stage-1 CAC toolkit for ModularCar2DEnv."""

from .agent import ReplayBuffer, SACAgent
from .cbf import SafetyCBF
from .checkpoints import load_agent, save_agent
from .config import CBFConfig, _default_env_cfg, default_env_config
from .env_setup import make_stage1_env
from .eval import evaluate_stage1, theoretical_Vmax
from .training import train_stage1
from .viz import _HAS_PLT, plot_action_Q_heatmap
from .wrappers import SafetyRewardWrapper

__all__ = [
    "CBFConfig",
    "SafetyCBF",
    "SafetyRewardWrapper",
    "ReplayBuffer",
    "SACAgent",
    "make_stage1_env",
    "train_stage1",
    "save_agent",
    "load_agent",
    "evaluate_stage1",
    "theoretical_Vmax",
    "plot_action_Q_heatmap",
    "default_env_config",
    "_default_env_cfg",
    "_HAS_PLT",
]
