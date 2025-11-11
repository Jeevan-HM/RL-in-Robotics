from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from stage1 import SafetyCBF
from stage1.wrappers import SafetyRewardWrapper

from .agent import Stage2Agent
from .goal_clf import GoalCLF

__all__ = ["evaluate_stage2"]


def _safety_reward_from_obs(cbf: SafetyCBF, obs: np.ndarray, next_obs: np.ndarray) -> float:
    h_curr, _ = cbf.h(obs)
    h_next, _ = cbf.h(next_obs)
    delta = min(h_next + (cbf.cfg.alpha0 - 1.0) * h_curr, 0.0)
    return float(math.exp(delta))


def evaluate_stage2(
    env: SafetyRewardWrapper,
    agent: Stage2Agent,
    goal_clf: GoalCLF,
    cbf: SafetyCBF,
    episodes: int = 20,
    gamma: float = 0.99,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed if seed is not None else 0)
    nav_returns = []
    safety_returns = []
    lengths = []
    safe_flags = []
    goal_hits = 0
    collisions = 0

    for _ in range(episodes):
        obs, _ = env.reset(seed=rng.randint(0, 10_000))
        disc = 1.0
        nav_ret = 0.0
        safety_ret = 0.0
        safe_all = True
        length = 0
        for _ in range(0, env.cfg.max_steps if max_steps is None else max_steps):
            action = agent.select_action(obs, eval_mode=True)
            next_obs, _, terminated, truncated, info = env.step(action)

            r1 = info.get("safety_r1")
            if r1 is None:
                r1 = _safety_reward_from_obs(cbf, obs, next_obs)
            r2, _ = goal_clf.reward(obs, next_obs)

            nav_ret += disc * r2
            safety_ret += disc * float(r1)
            disc *= gamma
            length += 1

            if float(r1) < 0.999999:
                safe_all = False

            done = terminated or truncated
            if done:
                reason = info.get("terminated_reason", "")
                if reason == "goal":
                    goal_hits += 1
                if reason == "collision":
                    collisions += 1
                break
            obs = next_obs

        nav_returns.append(nav_ret)
        safety_returns.append(safety_ret)
        lengths.append(length)
        safe_flags.append(1.0 if safe_all else 0.0)

    summary = {
        "avg_return_r2": float(np.mean(nav_returns)),
        "avg_return_r1": float(np.mean(safety_returns)),
        "avg_len": float(np.mean(lengths)),
        "safe_rate_all_r1": float(np.mean(safe_flags)),
        "goal_rate": float(goal_hits / episodes),
        "collision_rate": float(collisions / episodes),
    }

    print("\n=== Stage-2 Evaluation (Navigation + Safety) ===")
    for k, v in summary.items():
        print(f"{k:>22s}: {v:.4f}")
    print("goal_rate reflects fraction of episodes that terminate at the goal region.")
    return summary
