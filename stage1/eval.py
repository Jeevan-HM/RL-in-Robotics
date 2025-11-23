from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .agent import SACAgent
from .wrappers import SafetyRewardWrapper

__all__ = ["theoretical_Vmax", "evaluate_stage1"]


def theoretical_Vmax(gamma: float, kmax: int) -> float:
    return (1.0 - (gamma ** kmax)) / (1.0 - gamma) if gamma < 0.999999 else float(kmax)


def evaluate_stage1(
    env: SafetyRewardWrapper,
    agent: SACAgent,
    episodes: int = 20,
    gamma: float = 0.99,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed if seed is not None else 0)
    returns = []
    lengths = []
    safe_flags = []
    collisions = 0
    predicted_Vs = []

    for _ in range(episodes):
        o, _ = env.reset(seed=rng.randint(0, 10_000))
        predicted_Vs.append(agent.estimate_V(o, samples=64))

        disc = 1.0
        ret = 0.0
        safe_all = True
        length = 0
        for _ in range(0, env.cfg.max_steps if max_steps is None else max_steps):
            a = agent.select_action(o, eval_mode=True)
            o2, r, terminated, truncated, info = env.step(a)
            ret += disc * r
            disc *= gamma
            length += 1
            if info.get("safety_r1", 1.0) < 0.999999:
                safe_all = False
            if terminated or truncated:
                if info.get("terminated_reason", "") == "collision":
                    collisions += 1
                break
            o = o2

        returns.append(ret)
        lengths.append(length)
        safe_flags.append(1.0 if safe_all else 0.0)

    avg_ret = float(np.mean(returns))
    avg_len = float(np.mean(lengths))
    safe_rate = float(np.mean(safe_flags))
    coll_rate = float(collisions / episodes)
    avg_predV = float(np.mean(predicted_Vs))
    Vmax = theoretical_Vmax(gamma, int(avg_len))

    summary = {
        "avg_return_r1": avg_ret,
        "avg_len": avg_len,
        "safe_rate_all_r1": safe_rate,
        "collision_rate": coll_rate,
        "avg_predicted_V1": avg_predV,
        "Vmax_at_avg_len": Vmax,
    }
    print("\n=== Stage-1 Evaluation (Safety Certificate) ===")
    for k, v in summary.items():
        print(f"{k:>22s}: {v:.4f}")
    print("(Note: avg_return_r1 should approach Vmax if episodes are fully safe.)")
    return summary
