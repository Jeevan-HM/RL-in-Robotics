import argparse
import os

import numpy as np

from cac.algorithms import CACAgent
from envs.planar_nav import PlanarNavEnv


def run_episode(env, agent, deterministic=True, seed=None):
    o, _ = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    ep_len = 0
    min_clearances = []

    while not done:
        clearance = env._min_range(env.p, env.theta) - env.R
        min_clearances.append(clearance)

        a = agent.act(o, deterministic=deterministic)
        o, r, term, trunc, info = env.step(a)
        done = term or trunc
        ep_ret += r
        ep_len += 1

    # classify outcome
    success = float(np.linalg.norm(env.goal - env.p) < 1.5)
    timeout = float(ep_len >= env.max_steps)
    collision = float((not success) and (not timeout))

    return (
        ep_ret,
        ep_len,
        success,
        collision,
        timeout,
        np.min(min_clearances),
        np.mean(min_clearances),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to saved model .pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--enhanced_obs", action="store_true", help="Use enhanced observations"
    )
    args = ap.parse_args()

    # load agent
    agent, _ = CACAgent.load(args.ckpt)

    # Test with original safety margins
    print("=== Testing with ORIGINAL safety margins ===")
    env_orig = PlanarNavEnv(
        seed=args.seed, safety_margin=0.15, alpha=0.9, enhanced_obs=args.enhanced_obs
    )
    env_orig.set_stage("safety")

    results_orig = []
    for ep in range(args.episodes):
        ep_ret, ep_len, suc, col, tout, min_clear, avg_clear = run_episode(
            env_orig, agent, deterministic=True, seed=args.seed + ep
        )
        results_orig.append((ep_ret, ep_len, suc, col, tout, min_clear, avg_clear))

    # Test with improved safety margins
    print("\n=== Testing with IMPROVED safety margins ===")
    env_new = PlanarNavEnv(
        seed=args.seed, safety_margin=0.3, alpha=0.95, enhanced_obs=args.enhanced_obs
    )
    env_new.set_stage("safety")

    results_new = []
    for ep in range(args.episodes):
        ep_ret, ep_len, suc, col, tout, min_clear, avg_clear = run_episode(
            env_new, agent, deterministic=True, seed=args.seed + ep
        )
        results_new.append((ep_ret, ep_len, suc, col, tout, min_clear, avg_clear))

    # Compare results
    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change'}")
    print(f"{'-' * 60}")

    # Extract metrics
    orig_cols = [r[3] for r in results_orig]
    new_cols = [r[3] for r in results_new]
    orig_suc = [r[2] for r in results_orig]
    new_suc = [r[2] for r in results_new]
    orig_min_clear = [r[5] for r in results_orig]
    new_min_clear = [r[5] for r in results_new]
    orig_avg_clear = [r[6] for r in results_orig]
    new_avg_clear = [r[6] for r in results_new]

    print(
        f"{'Collision Rate':<20} {np.mean(orig_cols) * 100:>6.1f}%      {np.mean(new_cols) * 100:>6.1f}%      {(np.mean(new_cols) - np.mean(orig_cols)) * 100:+.1f}%"
    )
    print(
        f"{'Success Rate':<20} {np.mean(orig_suc) * 100:>6.1f}%      {np.mean(new_suc) * 100:>6.1f}%      {(np.mean(new_suc) - np.mean(orig_suc)) * 100:+.1f}%"
    )
    print(
        f"{'Min Clearance':<20} {np.mean(orig_min_clear):>8.3f}    {np.mean(new_min_clear):>8.3f}    {np.mean(new_min_clear) - np.mean(orig_min_clear):+.3f}"
    )
    print(
        f"{'Avg Clearance':<20} {np.mean(orig_avg_clear):>8.3f}    {np.mean(new_avg_clear):>8.3f}    {np.mean(new_avg_clear) - np.mean(orig_avg_clear):+.3f}"
    )

    print(f"\nSafety margin: {env_orig.margin:.3f} → {env_new.margin:.3f}")
    print(f"Alpha (CBF):   {env_orig.alpha:.2f} → {env_new.alpha:.2f}")


if __name__ == "__main__":
    main()
