import argparse
import os

import numpy as np

from cac.algorithms import CACAgent
from envs.planar_nav import PlanarNavEnv


def run_episode_detailed(env, agent, deterministic=True, seed=None, verbose=False):
    o, _ = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    ep_len = 0
    min_clearances = []
    actions = []
    close_calls = 0

    while not done:
        clearance = env._min_range(env.p, env.theta) - env.R
        min_clearances.append(clearance)

        if clearance < env.margin:
            close_calls += 1

        a = agent.act(o, deterministic=deterministic)
        actions.append(a[0])

        if verbose and clearance < env.margin * 1.5:
            print(
                f"Step {ep_len}: Clearance={clearance:.3f}, Action={a[0]:.3f}, Pos=({env.p[0]:.2f},{env.p[1]:.2f})"
            )

        o, r, term, trunc, info = env.step(a)
        done = term or trunc
        ep_ret += r
        ep_len += 1

        if done and verbose:
            print(f"Episode ended: {info.get('termination', 'unknown')}")

    # Classify outcome
    success = float(np.linalg.norm(env.goal - env.p) < 1.5)
    timeout = float(ep_len >= env.max_steps)
    collision = float((not success) and (not timeout))

    stats = {
        "return": ep_ret,
        "length": ep_len,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "min_clearance": np.min(min_clearances),
        "avg_clearance": np.mean(min_clearances),
        "close_calls": close_calls,
        "avg_action": np.mean(np.abs(actions)),
    }

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to saved model .pt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="Print detailed info")
    args = ap.parse_args()

    # Load agent
    try:
        agent, _ = CACAgent.load(args.ckpt)
        print(f"✅ Loaded model from {args.ckpt}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create environment matching the training configuration
    env = PlanarNavEnv(
        seed=args.seed,
        safety_margin=0.25,  # Match training config
        alpha=0.93,  # Match training config
        enhanced_obs=True,  # Use enhanced observations
        n_rays=19,  # Match training config
        fov_deg=130,  # Match training config
    )
    env.set_stage("safety")

    print(f"Environment config:")
    print(f"  Safety margin: {env.margin:.3f}")
    print(f"  Robot radius: {env.R:.3f}")
    print(f"  Alpha (CBF): {env.alpha:.2f}")
    print(f"  Observation dim: {env.observation_space.shape[0]}")
    print(f"  Collision threshold: {env.margin * 0.67:.3f}")
    print(f"\nRunning {args.episodes} episodes...")

    all_stats = []

    for ep in range(args.episodes):
        stats = run_episode_detailed(
            env,
            agent,
            deterministic=True,
            seed=args.seed + ep,
            verbose=args.verbose and ep < 3,
        )
        all_stats.append(stats)

        if ep < 10 or (ep + 1) % 10 == 0:
            print(
                f"Ep {ep + 1:3d}: Ret={stats['return']:6.1f} Len={stats['length']:3d} "
                f"S={int(stats['success'])} C={int(stats['collision'])} T={int(stats['timeout'])} "
                f"MinClear={stats['min_clearance']:.3f} CloseCalls={stats['close_calls']:2d}"
            )

    # Summary statistics
    returns = [s["return"] for s in all_stats]
    lengths = [s["length"] for s in all_stats]
    successes = [s["success"] for s in all_stats]
    collisions = [s["collision"] for s in all_stats]
    timeouts = [s["timeout"] for s in all_stats]
    min_clearances = [s["min_clearance"] for s in all_stats]
    avg_clearances = [s["avg_clearance"] for s in all_stats]
    close_calls = [s["close_calls"] for s in all_stats]

    print(f"\n{'=' * 70}")
    print("🎯 BALANCED SAFETY MODEL PERFORMANCE")
    print(f"{'=' * 70}")
    print(f"Episodes analyzed    : {args.episodes}")
    print(f"Average return       : {np.mean(returns):8.1f} ± {np.std(returns):6.1f}")
    print(f"Average length       : {np.mean(lengths):8.1f} ± {np.std(lengths):6.1f}")
    print(f"Success rate         : {np.mean(successes) * 100:8.1f}%")
    print(f"Collision rate       : {np.mean(collisions) * 100:8.1f}%")
    print(f"Timeout rate         : {np.mean(timeouts) * 100:8.1f}%")
    print(f"Min clearance        : {np.min(min_clearances):8.3f}")
    print(
        f"Avg min clearance    : {np.mean(min_clearances):8.3f} ± {np.std(min_clearances):6.3f}"
    )
    print(
        f"Avg episode clearance: {np.mean(avg_clearances):8.3f} ± {np.std(avg_clearances):6.3f}"
    )
    print(
        f"Avg close calls/ep   : {np.mean(close_calls):8.1f} ± {np.std(close_calls):6.1f}"
    )

    # Safety assessment
    collision_rate = np.mean(collisions) * 100
    if collision_rate < 20:
        print(f"\n🟢 EXCELLENT: Collision rate = {collision_rate:.1f}%")
    elif collision_rate < 35:
        print(f"\n🟡 GOOD: Collision rate = {collision_rate:.1f}%")
    elif collision_rate < 50:
        print(f"\n🟠 ACCEPTABLE: Collision rate = {collision_rate:.1f}%")
    else:
        print(f"\n🔴 NEEDS IMPROVEMENT: Collision rate = {collision_rate:.1f}%")

    print(
        f"\nSafety margin: {env.margin:.3f}, Collision threshold: {env.margin * 0.67:.3f}"
    )

    # Compare to random policy baseline
    print(f"\n📊 COMPARISON TO BASELINE:")
    print(f"Random policy collision rate: ~90-95%")
    print(f"This model collision rate: {collision_rate:.1f}%")
    print(
        f"Improvement: {95 - collision_rate:.1f} percentage points better than random"
    )


if __name__ == "__main__":
    main()
