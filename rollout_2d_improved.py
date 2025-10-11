import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cac.algorithms import CACAgent
from envs.planar_nav import PlanarNavEnv


def analyze_episode(env, agent, deterministic=True, seed=None, verbose=False):
    """Run an episode and collect detailed statistics"""
    o, _ = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    ep_len = 0
    trajectory = []
    clearances = []
    actions = []

    while not done:
        # Record current state
        clearance = env._min_range(env.p, env.theta) - env.R
        clearances.append(clearance)
        trajectory.append(env.p.copy())

        a = agent.act(o, deterministic=deterministic)
        actions.append(a[0])

        if verbose and clearance < env.margin * 2:
            print(
                f"Step {ep_len}: Close to obstacle! Clearance: {clearance:.3f}, Action: {a[0]:.3f}"
            )

        o, r, term, trunc, info = env.step(a)
        done = term or trunc
        ep_ret += r
        ep_len += 1

        if done and verbose:
            print(f"Episode ended: {info.get('termination', 'unknown')}")

    # Analyze trajectory
    trajectory = np.array(trajectory)
    clearances = np.array(clearances)
    actions = np.array(actions)

    # Classification
    success = float(np.linalg.norm(env.goal - env.p) < 1.5)
    timeout = float(ep_len >= env.max_steps)
    collision = float((not success) and (not timeout))

    stats = {
        "return": ep_ret,
        "length": ep_len,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "min_clearance": np.min(clearances),
        "avg_clearance": np.mean(clearances),
        "clearance_violations": np.sum(clearances < env.margin),
        "trajectory": trajectory,
        "clearances": clearances,
        "actions": actions,
    }

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to saved model .pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", choices=["goal", "safety"], default="safety")
    ap.add_argument(
        "--stochastic",
        action="store_true",
        help="Use sampling instead of deterministic mean action",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed info about close encounters",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of trajectories and clearances",
    )
    args = ap.parse_args()

    # Load agent
    try:
        agent, _ = CACAgent.load(args.ckpt)
        print(f"Loaded model from {args.ckpt}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create environment
    env = PlanarNavEnv(seed=args.seed)
    env.set_stage(args.stage)

    print(f"Environment: Safety margin = {env.margin:.3f}, Robot radius = {env.R:.3f}")
    print(f"Running {args.episodes} episodes...")

    det = not args.stochastic
    all_stats = []

    for ep in range(args.episodes):
        stats = analyze_episode(
            env, agent, deterministic=det, seed=args.seed + ep, verbose=args.verbose
        )
        all_stats.append(stats)

        print(
            f"Episode {ep:03d} | return={stats['return']:.3f} len={stats['length']:4d} "
            f"success={int(stats['success'])} collision={int(stats['collision'])} timeout={int(stats['timeout'])} "
            f"min_clearance={stats['min_clearance']:.3f} violations={stats['clearance_violations']}"
        )

    # Calculate summary statistics
    returns = [s["return"] for s in all_stats]
    lengths = [s["length"] for s in all_stats]
    successes = [s["success"] for s in all_stats]
    collisions = [s["collision"] for s in all_stats]
    timeouts = [s["timeout"] for s in all_stats]
    min_clearances = [s["min_clearance"] for s in all_stats]
    avg_clearances = [s["avg_clearance"] for s in all_stats]
    violations = [s["clearance_violations"] for s in all_stats]

    print("\n" + "=" * 60)
    print("SAFETY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Episodes          : {args.episodes}")
    print(f"Average return    : {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"Average length    : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Success rate      : {np.mean(successes) * 100:.1f}%")
    print(f"Collision rate    : {np.mean(collisions) * 100:.1f}%")
    print(f"Timeout rate      : {np.mean(timeouts) * 100:.1f}%")
    print(f"Min clearance     : {np.min(min_clearances):.3f}")
    print(
        f"Avg clearance     : {np.mean(avg_clearances):.3f} ± {np.std(avg_clearances):.3f}"
    )
    print(
        f"Safety violations : {np.mean(violations):.1f} ± {np.std(violations):.1f} per episode"
    )
    print(
        f"Critical episodes : {np.sum([s['min_clearance'] < env.margin * 0.5 for s in all_stats])}/{args.episodes}"
    )

    # Safety assessment
    if np.mean(collisions) > 0.1:  # More than 10% collision rate
        print(f"\n⚠️  HIGH COLLISION RATE: {np.mean(collisions) * 100:.1f}%")
        print("Recommendations:")
        print("- Increase safety margin in environment")
        print("- Retrain with stronger CBF constraints")
        print("- Reduce entropy coefficient in training")
    else:
        print(f"\n✅ Collision rate acceptable: {np.mean(collisions) * 100:.1f}%")

    if args.plot:
        # Plot collision episodes if any
        collision_episodes = [i for i, s in enumerate(all_stats) if s["collision"]]
        if collision_episodes:
            fig, axes = plt.subplots(
                2, min(3, len(collision_episodes)), figsize=(15, 8)
            )
            if len(collision_episodes) == 1:
                axes = axes.reshape(-1, 1)

            for i, ep_idx in enumerate(collision_episodes[:3]):
                stats = all_stats[ep_idx]
                traj = stats["trajectory"]
                clearances = stats["clearances"]

                # Plot trajectory
                ax1 = axes[0, i] if len(collision_episodes) > 1 else axes[0]
                ax1.plot(traj[:, 0], traj[:, 1], "b-", alpha=0.7)
                ax1.scatter(traj[0, 0], traj[0, 1], c="g", s=100, label="Start")
                ax1.scatter(traj[-1, 0], traj[-1, 1], c="r", s=100, label="End")
                ax1.set_title(f"Collision Episode {ep_idx}")
                ax1.legend()
                ax1.grid(True)
                ax1.axis("equal")

                # Plot clearances
                ax2 = axes[1, i] if len(collision_episodes) > 1 else axes[1]
                steps = range(len(clearances))
                ax2.plot(steps, clearances, "b-", label="Clearance")
                ax2.axhline(
                    y=env.margin, color="r", linestyle="--", label="Safety margin"
                )
                ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Clearance")
                ax2.set_title(f"Clearance History {ep_idx}")
                ax2.legend()
                ax2.grid(True)

            plt.tight_layout()
            plt.savefig(
                f"collision_analysis_{args.stage}.png", dpi=150, bbox_inches="tight"
            )
            print(
                f"\nSaved collision analysis plot to collision_analysis_{args.stage}.png"
            )


if __name__ == "__main__":
    main()
