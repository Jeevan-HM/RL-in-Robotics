"""Stage-1 training entry point.

Trains the safety critic/policy and saves the resulting SAC weights so that
evaluation can be run separately without repeating the expensive rollout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stage1 import (
    CBFConfig,
    _default_env_cfg,
    train_stage1,
    save_agent,
    save_actor_critic_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 (safety) SAC agent.")
    parser.add_argument("--total-steps", type=int, default=200_000, help="Total environment steps.")
    parser.add_argument("--checkpoint", type=Path, default=Path("stage1_sac_agent.pt"), help="Where to store the trained weights.")
    parser.add_argument("--seed", type=int, default=42, help="Environment and PyTorch seed.")
    parser.add_argument("--start-random-steps", type=int, default=5_000)
    parser.add_argument("--update-after", type=int, default=5_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument(
        "--stage2-weights",
        type=Path,
        default=None,
        help="Optional path to store actor/critic weights for Stage-2. "
        "Defaults to '<checkpoint_stem>_stage2.pt' next to the main checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage2_weights_path = args.stage2_weights
    if stage2_weights_path is None:
        stage2_weights_path = args.checkpoint.with_name(f"{args.checkpoint.stem}_stage2.pt")
    env_cfg = _default_env_cfg()
    env_cfg.n_obstacles = 0
    cbf_cfg = CBFConfig(alpha_cbf=5.0, alpha0=0.2, d_safe_point=0.8, d_safe_car=1.0)

    agent, out = train_stage1(
        total_steps=args.total_steps,
        start_random_steps=args.start_random_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        seed=args.seed,
        env_cfg=env_cfg,
        cbf_cfg=cbf_cfg,
        log_every=args.log_every,
    )

    metadata = {
        "total_steps": args.total_steps,
        "seed": args.seed,
        "env_cfg": env_cfg,
        "cbf_cfg": cbf_cfg,
    }
    save_agent(agent, args.checkpoint, metadata=metadata)
    save_actor_critic_weights(agent, stage2_weights_path)
    print(f"\nCheckpoint saved to {args.checkpoint}")
    print(f"Stage-2 actor/critic weights saved to {stage2_weights_path}")


if __name__ == "__main__":
    main()
