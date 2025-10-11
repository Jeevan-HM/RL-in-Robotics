import argparse
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from cac.algorithms import CACAgent, CACConfig
from envs.planar_nav import PlanarNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["safety", "goal"], default="safety")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_ent", type=float, default=0.1)  # Lower entropy
    parser.add_argument("--restrict_cos_min", type=float, default=-0.1)  # Stricter
    args = parser.parse_args()

    # Create environment with improved safety parameters
    env = PlanarNavEnv(
        seed=args.seed,
        safety_margin=0.3,  # Increased safety margin
        alpha=0.95,  # Stronger CBF constraint
        n_rays=21,  # More rays for better obstacle detection
        fov_deg=140,  # Wider field of view
    )
    env.set_stage(args.stage)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    cfg = CACConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        stage=args.stage,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_ent=args.alpha_ent,
        restrict_cos_min=args.restrict_cos_min,
        update_every=25,  # More frequent updates
        num_updates_per_step=2,  # More updates per step
    )
    agent = CACAgent(cfg)

    if args.resume_from:
        agent, _ = CACAgent.load(args.resume_from)
        agent.cfg.stage = args.stage

    run_name = args.run_name or f"{args.stage}_improved_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    ckpt_path = os.path.join("runs", f"{run_name}_last.pt")

    o, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_len = 0
    sum_r1 = 0.0
    sum_r2 = 0.0
    collision_count = 0
    total_episodes = 0
    min_clearance_sum = 0.0

    for t in trange(args.steps, desc=f"Training {args.stage} (Improved)"):
        if t < agent.cfg.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o)

        o2, r, term, trunc, info = env.step(a)
        done = term or trunc
        agent.replay.store(o, a, r, done, o2)
        ep_ret += r
        ep_len += 1
        sum_r1 += info.get("r1", 0.0)
        sum_r2 += info.get("r2", 0.0)

        # Track minimum clearance for safety monitoring
        current_clearance = env._min_range(env.p, env.theta) - env.R
        min_clearance_sum += current_clearance

        o = o2

        if done:
            total_episodes += 1
            if info.get("termination") == "collision":
                collision_count += 1

            # Enhanced logging for safety monitoring
            writer.add_scalar("ep/return", ep_ret, t)
            writer.add_scalar("ep/len", ep_len, t)
            writer.add_scalar("ep/r1_sum", sum_r1, t)
            writer.add_scalar("ep/r2_sum", sum_r2, t)
            writer.add_scalar("ep/avg_clearance", min_clearance_sum / ep_len, t)
            writer.add_scalar(
                "safety/collision_rate", collision_count / total_episodes, t
            )

            # Print safety statistics periodically
            if total_episodes % 10 == 0:
                collision_rate = collision_count / total_episodes * 100
                avg_clearance = min_clearance_sum / ep_len
                print(
                    f"Episodes: {total_episodes}, Collision Rate: {collision_rate:.1f}%, Avg Clearance: {avg_clearance:.3f}"
                )

            o, _ = env.reset()
            env.set_stage(args.stage)
            ep_ret = ep_len = 0
            sum_r1 = sum_r2 = 0
            min_clearance_sum = 0

        if t >= agent.cfg.update_after and t % agent.cfg.update_every == 0:
            for _ in range(agent.cfg.update_every * agent.cfg.num_updates_per_step):
                batch = agent.replay.sample_batch(agent.cfg.batch_size)
                stats_q = agent.update_critics(batch, stage=args.stage)
                for k, v in stats_q.items():
                    writer.add_scalar(f"train/{k}", v, t)

                if args.stage == "safety":
                    stats_pi = agent.update_actor_sac(batch, stage="safety")
                else:
                    stats_pi = agent.update_actor_restricted(batch)
                for k, v in stats_pi.items():
                    writer.add_scalar(f"train/{k}", v, t)

        if (t + 1) % 5000 == 0:
            agent.save(ckpt_path)

    agent.save(ckpt_path)
    print(f"\nTraining completed!")
    print(f"Final collision rate: {collision_count / total_episodes * 100:.1f}%")
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
