import argparse, os, time
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from cac.algorithms import CACAgent, CACConfig
from isaac.isaac_env import IsaacPlanarNavEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["safety","goal"], required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_ent", type=float, default=0.2)
    parser.add_argument("--restrict_cos_min", type=float, default=-0.25)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = IsaacPlanarNavEnv(headless=args.headless, seed=args.seed)
    env.set_stage(args.stage)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    cfg = CACConfig(
        obs_dim=obs_dim, act_dim=act_dim,
        stage=args.stage, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        alpha_ent=args.alpha_ent, restrict_cos_min=args.restrict_cos_min
    )
    agent = CACAgent(cfg)

    if args.resume_from:
        agent, _ = CACAgent.load(args.resume_from)
        agent.cfg.stage = args.stage

    run_name = args.run_name or f"isaac_{args.stage}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    ckpt_path = os.path.join("runs", f"{run_name}_last.pt")

    o, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_len = 0

    for t in trange(args.steps, desc=f"Training Isaac {args.stage}"):
        if t < agent.cfg.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o)

        o2, r, term, trunc, info = env.step(a)
        done = term or trunc
        agent.replay.store(o, a, r, done, o2)
        ep_ret += r
        ep_len += 1
        o = o2

        if done:
            writer.add_scalar("ep/return", ep_ret, t)
            writer.add_scalar("ep/len", ep_len, t)
            o, _ = env.reset()
            env.set_stage(args.stage)
            ep_ret = 0.0; ep_len = 0

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

        if (t+1) % 5000 == 0:
            agent.save(ckpt_path)

    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    env.close()

if __name__ == "__main__":
    main()
