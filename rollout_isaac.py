import argparse, os
import numpy as np
from isaac.isaac_env import IsaacPlanarNavEnv
from cac.algorithms import CACAgent

def run_episode(env, agent, deterministic=True, seed=None):
    o, _ = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    ep_len = 0
    while not done:
        a = agent.act(o, deterministic=deterministic)
        o, r, term, trunc, _ = env.step(a)
        done = term or trunc
        ep_ret += r
        ep_len += 1
    success = float(np.linalg.norm(env.goal - env.p) < 1.5)
    timeout = float(ep_len >= env.max_steps)
    collision = float((not success) and (not timeout))
    return ep_ret, ep_len, success, collision, timeout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to saved model .pt')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--stage', choices=['goal','safety'], default='goal')
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--headless', action='store_true')
    args = ap.parse_args()

    agent, _ = CACAgent.load(args.ckpt)

    env = IsaacPlanarNavEnv(headless=args.headless, seed=args.seed)
    env.set_stage(args.stage)

    det = not args.stochastic
    R = []; L = []; S = []; C = []; T = []
    for ep in range(args.episodes):
        ep_ret, ep_len, suc, col, tout = run_episode(env, agent, deterministic=det, seed=args.seed+ep)
        print(f"Episode {ep:03d} | return={ep_ret:.3f} len={ep_len:4d}  success={int(suc)}  collision={int(col)}  timeout={int(tout)}")
        R.append(ep_ret); L.append(ep_len); S.append(suc); C.append(col); T.append(tout)

    print("\n=== Summary ===")
    print(f"Episodes      : {args.episodes}")
    print(f"Avg return    : {np.mean(R):.3f}")
    print(f"Avg length    : {np.mean(L):.1f}")
    print(f"Success rate  : {np.mean(S)*100:.1f}%")
    print(f"Collision rate: {np.mean(C)*100:.1f}%")
    print(f"Timeout rate  : {np.mean(T)*100:.1f}%")

    env.close()

if __name__ == '__main__':
    main()
