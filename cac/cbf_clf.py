import numpy as np
import torch

def delta_h(h_next: np.ndarray, h_now: np.ndarray, alpha: float) -> np.ndarray:
    """Discrete-time CBF residual: δh = h(s') + (α - 1) h(s)."""
    return h_next + (alpha - 1.0) * h_now

def safety_reward(delta_h_val: np.ndarray, exponential=True) -> np.ndarray:
    """r1 = exp(min(δh,0)) in (0,1], or plain min(δh,0)."""
    m = np.minimum(delta_h_val, 0.0)
    if exponential:
        return np.exp(m)  # ∈ (0,1]
    else:
        return m

def delta_l(l_next: np.ndarray, l_now: np.ndarray, beta: float) -> np.ndarray:
    return l_next + (beta - 1.0) * l_now

def goal_reward(delta_l_val: np.ndarray) -> np.ndarray:
    """r2 = -max(δl,0)."""
    return -np.maximum(delta_l_val, 0.0)

@torch.no_grad()
def evaluate_certificate(model, env, n=64):
    returns = []
    for _ in range(n):
        o, _ = env.reset()
        done = False
        G = 0.0
        gamma = env.gamma if hasattr(env, 'gamma') else 0.99
        steps = 0
        while not done and steps < env.max_steps:
            obs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
            a, _, _ = model.policy.sample(obs)
            a = a.cpu().numpy()[0]
            o2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            G += r * (gamma ** steps)
            o = o2
            steps += 1
        returns.append(G)
    return float(np.mean(returns))
