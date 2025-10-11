# eval_cac.py
import argparse
import importlib
import inspect
import json
import os
import sys
import time

import numpy as np
import torch


def try_import(module_name, extra_paths=()):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception:
        for p in extra_paths:
            if p not in sys.path:
                sys.path.append(p)
        return __import__(module_name, fromlist=["*"])


PROJECT_HINTS = (".", "src", "RL-in-Robotics")
gymnasium = try_import("gymnasium", PROJECT_HINTS)

from gymnasium.envs.registration import register

try:
    register(
        id="PlanarNavEnv-v0",
        entry_point="envs.planar_nav:PlanarNavEnv",
    )
except Exception:
    pass

np.set_printoptions(precision=3, suppress=True)


# --------------------------
# Observation sanity wrapper
# --------------------------
class ObsSpaceSanityWrapper(gymnasium.Wrapper):
    """
    Cast to float32 and clip to Box bounds to help pass PassiveEnvChecker.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Box):
            raise RuntimeError("ObsSpaceSanityWrapper expects a Box observation_space.")
        self._low = env.observation_space.low.astype("float32")
        self._high = env.observation_space.high.astype("float32")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.clip(obs, self._low, self._high)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.clip(obs, self._low, self._high)
        return obs, rew, term, trunc, info


# --------------------------
# Tensor/numpy helpers
# --------------------------
def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.float32).unsqueeze(0)


def to_numpy_action(act_tensor):
    if isinstance(act_tensor, (tuple, list)):
        act_tensor = torch.tensor(act_tensor, dtype=torch.float32)
    if isinstance(act_tensor, torch.Tensor):
        act = act_tensor.detach().cpu().float().numpy()
    else:
        act = np.asarray(act_tensor, dtype=np.float32)
    act = np.squeeze(act)
    act = np.atleast_1d(act).astype(np.float32)
    return act


# --------------------------
# State dict + dim inference
# --------------------------
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise ValueError("Unsupported checkpoint format")


def infer_expected_in_out_from_actor_sd(actor_sd):
    """
    Try to infer (in_dim, out_dim) from Linear layer weights in the actor sd.
    - Find first weight with shape [H, I] -> input I
    - Find last weight with shape [O, H*] -> output O (best effort)
    """
    first_in = None
    last_out = None
    for k, v in actor_sd.items():
        if k.endswith("weight") and v.ndim == 2:
            H, I = v.shape
            if first_in is None:
                first_in = I
            last_out = H
    return first_in, last_out


# --------------------------
# Flexible checkpoint loader
# --------------------------
def load_actor_and_critics(
    ckpt_goal, ckpt_safety, model_ns, device, actor_class_name="", critic_class_name=""
):
    models_mod = try_import(model_ns, PROJECT_HINTS)

    def pick_class(preferred_name, fallback_names, predicate_tokens):
        if preferred_name:
            cls = getattr(models_mod, preferred_name, None)
            if inspect.isclass(cls):
                return cls
            raise RuntimeError(
                f"Requested class '{preferred_name}' not found in '{model_ns}'"
            )
        for nm in fallback_names:
            cls = getattr(models_mod, nm, None)
            if inspect.isclass(cls):
                return cls
        candidates = []
        for nm, obj in vars(models_mod).items():
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                lower = nm.lower()
                if any(tok in lower for tok in predicate_tokens):
                    candidates.append((nm, obj))
        if candidates:
            candidates.sort(key=lambda x: len(x[0]))
            return candidates[0][1]
        return None

    Actor = pick_class(
        actor_class_name,
        fallback_names=[
            "Actor",
            "Policy",
            "GaussianPolicy",
            "DeterministicPolicy",
            "MLPActor",
            "ActorNet",
            "TanhGaussianPolicy",
        ],
        predicate_tokens=["actor", "policy", "pi"],
    )
    Critic = pick_class(
        critic_class_name,
        fallback_names=[
            "Critic",
            "QNetwork",
            "ValueNetwork",
            "QNet",
            "VNet",
            "DoubleQ",
        ],
        predicate_tokens=["critic", "q", "value", "vnet", "qnet"],
    )

    if Actor is None:
        raise RuntimeError(
            f"Could not find an Actor-like class in '{model_ns}'. "
            f"Pass --actor_class <ClassName> to specify one explicitly."
        )

    actor_sd = None
    critic_safe_sd = None
    critic_goal_sd = None

    if ckpt_goal and os.path.exists(ckpt_goal):
        d = _extract_state_dict(torch.load(ckpt_goal, map_location=device))
        for k in ["actor", "policy", "pi"]:
            if k in d and isinstance(d[k], dict):
                actor_sd = d[k]
                break
        for k in ["critic_goal", "critic2", "Q2", "V2", "critic"]:
            if k in d and isinstance(d[k], dict):
                critic_goal_sd = d[k]
                break
        if actor_sd is None and any(k.startswith("actor.") for k in d.keys()):
            actor_sd = {
                k.replace("actor.", "", 1): v
                for k, v in d.items()
                if k.startswith("actor.")
            }

    if ckpt_safety and os.path.exists(ckpt_safety):
        d = _extract_state_dict(torch.load(ckpt_safety, map_location=device))
        for k in ["critic_safe", "critic1", "Q1", "V1", "safety_critic"]:
            if k in d and isinstance(d[k], dict):
                critic_safe_sd = d[k]
                break
        if actor_sd is None:
            for k in ["actor", "policy", "pi"]:
                if k in d and isinstance(d[k], dict):
                    actor_sd = d[k]
                    break
            if actor_sd is None and any(k.startswith("actor.") for k in d.keys()):
                actor_sd = {
                    k.replace("actor.", "", 1): v
                    for k, v in d.items()
                    if k.startswith("actor.")
                }

    return Actor, Critic, actor_sd, critic_safe_sd, critic_goal_sd


# --------------------------
# Model builder
# --------------------------
def maybe_init_model(cls, sample_in_dim: int, sample_out_dim: int | None = None):
    for kwargs in (
        dict(obs_dim=sample_in_dim, act_dim=sample_out_dim),
        dict(state_dim=sample_in_dim, action_dim=sample_out_dim),
        dict(input_dim=sample_in_dim, output_dim=sample_out_dim),
        dict(obs_dim=sample_in_dim),
        dict(input_dim=sample_in_dim),
        dict(state_dim=sample_in_dim),
        {},
    ):
        try:
            return cls(**kwargs)
        except Exception:
            continue
    raise RuntimeError(
        f"Could not construct {cls.__name__}; please edit maybe_init_model() with proper kwargs."
    )


class InferenceActor(torch.nn.Module):
    def __init__(
        self, base_actor, deterministic=True, action_low=None, action_high=None
    ):
        super().__init__()
        self.base = base_actor
        self.deterministic = deterministic
        self.register_buffer(
            "_low",
            None
            if action_low is None
            else torch.tensor(action_low, dtype=torch.float32),
        )
        self.register_buffer(
            "_high",
            None
            if action_high is None
            else torch.tensor(action_high, dtype=torch.float32),
        )

    @torch.no_grad()
    def act(self, obs):
        out = self.base(obs)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            mu = out[0]
            act = mu
        else:
            act = out
        if self._low is not None and self._high is not None:
            act = torch.minimum(torch.maximum(act, self._low), self._high)
        return act


def goal_reached_from_info(info: dict) -> bool:
    for k in ("goal_reached", "success", "is_success", "at_goal"):
        if k in info and bool(info[k]):
            return True
    return False


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CAC: load safety+goal models and run a rollout."
    )
    parser.add_argument("--env_id", type=str, default="PlanarNavEnv-v0")
    parser.add_argument("--models_module", type=str, default="cac.nets")
    parser.add_argument(
        "--actor_class",
        type=str,
        default="",
        help="Explicit actor class name e.g., TanhGaussianPolicy",
    )
    parser.add_argument("--critic_class", type=str, default="")
    parser.add_argument("--ckpt_safety", type=str, default="runs/demo_s1_last.pt")
    parser.add_argument("--ckpt_goal", type=str, default="runs/demo_s2_last.pt")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no_obs_fix", action="store_true")
    parser.add_argument(
        "--obs_index_map",
        type=str,
        default="",
        help="JSON list of indices to select from env obs to feed the actor, e.g. '[0,1,2,...]'. If omitted and dims mismatch, drops the tail.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Create env
    env = gymnasium.make(args.env_id, render_mode="human")
    if not args.no_obs_fix and isinstance(env.observation_space, gymnasium.spaces.Box):
        env = ObsSpaceSanityWrapper(env)

    # Reset & spaces
    obs, info = env.reset(seed=np.random.randint(0, 10_000))
    obs = np.asarray(obs, dtype=np.float32)
    if not isinstance(env.action_space, gymnasium.spaces.Box):
        raise RuntimeError("This script assumes a continuous Box action space.")
    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)
    act_dim_env = int(np.prod(env.action_space.shape))
    obs_dim_env = int(np.prod(obs.shape))

    # Load models + state dicts
    Actor, Critic, actor_sd, critic_safe_sd, critic_goal_sd = load_actor_and_critics(
        args.ckpt_goal,
        args.ckpt_safety,
        args.models_module,
        args.device,
        actor_class_name=args.actor_class,
        critic_class_name=args.critic_class,
    )

    # ---- Infer expected input/output dims from the actor checkpoint ----
    expected_obs_dim = obs_dim_env
    expected_act_dim = act_dim_env
    if actor_sd is not None:
        inf_in, inf_out = infer_expected_in_out_from_actor_sd(actor_sd)
        if inf_in is not None:
            expected_obs_dim = int(inf_in)
        if inf_out is not None:
            # Many Gaussian policies output 2*act_dim (mu, log_std) internally;
            # we keep env act_dim as runtime output size. Leave expected_act_dim as env's.
            pass

    # ---- Build actor & load weights with the expected input size ----
    actor = maybe_init_model(
        Actor, sample_in_dim=expected_obs_dim, sample_out_dim=act_dim_env
    ).to(args.device)
    if actor_sd is not None:
        actor.load_state_dict(actor_sd, strict=False)
    else:
        print("[WARN] No actor weights found; using randomly initialized actor.")

    # ---- Optionally build critics ----
    def maybe_build_critic(sd):
        if sd is None or Critic is None:
            return None
        try:
            c = maybe_init_model(
                Critic, sample_in_dim=expected_obs_dim, sample_out_dim=act_dim_env
            ).to(args.device)
            c.load_state_dict(sd, strict=False)
            return c
        except Exception as e:
            print(f"[WARN] Could not init/load critic: {e}")
            return None

    safety_critic = maybe_build_critic(critic_safe_sd)
    goal_critic = maybe_build_critic(critic_goal_sd)

    policy = (
        InferenceActor(
            actor,
            deterministic=args.deterministic,
            action_low=act_low,
            action_high=act_high,
        )
        .to(args.device)
        .eval()
    )

    # ---- Build observation index map (project env obs -> expected_obs_dim) ----
    idx_map = None
    if args.obs_index_map:
        try:
            idx_map = json.loads(args.obs_index_map)
            idx_map = [int(i) for i in idx_map]
            if len(idx_map) != expected_obs_dim:
                raise ValueError(
                    f"obs_index_map length {len(idx_map)} != expected_obs_dim {expected_obs_dim}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to parse --obs_index_map: {e}")
    else:
        # default behavior: if env obs is longer than expected, drop tail; if shorter, pad zeros
        if obs_dim_env != expected_obs_dim:
            if obs_dim_env > expected_obs_dim:
                idx_map = list(range(expected_obs_dim))  # keep first K features
                print(
                    f"[INFO] Projecting obs: env_dim={obs_dim_env} -> expected={expected_obs_dim} (dropping tail). "
                    f"Pass --obs_index_map to customize."
                )
            else:
                idx_map = list(range(obs_dim_env))  # we will zero-pad later
                print(
                    f"[INFO] Projecting obs: env_dim={obs_dim_env} -> expected={expected_obs_dim} (zero-padding tail). "
                    f"Pass --obs_index_map to customize."
                )

    def project_obs(o: np.ndarray) -> np.ndarray:
        o = np.asarray(o, dtype=np.float32).reshape(-1)
        if idx_map is not None:
            sel = np.take(o, idx_map, mode="clip")
            if len(sel) < expected_obs_dim:
                pad = np.zeros((expected_obs_dim - len(sel),), dtype=np.float32)
                sel = np.concatenate([sel, pad], axis=0)
            return sel
        # sizes match
        if o.size == expected_obs_dim:
            return o
        # fallback: drop or pad
        if o.size > expected_obs_dim:
            return o[:expected_obs_dim]
        else:
            pad = np.zeros((expected_obs_dim - o.size,), dtype=np.float32)
            return np.concatenate([o, pad], axis=0)

    # If first reset size mismatched, project it now
    obs = project_obs(obs)

    ep_ret = 0.0
    ep_len = 0
    dt = 1.0 / max(args.fps, 1)

    for t in range(args.max_steps):
        obs_t = to_tensor(obs, args.device)

        with torch.no_grad():
            act_t = policy.act(obs_t)
            if safety_critic is not None:
                try:
                    try:
                        q_val = safety_critic(obs_t, act_t)
                    except Exception:
                        q_val = safety_critic(obs_t)
                    q_val_np = float(q_val.squeeze().detach().cpu().numpy())
                    print(f"[t={t:04d}] safety_critic≈{q_val_np:.3f}")
                except Exception:
                    pass

        action = to_numpy_action(act_t)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = project_obs(next_obs)

        ep_ret += float(reward)
        ep_len += 1

        if args.render and hasattr(env, "render"):
            env.render()
            time.sleep(dt)

        if terminated or truncated or goal_reached_from_info(info):
            print(
                f"Episode done | return={ep_ret:.3f} length={ep_len} "
                f"{'(goal)' if goal_reached_from_info(info) else ''} "
                f"{'(terminated)' if terminated else ''} "
                f"{'(truncated)' if truncated else ''}"
            )
            break

        obs = next_obs

    env.close()


if __name__ == "__main__":
    main()
