# Stage-1 Safety CAC Pipeline

This workspace contains a minimal Certificated Actor-Critic (CAC) pipeline for the ModularCar2DEnv. Stage 1 trains a safety critic/policy using only the control-barrier-based reward, evaluates it, and optionally visualizes the learned policy.

## 0. Setup

1. **Install deps** (already done if you used the provided `.venv`, otherwise):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # or pip install numpy torch gymnasium matplotlib
   ```
2. Always activate the virtual environment before running the scripts:
   ```bash
   source /home/robo/RL/.venv/bin/activate
   ```

## 1. Train (stage1 safety critic)

`train_stage1.py` runs the Stage‑1 SAC training loop and saves the model weights.

```bash
python train_stage1.py \
  --total-steps 200000 \
  --checkpoint stage1_sac_agent.pt \
  --seed 42
```

Key flags:
- `--total-steps` – total number of env steps to collect before stopping.
- `--checkpoint` – path where the trained policy/critic (plus optimizer state and metadata) is written.
- `--seed` – random seed for env resets and PyTorch; controls reproducibility.
- `--start-random-steps` – number of purely random actions before using the policy (helps fill replay buffer).
- `--update-after` – wait this many steps before the first gradient update so the buffer has data.
- `--update-every` – how often (in env steps) to run an update block; each block runs `update_every` gradient steps.
- `--batch-size` – minibatch size sampled from replay for each gradient step.
- `--gamma` – reward discount factor used by the SAC critic.
- `--tau` – Polyak averaging rate for the target critics (smaller = slower target updates).
- `--lr` – learning rate shared by actor, critics, and entropy temperature optimizer.
- `--log-every` – how many steps between console summaries (avg reward, safe rate, etc.).

## 2. Evaluate saved agent

`eval_stage1.py` reloads a checkpoint and runs deterministic rollouts to estimate the Stage‑1 safety metrics (avg r1 return, safe episode rate, collision rate, etc.).

```bash
python eval_stage1.py \
  --checkpoint stage1_sac_agent.pt \
  --episodes 20 \
  --gamma 0.99 \
  --eval-seed 123
```

Key flags:
- `--checkpoint` – which `.pt` file to load (must match what `train_stage1.py` produced).
- `--episodes` – number of evaluation rollouts; more episodes = smoother stats but longer runtime.
- `--gamma` – discount assumed when reporting theoretical Vmax and discounted returns (should match training).
- `--tau`, `--lr` – optimizer hyperparameters for the temporary `SACAgent` instance (used only if you keep training post‑load; safe to leave at defaults).
- `--seed` – seed used when creating the wrapped environment (affects obstacle layouts).
- `--eval-seed` – RNG seed for the evaluation loop; controls which random resets are sampled.
- `--max-steps` – optional cap on steps per episode (defaults to env.cfg.max_steps when omitted).

## 3. Visualize / deploy the policy

`deploy_stage1.py` renders the trained agent inside the ModularCar2DEnv, similar to `manual_drive.py`. Requires Matplotlib with an interactive backend (Qt/Tk).

```bash
MPLBACKEND=QtAgg python deploy_stage1.py \
  --checkpoint stage1_sac_agent.pt \
  --episodes 3 \
  --sleep-scale 1.0
```

Useful flags:
- `--checkpoint` – trained weights to load.
- `--episodes` – how many episodes to roll out in one run.
- `--max-steps` – optional upper bound per episode (defaults to env cfg).
- `--seed` – RNG seed used when drawing maps/reset states during visualization.
- `--sleep-scale` – scales the pause between frames (`env.dt * sleep_scale`); <1 speeds the video up.
- `--headless` – skip calls to `env.render()` so you can do a quick textual smoke test (no GUI required).

## Tips

- Checkpoints embed both actor/critic weights and metadata (env/CBF configs) so future evaluations use identical settings.
- If you run on a machine without GPU, the scripts automatically fall back to CPU; CUDA warnings can be ignored.
- Use `python -m pip install pyqt5` (or another GUI toolkit) if Matplotlib complains about non-interactive backends during visualization.
