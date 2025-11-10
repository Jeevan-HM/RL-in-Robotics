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
- `--total-steps`: environment interaction budget.
- `--checkpoint`: output file for the trained weights + optimizer state.
- Tune `--log-every`, `--start-random-steps`, etc. as needed.

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
- `--episodes`: number of Monte-Carlo rollouts used for the statistics.
- `--max-steps`: optional cap per episode (defaults to env config).
- `--seed` / `--eval-seed`: control environment construction and rollout RNGs.

## 3. Visualize / deploy the policy

`deploy_stage1_agent.py` renders the trained agent inside the ModularCar2DEnv, similar to `manual_drive.py`. Requires Matplotlib with an interactive backend (Qt/Tk).

```bash
MPLBACKEND=QtAgg python deploy_stage1_agent.py \
  --checkpoint stage1_sac_agent.pt \
  --episodes 3 \
  --sleep-scale 1.0
```

Useful flags:
- `--headless`: skip rendering (quick smoke test).
- `--sleep-scale`: multiply `env.dt` between frames (<1 speeds up playback).
- `--max-steps`: stop episodes early if you only want short previews.

## Tips

- Checkpoints embed both actor/critic weights and metadata (env/CBF configs) so future evaluations use identical settings.
- If you run on a machine without GPU, the scripts automatically fall back to CPU; CUDA warnings can be ignored.
- Use `python -m pip install pyqt5` (or another GUI toolkit) if Matplotlib complains about non-interactive backends during visualization.
