# Certificated Actor–Critic (CAC) — Reference Implementation (Linux)

This repo provides a minimal **Certificated Actor–Critic (CAC)** implementation in PyTorch, with:
- a fast **2D planar obstacle‑avoidance** environment,
- a two‑stage training schedule (Safety then Goal),
- the **restricted policy‑gradient** update from the paper,
- and an **optional Isaac Sim** environment stub to run the same task inside Omniverse.

> **Heads‑up:** Isaac Sim integration is provided as a *lightweight* script that builds the scene and steps the sim using `omni.isaac.core`. It avoids hardware‑specific sensors and computes range readings in Python for simplicity. It should run on Isaac Sim 2023.1+ / 4.x, but you may need to tweak asset paths.

---

## Quick start (2D env)

```bash
# 1) Create an environment (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Train Stage 1 (safety critic)
python train_2dnav.py --stage safety --run_name demo_s1

# 3) Train Stage 2 (goal with safety restriction)
python train_2dnav.py --stage goal --run_name demo_s2 --resume_from runs/demo_s1_last.pt
```

TensorBoard logs are in `runs/`. Saved checkpoints are in `runs/<run_name>_last.pt`.

---

## Isaac Sim (optional)

> Tested with: Isaac Sim 2023.1/2024.x; headless mode is supported.
> Install Isaac Sim via Omniverse Launcher. Then run:

```bash
# Activate your CAC venv, then:
python train_isaac.py --stage safety --headless --steps 20000
python train_isaac.py --stage goal   --resume_from runs/isaac_s1_last.pt --headless --steps 20000
```

`train_isaac.py` automatically uses the same CAC algorithm but wraps an Isaac Sim scene: a disc‑robot navigating among randomly placed boxes. We compute "rangefinder" beams by querying obstacle prim transforms and doing CPU ray‑casts (fast enough for small worlds).

If Isaac Sim is not found on your system, the script will exit with a clear message.

---

## What’s inside

- `cac/algorithms.py` — CAC (Stage 1 & Stage 2) with the **restricted gradient**.
- `cac/nets.py` — MLP policy (tanh‑squashed Gaussian) and twin Q‑networks.
- `cac/replay.py` — Replay buffer, n‑step returns (1‑step by default).
- `cac/cbf_clf.py` — Generic CBF/CLF utilities (δh, rewards, etc.).
- `envs/planar_nav.py` — 2D world with obstacles, CBF = min_range − margin, CLF = goal distance²/2.
- `train_2dnav.py` — CLI to run Stage 1/Stage 2 on the 2D env.
- `isaac/isaac_env.py` — Isaac Sim environment wrapper (scene creation + obs + stepping).
- `train_isaac.py` — CLI to run CAC with `isaac_env.py`.

---

## Notes on CAC details

- **Stage 1 (Safety):** reward `r1 = exp(min(δh, 0)) ∈ (0,1]` where `δh = h(s′) + (α−1)h(s)`.
- **Stage 2 (Goal):** reward `r2 = −max(δl, 0)` with `δl = l(s′) + (β−1)l(s)`.
- **Restricted update:** with `g2 = ∇θJ2`, `g1 = ∇θJ1`, actor step uses projection
  `e = g2` if `g1·g2 ≥ 0`, else `e = g2 − (g2·g1)/(||g1||²+ε) * g1`, and we clamp `||e|| ≤ ||g2||`.

This code keeps the implementation compact and readable; it’s not hyper‑optimized.

---

## Requirements

- Python 3.10+
- PyTorch 2.x, Gymnasium, NumPy, TensorBoard
- Optional: Isaac Sim 2023.1+ (Omniverse), `omni.isaac.*` Python modules on `PYTHONPATH`

---

## Troubleshooting

- If Stage 2 becomes unsafe, increase `--restrict_cos_min` to enforce stronger alignment
  or lower the actor step size `--actor_lr`.
- For Isaac Sim, ensure you run with the Isaac Sim Python (`python.sh`) or add its site‑packages to `PYTHONPATH`.
- On headless servers, use `--headless` and set `export NVIDIA_VISIBLE_DEVICES=...` if needed.

Enjoy and feel free to extend!
