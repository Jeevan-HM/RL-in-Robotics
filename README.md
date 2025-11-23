# Certificated Actor-Critic (CAC) for Safe Robot Navigation

## Overview

Implementation of the **Certificated Actor-Critic (CAC)** algorithm from the paper:
> "Certificated Actor-Critic: Hierarchical Reinforcement Learning with Control Barrier Functions for Safe Navigation"  
> Xie, Junjun and Zhao, Shuhao and Hu, Liang and Gao, Huijun  
> arXiv preprint arXiv:2501.17424 (2025)

This is a **model-free reinforcement learning framework** that achieves **safe goal-reaching navigation** through a two-stage hierarchical learning process:

1. **Stage 1: Safety Critic Construction** - Learn collision-free navigation using Control Barrier Function (CBF)-derived rewards
2. **Stage 2: Restricted Policy Update** - Improve goal-reaching performance while maintaining safety guarantees through restricted gradient updates

## 🎯 Key Features

✅ **Safety Guarantees** - CBF-based forward invariance ensures collision avoidance  
✅ **Safety Certificates** - Quantitative safety evaluation via learned critics  
✅ **Hierarchical Learning** - Separate stages for safety and goal-reaching  
✅ **Restricted Gradients** - Maintains safety during policy improvement  
✅ **Model-Free** - No explicit system model required  
✅ **Realistic Physics** - High-fidelity vehicle dynamics simulation  
✅ **Dynamic Obstacles** - Handles both static and moving obstacles

## 📁 Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
│
├── main.py                     # Q-Learning GridWorld (legacy)
├── train_cac.py                # Main CAC training script (Algorithm 1)
├── goal_reaching_test.py       # Improved training with optimizations
├── demo_cac.py                 # Interactive demo with moving obstacles
├── device_config.py            # Device configuration (CPU/GPU/MPS)
│
├── realistic_car_env.py        # Realistic car physics environment
├── scenarios.py                # Environment wrappers and scenarios
│
├── stage1/                     # Stage 1: Safety Critic Construction
│   ├── __init__.py
│   ├── agent.py                # SAC agent implementation
│   ├── cbf.py                  # Control Barrier Function (CBF)
│   ├── checkpoints.py          # Model saving/loading
│   ├── config.py               # Configuration classes
│   ├── env_setup.py            # Environment setup utilities
│   ├── eval.py                 # Evaluation utilities
│   ├── networks.py             # Neural network architectures
│   ├── training.py             # Training utilities
│   ├── viz.py                  # Visualization tools
│   └── wrappers.py             # Environment wrappers
│
├── stage2/                     # Stage 2: Restricted Policy Update
│   ├── __init__.py
│   ├── agent.py                # Stage 2 agent
│   ├── checkpoints.py          # Checkpoint management
│   ├── eval.py                 # Evaluation tools
│   ├── goal_clf.py             # Control Lyapunov Function (CLF)
│   ├── replay.py               # Replay buffer
│   └── training.py             # Training utilities
│
├── checkpoints/                # Pre-trained models
│   ├── responsive_stage1.pt    # Stage 1 trained policy
│   ├── responsive_stage2.pt    # Stage 2 trained policy
│   ├── stage1_safe_policy.pt   # Alternative Stage 1 checkpoint
│   └── stage2_final_policy.pt  # Alternative Stage 2 checkpoint
│
├── Documents/                  # Research papers
│   └── 2501.17424v1.pdf       # CAC paper
│
├── images/                     # Visualization outputs
└── Report/                     # Project reports
```

## 🧠 Algorithm Overview

### Control Barrier Function (CBF)

The CBF `h(s)` defines a safe set `C = {s : h(s) ≥ 0}`. For safety:

```
h(s_{t+1}) + (α₀ - 1)h(s_t) ≥ 0
```

### Stage 1: Safety Critic Construction

**Objective**: Learn a safe policy `π*_safe` using CBF-derived rewards

**Safety Reward** (Equations 7 & 12 from paper):
```python
r₁(s_t, a_t) = exp(min(h(s_{t+1}) + (α₀ - 1)h(s_t), 0))
```

**Output**: Safe policy + Safety critics `V^π₁`, `Q^π₁`

The safety critics serve as **safety certificates**:
- If `V^π₁(s₀) ≈ 1.0`, the system is safe from state `s₀`
- Can compare relative safety between different policies

### Stage 2: Restricted Policy Update  

**Objective**: Improve goal-reaching while maintaining safety

**Navigation Reward** (Equation 9 from paper):
```python
r₂(s_t, a_t) = -max(l(s_{t+1}) + (β₀ - 1)l(s_t), 0)
```
where `l(s)` is a Control Lyapunov Function (CLF), e.g., squared distance to goal.

**Restricted Gradient Update** (Equation 10 from paper):
```
∇θ = argmax_e e·∇θJ₂(θ)
s.t. e·∇θJ₁(θ) ≥ 0, ||e|| ≤ ||∇θJ₂(θ)||
```

This ensures the safety critic doesn't decrease while improving goal-reaching performance.

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Jeevan-HM/RL-in-Robotics.git
cd RL-in-Robotics
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or use UV for faster installation:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV
uv sync  # Install dependencies
```

### Training CAC Agent

#### Option 1: Train Both Stages Sequentially
```bash
# Train complete CAC pipeline (Stage 1 + Stage 2)
python train_cac.py --steps-stage1 250000 --steps-stage2 250000
```

#### Option 2: Train Stages Separately
```bash
# Stage 1 only: Learn safe policy
python train_cac.py --stage 1 --steps-stage1 250000

# Stage 2 only: Improve goal-reaching (requires Stage 1 checkpoint)
python train_cac.py --stage 2 --steps-stage2 250000
```

#### Advanced Training Options
```bash
python train_cac.py \
    --steps-stage1 500000 \
    --steps-stage2 500000 \
    --alpha0 0.2 \           # CBF decay rate (lower = stricter safety)
    --beta0 0.9 \            # CLF decay rate (lower = faster goal approach)
    --device cuda \          # Use GPU (auto/cpu/cuda/mps)
    --checkpoint-dir ./my_checkpoints
```

#### Improved Training with Optimizations
```bash
# Use the improved training script with better hyperparameters
python goal_reaching_test.py \
    --stage 0 \              # Train both stages
    --steps-stage1 250000 \
    --steps-stage2 250000 \
    --difficulty medium \    # easy/medium/hard
    --device auto
```

### Evaluation

Evaluation functions are available in the stage1 and stage2 modules:

```python
# In your code, use the evaluation functions:
from stage1.eval import evaluate_stage1
from stage2.eval import evaluate_stage2

# Evaluate Stage 1 safety policy
stage1_results = evaluate_stage1(env, agent, episodes=100)

# Evaluate Stage 2 goal-reaching policy
stage2_results = evaluate_stage2(env, agent, goal_clf, cbf, episodes=100)
```

### Interactive Demo

```bash
# Visualize trained agent with moving obstacles
python demo_cac.py \
    --checkpoint checkpoints/responsive_stage2.pt \
    --episodes 3 \
    --moving-obstacles 3 \
    --obstacle-speed 3.0
```

## 🏗️ Environment Features

### Realistic Car Physics

The environment simulates a realistic vehicle using a bicycle model with:

- **Vehicle Dynamics**: Mass (800kg), inertia (1000 kg⋅m²), wheelbase (2.7m)
- **Tire Physics**: Friction coefficient (0.9), rolling resistance (0.01)
- **Aerodynamics**: Drag coefficient (0.25), frontal area (2.0 m²)
- **Engine/Brake**: Max engine force (8000N), max brake force (12000N)
- **Steering**: Max angle (50°), max steering rate (90°/s)

### Sensors

- **LIDAR**: 32-ray distance sensor with 25m range and 270° FOV
- **State Information**: Position, velocity, heading, acceleration
- **Safety Metrics**: Distance to nearest obstacle, collision detection

### Obstacles

- **Static Obstacles**: Fixed circular obstacles with configurable sizes
- **Dynamic Obstacles**: Moving obstacles with realistic velocities (up to 5 m/s)
- **Walls**: Boundary walls to constrain navigation space

### Goals

- **Goal Radius**: 3.0m (configurable)
- **CLF-based Rewards**: Encourage efficient goal approach
- **Waypoints**: Optional intermediate goals (future feature)

## 🎓 Implementation Details

### Neural Networks

- **Actor (Policy)**: 2-layer MLP with 256 units per layer
- **Critics**: 2 twin Q-networks, 3-layer MLP with 256 units each
- **Activation**: ReLU for hidden layers, Tanh for policy output
- **Output**: Continuous actions (steering, throttle/brake)

### Soft Actor-Critic (SAC) Algorithm

- **Optimization**: Adam optimizer with learning rate 3×10⁻⁴
- **Discount Factor**: γ = 0.99
- **Target Network Update**: Soft update with τ = 0.005
- **Entropy Tuning**: Automatic temperature adjustment
- **Replay Buffer**: 500k transitions

### Training Configuration

- **Batch Size**: 256
- **Initial Exploration**: 5000-10000 random steps
- **Update Frequency**: Every 50-100 steps
- **Episode Length**: Up to 3000 steps (5 minutes at 0.1s timestep)

## 📊 Expected Results

### Stage 1: Safety Critic Construction

After ~250k steps (approximately 2-4 hours on CPU):
- **Safe Rate**: >95% collision-free episodes
- **Safety Critic**: `V^π₁(s) ≈ 1.0` for safe states
- **Behavior**: Conservative, collision-avoiding navigation
- **Goal Rate**: ~20-40% (safety prioritized over goal-reaching)

### Stage 2: Restricted Policy Update  

After additional ~250k steps:
- **Safe Rate**: Maintained >95% (safety preserved)
- **Goal Rate**: >80% goal-reaching success
- **Behavior**: Efficient goal-reaching while staying safe
- **Navigation**: Smooth, human-like trajectories

### Dynamic Obstacle Performance

With 3 moving obstacles at 3 m/s:
- **Collision Avoidance**: >90% success
- **Close Calls**: <5 per episode (within 5m of moving obstacle)
- **Goal Success**: >70% in dynamic environments

## 🛠️ Troubleshooting

### Import Errors

If you get module import errors:
```bash
# Ensure you're in the project root directory
cd /path/to/RL-in-Robotics

# Install all dependencies
pip install -r requirements.txt
```

### Training Issues

**Low safe rate in Stage 1**:
- Increase `--steps-stage1` to 500k or more
- Adjust `--alpha0` to a lower value (e.g., 0.1) for stricter safety

**Poor goal-reaching in Stage 2**:
- Increase `--steps-stage2` to 500k or more
- Adjust `--beta0` to a lower value (e.g., 0.8) for more aggressive goal approach
- Use the improved training script: `goal_reaching_test.py`

**Memory issues**:
- Reduce replay buffer capacity in code (edit `agent.py`)
- Use CPU instead of GPU: `--device cpu`

### GPU/Device Issues

```bash
# Force CPU
python train_cac.py --device cpu

# Force CUDA (NVIDIA GPU)
python train_cac.py --device cuda

# Force Apple Silicon (MPS)
python train_cac.py --device mps

# Auto-detect best device
python train_cac.py --device auto
```

### Visualization Issues

If PyQt5 rendering doesn't work:
```bash
# Try installing PyQt5 separately
pip install pyqt5 --force-reinstall

# Or use matplotlib backend
pip install matplotlib pyqt5
```

## 📚 Legacy Q-Learning GridWorld

This repository also contains a classical Q-learning implementation for educational purposes:

```bash
# Run Q-learning on GridWorld with hyperparameter analysis
python main.py --episodes 10000
```

Features:
- Stochastic GridWorld environment (4x3 grid)
- Comprehensive hyperparameter analysis (α, γ, ε decay)
- Rich visualizations of convergence and Q-table heatmaps
- Educational walkthrough of RL fundamentals

See `main.py` for the complete implementation.

## 🔬 Research Context

This implementation is based on the research paper:

```bibtex
@article{xie2025certificated,
  title={Certificated Actor-Critic: Hierarchical Reinforcement Learning 
         with Control Barrier Functions for Safe Navigation},
  author={Xie, Junjun and Zhao, Shuhao and Hu, Liang and Gao, Huijun},
  journal={arXiv preprint arXiv:2501.17424},
  year={2025}
}
```

**Paper available in**: `Documents/2501.17424v1.pdf`

### Key Contributions from Paper

1. **Hierarchical Safety Framework**: Separates safety learning from goal-reaching
2. **Safety Certificates**: Uses critic networks as quantitative safety measures
3. **Restricted Gradients**: Novel gradient constraint preserves safety during policy updates
4. **Model-Free Approach**: No explicit dynamics model required
5. **Theoretical Guarantees**: Provable safety under CBF conditions

## 🔮 Future Improvements

The current implementation provides a solid foundation. Potential enhancements:

1. **Full Restricted Gradient Implementation**: Implement exact Equation 10 optimization with constrained gradient projection
2. **Dual Critics**: Separate safety critics (`V^π₁`, `Q^π₁`) and navigation critics (`V^π₂`, `Q^π₂`)
3. **High-Order CBFs**: Smoother control using second-order or adaptive CBFs
4. **Multi-Agent Extension**: Coordinate multiple robots safely
5. **Real Robot Deployment**: Port to physical platforms (ROS integration)
6. **Curriculum Learning**: Progressive difficulty increase during training
7. **Attention Mechanisms**: Better obstacle tracking for dynamic environments

## 📖 Additional Resources

### Learning Materials

- **Control Barrier Functions**: [Ames et al., 2019 - "Control Barrier Functions: Theory and Applications"](https://ieeexplore.ieee.org/document/8796030)
- **Soft Actor-Critic**: [Haarnoja et al., 2018 - "Soft Actor-Critic Algorithms and Applications"](https://arxiv.org/abs/1812.05905)
- **Safe RL Survey**: [García & Fernández, 2015 - "A Comprehensive Survey on Safe Reinforcement Learning"](https://jmlr.org/papers/v16/garcia15a.html)

### Related Projects

- **safety-gym**: OpenAI's safe RL benchmark environments
- **safe-control-gym**: Benchmark for safe learning-based control
- **CBF-QP**: Real-time CBF-based quadratic programming controllers

## 📄 License

This implementation is for research and educational purposes. Please cite the original paper if you use this code in your research.

## 🙏 Acknowledgments

- Original paper authors: Junjun Xie, Shuhao Zhao, Liang Hu, Huijun Gao
- Soft Actor-Critic implementation inspired by [Spinning Up](https://spinningup.openai.com/)
- Physics simulation based on realistic vehicle dynamics models

## 📧 Contact

For questions about the implementation:
- Open an issue on GitHub: [https://github.com/Jeevan-HM/RL-in-Robotics/issues](https://github.com/Jeevan-HM/RL-in-Robotics/issues)

For questions about the original paper:
- Contact the authors (see paper for details)

---

**Built with** ❤️ **for safe and intelligent robot navigation**
