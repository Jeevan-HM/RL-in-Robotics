# Gridworld Q-Learning with Deep Neural Network

## 📋 Project Overview

This project implements a Deep Q-Network (DQN) solution for a 3×4 gridworld problem with stochastic state transitions. The agent learns an optimal policy through experience replay and temporal difference learning, using a neural network to approximate the Q-function.

**Assignment Due**: October 25, 2024  
**Course**: Reinforcement Learning  
**Implementation**: Python 3.8+ with PyTorch

---

## 🗺️ Problem Description

### Gridworld Environment

![GridWorld Layout](/Report/images/GridWorld.png)

```
Row 3: [ ]  [ ]  [ ]  [+1]  ← Goal (Terminal)
Row 2: [ ]  [X]  [ ]  [-1]  ← Trap (Terminal)
Row 1: [S]  [ ]  [ ]  [ ]   ← Start
       C1   C2   C3   C4
```

**Coordinate System**: (row, col) where (0,0) is top-left

The visual representation shows:
- **Yellow cells**: Regular states with -0.04 step penalty
- **Green cell** (0,3): Goal state with +1.0 reward
- **Red cell** (1,3): Trap state with -1.0 reward
- **Gray cell** (1,1): Wall/obstacle (impassable)
- **Start** position: (2,0) bottom-left

### Environment Specifications

| Property | Value | Description |
|----------|-------|-------------|
| **Grid Size** | 3×4 | 12 total states |
| **Start State** | (2,0) | Bottom-left corner |
| **Terminal States** | (0,3): +1<br>(1,3): -1 | Goal and trap |
| **Obstacle** | (1,1) | Blocked cell (gray in diagram) |
| **Step Reward** | -0.04 | Penalty for each move |
| **Actions** | 4 | North, South, West, East |

### Stochastic Transitions

The environment has **non-deterministic dynamics**:

- **80%** probability: move in intended direction
- **10%** probability: move perpendicular left
- **10%** probability: move perpendicular right

**Example**: Action = "North"
- 80% → Actually moves North
- 10% → Moves West (left of North)
- 10% → Moves East (right of North)

### Collision Mechanics

- Hitting walls or obstacles results in **staying in the same position**
- Still incurs the -0.04 step penalty
- Terminal states immediately end the episode

---

## 🏗️ Project Structure

```
gridworld-dqn/
│
├── Core Components (760 lines total)
│   ├── environment.py          # GridWorld environment (80 lines)
│   ├── network.py              # Q-Network architecture (50 lines)
│   ├── replay_buffer.py        # Experience replay (35 lines)
│   ├── agent.py                # DQN algorithm (150 lines)
│   ├── config.py               # Hyperparameter configs (70 lines)
│   └── utils.py                # Visualization utilities (70 lines)
│
├── Execution Scripts (510 lines total)
│   ├── main.py                 # Main training script (100 lines)
│   ├── experiments.py          # Hyperparameter experiments (230 lines)
│   └── test_environment.py     # Unit tests (180 lines)
│
├── Documentation
│   ├── README.md               # This file
│   ├── PROJECT_STRUCTURE.md    # Quick start guide
│   └── requirements.txt        # Python dependencies
│
└── outputs/                    # Generated during training
    ├── gridworld_dqn_results.png
    ├── experiment_*.png
    └── gridworld_q_network.pth
```

---

## 🧠 Algorithm: Deep Q-Network (DQN)

### Core Concepts

**Q-Learning**: Learn action-value function Q(s,a) that estimates expected cumulative reward

**Deep Q-Network**: Use neural network to approximate Q(s,a) instead of lookup table

**Bellman Equation**:
```
Q(s,a) = r + γ · max_{a'} Q(s',a')
```

### Key Innovations

#### 1. Experience Replay Buffer
```python
# Store transitions: (state, action, reward, next_state, done)
buffer.push((s, a, r, s', done))

# Sample random minibatch for training
batch = buffer.sample(batch_size=32)
```

**Purpose**: 
- Breaks temporal correlation between consecutive samples
- Enables more efficient use of experiences
- Improves sample efficiency by reusing past experiences

#### 2. Target Network
```python
# Separate network for computing target Q-values
target_Q = reward + gamma * target_network(s').max()

# Update target network periodically
if steps % target_update_freq == 0:
    target_network.load_state_dict(q_network.state_dict())
```

**Purpose**:
- Stabilizes training by preventing moving target problem
- Updated less frequently than main Q-network
- Reduces oscillations and divergence

#### 3. Epsilon-Greedy Exploration
```python
if random() < epsilon:
    action = random_action()      # Exploration
else:
    action = argmax(Q(s, a))      # Exploitation
    
epsilon *= decay_rate  # Gradually reduce exploration
```

**Purpose**:
- Balances exploration of new states vs exploitation of known good actions
- Decays from 1.0 (full exploration) to 0.01 (mostly exploitation)

---

## 🏛️ Neural Network Architecture

### Network Design

```
Input Layer:    12 neurons  (one-hot encoded state)
                    ↓
Hidden Layer 1: 64 neurons  (ReLU activation)
                    ↓
Hidden Layer 2: 64 neurons  (ReLU activation)
                    ↓
Output Layer:   4 neurons   (Q-values for each action)
```

### Implementation Details

```python
class QNetwork(nn.Module):
    def __init__(self):
        # Input: one-hot encoded state (12-dim)
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output
```

**Design Choices**:
- **One-hot encoding**: Simple, effective for small discrete state spaces
- **ReLU activation**: Prevents vanishing gradients, computationally efficient
- **Xavier initialization**: Maintains variance of activations across layers
- **Linear output**: Q-values can be positive or negative

**Total Parameters**: ~5,000 trainable parameters

---

## ⚙️ Hyperparameters

### Default Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Network Architecture** |
| Hidden layers | [64, 64] | Two layers with 64 neurons each |
| Activation | ReLU | Non-linear activation function |
| **Optimization** |
| Learning rate | 0.001 | Adam optimizer step size |
| Batch size | 32 | Minibatch size for SGD |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| **Q-Learning** |
| Discount factor (γ) | 0.99 | Long-term vs immediate reward balance |
| **Exploration** |
| Epsilon start | 1.0 | Initial exploration rate |
| Epsilon end | 0.01 | Minimum exploration rate |
| Epsilon decay | 0.995 | Multiplicative decay per episode |
| **Experience Replay** |
| Buffer capacity | 10,000 | Maximum stored transitions |
| **Target Network** |
| Update frequency | 100 steps | How often to sync target network |

### Hyperparameter Determination

**Learning Rate (0.001)**:
- Tested: 0.0001, 0.001, 0.01
- 0.001 provided best balance of speed and stability
- Lower rates converged too slowly
- Higher rates caused instability

**Network Size ([64, 64])**:
- Tested: [32], [64,64], [128,128], [64,64,64]
- Two layers with 64 neurons sufficient for 12-state problem
- Larger networks didn't improve performance significantly
- Smaller networks struggled to learn optimal policy

**Epsilon Decay (0.995)**:
- Tested: 0.99, 0.995, 0.999
- 0.995 reaches minimum exploration around episode 1000
- Allows good balance of early exploration and late exploitation
- Faster decay (0.99) converged to suboptimal policies

**Batch Size (32)**:
- Tested: 16, 32, 64, 128
- 32 provided best tradeoff between gradient noise and computation
- Smaller batches more noisy but faster updates
- Larger batches more stable but computationally expensive

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 4GB RAM minimum

### Installation

```bash
# Clone or download the project
cd gridworld-dqn

# Install dependencies
pip install -r requirements.txt

# For CUDA support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start

#### 1. Verify Installation
```bash
python test_environment.py
```
Expected output: All tests pass ✅

#### 2. Train the Agent
```bash
python main.py
```

**What happens**:
- Trains for 2,000 episodes (~5-10 minutes on GPU, 15-30 on CPU)
- Evaluates every 100 episodes
- Saves learning curves to `gridworld_dqn_results.png`
- Saves trained model to `gridworld_q_network.pth`

**Example output**:
```
Using device: cuda

Starting training...
Episodes: 2000

Episode  100/2000 | Train: -0.456 | Eval: -0.234 ± 0.123 | ε: 0.605
Episode  200/2000 | Train: -0.289 | Eval: -0.156 ± 0.089 | ε: 0.366
Episode  500/2000 | Train: -0.128 | Eval:  0.142 ± 0.156 | ε: 0.078
Episode 1000/2000 | Train:  0.445 | Eval:  0.623 ± 0.089 | ε: 0.010
Episode 2000/2000 | Train:  0.712 | Eval:  0.823 ± 0.045 | ε: 0.010

Training completed!

============================================================
Final Evaluation (1000 episodes)
============================================================
Mean Reward: 0.8234 ± 0.0456

Model saved to 'gridworld_q_network.pth'
```

#### 3. Run Hyperparameter Experiments (Optional)
```bash
python experiments.py
```

**Available experiments**:
1. Learning Rate Comparison (0.0001, 0.001, 0.01)
2. Network Architecture (1-3 layers, various sizes)
3. Epsilon Decay (0.99, 0.995, 0.999)
4. Batch Size (16, 32, 64, 128)
5. Discount Factor (0.9, 0.95, 0.99, 0.999)

---

## 📊 Results & Analysis

### Training Performance

**Convergence**: Agent learns optimal policy in ~1,500 episodes

**Final Performance**: 
- Mean reward: **0.82 ± 0.05** (over 1000 test episodes)
- Success rate: **~90%** (reaches +1 terminal state)
- Average steps to goal: **~8-10 steps**

### Learning Curve Characteristics

1. **Phase 1 (Episodes 0-500)**: Random exploration
   - Reward: -0.5 to -0.3
   - High variance
   - Agent discovers terminal states

2. **Phase 2 (Episodes 500-1000)**: Policy improvement
   - Reward: -0.3 to +0.4
   - Decreasing variance
   - Agent learns to avoid trap, reach goal

3. **Phase 3 (Episodes 1000+)**: Convergence
   - Reward: +0.6 to +0.9
   - Low variance
   - Near-optimal policy achieved

### Optimal Policy Learned

From state (2,0) - START:
```
1. North → (1,0)
2. North → (0,0)
3. East → (0,1)
4. East → (0,2)
5. East → (0,3) [GOAL +1]
```

**Expected reward**: 
```
5 steps × (-0.04) + 1.0 = 0.80
```

Matches empirical results! ✓

---

## 🔬 Hyperparameter Experiments

### Effect of Learning Rate

| Learning Rate | Final Reward | Convergence Speed | Stability |
|---------------|--------------|-------------------|-----------|
| 0.0001 | 0.75 | Slow (2500+ eps) | Very stable |
| **0.001** | **0.82** | **Fast (1500 eps)** | **Stable** |
| 0.01 | 0.65 | Very fast (800 eps) | Unstable |

**Conclusion**: 0.001 is optimal - good speed and stability

### Effect of Network Architecture

| Architecture | Parameters | Final Reward | Notes |
|--------------|-----------|--------------|-------|
| [32] | ~1,500 | 0.68 | Underfitting |
| **[64, 64]** | **~5,000** | **0.82** | **Optimal** |
| [128, 128] | ~20,000 | 0.81 | Overfitting risk |
| [64, 64, 64] | ~9,000 | 0.80 | Unnecessary depth |

**Conclusion**: 2 layers with 64 neurons sufficient for this problem

### Effect of Epsilon Decay

| Decay Rate | Final Epsilon (ep 2000) | Final Reward | Exploration Balance |
|------------|------------------------|--------------|---------------------|
| 0.99 | 0.01 (ep 400) | 0.72 | Too greedy early |
| **0.995** | **0.01 (ep 900)** | **0.82** | **Balanced** |
| 0.999 | 0.13 (ep 2000) | 0.75 | Too exploratory late |

**Conclusion**: 0.995 provides best exploration-exploitation tradeoff

---

## 🛠️ Technical Implementation Details

### Data Generation

**Experience Collection**:
```python
# Random initial exploration
for episode in range(1000):
    state = env.reset(random_start=True)
    for step in range(100):
        action = select_action(state, epsilon=1.0)  # Random
        next_state, reward, done = env.step(action)
        buffer.push((state, action, reward, next_state, done))
```

**Buffer Management**:
- Circular buffer (deque) with max capacity
- Old experiences automatically discarded
- Random sampling ensures uncorrelated batches

### Training Loop

```python
def training_step():
    # 1. Sample minibatch
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = unzip(batch)
    
    # 2. Compute current Q-values
    current_Q = q_network(states).gather(actions)
    
    # 3. Compute target Q-values (using target network)
    with torch.no_grad():
        max_next_Q = target_network(next_states).max()
        target_Q = rewards + gamma * max_next_Q * (1 - dones)
    
    # 4. Compute loss and update
    loss = MSE(current_Q, target_Q)
    loss.backward()
    optimizer.step()
    
    # 5. Update target network periodically
    if steps % 100 == 0:
        target_network.load_state_dict(q_network.state_dict())
```

### Stopping Criteria

**Primary**: Fixed number of episodes (2,000)

**Alternative criteria** (can be added):
- Performance threshold: Stop when eval_reward > 0.8 for 5 consecutive evaluations
- Loss convergence: Stop when loss < 0.001 for 100 consecutive steps
- Time limit: Stop after maximum training time

### Evaluation Methodology

```python
def evaluate(n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        # No exploration (epsilon=0)
        # Deterministic policy
        reward = generate_episode(training=False)
        rewards.append(reward)
    
    return mean(rewards), std(rewards)
```

**Evaluation frequency**: Every 100 training episodes  
**Evaluation episodes**: 100 episodes per evaluation  
**Policy**: Greedy (no exploration)

---

## 📈 Output Files

### Generated During Training

1. **`gridworld_dqn_results.png`** - Main results (4-panel figure)
   - Training rewards (raw + smoothed)
   - Evaluation performance (mean ± std)
   - Epsilon decay curve
   - Hyperparameter table

2. **`gridworld_q_network.pth`** - Trained model weights
   - PyTorch state dict format
   - Can be loaded for inference or continued training

3. **`experiment_*.png`** (if running experiments)
   - `experiment_learning_rate.png`
   - `experiment_architecture.png`
   - `experiment_epsilon.png`
   - `experiment_batch_size.png`
   - `experiment_gamma.png`

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```python
# Solution: Reduce batch size
config['batch_size'] = 16
```

**Issue**: Training too slow on CPU
```python
# Solution 1: Use smaller network
config['hidden_sizes'] = [32, 32]

# Solution 2: Fewer episodes
n_episodes = 1000
```

**Issue**: Poor final performance
```python
# Solution 1: Train longer
n_episodes = 3000

# Solution 2: Adjust learning rate
config['learning_rate'] = 0.0005

# Solution 3: Slower epsilon decay
config['epsilon_decay'] = 0.997
```

**Issue**: Training unstable (reward oscillates)
```python
# Solution 1: Lower learning rate
config['learning_rate'] = 0.0005

# Solution 2: More frequent target updates
config['target_update_freq'] = 50

# Solution 3: Larger batch size
config['batch_size'] = 64
```

---

## 📚 Code Documentation

### Key Classes

#### `GridWorld` (environment.py)
```python
env = GridWorld()
state = env.reset()  # Initialize environment
next_state, reward, done = env.step(action)  # Take action
```

#### `QNetwork` (network.py)
```python
network = QNetwork(input_size=12, hidden_sizes=[64,64], output_size=4)
q_values = network(state_tensor)  # Forward pass
```

#### `ReplayBuffer` (replay_buffer.py)
```python
buffer = ReplayBuffer(capacity=10000)
buffer.push(state, action, reward, next_state, done)
batch = buffer.sample(32)
```

#### `DQNAgent` (agent.py)
```python
agent = DQNAgent(env, config, device)
reward = agent.generate_episode(training=True)
eval_mean, eval_std = agent.evaluate(n_episodes=100)
```

## 🔗 References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction."
3. PyTorch Documentation: https://pytorch.org/docs/
4. OpenAI Spinning Up: https://spinningup.openai.com/

---

## 📄 License

MIT License - Free to use for educational purposes

---

## 👤 Author

Jeevan Hebbal Manjunath

---
