# Q-Learning GridWorld - Reinforcement Learning in Robotics

A comprehensive implementation of the Q-learning algorithm applied to a classic GridWorld navigation problem. This project demonstrates fundamental reinforcement learning concepts commonly used in robotics applications.

## 🎯 Project Overview

This project implements a **Q-learning agent** that learns to navigate a 4×3 grid world to reach a goal while avoiding penalties. The environment simulates realistic robotic movement with stochastic transitions, making it an excellent introduction to reinforcement learning concepts used in real-world robotics.

### Key Features

- **Q-Learning Algorithm**: Model-free reinforcement learning with temporal difference updates
- **Stochastic Environment**: Realistic movement model with 80% success rate and 10% drift
- **Epsilon-Greedy Exploration**: Balanced exploration vs exploitation strategy
- **Visual Policy Display**: Clear visualization of the learned optimal policy
- **Comprehensive Documentation**: Detailed comments explaining all algorithmic components

## 🏗️ Grid World Environment

```
+---+---+---+---+
|   |   |   | +1| ← Goal state (reward: +1.0)
+---+---+---+---+
|   |XXX|   | -1| ← Penalty state (reward: -200.0)  
+---+---+---+---+
| S |   |   |   | ← Start state
+---+---+---+---+
```

**Legend:**
- `S`: Start position (1,1)
- `XXX`: Wall/obstacle (2,2) - blocks movement
- `+1`: Goal state (4,3) - positive reward
- `-1`: Penalty state (4,2) - large negative reward
- Empty cells: Normal states with small negative step cost (-0.04)

### Environment Dynamics

- **States**: 11 valid positions (excluding the wall)
- **Actions**: 4 possible movements (North, South, East, West)
- **Stochastic Transitions**: 
  - 80% chance of moving in intended direction
  - 10% chance of moving 90° left of intended direction
  - 10% chance of moving 90° right of intended direction
- **Collision Handling**: Agent stays in place when hitting walls or boundaries

## 🧠 Q-Learning Algorithm

The agent learns using the Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

**Parameters:**
- **α (alpha)**: Learning rate = 0.1
- **γ (gamma)**: Discount factor = 0.9  
- **ε (epsilon)**: Exploration rate = 1.0 → 0.01 (with decay)

**Key Components:**
- **Q-table**: Stores expected future rewards for each state-action pair
- **Epsilon-greedy policy**: Balances exploration of new actions vs exploitation of known good actions
- **Temporal difference learning**: Updates estimates based on immediate rewards and future value estimates

## 📋 Requirements

### System Requirements
- Python 3.11 or higher
- macOS, Linux, or Windows

### Dependencies
- `numpy>=2.3.3` (for numerical computations and random sampling)
- `random` (built-in Python module)

## ⚙️ Installation

### Option 1: Using UV (Recommended)

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python package management.

1. **Install UV** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and setup the project**:
   ```bash
   git clone https://github.com/Jeevan-HM/RL-in-Robotics.git
   cd RL-in-Robotics
   ```

3. **Install dependencies** (UV will automatically create a virtual environment):
   ```bash
   uv sync
   ```

### Option 2: Using pip/conda

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jeevan-HM/RL-in-Robotics.git
   cd RL-in-Robotics
   ```

2. **Create a virtual environment**:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n rl-robotics python=3.11
   conda activate rl-robotics
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy>=2.3.3
   ```

## 🚀 Usage

### Running the Training

Execute the main script to train the Q-learning agent:

```bash
# If using UV
uv run python main.py

# If using pip/conda (with activated environment)
python main.py
```

### Expected Output

The training process will display:

1. **Environment Setup Information**:
   ```
   ============================================================
   Q-LEARNING GRIDWORLD TRAINING
   ============================================================
   Environment: 4x3 grid
   Start state: (1, 1)
   Goal state: (4,3) with reward +1.0
   Penalty state: (4,2) with reward -200.0
   Wall state: (2, 2)
   Move reward: -0.04
   ```

2. **Training Progress** (every 10,000 episodes):
   ```
   Episode 10,000/50,000 | Epsilon: 0.6065 | Steps: 15
   Episode 20,000/50,000 | Epsilon: 0.3679 | Steps: 8  
   Episode 30,000/50,000 | Epsilon: 0.2231 | Steps: 6
   Episode 40,000/50,000 | Epsilon: 0.1353 | Steps: 6
   Episode 50,000/50,000 | Epsilon: 0.0821 | Steps: 6
   ```

3. **Final Learned Policy**:
   ```
   Learned Optimal Policy:
   (Arrows show best action from each state)
   +---+----+---+------+
   | → |  → | → | 1.0  |
   +---+----+---+------+
   | ↑ |WALL| ↑ |-200.0|
   +---+----+---+------+
   | ↑ |  → | → |   ↑  |
   +---+----+---+------+
   ```

### Understanding the Output

- **Arrows (↑↓←→)**: Indicate the optimal action to take from each state
- **Numbers**: Show terminal state rewards
- **WALL**: Marks the obstacle position
- **Training Progress**: Shows how exploration (epsilon) decreases and performance improves over time

## 📊 Hyperparameter Tuning

You can modify the learning parameters by editing the constants in `main.py`:

```python
# Environment parameters
PENALTY = -1.0 or -200.0        # Penalty state reward (more negative = stronger avoidance)
EPISODES = 50000        # Training episodes (more = better learning but longer time)

# Agent parameters (in QLearningAgent constructor)
alpha=0.1               # Learning rate (0.1-0.3 typical)
gamma=0.9               # Discount factor (0.9-0.99 typical)
epsilon=1.0             # Initial exploration rate
```

### Parameter Guidelines

- **PENALTY**: More negative values create stronger avoidance of the penalty state
- **EPISODES**: 50,000+ recommended for full convergence
- **alpha (learning rate)**: 
  - Higher (0.3): Faster learning, less stable
  - Lower (0.05): Slower learning, more stable
- **gamma (discount factor)**:
  - Higher (0.99): Values future rewards more
  - Lower (0.8): More myopic, focuses on immediate rewards
- **epsilon decay**: Controls exploration schedule
  - Faster decay: Quicker shift to exploitation
  - Slower decay: Longer exploration period

## 🔬 Experiment Ideas

1. **Different Penalties**: Try penalty values from -1 to -1000 to see policy changes
2. **Learning Rates**: Compare alpha values 0.01, 0.1, 0.5
3. **Discount Factors**: Test gamma values 0.5, 0.9, 0.99
4. **Grid Modifications**: Add more walls or change terminal state positions
5. **Stochastic Variations**: Modify action success probabilities

## 🛠️ Development

### Project Structure

```
RL-in-Robotics/
├── main.py              # Main Q-learning implementation
├── README.md            # This file
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Locked dependency versions (UV)
└── Report/             # Documentation and analysis
    ├── report.pdf
    └── teamwork_gridworld Q-learning due 10-1.pdf
```

### Code Organization

- **`GridWorld` class**: Environment implementation with stochastic transitions
- **`QLearningAgent` class**: Q-learning algorithm with epsilon-greedy policy
- **`print_policy()` function**: Visualization of learned policy
- **Main training loop**: Episode management and progress tracking

## 🤔 Troubleshooting

### Common Issues

1. **Slow Convergence**: 
   - Increase number of episodes
   - Adjust learning rate (try 0.05-0.3)
   - Check epsilon decay rate

2. **Poor Final Policy**:
   - Ensure sufficient training episodes (50,000+)
   - Verify penalty value is sufficiently negative
   - Check that epsilon reaches low values (&lt;0.1)

3. **Import Errors**:
   - Ensure numpy is installed: `pip install numpy`
   - Check Python version (3.11+ required)

### Performance Tips

- **Training Speed**: Reduce episodes for faster testing, increase for better policies
- **Memory Usage**: Current implementation scales O(states × actions) - efficient for small grids
- **Visualization**: Set `epsilon=0.0` before calling `print_policy()` for deterministic display

## 📚 Learning Resources

### Reinforcement Learning Concepts
- **Q-Learning**: [Sutton & Barto Chapter 6](http://incompleteideas.net/book/RLbook2020.pdf)
- **Epsilon-Greedy**: Exploration vs Exploitation tradeoff
- **Temporal Difference Learning**: Learning from prediction errors

### Extensions and Advanced Topics
- **Deep Q-Networks (DQN)**: Neural network function approximation
- **Policy Gradient Methods**: Direct policy optimization
- **Actor-Critic Methods**: Combining value and policy learning
- **Multi-Agent RL**: Multiple agents in shared environments

## 👨‍💻 Author

**Jeevan-Hebbal Manjunath**
- GitHub: [@Jeevan-HM](https://github.com/Jeevan-HM)
- Project: [RL-in-Robotics](https://github.com/Jeevan-HM/RL-in-Robotics)

---
