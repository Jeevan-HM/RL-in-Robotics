# Q-Learning GridWorld - Reinforcement Learning with Hyperparameter Analysis

A comprehensive implementation of the Q-learning algorithm applied to a classic GridWorld navigation problem. This project demonstrates fundamental reinforcement learning concepts and includes powerful tools for hyperparameter analysis and visualization.

## 🎯 Project Overview

This project implements a **Q-learning agent** that learns to navigate a 4×3 grid world. Beyond a basic implementation, this script provides a full suite of tools to analyze and visualize the impact of different hyperparameters on agent performance, making it an excellent framework for understanding the nuances of reinforcement learning.

### Key Features

- **Q-Learning Algorithm**: A robust implementation of the model-free reinforcement learning algorithm.
- **Stochastic Environment**: A realistic movement model where actions have an 80% success rate, with a 10% chance of drifting left or right.
- **Comprehensive Hyperparameter Analysis**: Automatically runs experiments and generates detailed plots for learning rate (α), discount factor (γ), and exploration decay (ε).
- **Advanced Data Visualization**: Generates a detailed 12-panel plot including convergence graphs, performance comparisons, stability analysis, and a performance heatmap.
- **Scenario Comparison**: Includes a high-penalty scenario to demonstrate how the agent adapts to different environmental constraints.
- **Modular and Refactored Code**: The code is organized with clear separation of concerns, making it easy to understand and extend.

## 📊 Analysis and Visualization

The script automatically generates a comprehensive hyperparameter analysis plot that provides deep insights into the learning process.


This visualization includes:
1.  **Q-Value Convergence**: Tracks how Q-values stabilize over time for different hyperparameters.
2.  **Learning Performance**: Shows the moving average of episode rewards to evaluate policy effectiveness.
3.  **Final Q-Value Stability**: Bar charts comparing the final convergence of Q-values.
4.  **Exploration Schedule**: Visualizes how the exploration rate (ε) decreases over time with different decay settings.
5.  **Learning Efficiency**: Compares episode lengths to see how quickly the agent finds the goal.
6.  **Performance Heatmap**: A 2D grid showing the final performance for different combinations of `alpha` and `gamma`.

## 🏗️ Grid World Environment

```
+---+---+---+---+
|   |   |   | +1| ← Goal state (reward: +1.0)
+---+---+---+---+
|   |XXX|   | -1| ← Penalty state (reward: -1.0 or -200.0)
+---+---+---+---+
| S |   |   |   | ← Start state
+---+---+---+---+
```

- **States**: 11 valid positions (excluding the wall).
- **Actions**: 4 movements (North, South, East, West).
- **Stochastic Transitions**: 80% chance of intended move, 10% chance of moving 90° left, 10% chance of moving 90° right.

## 🧠 Q-Learning Algorithm

The agent learns using the Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

- **α (alpha)**: Learning Rate
- **γ (gamma)**: Discount Factor
- **ε (epsilon)**: Exploration Rate

## 📋 Requirements

### System Requirements
- Python 3.11 or higher
- macOS, Linux, or Windows

### Dependencies
- `numpy`
- `matplotlib`

## ⚙️ Installation

### Using UV (Recommended)

This project uses [UV](https://docs.astral.sh/uv/) for fast Python package management.

1.  **Install UV**:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone and setup**:
    ```bash
    git clone https://github.com/Jeevan-HM/RL-in-Robotics.git
    cd RL-in-Robotics
    ```
3.  **Install dependencies** (UV creates a virtual environment automatically):
    ```bash
    uv sync
    ```

### Using pip/conda

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Jeevan-HM/RL-in-Robotics.git
    cd RL-in-Robotics
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install numpy matplotlib
    ```

## 🚀 Usage

Execute the main script to run the full analysis pipeline. The script will:
1.  Run a single training session with default parameters.
2.  Conduct a detailed hyperparameter analysis.
3.  Run a comparative scenario with a high penalty.
4.  Generate and display plots.

```bash
# If using UV
uv run python main.py

# If using pip/conda (with activated environment)
python main.py
```

### Customizing Experiments

You can easily customize the experiments by modifying the configuration parameters inside the `main()` function in `main.py`:

```python
def main():
    # Training Configuration
    episodes = 100
    base_penalty = -1.0
    high_penalty = -200.0

    # Hyperparameter Tuning Configuration
    alpha_values = [0.01, 0.1, 0.5, 0.9]
    gamma_values = [0.5, 0.9, 0.99]
    epsilon_decay_values = [0.001, 0.99, 0.995, 0.999, 0.9999]
    
    # ... rest of the main function
```

## �️ Code Structure

The project is organized into modular components for clarity and extensibility.

```
RL-in-Robotics/
├── main.py              # Main script with analysis functions
├── README.md            # This file
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # Locked dependency versions (UV)
└── Figure_1.png         # Example output plot
```

### Key Functions in `main.py`

- **`GridWorld` class**: Implements the environment, including state transitions and rewards.
- **`QLearningAgent` class**: Implements the Q-learning algorithm, including the learning rule and exploration strategy.
- **`run_training_session()`**: A unified function to run a complete training loop for a given agent and environment.
- **`analyze_hyperparameters()`**: Orchestrates the hyperparameter study by running experiments for different values of α, γ, and ε.
- **`plot_hyperparameter_analysis()`**: Generates the comprehensive 12-panel plot for analysis.
- **`main()`**: The main entry point that configures and runs all experiments and visualizations.

## 👨‍💻 Author

**Jeevan-Hebbal Manjunath**
- GitHub: [@Jeevan-HM](https://github.com/Jeevan-HM)

**Yeshwanth Reddy Gurreddy**

**Varun Karthik**

Project: [RL-in-Robotics](https://github.com/Jeevan-HM/RL-in-Robotics)
