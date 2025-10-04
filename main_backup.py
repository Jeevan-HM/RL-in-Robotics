"""
Q-Learning implementation for GridWorld environment with analysis and visualization.

This module implements a classic reinforcement learning problem where an agent
learns to navigate a 4x3 grid. It has been enhanced to track key metrics
during training and generate plots to analyze convergence and the impact of
hyperparameters.

Key Components:
- GridWorld: The environment that defines states, actions, and rewards
- QLearningAgent: The learning agent that updates its Q-table based on experience
- Stochastic transitions: Actions have uncertainty (80% intended, 10% left, 10% right)
- Analysis functions: Plotting for convergence and hyperparameter tuning

Author: Jeevan Hebbal Manjunath (Enhanced by Gemini)
Date: October 2024
"""

import copy
import random

import matplotlib.pyplot as plt
import numpy as np


class GridWorld:
    """GridWorld environment for the Q-learning problem.

    Manages states, actions, rewards, and transitions in a 4x3 grid world.
    The grid coordinates are (column, row), starting from (1, 1).

    Grid Layout:
    +---+---+---+---+
    |   |   |   | +1| (4,3) - Goal state (+1 reward)
    +---+---+---+---+
    |   |XXX|   | -1| (4,2) - Terminal penalty state
    +---+---+---+---+
    | S |   |   |   | (1,1) - Start state
    +---+---+---+---+

    Where: S=Start, XXX=Wall, +1=Goal, -1=Penalty
    """

    def __init__(self, penalty=-1.0):
        """Initialize the GridWorld environment."""
        self.grid_size = (4, 3)
        self.wall_state = (2, 2)
        self.start_state = (1, 1)
        self.terminal_states = {(4, 3): 1.0, (4, 2): penalty}
        self.move_reward = -0.04
        self.actions = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
        self.action_names = list(self.actions.keys())

    def get_states(self):
        """Generate all valid states in the grid world."""
        states = []
        for x in range(1, self.grid_size[0] + 1):
            for y in range(1, self.grid_size[1] + 1):
                if (x, y) != self.wall_state:
                    states.append((x, y))
        return states

    def step(self, state, action_name):
        """Execute one step in the environment with stochastic transitions."""
        if state in self.terminal_states:
            return state, 0

        outcomes = self._get_stochastic_outcomes(action_name)
        rng = np.random.default_rng()
        chosen_action_name = rng.choice(
            [outcomes["intent"], outcomes["left"], outcomes["right"]],
            p=[0.8, 0.1, 0.1],
        )

        action = self.actions[chosen_action_name]
        next_state = (state[0] + action[0], state[1] + action[1])

        if (
            next_state == self.wall_state
            or not (1 <= next_state[0] <= self.grid_size[0])
            or not (1 <= next_state[1] <= self.grid_size[1])
        ):
            next_state = state

        reward = self.terminal_states.get(next_state, self.move_reward)
        return next_state, reward

    def _get_stochastic_outcomes(self, action_name):
        """Determine possible action outcomes for stochastic transitions."""
        if action_name == "N":
            return {"intent": "N", "left": "W", "right": "E"}
        if action_name == "S":
            return {"intent": "S", "left": "E", "right": "W"}
        if action_name == "E":
            return {"intent": "E", "left": "N", "right": "S"}
        if action_name == "W":
            return {"intent": "W", "left": "S", "right": "N"}
        raise ValueError(f"Invalid action name: {action_name}")


class QLearningAgent:
    """Q-Learning Agent that learns optimal policies through experience."""

    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        """Initialize the Q-Learning agent with hyperparameters."""
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.q_table = {
            state: {action: 0.0 for action in self.actions} for state in self.states
        }

    def get_policy(self):
        """Extract the current optimal policy from the Q-table."""
        policy = {}
        for state in self.states:
            if state not in self.q_table:
                continue
            policy[state] = max(self.q_table[state], key=self.q_table[state].get)
        return policy

    def choose_action(self, state):
        """Select action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state]
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning algorithm."""
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Gradually reduce exploration rate over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def print_policy(agent, env):
    """Visualize the learned policy as a grid with directional arrows."""
    policy_grid = [
        [" " for _ in range(env.grid_size[0])] for _ in range(env.grid_size[1])
    ]
    action_arrows = {"N": "↑", "S": "↓", "E": "→", "W": "←"}

    # Set agent to exploitation mode for accurate policy display
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for state in agent.states:
        x, y = state
        grid_row = env.grid_size[1] - y
        grid_col = x - 1
        if state in env.terminal_states:
            policy_grid[grid_row][grid_col] = f"{env.terminal_states[state]:>+g}"
        elif state == env.wall_state:
            policy_grid[grid_row][grid_col] = "WALL"
        else:
            best_action = agent.choose_action(state)
            policy_grid[grid_row][grid_col] = action_arrows[best_action]

    # Restore original epsilon
    agent.epsilon = original_epsilon

    print("\nLearned Optimal Policy:")
    for row in policy_grid:
        print("+---" * len(row) + "+")
        print("| " + " | ".join(f"{cell:^1}" for cell in row) + " |")
    print("+---" * len(policy_grid[0]) + "+")


def plot_convergence(q_changes, policy_changes):
    """Plots the convergence of Q-values and policy over episodes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Q-value changes
    ax1.plot(q_changes, label="Sum of Q-value Changes", color="b")
    ax1.set_ylabel("Total Q-Value Change per Episode")
    ax1.set_title("Q-Value Convergence Over Time")
    ax1.grid(True)
    ax1.legend()

    # Plot policy changes
    ax2.plot(policy_changes, label="Number of Policy Changes", color="r")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Number of State-Action Changes")
    ax2.set_title("Policy Convergence Over Time")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    plt.show()


def run_training(env, agent, episodes):
    """Runs the main Q-learning training loop and returns tracked metrics."""
    q_value_changes = []
    policy_changes = []

    for episode in range(episodes):
        # Store state before the episode for comparison
        q_table_old = copy.deepcopy(agent.q_table)
        policy_old = agent.get_policy()

        state = env.start_state
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward = env.step(state, action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if state in env.terminal_states:
                done = True

        agent.decay_epsilon()

        # Track metrics
        q_change = sum(
            abs(agent.q_table[s][a] - q_table_old[s][a])
            for s in agent.states
            for a in agent.actions
        )
        q_value_changes.append(q_change)

        policy_new = agent.get_policy()
        p_change = sum(
            1 for s in agent.states if policy_old.get(s) != policy_new.get(s)
        )
        policy_changes.append(p_change)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:,}/{episodes:,} | Epsilon: {agent.epsilon:.4f}",
            )

    return q_value_changes, policy_changes


def calculate_convergence_metrics(results, q_threshold=0.001, stability_window=10):
    """Calculate convergence metrics for a given set of training results."""
    q_changes = results["q_changes"]
    policy_changes = results["policy_changes"]

    # Find Q-value convergence episode
    q_conv_episode = None
    for i, change in enumerate(q_changes):
        if change < q_threshold:
            q_conv_episode = i
            break

    # Find policy stabilization episode (adjusted for shorter episodes)
    policy_conv_episode = None
    for i in range(stability_window, len(policy_changes)):
        window_changes = sum(policy_changes[i - stability_window : i])
        if window_changes == 0:
            policy_conv_episode = i - stability_window
            break

    final_window = 10  # Adjusted for 100 episodes
    final_q_change = (
        q_changes[-final_window] if len(q_changes) >= final_window else q_changes[-1]
    )

    return {
        "q_convergence": q_conv_episode,
        "policy_convergence": policy_conv_episode,
        "final_q_change": final_q_change,
        "total_policy_changes": sum(policy_changes),
    }


def plot_smoothed_convergence(ax, results_dict, param_name, title_suffix):
    """Plot smoothed Q-value convergence for multiple parameter values."""
    window = 5  # Smaller window for 100 episodes
    for param_value, results in results_dict.items():
        q_changes = results["q_changes"]
        if len(q_changes) >= window:
            q_smooth = np.convolve(q_changes, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(q_changes)),
                q_smooth,
                label=f"{param_name} = {param_value}",
            )
        else:
            ax.plot(q_changes, label=f"{param_name} = {param_value}")

    ax.set_title(f"Smoothed Q-Value Convergence ({title_suffix})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Moving Average Q-Value Change")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")


def plot_convergence_bars(ax, convergence_data, param_key, param_name, title_suffix):
    """Plot bar charts comparing convergence metrics."""
    param_values = [d[param_key] for d in convergence_data]
    q_conv_episodes = [
        d["q_convergence"] if d["q_convergence"] else 100 for d in convergence_data
    ]
    total_changes = [d["total_policy_changes"] for d in convergence_data]

    x = np.arange(len(param_values))
    width = 0.35

    ax.bar(x, q_conv_episodes, width, label="Q-Value Convergence Episode", alpha=0.8)
    ax_twin = ax.twinx()
    ax_twin.bar(
        x + width,
        total_changes,
        width,
        label="Total Policy Changes",
        alpha=0.8,
        color="orange",
    )

    ax.set_xlabel(f"{param_name}")
    ax.set_ylabel("Episodes to Q-Convergence", color="blue")
    ax_twin.set_ylabel("Total Policy Changes", color="orange")
    ax.set_title(f"Convergence Metrics: {title_suffix}")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{v}" for v in param_values])
    ax.grid(True, alpha=0.3)


def analyze_hyperparameters():
    """Trains agents with different hyperparameters and plots comprehensive convergence results."""
    episodes = 100
    env = GridWorld(penalty=-1.0)

    # --- Learning Rate (alpha) Analysis ---
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))

    print("Analyzing Learning Rate (alpha) effects...")
    alpha_results = {}
    for alpha in [0.01, 0.1, 0.5, 0.9]:
        print(f"  Training with alpha = {alpha}...")
        agent = QLearningAgent(env.get_states(), env.action_names, alpha=alpha)
        q_changes, policy_changes = run_training(env, agent, episodes)
        alpha_results[alpha] = {
            "q_changes": q_changes,
            "policy_changes": policy_changes,
            "final_policy": agent.get_policy(),
        }

        # Plot Q-value convergence
        ax1.plot(q_changes, label=f"alpha = {alpha}", alpha=0.8)
        # Plot cumulative policy changes
        ax2.plot(np.cumsum(policy_changes), label=f"alpha = {alpha}", alpha=0.8)

    ax1.set_title("Q-Value Convergence: Learning Rate (alpha)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Q-Value Change per Episode")
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale("log")

    ax2.set_title("Policy Convergence: Learning Rate (alpha)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Policy Changes")
    ax2.legend()
    ax2.grid(True)

    # --- Discount Factor (gamma) Analysis ---
    print("\nAnalyzing Discount Factor (gamma) effects...")
    gamma_results = {}
    for gamma in [0.5, 0.9, 0.99]:
        print(f"  Training with gamma = {gamma}...")
        agent = QLearningAgent(env.get_states(), env.action_names, gamma=gamma)
        q_changes, policy_changes = run_training(env, agent, episodes)
        gamma_results[gamma] = {
            "q_changes": q_changes,
            "policy_changes": policy_changes,
            "final_policy": agent.get_policy(),
        }

        # Plot Q-value convergence
        ax3.plot(q_changes, label=f"gamma = {gamma}", alpha=0.8)
        # Plot cumulative policy changes
        ax4.plot(np.cumsum(policy_changes), label=f"gamma = {gamma}", alpha=0.8)

    ax3.set_title("Q-Value Convergence: Discount Factor (gamma)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Total Q-Value Change per Episode")
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale("log")

    ax4.set_title("Policy Convergence: Discount Factor (gamma)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Cumulative Policy Changes")
    ax4.legend()
    ax4.grid(True)

    # --- Epsilon Analysis ---
    print("\nAnalyzing Epsilon effects...")
    epsilon_results = {}
    for epsilon in [0.1, 0.3, 0.7, 1.0]:
        print(f"  Training with epsilon = {epsilon}...")
        agent = QLearningAgent(env.get_states(), env.action_names, epsilon=epsilon)
        q_changes, policy_changes = run_training(env, agent, episodes)
        epsilon_results[epsilon] = {
            "q_changes": q_changes,
            "policy_changes": policy_changes,
            "final_policy": agent.get_policy(),
        }

        # Plot Q-value convergence
        ax5.plot(q_changes, label=f"epsilon = {epsilon}", alpha=0.8)
        # Plot cumulative policy changes
        ax6.plot(np.cumsum(policy_changes), label=f"epsilon = {epsilon}", alpha=0.8)

    ax5.set_title("Q-Value Convergence: Epsilon")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Total Q-Value Change per Episode")
    ax5.legend()
    ax5.grid(True)
    ax5.set_yscale("log")

    ax6.set_title("Policy Convergence: Epsilon")
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Cumulative Policy Changes")
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Convergence Speed Analysis ---
    analyze_convergence_speed(alpha_results, gamma_results, epsilon_results)


def analyze_convergence_speed(alpha_results, gamma_results, epsilon_results):
    """Analyze and visualize convergence speed metrics for different hyperparameters."""
    print("\n" + "=" * 60)
    print("CONVERGENCE SPEED ANALYSIS")
    print("=" * 60)

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))

    # Calculate convergence metrics for alpha values
    alpha_convergence_data = []
    for alpha, results in alpha_results.items():
        metrics = calculate_convergence_metrics(results)
        metrics["alpha"] = alpha
        alpha_convergence_data.append(metrics)

    # Calculate convergence metrics for gamma values
    gamma_convergence_data = []
    for gamma, results in gamma_results.items():
        metrics = calculate_convergence_metrics(results)
        metrics["gamma"] = gamma
        gamma_convergence_data.append(metrics)

    # Calculate convergence metrics for epsilon values
    epsilon_convergence_data = []
    for epsilon, results in epsilon_results.items():
        metrics = calculate_convergence_metrics(results)
        metrics["epsilon"] = epsilon
        epsilon_convergence_data.append(metrics)

    # Plot smoothed convergence curves
    plot_smoothed_convergence(ax1, alpha_results, "alpha", "Learning Rate")
    plot_smoothed_convergence(ax2, gamma_results, "gamma", "Discount Factor")
    plot_smoothed_convergence(ax3, epsilon_results, "epsilon", "Epsilon")

    # Plot convergence comparison bar charts
    plot_convergence_bars(
        ax4, alpha_convergence_data, "alpha", "Learning Rate (alpha)", "Learning Rate"
    )
    plot_convergence_bars(
        ax5,
        gamma_convergence_data,
        "gamma",
        "Discount Factor (gamma)",
        "Discount Factor",
    )
    plot_convergence_bars(
        ax6, epsilon_convergence_data, "epsilon", "Epsilon", "Epsilon"
    )

    plt.tight_layout()
    plt.show()

    # Print convergence summary
    print_convergence_summary(
        alpha_convergence_data, gamma_convergence_data, epsilon_convergence_data
    )


def print_convergence_summary(alpha_data, gamma_data, epsilon_data):
    """Print a detailed summary of convergence analysis results."""
    # Constants
    max_episodes = 100
    fast_convergence_threshold = 25  # Adjusted for 100 episodes

    print("\nCONVERGENCE ANALYSIS SUMMARY:")
    print("-" * 50)

    print("\nLearning Rate (alpha) Analysis:")
    print(
        f"{'alpha':<6} {'Q-Conv Episode':<15} {'Policy Changes':<15} {'Convergence Quality':<20}"
    )
    print("-" * 70)
    for data in alpha_data:
        q_conv = data["q_convergence"] if data["q_convergence"] else f">{max_episodes}"
        if data["q_convergence"] and data["q_convergence"] < fast_convergence_threshold:
            quality = "Fast"
        elif data["q_convergence"]:
            quality = "Slow"
        else:
            quality = "No Convergence"
        print(
            f"{data['alpha']:<6} {q_conv!s:<15} {data['total_policy_changes']:<15} {quality:<20}"
        )

    print("\nDiscount Factor (gamma) Analysis:")
    print(
        f"{'gamma':<6} {'Q-Conv Episode':<15} {'Policy Changes':<15} {'Convergence Quality':<20}"
    )
    print("-" * 70)
    for data in gamma_data:
        q_conv = data["q_convergence"] if data["q_convergence"] else f">{max_episodes}"
        if data["q_convergence"] and data["q_convergence"] < fast_convergence_threshold:
            quality = "Fast"
        elif data["q_convergence"]:
            quality = "Slow"
        else:
            quality = "No Convergence"
        print(
            f"{data['gamma']:<6} {q_conv!s:<15} {data['total_policy_changes']:<15} {quality:<20}"
        )

    print("\nEpsilon Analysis:")
    print(
        f"{'epsilon':<8} {'Q-Conv Episode':<15} {'Policy Changes':<15} {'Convergence Quality':<20}"
    )
    print("-" * 70)
    for data in epsilon_data:
        q_conv = data["q_convergence"] if data["q_convergence"] else f">{max_episodes}"
        if data["q_convergence"] and data["q_convergence"] < fast_convergence_threshold:
            quality = "Fast"
        elif data["q_convergence"]:
            quality = "Slow"
        else:
            quality = "No Convergence"
        print(
            f"{data['epsilon']:<8} {q_conv!s:<15} {data['total_policy_changes']:<15} {quality:<20}"
        )

    # Find best parameters
    best_alpha = min(
        alpha_data,
        key=lambda x: x["q_convergence"] if x["q_convergence"] else float("inf"),
    )
    best_gamma = min(
        gamma_data,
        key=lambda x: x["q_convergence"] if x["q_convergence"] else float("inf"),
    )
    best_epsilon = min(
        epsilon_data,
        key=lambda x: x["q_convergence"] if x["q_convergence"] else float("inf"),
    )

    print("\nRECOMMENDED HYPERPARAMETERS:")
    print(
        f"Best Learning Rate (alpha): {best_alpha['alpha']} (converged at episode {best_alpha['q_convergence']})"
    )
    print(
        f"Best Discount Factor (gamma): {best_gamma['gamma']} (converged at episode {best_gamma['q_convergence']})"
    )
    print(
        f"Best Epsilon: {best_epsilon['epsilon']} (converged at episode {best_epsilon['q_convergence']})"
    )
    print(
        "\nNote: 'Best' is defined as fastest Q-value convergence to threshold of 0.001"
    )


if __name__ == "__main__":
    EPISODES = 100

    # --- Part 1 & 2: Base Case and Convergence Analysis ---
    print("=" * 60)
    print("RUNNING BASE CASE (Penalty=-1) & CONVERGENCE ANALYSIS")
    print("=" * 60)
    env_base = GridWorld(penalty=-1.0)
    agent_base = QLearningAgent(env_base.get_states(), env_base.action_names)

    q_changes, p_changes = run_training(env_base, agent_base, EPISODES)
    print("\nTRAINING COMPLETED FOR BASE CASE")
    print_policy(agent_base, env_base)
    plot_convergence(q_changes, p_changes)

    # --- Part 3: High Penalty Analysis ---
    print("\n" + "=" * 60)
    print("RUNNING HIGH PENALTY CASE (Penalty=-200)")
    print("=" * 60)
    env_high_penalty = GridWorld(penalty=-200.0)
    agent_high_penalty = QLearningAgent(
        env_high_penalty.get_states(),
        env_high_penalty.action_names,
    )

    run_training(env_high_penalty, agent_high_penalty, EPISODES)
    print("\nTRAINING COMPLETED FOR HIGH PENALTY CASE")
    print_policy(agent_high_penalty, env_high_penalty)

    # --- Hyperparameter analysis ---
    print("\n" + "=" * 60)
    print("RUNNING HYPERPARAMETER ANALYSIS")
    print("=" * 60)
    analyze_hyperparameters()
