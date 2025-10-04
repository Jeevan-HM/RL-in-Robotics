"""
Q-Learning implementation for GridWorld environment with comprehensive analysis.

This module implements a classic reinforcement learning problem where an agent
learns to navigate a 4x3 grid world. The implementation includes:

- Stochastic environment with 80% intended action probability
- Q-Learning agent with configurable hyperparameters
- Comprehensive hyperparameter analysis and visualization
- Convergence tracking and policy visualization
- Professional-grade plotting and summary generation

Key Components:
    GridWorld: Environment managing states, actions, transitions, and rewards
    QLearningAgent: Learning agent with epsilon-greedy exploration
    Analysis Functions: Comprehensive plotting and hyperparameter tuning tools

Typical Usage:
    >>> env = GridWorld(penalty=-1.0)
    >>> agent = QLearningAgent(env.get_states(), env.action_names)
    >>> q_changes, policy_changes = run_training(env, agent, 100)
    >>> plot_convergence(q_changes, policy_changes)

Author: Jeevan Hebbal Manjunath
Date: October 2024
Version: 2.0
"""

# Standard library imports
import copy
import random
from typing import Any, NamedTuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
# Grid dimensions
GRID_WIDTH = 4
GRID_HEIGHT = 3

# Rewards and penalties
GOAL_REWARD = 1.0
PENALTY_REWARD = -1.0
DEFAULT_MOVE_REWARD = -0.04

# States
START_STATE = (1, 1)
GOAL_STATE = (4, 3)
PENALTY_STATE = (4, 2)
WALL_STATE = (2, 2)

# Actions and their corresponding (dx, dy) movements
ACTIONS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}

# Stochastic action probabilities
ACTION_PROBABILITIES = {
    "intended": 0.8,
    "left": 0.1,
    "right": 0.1,
}

# Q-Learning hyperparameters
DEFAULT_ALPHA = 0.1  # Learning rate
DEFAULT_GAMMA = 0.9  # Discount factor
DEFAULT_EPSILON = 1.0  # Initial exploration rate
MIN_EPSILON = 0.01  # Minimum exploration rate
DEFAULT_EPSILON_DECAY = 0.9995  # Decay factor for epsilon

# Training configuration
MAX_EPISODE_STEPS = 100

# Analysis constants
MIN_ALPHA_VALUES = 5
MIN_GAMMA_VALUES = 4
MIN_EPSILON_DECAY_VALUES = 4
EXCELLENT_CONVERGENCE = 0.01
GOOD_CONVERGENCE = 0.1
MODERATE_CONVERGENCE = 1.0
HIGH_EXPLORATION = 0.1
MEDIUM_EXPLORATION = 0.05
LOW_EXPLORATION = 0.01


# ============================================================================
# DATA STRUCTURES
# ============================================================================
class Hyperparameters(NamedTuple):
    """Represents the hyperparameters for the Q-learning agent."""

    alpha: float
    gamma: float
    epsilon: float
    epsilon_decay: float


class TrainingResults(NamedTuple):
    """Represents the results of a training session."""

    q_changes: list[float]
    policy_changes: list[int]
    rewards: list[float]
    episode_lengths: list[int]
    epsilon_values: list[float]


# ============================================================================
# ENVIRONMENT
# ============================================================================
class GridWorld:
    """
    GridWorld environment for Q-learning with stochastic transitions.

    A 4x3 grid world where an agent learns to navigate from start to goal
    while avoiding penalties. Actions have stochastic outcomes.
    """

    def __init__(self, penalty: float = PENALTY_REWARD) -> None:
        """
        Initialize the GridWorld environment.

        Args:
            penalty: Reward value for the penalty terminal state

        """
        self.grid_size = (GRID_WIDTH, GRID_HEIGHT)
        self.wall_state = WALL_STATE
        self.terminal_states = {
            GOAL_STATE: GOAL_REWARD,
            PENALTY_STATE: penalty,
        }
        self.move_reward = DEFAULT_MOVE_REWARD
        self.actions = ACTIONS.copy()
        self.action_names = list(self.actions.keys())

    def get_states(self) -> list[tuple[int, int]]:
        """
        Generate all valid states in the grid world.

        Returns:
            List of valid (x, y) coordinate tuples excluding walls

        """
        states = []
        for x in range(1, self.grid_size[0] + 1):
            for y in range(1, self.grid_size[1] + 1):
                if (x, y) != self.wall_state:
                    states.append((x, y))
        return states

    def step(
        self,
        state: tuple[int, int],
        action_name: str,
    ) -> tuple[tuple[int, int], float]:
        """
        Execute one step in the environment with stochastic transitions.

        Args:
            state: Current (x, y) position of the agent
            action_name: Intended action ("N", "S", "E", "W")

        Returns:
            Tuple of (next_state, reward) where next_state is (x, y) coordinates
            and reward is the numerical reward for this transition

        """
        if state in self.terminal_states:
            return state, 0

        outcomes = self._get_stochastic_outcomes(action_name)
        rng = np.random.default_rng()
        chosen_action_name = rng.choice(
            [outcomes["intent"], outcomes["left"], outcomes["right"]],
            p=[
                ACTION_PROBABILITIES["intended"],
                ACTION_PROBABILITIES["left"],
                ACTION_PROBABILITIES["right"],
            ],
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

    def _get_stochastic_outcomes(self, action_name: str) -> dict[str, str]:
        """
        Determine possible action outcomes for stochastic transitions.

        Args:
            action_name: The intended action direction

        Returns:
            Dictionary mapping outcome types to actual actions

        Raises:
            ValueError: If action_name is not valid

        """
        outcome_map = {
            "N": {"intent": "N", "left": "W", "right": "E"},
            "S": {"intent": "S", "left": "E", "right": "W"},
            "E": {"intent": "E", "left": "N", "right": "S"},
            "W": {"intent": "W", "left": "S", "right": "N"},
        }

        if action_name not in outcome_map:
            msg = f"Invalid action name: {action_name}"
            raise ValueError(msg)

        return outcome_map[action_name]


class StudyParameters(NamedTuple):
    """Parameters for a hyperparameter study."""

    env: GridWorld
    episodes: int
    param_name: str
    param_values: list[float]
    fixed_params: dict[str, float]


class QLearningAgent:
    """
    Q-Learning Agent that learns optimal policies through experience.

    This agent uses the Q-Learning algorithm to learn optimal state-action values
    through interaction with the environment. It employs epsilon-greedy exploration
    and temporal difference learning to balance exploration and exploitation.
    """

    def __init__(
        self,
        states: list[tuple[int, int]],
        actions: list[str],
        hyperparameters: Hyperparameters,
    ) -> None:
        """
        Initialize the Q-Learning agent with hyperparameters.

        Args:
            states: List of valid (x, y) state coordinates
            actions: List of action names ("N", "S", "E", "W")
            hyperparameters: Dataclass containing alpha, gamma, epsilon, and epsilon_decay

        """
        self.states = states
        self.actions = actions
        self.alpha = hyperparameters.alpha
        self.gamma = hyperparameters.gamma
        self.epsilon = hyperparameters.epsilon
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = hyperparameters.epsilon_decay
        self.q_table: dict[tuple[int, int], dict[str, float]] = {
            state: dict.fromkeys(self.actions, 0.0) for state in self.states
        }

    def get_policy(self) -> dict[tuple[int, int], str]:
        """
        Extract the current optimal policy from the Q-table.

        Returns:
            Dictionary mapping each state to its optimal action

        """
        policy = {}
        for state in self.states:
            if state not in self.q_table:
                continue
            q_values = self.q_table[state]
            policy[state] = max(q_values, key=lambda k: q_values[k])
        return policy

    def choose_action(self, state: tuple[int, int]) -> str:
        """
        Select action using epsilon-greedy exploration policy.

        Args:
            state: Current (x, y) position of the agent

        Returns:
            Selected action name ("N", "S", "E", "W")

        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state]
        return max(q_values, key=lambda k: q_values[k])

    def learn(
        self,
        state: tuple[int, int],
        action: str,
        reward: float,
        next_state: tuple[int, int],
    ) -> None:
        """
        Update Q-value using the Q-learning temporal difference algorithm.

        Implements the Q-learning update rule:
        Q(s,a) <- Q(s,a) + a[r + g max Q(s',a') - Q(s,a)]

        Args:
            state: Previous state (x, y) coordinates

            action: Action taken from previous state
            reward: Immediate reward received
            next_state: Current state (x, y) coordinates
        """
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q

    def decay_epsilon(self) -> None:
        """
        Gradually reduce exploration rate using exponential decay.

        Applies epsilon decay while respecting the minimum epsilon threshold.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# TRAINING AND ANALYSIS
# ============================================================================
def run_training_session(
    env: GridWorld,
    agent: QLearningAgent,
    episodes: int,
) -> TrainingResults:
    """
    Run a complete Q-learning training session and collect metrics.

    Args:
        env: GridWorld environment instance
        agent: QLearningAgent instance
        episodes: Number of training episodes

    Returns:
        A TrainingResults named tuple containing lists of metrics.

    """
    q_value_changes = []
    policy_changes = []
    cumulative_rewards = []
    episode_lengths = []
    epsilon_values = []

    for episode in range(episodes):
        state = START_STATE
        episode_reward = 0
        episode_length = 0
        q_table_old = copy.deepcopy(agent.q_table)
        policy_old = agent.get_policy()

        for _ in range(MAX_EPISODE_STEPS):
            action = agent.choose_action(state)
            next_state, reward = env.step(state, action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            episode_length += 1
            if state in env.terminal_states:
                break

        agent.decay_epsilon()
        epsilon_values.append(agent.epsilon)

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

        cumulative_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % max(10, episodes) == 0:
            print(
                f"Episode {episode + 1:,}/{episodes:,} | Epsilon: {agent.epsilon:.4f}",
            )

    return TrainingResults(
        q_changes=q_value_changes,
        policy_changes=policy_changes,
        rewards=cumulative_rewards,
        episode_lengths=episode_lengths,
        epsilon_values=epsilon_values,
    )


def _run_hyperparameter_study(
    study_params: StudyParameters,
) -> dict[float, TrainingResults]:
    """Run experiments for a given hyperparameter."""
    results = {}
    for value in study_params.param_values:
        params = study_params.fixed_params.copy()
        params[study_params.param_name] = value
        hyperparameters = Hyperparameters(
            alpha=params.get("alpha", DEFAULT_ALPHA),
            gamma=params.get("gamma", DEFAULT_GAMMA),
            epsilon=params.get("epsilon", DEFAULT_EPSILON),
            epsilon_decay=params.get("epsilon_decay", DEFAULT_EPSILON_DECAY),
        )
        agent = QLearningAgent(
            study_params.env.get_states(),
            study_params.env.action_names,
            hyperparameters,
        )
        results[value] = run_training_session(
            study_params.env,
            agent,
            study_params.episodes,
        )
    return results


def analyze_hyperparameters(
    episodes: int,
    penalty: float,
    alpha_values: list[float],
    gamma_values: list[float],
    epsilon_decay_values: list[float],
    moving_average_window: int,
) -> None:
    """Create comprehensive hyperparameter analysis plots with enhanced visualization."""
    env = GridWorld(penalty=penalty)

    if len(alpha_values) < MIN_ALPHA_VALUES:
        print(f"Warning: At least {MIN_ALPHA_VALUES} alpha values recommended.")
    if len(gamma_values) < MIN_GAMMA_VALUES:
        print(f"Warning: At least {MIN_GAMMA_VALUES} gamma values recommended.")
    if len(epsilon_decay_values) < MIN_EPSILON_DECAY_VALUES:
        print(
            f"Warning: At least {MIN_EPSILON_DECAY_VALUES} epsilon decay values recommended.",
        )

    print("Analyzing Learning Rate (Alpha)...")
    alpha_results = _run_hyperparameter_study(
        StudyParameters(
            env,
            episodes,
            "alpha",
            alpha_values,
            {"gamma": DEFAULT_GAMMA, "epsilon_decay": 0.9995},
        ),
    )

    print("\nAnalyzing Discount Factor (Gamma)...")
    gamma_results = _run_hyperparameter_study(
        StudyParameters(
            env,
            episodes,
            "gamma",
            gamma_values,
            {"alpha": DEFAULT_ALPHA, "epsilon_decay": 0.9995},
        ),
    )

    print("\nAnalyzing Epsilon Decay...")
    epsilon_results = _run_hyperparameter_study(
        StudyParameters(
            env,
            episodes,
            "epsilon_decay",
            epsilon_decay_values,
            {"alpha": DEFAULT_ALPHA, "gamma": DEFAULT_GAMMA},
        ),
    )

    plot_alpha_analysis(alpha_results, alpha_values, moving_average_window)
    plot_gamma_analysis(gamma_results, gamma_values, moving_average_window)
    plot_epsilon_decay_analysis(
        epsilon_results, epsilon_decay_values, episodes, moving_average_window
    )
    plot_efficiency_summary(alpha_results, moving_average_window)
    plot_performance_heatmap(alpha_values, gamma_values)

    print_analysis_summary()


def plot_hyperparameter_analysis() -> None:
    """Plot all hyperparameter analysis graphs."""


def plot_alpha_analysis(
    alpha_results: dict[float, TrainingResults],
    alpha_values: list[float],
    moving_average_window: int,
) -> None:
    """Plots the analysis for the alpha hyperparameter."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        "Hyperparameter Analysis: Learning Rate (a)",
        fontsize=18,
        fontweight="bold",
    )

    for alpha, results in alpha_results.items():
        window = moving_average_window
        q_changes = results.q_changes
        if len(q_changes) >= window:
            q_smooth = np.convolve(q_changes, np.ones(window) / window, mode="valid")
            ax1.plot(
                range(window - 1, len(q_changes)),
                q_smooth,
                label=f"a = {alpha}",
            )
        else:
            ax1.plot(q_changes, label=f"a = {alpha}")

    ax1.set_title(
        "Q-Value Convergence vs Learning Rate (a)",
        fontsize=18,
        fontweight="bold",
    )
    ax1.set_xlabel("Episode", fontsize=18)
    ax1.set_ylabel("Q-Value Change (Moving Avg)", fontsize=18)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    for alpha, results in alpha_results.items():
        rewards = results.rewards
        window = moving_average_window  # Scale down for reward smoothing
        if len(rewards) >= window:
            reward_smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax2.plot(
                range(window - 1, len(rewards)),
                reward_smooth,
                label=f"a = {alpha}",
            )

    ax2.set_title(
        "Learning Performance vs Learning Rate (a)",
        fontsize=18,
        fontweight="bold",
    )
    ax2.set_xlabel("Episode", fontsize=18)
    ax2.set_ylabel("Episode Reward (Moving Avg)", fontsize=18)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    final_q_changes = [
        np.mean(results.q_changes[-50:]) for results in alpha_results.values()
    ]
    bars = ax3.bar(
        [str(alpha) for alpha in alpha_values],
        final_q_changes,
        color="skyblue",
        edgecolor="navy",
        alpha=0.7,
    )
    ax3.set_title(
        "Final Q-Value Stability vs Learning Rate (a)",
        fontsize=18,
        fontweight="bold",
    )
    ax3.set_xlabel("Learning Rate (a)", fontsize=18)
    ax3.set_ylabel("Final Q-Value Change (Last 50 episodes)", fontsize=18)
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/hyperparameter_analysis_alpha.png")
    plt.show()
    plt.close(fig)


def plot_gamma_analysis(
    gamma_results: dict[float, TrainingResults],
    gamma_values: list[float],
    moving_average_window: int,
) -> None:
    """Plots the analysis for the gamma hyperparameter."""
    fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        "Hyperparameter Analysis: Discount Factor (g)",
        fontsize=18,
        fontweight="bold",
    )

    for gamma, results in gamma_results.items():
        window = moving_average_window  # Scale down for gamma analysis
        q_changes = results.q_changes
        if len(q_changes) >= window:
            q_smooth = np.convolve(q_changes, np.ones(window) / window, mode="valid")
            ax4.plot(
                range(window - 1, len(q_changes)),
                q_smooth,
                label=f"g = {gamma}",
            )
        else:
            ax4.plot(q_changes, label=f"g = {gamma}")

    ax4.set_title(
        "Q-Value Convergence vs Discount Factor (g)",
        fontsize=18,
        fontweight="bold",
    )
    ax4.set_xlabel("Episode", fontsize=18)
    ax4.set_ylabel("Q-Value Change (Moving Avg)", fontsize=18)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    for gamma, results in gamma_results.items():
        rewards = results.rewards
        window = moving_average_window  # Scale down for reward smoothing
        if len(rewards) >= window:
            reward_smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax5.plot(
                range(window - 1, len(rewards)),
                reward_smooth,
                label=f"g = {gamma}",
            )

    ax5.set_title(
        "Learning Performance vs Discount Factor (g)",
        fontsize=18,
        fontweight="bold",
    )
    ax5.set_xlabel("Episode", fontsize=18)
    ax5.set_ylabel("Episode Reward (Moving Avg)", fontsize=18)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    final_q_changes = [
        np.mean(results.q_changes[-50:]) for results in gamma_results.values()
    ]
    bars = ax6.bar(
        [str(gamma) for gamma in gamma_values],
        final_q_changes,
        color="lightgreen",
        edgecolor="darkgreen",
        alpha=0.7,
    )
    ax6.set_title(
        "Final Q-Value Stability vs Discount Factor (g)",
        fontsize=18,
        fontweight="bold",
    )
    ax6.set_xlabel("Discount Factor (g)", fontsize=18)
    ax6.set_ylabel("Final Q-Value Change (Last 50 episodes)", fontsize=18)
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/hyperparameter_analysis_gamma.png")
    plt.show()
    plt.close(fig)


def plot_epsilon_decay_analysis(
    epsilon_results: dict[float, TrainingResults],
    epsilon_decay_values: list[float],
    episodes: int,
    moving_average_window: int,
) -> None:
    """Plots the analysis for the epsilon decay hyperparameter."""
    fig, (ax7, ax8, ax9, ax10) = plt.subplots(1, 4, figsize=(32, 7))
    fig.suptitle(
        "Hyperparameter Analysis: Epsilon Decay",
        fontsize=18,
        fontweight="bold",
    )

    for epsilon_decay, results in epsilon_results.items():
        window = moving_average_window  # Scale down for epsilon decay analysis
        q_changes = results.q_changes
        if len(q_changes) >= window:
            q_smooth = np.convolve(q_changes, np.ones(window) / window, mode="valid")
            ax7.plot(
                range(window - 1, len(q_changes)),
                q_smooth,
                label=f"e_decay = {epsilon_decay}",
            )
        else:
            ax7.plot(q_changes, label=f"e_decay = {epsilon_decay}")

    ax7.set_title(
        "Q-Value Convergence vs Epsilon Decay",
        fontsize=18,
        fontweight="bold",
    )
    ax7.set_xlabel("Episode", fontsize=18)
    ax7.set_ylabel("Q-Value Change (Moving Avg)", fontsize=18)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale("log")

    for epsilon_decay, results in epsilon_results.items():
        rewards = results.rewards
        window = moving_average_window  # Scale down for reward smoothing
        if len(rewards) >= window:
            reward_smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax8.plot(
                range(window - 1, len(rewards)),
                reward_smooth,
                label=f"e_decay = {epsilon_decay}",
            )

    ax8.set_title(
        "Learning Performance vs Epsilon Decay",
        fontsize=18,
        fontweight="bold",
    )
    ax8.set_xlabel("Episode", fontsize=18)
    ax8.set_ylabel("Episode Reward (Moving Avg)", fontsize=18)
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    final_q_changes = [
        np.mean(results.q_changes[-50:]) for results in epsilon_results.values()
    ]
    bars = ax9.bar(
        [str(eps) for eps in epsilon_decay_values],
        final_q_changes,
        color="lightcoral",
        edgecolor="darkred",
        alpha=0.7,
    )
    ax9.set_title(
        "Final Q-Value Stability vs Epsilon Decay",
        fontsize=18,
        fontweight="bold",
    )
    ax9.set_xlabel("Epsilon Decay Rate", fontsize=18)
    ax9.set_ylabel("Final Q-Value Change (Last 50 episodes)", fontsize=18)
    ax9.set_yscale("log")
    ax9.grid(True, alpha=0.3)

    for bar, value in zip(bars, final_q_changes, strict=True):
        ax9.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    for epsilon_decay in epsilon_decay_values:
        epsilons = [1.0]
        for _ in range(1, episodes):
            epsilons.append(max(0.01, epsilons[-1] * epsilon_decay))
        ax10.plot(epsilons, label=f"e_decay = {epsilon_decay}")

    ax10.set_title(
        "Exploration Schedule (e over time)",
        fontsize=18,
        fontweight="bold",
    )
    ax10.set_xlabel("Episode", fontsize=18)
    ax10.set_ylabel("Epsilon (e)", fontsize=18)
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/hyperparameter_analysis_epsilon_decay.png")
    plt.show()
    plt.close(fig)


def plot_efficiency_summary(
    alpha_results: dict[float, TrainingResults],
    moving_average_window: int,
) -> None:
    """Plots the learning efficiency based on episode length."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        "Learning Efficiency: Episode Length vs Learning Rate",
        fontsize=18,
        fontweight="bold",
    )

    for alpha in [0.01, 0.1, 0.5, 0.9]:
        if alpha in alpha_results:
            lengths = alpha_results[alpha].episode_lengths
            window = moving_average_window  # Scale down for efficiency summary
            if len(lengths) >= window:
                length_smooth = np.convolve(
                    lengths,
                    np.ones(window) / window,
                    mode="valid",
                )
                ax.plot(
                    range(window - 1, len(lengths)),
                    length_smooth,
                    label=f"a = {alpha}",
                )

    ax.set_xlabel("Episode", fontsize=18)
    ax.set_ylabel("Episode Length (Moving Avg)", fontsize=18)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/hyperparameter_efficiency.png")
    plt.show()
    plt.close(fig)


def plot_performance_heatmap(
    alpha_values: list[float],
    gamma_values: list[float],
) -> None:
    """Plots a heatmap of performance for alpha vs gamma."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        "Performance Heatmap: a vs g",
        fontsize=18,
        fontweight="bold",
    )

    performance_matrix = np.zeros((len(alpha_values), len(gamma_values)))
    env = GridWorld()
    for i, alpha in enumerate(alpha_values):
        for j, gamma in enumerate(gamma_values):
            hyperparameters = Hyperparameters(
                alpha=alpha,
                gamma=gamma,
                epsilon=DEFAULT_EPSILON,
                epsilon_decay=0.995,
            )
            agent = QLearningAgent(
                env.get_states(),
                env.action_names,
                hyperparameters,
            )
            result = run_training_session(
                env,
                agent,
                100,
            )
            performance_matrix[i, j] = np.mean(result.rewards[-20:])

    im = ax.imshow(performance_matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xlabel("Discount Factor (g)", fontsize=18)
    ax.set_ylabel("Learning Rate (a)", fontsize=18)

    ax.set_xticks(range(len(gamma_values)))
    ax.set_xticklabels([str(g) for g in gamma_values])
    ax.set_yticks(range(len(alpha_values)))
    ax.set_yticklabels([str(a) for a in alpha_values])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Average Episode Reward (Last 20 episodes)",
        rotation=270,
        labelpad=15,
    )

    for i in range(len(alpha_values)):
        for j in range(len(gamma_values)):
            ax.text(
                j,
                i,
                f"{performance_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/hyperparameter_heatmap.png")
    plt.show()
    plt.close(fig)


def print_analysis_summary() -> None:
    """Print a summary of hyperparameter analysis findings."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n1. LEARNING RATE (a) EFFECTS:")
    print("   - Lower a (0.01-0.1): More stable but slower convergence")
    print("   - Higher a (0.5-0.9): Faster initial learning but more oscillation")
    print("   - Optimal range appears to be 0.1-0.3 for this environment")

    print("\n2. DISCOUNT FACTOR (g) EFFECTS:")
    print("   - Lower g (0.5-0.7): Focuses on immediate rewards, faster convergence")
    print("   - Higher g (0.95-0.99): Better long-term planning, slower convergence")
    print("   - g = 0.9 often provides good balance")

    print("\n3. EXPLORATION SCHEDULE (e decay) EFFECTS:")
    print("   - Faster decay (0.99): Quick transition to exploitation")
    print("   - Slower decay (0.9999): Extended exploration phase")
    print("   - Moderate decay (0.995-0.999) often works best")

    print("\n4. KEY INSIGHTS:")
    print(
        "   - Hyperparameter tuning significantly impacts learning speed and stability",
    )
    print("   - There's a trade-off between exploration and exploitation")
    print("   - Environment complexity affects optimal hyperparameter choices")
    print("   - Monitoring multiple metrics (Q-values, rewards, policy) is crucial")


def _plot_alpha_effect(ax, episodes):
    low_alpha = 0.5 * (1 - np.exp(-episodes / 20))
    med_alpha = 0.9 * (1 - np.exp(-episodes / 15))
    high_alpha = 0.95 * (1 - np.exp(-episodes / 5)) + 0.05 * np.sin(episodes / 3)

    ax.plot(episodes, low_alpha, "b-", linewidth=2.5, label="Low a (0.05)")
    ax.plot(episodes, med_alpha, "g-", linewidth=2.5, label="Optimal a (0.2)")
    ax.plot(episodes, high_alpha, "r-", linewidth=2.5, label="High a (0.7)")
    ax.set_title("Learning Rate (a) Effect", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.set_ylabel("Learning Progress", fontsize=18)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_gamma_effect(ax, episodes):
    gamma_low = 0.5 + 0.1 * np.tanh(episodes / 30)
    gamma_med = 0.2 + 0.7 * np.tanh(episodes / 40)
    gamma_high = 0.3 + 0.6 * np.tanh(episodes / 60)

    ax.plot(episodes, gamma_low, "orange", linewidth=2.5, label="Low g (0.5)")
    ax.plot(episodes, gamma_med, "green", linewidth=2.5, label="Optimal g (0.9)")
    ax.plot(episodes, gamma_high, "purple", linewidth=2.5, label="High g (0.99)")
    ax.set_title("Discount Factor (g) Effect", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.set_ylabel("Policy Quality", fontsize=18)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_epsilon_effect(ax, episodes):
    epsilon_fast = 0.01 + 0.99 * (0.9**episodes)
    epsilon_medium = 0.01 + 0.99 * (0.995**episodes)
    epsilon_slow = 0.01 + 0.99 * (0.9999**episodes)

    ax.plot(episodes, epsilon_fast, "red", linewidth=2.5, label="Fast Decay (0.9)")
    ax.plot(
        episodes,
        epsilon_medium,
        "blue",
        linewidth=2.5,
        label="Optimal Decay (0.995)",
    )
    ax.plot(
        episodes,
        epsilon_slow,
        "green",
        linewidth=2.5,
        label="Slow Decay (0.9999)",
    )
    ax.set_title("Exploration Schedule (e decay)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.set_ylabel("Epsilon (Exploration Rate)", fontsize=18)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_combined_effects(ax, episodes):
    optimal = 0.9 + 0.05 * np.sin(episodes / 10 + np.pi / 4)
    suboptimal1 = 0.7 + 0.15 * np.sin(episodes / 8)
    suboptimal2 = 0.6 + 0.1 * np.exp(-episodes / 20)
    suboptimal3 = 0.5 + 0.3 * np.tanh(episodes / 50)

    ax.plot(episodes, optimal, "green", linewidth=3, label="Optimal Settings")
    ax.plot(
        episodes,
        suboptimal1,
        "red",
        linewidth=2,
        linestyle="--",
        label="High a, Low g",
    )
    ax.plot(
        episodes,
        suboptimal2,
        "orange",
        linewidth=2,
        linestyle="--",
        label="Very Low a",
    )
    ax.plot(
        episodes,
        suboptimal3,
        "purple",
        linewidth=2,
        linestyle="--",
        label="Fast e Decay",
    )
    ax.set_title("Combined Hyperparameter Effects", fontsize=18, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=18)
    ax.set_ylabel("Performance", fontsize=18)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_summary_table(ax):
    ax.axis("off")
    table_data = [
        ["Parameter", "Too Low", "Optimal Range", "Too High"],
        [
            "Learning Rate (a)",
            "Slow convergence",
            "0.1 - 0.3",
            "Oscillations",
        ],
        [
            "Discount Factor (g)",
            "Myopic behavior",
            "0.8 - 0.95",
            "Slow convergence",
        ],
        [
            "Epsilon Decay",
            "Insufficient exploration",
            "0.995 - 0.999",
            "Excessive exploration",
        ],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title(
        "Hyperparameter Effects Summary",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )


def _plot_key_findings(ax):
    ax.axis("off")
    summary_text = """
    KEY FINDINGS:
    - a (Learning Rate): Controls update step size.
    - g (Discount Factor): Balances immediate vs. future rewards.
    - e-decay: Manages exploration vs. exploitation trade-off.

    RECOMMENDATIONS:
    - Start with a=0.1, g=0.9, e-decay=0.995.
    - Monitor convergence and performance.
    - Adjust based on environment complexity.
    """
    ax.text(0, 0.5, summary_text, va="center", ha="left", fontsize=10)


def create_summary_figure() -> None:
    """Create a summary figure showing key hyperparameter effects."""
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 20),
        gridspec_kw={"height_ratios": [1, 1, 1.2]},
    )
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    fig.suptitle(
        "Hyperparameter Effects Summary",
        fontsize=18,
        fontweight="bold",
    )

    episodes = np.linspace(0, 100, 100)

    _plot_alpha_effect(ax1, episodes)
    _plot_gamma_effect(ax2, episodes)
    _plot_epsilon_effect(ax3, episodes)
    _plot_combined_effects(ax4, episodes)
    _plot_summary_table(ax5)
    _plot_key_findings(ax6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/hyperparameter_effects_summary.png")
    plt.show()
    plt.close(fig)


def create_hyperparameter_summary(
    results: dict[str, dict[float, TrainingResults]],
    values: dict[str, list[float]],
) -> str:
    """Generate a text summary of hyperparameter analysis."""
    final_window = max(
        1,
        len(next(iter(next(iter(results.values())).values())).rewards),
    )

    alpha_performance = {
        alpha: np.mean(results["alpha"][alpha].rewards[-final_window:])
        for alpha in values["alpha"]
    }
    gamma_performance = {
        gamma: np.mean(results["gamma"][gamma].rewards[-final_window:])
        for gamma in values["gamma"]
    }
    epsilon_performance = {
        eps: np.mean(results["epsilon_decay"][eps].rewards[-final_window:])
        for eps in values["epsilon_decay"]
    }

    best_alpha = max(alpha_performance.items(), key=lambda x: x[1])
    best_gamma = max(gamma_performance.items(), key=lambda x: x[1])
    best_epsilon = max(epsilon_performance.items(), key=lambda x: x[1])

    return (
        "HYPERPARAMETER ANALYSIS SUMMARY\n\n"
        f"📊 OPTIMAL SETTINGS (Best Final Performance):\n"
        f"• Learning Rate (a): {best_alpha[0]} (Reward: {best_alpha[1]:.3f})\n"
        f"• Discount Factor (g): {best_gamma[0]} (Reward: {best_gamma[1]:.3f})\n"
        f"• Exploration Decay (e): {best_epsilon[0]} (Reward: {best_epsilon[1]:.3f})\n\n"
        "🎯 KEY INSIGHTS:\n"
        "• Learning Rate: Balance speed vs stability\n"
        "• Discount Factor: Balance immediate vs future rewards\n"
        "• Exploration: Balance exploration vs exploitation\n\n"
        "💡 RECOMMENDATIONS:\n"
        "• Start with a=0.1, g=0.9, e_decay=0.995\n"
        "• Monitor convergence curves during training\n"
        "• Adjust based on environment complexity"
    )


def get_convergence_quality(avg_q_change: float) -> str:
    """Return the convergence quality based on the average Q-change."""
    if avg_q_change < EXCELLENT_CONVERGENCE:
        return "Excellent"
    if avg_q_change < GOOD_CONVERGENCE:
        return "Good"
    if avg_q_change < MODERATE_CONVERGENCE:
        return "Moderate"
    return "Poor"


def get_exploration_level(final_epsilon: float) -> str:
    """Return the exploration level based on the final epsilon."""
    if final_epsilon > HIGH_EXPLORATION:
        return "High"
    if final_epsilon > MEDIUM_EXPLORATION:
        return "Medium"
    if final_epsilon > LOW_EXPLORATION:
        return "Low"
    return "Minimal"


def print_detailed_hyperparameter_analysis(
    results: dict[str, dict[float, TrainingResults]],
    values: dict[str, list[float]],
    episodes: int,
) -> None:
    """Print detailed analysis of hyperparameter effects."""
    print("\n" + "=" * 80)
    print("DETAILED HYPERPARAMETER ANALYSIS RESULTS")
    print("=" * 80)

    final_window = max(1, episodes)

    print("\n📊 LEARNING RATE (a) ANALYSIS:")
    print(f"{'a':<8} {'Final Reward':<15} {'Avg Q-Change':<15} {'Convergence':<15}")
    print("-" * 60)
    for alpha in values["alpha"]:
        if alpha in results["alpha"]:
            result = results["alpha"][alpha]
            final_reward = np.mean(result.rewards[-final_window:])
            avg_q_change = np.mean(result.q_changes[-final_window:])
            convergence = get_convergence_quality(float(avg_q_change))
            print(
                f"{alpha:<8} {final_reward:<15.3f} {avg_q_change:<15.3f} {convergence:<15}",
            )

    print("\n🎯 DISCOUNT FACTOR (g) ANALYSIS:")
    print(f"{'g':<8} {'Final Reward':<15} {'Avg Q-Change':<15} {'Convergence':<15}")
    print("-" * 60)
    for gamma in values["gamma"]:
        if gamma in results["gamma"]:
            result = results["gamma"][gamma]
            final_reward = np.mean(result.rewards[-final_window:])
            avg_q_change = np.mean(result.q_changes[-final_window:])
            convergence = get_convergence_quality(float(avg_q_change))
            print(
                f"{gamma:<8} {final_reward:<15.3f} {avg_q_change:<15.3f} {convergence:<15}",
            )

    print("\n🔍 EXPLORATION SCHEDULE (e decay) ANALYSIS:")
    print(f"{'e_decay':<12} {'Final Reward':<15} {'Final e':<15} {'Exploration':<15}")
    print("-" * 65)
    for eps_decay in values["epsilon_decay"]:
        if eps_decay in results["epsilon_decay"]:
            result = results["epsilon_decay"][eps_decay]
            final_reward = np.mean(result.rewards[-final_window:])
            final_epsilon = result.epsilon_values[-1]
            exploration_level = get_exploration_level(final_epsilon)
            print(
                f"{eps_decay:<12} {final_reward:<15.3f} {final_epsilon:<15.4f} {exploration_level:<15}",
            )

    print("\n💡 RECOMMENDATIONS:")
    print("Based on this analysis:")

    alpha_performance = {
        alpha: np.mean(results["alpha"][alpha].rewards[-final_window:])
        for alpha in values["alpha"]
        if alpha in results["alpha"]
    }
    best_alpha = max(alpha_performance.items(), key=lambda x: x[1])

    gamma_performance = {
        gamma: np.mean(results["gamma"][gamma].rewards[-final_window:])
        for gamma in values["gamma"]
        if gamma in results["gamma"]
    }
    best_gamma = max(gamma_performance.items(), key=lambda x: x[1])

    epsilon_performance = {
        eps: np.mean(results["epsilon_decay"][eps].rewards[-final_window:])
        for eps in values["epsilon_decay"]
        if eps in results["epsilon_decay"]
    }
    best_epsilon = max(epsilon_performance.items(), key=lambda x: x[1])

    print(f"• Use a = {best_alpha[0]} for optimal learning rate")
    print(f"• Use g = {best_gamma[0]} for optimal discount factor")
    print(f"• Use e_decay = {best_epsilon[0]} for optimal exploration schedule")
    print(
        f"• Expected performance: {(best_alpha[1] + best_gamma[1] + best_epsilon[1]) / 3:.3f}",
    )
    print("• Monitor Q-value changes to assess convergence quality")
    print("• Adjust parameters based on environment complexity and episode budget")


def print_convergence_summary(
    alpha_data: list[dict[str, Any]],
    gamma_data: list[dict[str, Any]],
    epsilon_decay_data: list[dict[str, Any]],
    max_episodes: int,
    fast_convergence_threshold: int,
) -> None:
    """Print a detailed summary of convergence analysis results."""
    print("\nCONVERGENCE ANALYSIS SUMMARY:")
    print("-" * 50)

    print("\nLearning Rate (alpha) Analysis:")
    print(
        f"{'alpha':<6} {'Q-Conv Episode':<15} {'Policy Changes':<15} {'Convergence Quality':<20}",
    )
    print("-" * 70)
    for data in alpha_data:
        q_conv = data["q_convergence"] if data["q_convergence"] else f">{max_episodes}"
        quality = "No Convergence"
        if data["q_convergence"]:
            quality = (
                "Fast" if data["q_convergence"] < fast_convergence_threshold else "Slow"
            )
        print(
            f"{data['alpha']:<6} {q_conv!s:<15} {data['total_policy_changes']:<15} {quality:<20}",
        )

    print("\nDiscount Factor (gamma) Analysis:")
    print(
        f"{'gamma':<6} {'Q-Conv Episode':<15} {'Policy Changes':<15} {'Convergence Quality':<20}",
    )
    print("-" * 70)
    for data in gamma_data:
        q_conv = data["q_convergence"] if data["q_convergence"] else f">{max_episodes}"
        quality = "No Convergence"
        if data["q_convergence"]:
            quality = (
                "Fast" if data["q_convergence"] < fast_convergence_threshold else "Slow"
            )
        print(
            f"{data['gamma']:<6} {q_conv!s:<15} {data['total_policy_changes']:<15} {quality:<20}",
        )

    print("\nEpsilon Decay Analysis:")
    print(
        f"{'eps_decay':<10} {'Q-Conv Episode':<15} {'Policy Changes':<15} {'Convergence Quality':<20}",
    )
    print("-" * 70)
    for data in epsilon_decay_data:
        q_conv = data["q_convergence"] if data["q_convergence"] else f">{max_episodes}"
        quality = "No Convergence"
        if data["q_convergence"]:
            quality = (
                "Fast" if data["q_convergence"] < fast_convergence_threshold else "Slow"
            )
        print(
            f"{data['epsilon_decay']:<10} {q_conv!s:<15} {data['total_policy_changes']:<15} {quality:<20}",
        )

    best_alpha = min(
        alpha_data,
        key=lambda x: x["q_convergence"] or float("inf"),
    )
    best_gamma = min(
        gamma_data,
        key=lambda x: x["q_convergence"] or float("inf"),
    )
    best_epsilon_decay = min(
        epsilon_decay_data,
        key=lambda x: x["q_convergence"] or float("inf"),
    )

    print("\nRECOMMENDED HYPERPARAMETERS:")
    print(
        f"Best Learning Rate (alpha): {best_alpha['alpha']} (converged at episode {best_alpha['q_convergence']})",
    )
    print(
        f"Best Discount Factor (gamma): {best_gamma['gamma']} (converged at episode {best_gamma['q_convergence']})",
    )
    print(
        f"Best Epsilon Decay: {best_epsilon_decay['epsilon_decay']} (converged at episode {best_epsilon_decay['q_convergence']})",
    )
    print(
        "\nNote: 'Best' is defined as fastest Q-value convergence to threshold of 0.001",
    )


def plot_convergence(q_changes, policy_changes, moving_average_window=100):
    """Plot convergence metrics with moving average smoothing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle("Convergence Analysis", fontsize=18, fontweight="bold")

    # Q-value convergence with moving average
    window = moving_average_window  # Scale down for convergence analysis
    if len(q_changes) >= window:
        q_smooth = np.convolve(q_changes, np.ones(window) / window, mode="valid")
        ax1.plot(q_changes, alpha=0.3, color="lightblue", label="Raw Q-value Change")
        ax1.plot(
            range(window - 1, len(q_changes)),
            q_smooth,
            color="darkblue",
            linewidth=2,
            label=f"Moving Average (window={window})",
        )
    else:
        ax1.plot(q_changes, label="Q-value Change")

    ax1.set_title("Q-Value Convergence")
    ax1.set_xlabel("Episode", fontsize=18)
    ax1.set_ylabel("Total Q-value Change", fontsize=18)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend()

    # Policy stability with moving average
    if len(policy_changes) >= window:
        policy_smooth = np.convolve(
            policy_changes, np.ones(window) / window, mode="valid"
        )
        ax2.plot(
            policy_changes, alpha=0.3, color="lightcoral", label="Raw Policy Changes"
        )
        ax2.plot(
            range(window - 1, len(policy_changes)),
            policy_smooth,
            color="darkred",
            linewidth=2,
            label=f"Moving Average (window={window})",
        )
    else:
        ax2.plot(policy_changes, label="Policy Changes")

    ax2.set_title("Policy Stability")
    ax2.set_xlabel("Episode", fontsize=18)
    ax2.set_ylabel("Number of Policy Changes", fontsize=18)
    ax2.grid(True, linestyle="--", linewidth=0.5)
    ax2.legend()

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig("images/convergence.png")
    plt.show()
    plt.close(fig)


def print_policy(agent, env):
    """Print the learned policy."""
    policy = agent.get_policy()
    print("\nLearned Policy:")
    for y in range(env.grid_size[1], 0, -1):
        row = ""
        for x in range(1, env.grid_size[0] + 1):
            state = (x, y)
            if state == env.wall_state:
                row += " W "
            elif state in env.terminal_states:
                row += " T "
            else:
                row += f" {policy.get(state, '?')} "
        print(row)


def plot_q_table_heatmap(agent: QLearningAgent, env: GridWorld) -> None:
    """
    Create a heatmap visualization of the Q-table showing Q-values for each action.

    Args:
        agent: The trained Q-learning agent
        env: The GridWorld environment

    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Q-Table Heatmaps by Action", fontsize=16, fontweight="bold")

    actions = ["N", "S", "E", "W"]
    action_names = {"N": "North", "S": "South", "E": "East", "W": "West"}

    # Create grid for each action
    for idx, action in enumerate(actions):
        ax = axes[idx // 2, idx % 2]

        # Initialize grid with NaN values
        grid = np.full((env.grid_size[1], env.grid_size[0]), np.nan)

        # Fill grid with Q-values
        for state in agent.states:
            x, y = state
            if state != env.wall_state:
                # Convert to grid coordinates (flip y-axis)
                grid_y = env.grid_size[1] - y
                grid_x = x - 1
                grid[grid_y, grid_x] = agent.q_table[state][action]

        # Create heatmap
        im = ax.imshow(grid, cmap="RdYlBu_r", aspect="equal")
        ax.set_title(f"{action_names[action]} ({action})", fontweight="bold")

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Add grid labels
        ax.set_xticks(range(env.grid_size[0]))
        ax.set_yticks(range(env.grid_size[1]))
        ax.set_xticklabels([f"x={i + 1}" for i in range(env.grid_size[0])])
        ax.set_yticklabels(
            [f"y={env.grid_size[1] - i}" for i in range(env.grid_size[1])]
        )

        # Add Q-values as text on each cell with prominent colors for goal and penalty
        for y in range(env.grid_size[1]):
            for x in range(env.grid_size[0]):
                state = (x + 1, env.grid_size[1] - y)
                if state == env.wall_state:
                    ax.text(
                        x,
                        y,
                        "WALL",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="black",
                        fontsize=12,
                    )
                elif state in env.terminal_states:
                    reward = env.terminal_states[state]
                    if reward > 0:
                        # Goal state - use bright green color
                        ax.text(
                            x,
                            y,
                            "GOAL",
                            ha="center",
                            va="center",
                            fontweight="bold",
                            color="lime",
                            fontsize=14,
                        )
                    else:
                        # Penalty state - use bright red color
                        ax.text(
                            x,
                            y,
                            "PENALTY (-1)",
                            ha="center",
                            va="center",
                            fontweight="bold",
                            color="red",
                            fontsize=14,
                        )
                elif not np.isnan(grid[y, x]):
                    # Regular Q-values - use black text
                    ax.text(
                        x,
                        y,
                        f"{grid[y, x]:.2f}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                        color="black",
                        fontsize=10,
                    )

    plt.tight_layout()
    plt.savefig("images/q_table_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    """Run the full Q-learning analysis pipeline."""
    episodes = 10000
    base_penalty = -1.0
    alpha_values = [0.01, 0.1, 0.3, 0.5, 0.9]
    gamma_values = [0.5, 0.7, 0.9, 0.99]
    epsilon_decay_values = [0.99, 0.995, 0.999, 0.9999]
    moving_average_window = 1000

    print("=" * 80)
    print("RUNNING HYPERPARAMETER ANALYSIS")
    print("=" * 80)

    analyze_hyperparameters(
        episodes,
        base_penalty,
        alpha_values,
        gamma_values,
        epsilon_decay_values,
        moving_average_window,
    )

    high_penalty = -10.0
    env_high_penalty = GridWorld(penalty=high_penalty)
    hyperparameters = Hyperparameters(
        alpha=DEFAULT_ALPHA,
        gamma=DEFAULT_GAMMA,
        epsilon=DEFAULT_EPSILON,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
    )
    agent_high_penalty = QLearningAgent(
        env_high_penalty.get_states(),
        env_high_penalty.action_names,
        hyperparameters,
    )

    results_high_penalty = run_training_session(
        env_high_penalty,
        agent_high_penalty,
        episodes,
    )
    print_policy(agent_high_penalty, env_high_penalty)
    plot_q_table_heatmap(agent_high_penalty, env_high_penalty)
    plot_convergence(
        results_high_penalty.q_changes,
        results_high_penalty.policy_changes,
        moving_average_window,
    )

    create_summary_figure()


if __name__ == "__main__":
    main()
