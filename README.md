# Safety-Critical Autonomous Navigation using Reinforcement Learning

This project implements and evaluates the **Certificated Actor-Critic (CAC)** framework, a hierarchical reinforcement learning approach that uses Control Barrier Functions (CBFs) to enforce safety constraints for autonomous navigation. Unlike standard RL methods that offer weak guarantees, CAC provides a structured method to prioritize safety while achieving task objectives.

The framework operates in two stages:
1.  A **safety critic** is trained using a CBF-derived reward, yielding a quantitative safety certificate.
2.  A task objective, such as goal-reaching, is optimized via a **restricted policy update** that preserves the learned safety guarantees.

This ensures the agent learns a policy that is verifiably safe and effective for navigation in cluttered environments.

## Problem Statement

This work aims to learn a policy, $\pi_\theta$, that satisfies three primary objectives:

-   **Safety Guarantee:** Ensure that the robot's state remains within a predefined safe set for all timesteps (forward invariance).
-   **Goal Reaching:** Reach the target goal efficiently, with near-minimal time or path cost.
-   **Quantified Safety:** Provide a numeric, verifiable certificate of safety for any given policy and state.

## Key Features

-   **Safety-First RL:** Implements a CAC agent that learns to navigate while satisfying safety constraints defined by a CBF.
-   **Two-Stage Training:** The agent is trained in two distinct stages: safety-critic construction and restricted policy updates.
-   **2D Planar Navigation Environment:** A custom `gymnasium` environment for training and evaluating the navigation agent.
-   **Dual Model Visualization:** A powerful visualization tool to observe the behavior of both the safety and goal-oriented models in real-time.

## � A Note on Isaac Sim

The implementation for the Isaac Sim environment is currently **under development and is not yet functional**. The core logic is based on the 2D planar navigation environment, but integration with Isaac Sim is a work in progress.

## Getting Started

Follow these instructions to set up the project and run the code on your local machine.

### Prerequisites

-   Python 3.8 or later
-   A virtual environment manager like `venv` or `conda` is recommended.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jeevan-HM/RL-in-Robotics.git
    cd RL-in-Robotics
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is divided into two main parts: training the agent and visualizing its performance.

### Training the Agent

The training process is split into two stages. You must train the safety model first, and then use its weights to train the goal-seeking model.

1.  **Train the Safety Model:**
    This command trains the agent to avoid obstacles. The trained model will be saved in the `runs/` directory.

    ```bash
    python train_2dnav.py --stage safety --run_name safety_model
    ```

2.  **Train the Goal-Seeking Model:**
    This command loads the pre-trained safety model and continues training the agent to reach the goal.

    ```bash
    python train_2dnav.py --stage goal --resume_from runs/safety_model_last.pt --run_name goal_model
    ```

### Visualizing the Models

The `model_visualization.py` script allows you to see the agent's behavior in the 2D environment. It loads both the safety and goal-seeking models to provide a comprehensive visualization.

```bash
python model_visualization.py --safety_model runs/safety_model_last.pt --goal_model runs/goal_model_last.pt
```

### Training Parameters

You can customize the training process using the following command-line arguments in `train_2dnav.py`:

-   `--stage`: (Required) The training stage. Choose between `safety` and `goal`.
-   `--steps`: The total number of training steps (default: `200000`).
-   `--seed`: The random seed for reproducibility (default: `1`).
-   `--run_name`: A custom name for the training run, used for logging (default: generated from stage and timestamp).
-   `--resume_from`: Path to a pre-trained model checkpoint to resume training from (e.g., for the `goal` stage).
-   `--actor_lr`: The learning rate for the actor network (default: `3e-4`).
-   `--critic_lr`: The learning rate for the critic network (default: `3e-4`).
-   `--alpha_ent`: The entropy regularization coefficient (default: `0.2`).
-   `--restrict_cos_min`: The cosine similarity threshold for the restricted policy update (default: `-0.25`).

## File Structure

```
.
├── cac/                # Core CAC algorithm, networks, and replay buffer
├── envs/               # Gymnasium environments (2D planar navigation)
├── isaac/              # (Work in Progress) Isaac Sim environment
├── runs/               # Directory for TensorBoard logs and saved models
├── train_2dnav.py      # Script for training the agent in the 2D environment
├── model_visualization.py # Script for visualizing the trained models
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
