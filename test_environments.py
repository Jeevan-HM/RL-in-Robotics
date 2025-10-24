"""
test_environment.py - Test script to verify environment and components

Run with: python test_environment.py
"""

import torch
from environment import GridWorld
from network import QNetwork
from replay_buffer import ReplayBuffer
from agent import DQNAgent
from config import get_default_config


def test_environment():
    """Test GridWorld environment."""
    print("\n" + "="*60)
    print("Testing GridWorld Environment")
    print("="*60)
    
    env = GridWorld()
    
    # Test initialization
    print(f"✓ Environment initialized: {env.rows}x{env.cols} grid")
    print(f"✓ Terminal states: {env.terminal_states}")
    print(f"✓ Actions: {env.actions}")
    
    # Test reset
    state = env.reset()
    print(f"✓ Reset state: {state}")
    
    # Test step
    next_state, reward, done = env.step(0)
    print(f"✓ Step executed: {state} -> {next_state}, reward={reward}, done={done}")
    
    # Test terminal state
    env.state = (0, 3)
    assert env.is_terminal(env.state), "Terminal state check failed"
    print(f"✓ Terminal state detection works")
    
    # Test obstacle
    assert not env.is_valid((1, 1)), "Obstacle check failed"
    print(f"✓ Obstacle detection works")
    
    # Test state conversion
    state_idx = env.state_to_index((1, 2))
    state_back = env.index_to_state(state_idx)
    assert state_back == (1, 2), "State conversion failed"
    print(f"✓ State conversion works: (1,2) <-> {state_idx}")
    
    print("\n✅ All environment tests passed!\n")


def test_network():
    """Test Q-Network."""
    print("="*60)
    print("Testing Q-Network")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Create network
    input_size = 12  # 3x4 grid
    hidden_sizes = [64, 64]
    output_size = 4  # 4 actions
    
    network = QNetwork(input_size, hidden_sizes, output_size).to(device)
    print(f"✓ Network created with architecture: {input_size} -> {hidden_sizes} -> {output_size}")
    
    # Test forward pass
    dummy_input = torch.randn(1, input_size).to(device)
    output = network(dummy_input)
    
    assert output.shape == (1, output_size), "Output shape mismatch"
    print(f"✓ Forward pass works: input {dummy_input.shape} -> output {output.shape}")
    
    # Test parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    print("\n✅ All network tests passed!\n")


def test_replay_buffer():
    """Test replay buffer."""
    print("="*60)
    print("Testing Replay Buffer")
    print("="*60)
    
    buffer = ReplayBuffer(capacity=100)
    print(f"✓ Buffer created with capacity: 100")
    
    # Add transitions
    for i in range(50):
        buffer.push((0, 0), i % 4, -0.04, (0, 1), False)
    
    print(f"✓ Added 50 transitions, buffer size: {len(buffer)}")
    
    # Sample batch
    batch = buffer.sample(32)
    assert len(batch) == 32, "Batch size mismatch"
    print(f"✓ Sampled batch of size: {len(batch)}")
    
    # Test overflow
    for i in range(100):
        buffer.push((1, 1), i % 4, -0.04, (1, 2), False)
    
    assert len(buffer) == 100, "Buffer overflow handling failed"
    print(f"✓ Buffer capacity maintained at: {len(buffer)}")
    
    print("\n✅ All replay buffer tests passed!\n")


def test_agent():
    """Test DQN agent."""
    print("="*60)
    print("Testing DQN Agent")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld()
    config = get_default_config()
    
    agent = DQNAgent(env, config, device)
    print(f"✓ Agent created with config")
    
    # Test state conversion
    state = (1, 2)
    state_tensor = agent.state_to_tensor(state)
    assert state_tensor.shape == (1, 12), "State tensor shape mismatch"
    print(f"✓ State to tensor conversion works: {state} -> {state_tensor.shape}")
    
    # Test action selection
    action = agent.select_action(state)
    assert 0 <= action < 4, "Invalid action selected"
    print(f"✓ Action selection works: action={action}")
    
    # Test episode generation
    reward = agent.generate_episode(max_steps=50, training=True)
    print(f"✓ Episode generated: total_reward={reward:.3f}")
    
    # Test training step
    for _ in range(100):
        agent.generate_episode(max_steps=50, training=True)
    
    loss = agent.train_step()
    if loss is not None:
        print(f"✓ Training step works: loss={loss:.4f}")
    
    # Test evaluation
    eval_mean, eval_std = agent.evaluate(n_episodes=10)
    print(f"✓ Evaluation works: mean={eval_mean:.3f}, std={eval_std:.3f}")
    
    print("\n✅ All agent tests passed!\n")


def quick_training_test():
    """Run a quick training test."""
    print("="*60)
    print("Quick Training Test (50 episodes)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorld()
    config = get_default_config()
    config['epsilon_decay'] = 0.95  # Faster decay for testing
    
    agent = DQNAgent(env, config, device)
    
    rewards = []
    for episode in range(50):
        reward = agent.generate_episode(training=True)
        rewards.append(reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards[-10:]) / 10
            print(f"Episode {episode+1}: avg_reward={avg_reward:.3f}, epsilon={agent.epsilon:.3f}")
    
    print(f"\n✓ Training completed: final avg reward={sum(rewards[-10:])/10:.3f}")
    print("\n✅ Quick training test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GRIDWORLD DQN - COMPONENT TESTS")
    print("="*60)
    
    try:
        test_environment()
        test_network()
        test_replay_buffer()
        test_agent()
        quick_training_test()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nYou're ready to run the full training:")
        print("  python main.py")
        print("\nOr run hyperparameter experiments:")
        print("  python experiments.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()