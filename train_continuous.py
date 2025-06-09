"""
Training script for continuous environment with PPO agent.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import trange
import torch

from world.continuous_environment import ContinuousEnvironment
from agents.ppo_agent import PPOAgent


def train_continuous_ppo(grid_path: str = "grid_configs/small_grid.npy",
                        episodes: int = 1000,
                        max_steps: int = 500,
                        step_size: float = 0.1,
                        learning_rate: float = 3e-4,
                        gamma: float = 0.99,
                        save_model: bool = True,
                        show_progress: bool = True):
    """Train a PPO agent in the continuous environment.
    
    Args:
        grid_path: Path to the grid configuration file
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        step_size: Step size for agent movement
        learning_rate: Learning rate for neural networks
        gamma: Discount factor
        save_model: Whether to save the trained model
        show_progress: Whether to show training progress
    """
    
    # Create environment
    env = ContinuousEnvironment(
        grid_fp=Path(grid_path),
        no_gui=True,
        sigma=0.1,  # 10% stochasticity
        step_size=step_size,
        collision_radius=0.1
    )
    
    # Get environment bounds
    state_bounds = env.get_state_bounds()
    action_dim = env.get_action_space_size()
    
    # Create agent
    agent = PPOAgent(
        state_bounds=state_bounds,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    
    # Training loop
    for episode in trange(episodes, desc="Training PPO", disable=not show_progress):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Take action
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            agent.update(state, reward, action, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate success rate (rolling average over last 100 episodes)
        recent_episodes = episode_rewards[-100:]
        success_count = sum(1 for r in recent_episodes if r > 0)  # Positive reward indicates success
        success_rate.append(success_count / len(recent_episodes))
        
        # Print progress every 100 episodes
        if show_progress and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            current_success_rate = success_rate[-1]
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Success Rate: {current_success_rate:.2%}")
    
    # Save model if requested
    if save_model:
        model_path = f"continuous_ppo_model_{episodes}ep.pth"
        agent.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    # Plot training curves
    if show_progress:
        plot_training_curves(episode_rewards, episode_lengths, success_rate)
    
    return agent, episode_rewards, episode_lengths, success_rate


def plot_training_curves(episode_rewards, episode_lengths, success_rate):
    """Plot training progress curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Smooth the curves with rolling average
    window = max(1, min(100, len(episode_rewards) // 10))
    
    def smooth(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    episodes = range(len(episode_rewards))
    
    # Episode rewards
    ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(episode_rewards) >= window:
        smooth_rewards = smooth(episode_rewards, window)
        smooth_episodes = range(window-1, len(episode_rewards))
        ax1.plot(smooth_episodes, smooth_rewards, color='blue', label=f'Smoothed ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ax2.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Raw')
    if len(episode_lengths) >= window:
        smooth_lengths = smooth(episode_lengths, window)
        ax2.plot(smooth_episodes, smooth_lengths, color='green', label=f'Smoothed ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Success rate
    ax3.plot(episodes, success_rate, color='red')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Success Rate (Rolling 100 episodes)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(episode_rewards)
    ax4.plot(episodes, cumulative_rewards, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Reward')
    ax4.set_title('Cumulative Reward')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('continuous_ppo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_trained_agent(model_path: str,
                          grid_path: str = "grid_configs/small_grid.npy",
                          num_eval_episodes: int = 10,
                          max_steps: int = 500,
                          step_size: float = 0.1,
                          render: bool = False):
    """Evaluate a trained PPO agent.
    
    Args:
        model_path: Path to the saved model
        grid_path: Path to the grid configuration
        num_eval_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        step_size: Step size for agent movement
        render: Whether to render the environment
    """
    # Create environment
    env = ContinuousEnvironment(
        grid_fp=Path(grid_path),
        no_gui=not render,
        sigma=0.0,  # No stochasticity during evaluation
        step_size=step_size,
        collision_radius=0.1
    )
    
    # Create and load agent
    state_bounds = env.get_state_bounds()
    action_dim = env.get_action_space_size()
    
    agent = PPOAgent(state_bounds=state_bounds, action_dim=action_dim)
    agent.load_model(model_path)
    agent.set_training_mode(False)  # Set to evaluation mode
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Evaluating agent for {num_eval_episodes} episodes...")
    
    for episode in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.take_action(state)
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                if info.get('target_reached', False):
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:6.2f}, "
              f"Length = {episode_length:3d}, "
              f"Success = {'Yes' if episode_reward > 0 else 'No'}")
    
    # Print summary statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = success_count / num_eval_episodes
    
    print(f"\nEvaluation Summary:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"Success Rate: {success_rate:.1%}")
    
    return episode_rewards, episode_lengths, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent in continuous environment")
    parser.add_argument("--grid", type=str, default="grid_configs/small_grid.npy",
                       help="Path to grid configuration file")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--step-size", type=float, default=0.1,
                       help="Step size for agent movement")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the trained model")
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Path to model to evaluate (skips training)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluation mode
        evaluate_trained_agent(
            model_path=args.evaluate,
            grid_path=args.grid,
            num_eval_episodes=10,
            max_steps=args.max_steps,
            step_size=args.step_size,
            render=True
        )
    else:
        # Training mode
        train_continuous_ppo(
            grid_path=args.grid,
            episodes=args.episodes,
            max_steps=args.max_steps,
            step_size=args.step_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            save_model=not args.no_save,
            show_progress=True
        )
