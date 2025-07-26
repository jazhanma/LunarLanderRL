#!/usr/bin/env python3
"""
Evaluation script for trained LunarLander DQN agent.
"""

import argparse
import yaml
import torch
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.dqn import DQNAgent
from envs.make_env import make_env, get_env_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DQN agent on LunarLander')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_arg: str) -> str:
    """Determine device to use."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    else:
        return device_arg


def evaluate_agent(agent: DQNAgent, env, num_episodes: int, render: bool = False):
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained DQN agent
        env: Environment to evaluate in
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    successful_landings = 0
    
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Count successful landings (reward > 200)
        if episode_reward > 200:
            successful_landings += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Length: {episode_length}")
    
    # Calculate metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    success_rate = successful_landings / num_episodes * 100
    mean_length = np.mean(episode_lengths)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'success_rate': success_rate,
        'mean_length': mean_length,
        'episode_rewards': episode_rewards
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create environment
    env_name = config['env']['name']
    env_seed = config['env']['seed'] + 2000  # Different seed for evaluation
    env = make_env(env_name, env_seed, render_mode='human' if args.render else None)
    
    # Get environment information
    env_info = get_env_info(env_name)
    state_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config, device)
    
    # Load trained model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    agent.load_model(args.model_path)
    print(f"Loaded model from: {args.model_path}")
    
    # Evaluate agent
    eval_metrics = evaluate_agent(agent, env, args.num_episodes, args.render)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"Reward range: [{eval_metrics['min_reward']:.1f}, {eval_metrics['max_reward']:.1f}]")
    print(f"Success rate: {eval_metrics['success_rate']:.1f}%")
    print(f"Mean episode length: {eval_metrics['mean_length']:.1f}")
    
    # Performance assessment
    if eval_metrics['mean_reward'] >= 200:
        print("\n🎉 EXCELLENT PERFORMANCE! Agent has solved the environment.")
    elif eval_metrics['mean_reward'] >= 100:
        print("\n✅ GOOD PERFORMANCE! Agent is learning well.")
    elif eval_metrics['mean_reward'] >= 0:
        print("\n⚠️  MODERATE PERFORMANCE! Agent needs more training.")
    else:
        print("\n❌ POOR PERFORMANCE! Agent needs significant improvement.")
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main() 