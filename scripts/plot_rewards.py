#!/usr/bin/env python3
"""
Script to plot training rewards and metrics.
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot training rewards')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory containing training logs')
    parser.add_argument('--rewards_file', type=str, default='rewards.json',
                       help='Name of rewards file')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for moving average')
    return parser.parse_args()


def load_rewards_data(log_dir: str, rewards_file: str) -> dict:
    """Load rewards data from JSON file."""
    rewards_path = os.path.join(log_dir, rewards_file)
    
    if not os.path.exists(rewards_path):
        print(f"Error: Rewards file not found at {rewards_path}")
        return None
    
    with open(rewards_path, 'r') as f:
        data = json.load(f)
    
    return data


def plot_training_progress(data: dict, output_dir: str, window_size: int = 100):
    """
    Plot comprehensive training progress.
    
    Args:
        data: Dictionary containing training data
        output_dir: Directory to save plots
        window_size: Window size for moving average
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LunarLander DQN Training Progress', fontsize=16, fontweight='bold')
    
    episode_rewards = data.get('episode_rewards', [])
    episode_lengths = data.get('episode_lengths', [])
    losses = data.get('losses', [])
    q_values = data.get('q_values', [])
    epsilons = data.get('epsilons', [])
    eval_rewards = data.get('eval_rewards', [])
    
    # 1. Episode Rewards
    if episode_rewards:
        axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.8, label='Episode Reward')
        
        # Moving average
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(episode_rewards)), 
                           moving_avg, color='red', linewidth=2, 
                           label=f'{window_size}-Episode Moving Average')
        
        # Success threshold
        axes[0, 0].axhline(y=200, color='green', linestyle='--', alpha=0.7, 
                          label='Success Threshold (200)')
        
        axes[0, 0].set_title('Episode Rewards', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training Loss
    if losses:
        axes[0, 1].plot(losses, alpha=0.7, color='orange')
        axes[0, 1].set_title('Training Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(losses) > 10:
            x = np.arange(len(losses))
            z = np.polyfit(x, losses, 1)
            p = np.poly1d(z)
            axes[0, 1].plot(x, p(x), "r--", alpha=0.8, linewidth=2)
    
    # 3. Q-Values
    if q_values:
        axes[1, 0].plot(q_values, alpha=0.7, color='purple')
        axes[1, 0].set_title('Average Q-Values', fontweight='bold')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Q-Value')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Epsilon Decay
    if epsilons:
        axes[1, 1].plot(epsilons, alpha=0.7, color='brown')
        axes[1, 1].set_title('Epsilon Decay', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved to: {plot_path}")
    plt.close()


def plot_evaluation_results(data: dict, output_dir: str):
    """
    Plot evaluation results.
    
    Args:
        data: Dictionary containing training data
        output_dir: Directory to save plots
    """
    eval_rewards = data.get('eval_rewards', [])
    
    if not eval_rewards:
        print("No evaluation data found.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(eval_rewards, marker='o', linewidth=2, markersize=4, color='green')
    plt.title('Evaluation Performance', fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3)
    
    # Add success threshold
    plt.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
    plt.legend()
    
    plot_path = os.path.join(output_dir, 'evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation results plot saved to: {plot_path}")
    plt.close()


def plot_reward_distribution(data: dict, output_dir: str):
    """
    Plot reward distribution.
    
    Args:
        data: Dictionary containing training data
        output_dir: Directory to save plots
    """
    episode_rewards = data.get('episode_rewards', [])
    
    if not episode_rewards:
        print("No episode rewards data found.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax1.axvline(200, color='green', linestyle='--', label='Success Threshold')
    ax1.set_title('Reward Distribution', fontweight='bold')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(episode_rewards, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_title('Reward Statistics', fontweight='bold')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'reward_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Reward distribution plot saved to: {plot_path}")
    plt.close()


def print_summary_stats(data: dict):
    """Print summary statistics."""
    episode_rewards = data.get('episode_rewards', [])
    
    if not episode_rewards:
        print("No data available for summary statistics.")
        return
    
    recent_rewards = episode_rewards[-100:]  # Last 100 episodes
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Overall mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Recent mean reward (last 100): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.1f}")
    print(f"Success rate (recent): {np.mean([r > 200 for r in recent_rewards]) * 100:.1f}%")
    
    if data.get('eval_rewards'):
        print(f"Best evaluation reward: {max(data['eval_rewards']):.2f}")


def main():
    """Main function."""
    args = parse_args()
    
    # Load data
    data = load_rewards_data(args.log_dir, args.rewards_file)
    
    if data is None:
        return
    
    # Create plots
    plot_training_progress(data, args.output_dir, args.window_size)
    plot_evaluation_results(data, args.output_dir)
    plot_reward_distribution(data, args.output_dir)
    
    # Print summary
    print_summary_stats(data)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 