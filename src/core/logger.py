import os
import json
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for tracking training progress and metrics.
    
    Args:
        log_dir: Directory to save logs
        config: Configuration dictionary
        use_tensorboard: Whether to use TensorBoard logging
    """
    
    def __init__(self, log_dir: str, config: Dict[str, Any], use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.config = config
        self.use_tensorboard = use_tensorboard
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.q_values = []
        self.epsilons = []
        self.eval_rewards = []
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save configuration to log directory."""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   loss: float = None, q_value: float = None, epsilon: float = None):
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Episode reward
            length: Episode length
            loss: Training loss
            q_value: Average Q-value
            epsilon: Current epsilon value
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_scalar('Training/Episode_Reward', reward, episode)
            self.writer.add_scalar('Training/Episode_Length', length, episode)
            if loss is not None:
                self.writer.add_scalar('Training/Loss', loss, episode)
            if q_value is not None:
                self.writer.add_scalar('Training/Q_Value', q_value, episode)
            if epsilon is not None:
                self.writer.add_scalar('Training/Epsilon', epsilon, episode)
    
    def log_evaluation(self, episode: int, eval_metrics: Dict[str, float]):
        """
        Log evaluation metrics.
        
        Args:
            episode: Episode number
            eval_metrics: Dictionary containing evaluation metrics
        """
        self.eval_rewards.append(eval_metrics['mean_reward'])
        
        if self.use_tensorboard:
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f'Evaluation/{key}', value, episode)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if self.use_tensorboard:
            self.writer.add_hparams(hyperparams, {})
    
    def save_rewards(self, filename: str = 'rewards.json'):
        """
        Save training rewards to file.
        
        Args:
            filename: Name of the file to save rewards
        """
        rewards_path = os.path.join(self.log_dir, filename)
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'q_values': self.q_values,
            'epsilons': self.epsilons,
            'eval_rewards': self.eval_rewards
        }
        
        with open(rewards_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot_rewards(self, save_path: str = None, window_size: int = 100):
        """
        Plot training rewards and metrics.
        
        Args:
            save_path: Path to save the plot
            window_size: Window size for moving average
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(self.episode_rewards)), 
                           moving_avg, label=f'{window_size}-Episode Moving Average', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        if self.losses:
            axes[0, 1].plot(self.losses, alpha=0.6)
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Q-values
        if self.q_values:
            axes[1, 0].plot(self.q_values, alpha=0.6)
            axes[1, 0].set_title('Average Q-Values')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Q-Value')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Epsilon
        if self.epsilons:
            axes[1, 1].plot(self.epsilons, alpha=0.6)
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_evaluation(self, save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.eval_rewards:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_rewards, marker='o', linewidth=2, markersize=4)
        plt.title('Evaluation Performance')
        plt.xlabel('Evaluation Episode')
        plt.ylabel('Mean Reward')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics of training.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'recent_mean_reward': np.mean(recent_rewards),
            'recent_std_reward': np.std(recent_rewards),
            'best_eval_reward': max(self.eval_rewards) if self.eval_rewards else 0.0
        }
    
    def close(self):
        """Close the logger and TensorBoard writer."""
        if self.use_tensorboard:
            self.writer.close() 