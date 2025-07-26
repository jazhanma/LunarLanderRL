import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional
from tqdm import tqdm

from agents.dqn import DQNAgent
from envs.make_env import make_env, get_env_info
from core.logger import Logger


class Trainer:
    """
    Main trainer class for DQN training.
    
    Args:
        config: Configuration dictionary
        device: Device to run training on
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Environment setup
        self.env_name = config['env']['name']
        self.env_seed = config['env']['seed']
        self.env = make_env(self.env_name, self.env_seed)
        self.eval_env = make_env(self.env_name, self.env_seed + 1000)  # Different seed for eval
        
        # Get environment information
        env_info = get_env_info(self.env_name)
        self.state_dim = env_info['observation_dim']
        self.action_dim = env_info['action_dim']
        
        # Training parameters
        self.max_episodes = config['training']['max_episodes']
        self.max_steps_per_episode = config['training']['max_steps_per_episode']
        self.update_freq = config['training']['update_freq']
        self.eval_freq = config['training']['eval_freq']
        self.save_freq = config['training']['save_freq']
        
        # Logging setup
        log_dir = config['logging']['log_dir']
        use_tensorboard = config['logging']['tensorboard']
        self.logger = Logger(log_dir, config, use_tensorboard)
        
        # Model saving
        self.save_dir = config['model']['save_dir']
        self.save_best = config['model']['save_best']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize agent
        self.agent = DQNAgent(self.state_dim, self.action_dim, config, device)
        
        # Load pretrained model if specified
        load_path = config['model']['load_path']
        if load_path and os.path.exists(load_path):
            self.agent.load_model(load_path)
            print(f"Loaded pretrained model from {load_path}")
        
        # Training state
        self.best_eval_reward = -float('inf')
        self.episode_count = 0
        self.step_count = 0
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode.
        
        Returns:
            Dictionary containing episode metrics
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        episode_q_values = []
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update network
            if len(self.agent.replay_buffer) >= self.agent.min_buffer_size and step % self.update_freq == 0:
                update_metrics = self.agent.update()
                if update_metrics['loss'] > 0:
                    episode_losses.append(update_metrics['loss'])
                    episode_q_values.append(update_metrics['q_value'])
            
            episode_reward += reward
            episode_length += 1
            self.step_count += 1
            
            if done:
                break
            
            state = next_state
        
        # Calculate average metrics for the episode
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0.0
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'loss': avg_loss,
            'q_value': avg_q_value,
            'epsilon': self.agent.epsilon
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        eval_config = self.config['evaluation']
        num_episodes = eval_config['num_episodes']
        render = eval_config['render']
        
        return self.agent.evaluate(self.eval_env, num_episodes)
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
            is_best: Whether this is the best model so far
        """
        # Regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth')
        self.agent.save_model(checkpoint_path)
        
        # Best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            self.agent.save_model(best_path)
            print(f"New best model saved! Eval reward: {self.best_eval_reward:.2f}")
    
    def train(self):
        """
        Main training loop.
        """
        print(f"Starting training on {self.device}")
        print(f"Environment: {self.env_name}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Max episodes: {self.max_episodes}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            for episode in range(1, self.max_episodes + 1):
                self.episode_count = episode
                
                # Train one episode
                episode_metrics = self.train_episode()
                
                # Log episode
                self.logger.log_episode(
                    episode=episode,
                    reward=episode_metrics['reward'],
                    length=episode_metrics['length'],
                    loss=episode_metrics['loss'],
                    q_value=episode_metrics['q_value'],
                    epsilon=episode_metrics['epsilon']
                )
                
                # Print progress
                if episode % 10 == 0 or episode == 1:
                    print(f"Episode {episode:4d} | "
                          f"Reward: {episode_metrics['reward']:6.1f} | "
                          f"Length: {episode_metrics['length']:3d} | "
                          f"Epsilon: {episode_metrics['epsilon']:.3f} | "
                          f"Loss: {episode_metrics['loss']:.4f}")
                
                # Evaluation
                if episode % self.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    self.logger.log_evaluation(episode, eval_metrics)
                    
                    print(f"Evaluation | Mean Reward: {eval_metrics['mean_reward']:.2f} ± "
                          f"{eval_metrics['std_reward']:.2f} | "
                          f"Range: [{eval_metrics['min_reward']:.1f}, {eval_metrics['max_reward']:.1f}]")
                    
                    # Save best model
                    if eval_metrics['mean_reward'] > self.best_eval_reward:
                        self.best_eval_reward = eval_metrics['mean_reward']
                        self.save_checkpoint(episode, is_best=True)
                
                # Regular checkpoint
                if episode % self.save_freq == 0:
                    self.save_checkpoint(episode)
                
                # Early stopping (optional)
                if episode > 1000 and episode_metrics['reward'] > 200:
                    recent_rewards = self.logger.episode_rewards[-100:]
                    if np.mean(recent_rewards) > 200:
                        print(f"Early stopping: Achieved target performance!")
                        break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Final evaluation and cleanup
            print("\n" + "="*50)
            print("Training completed!")
            
            final_eval = self.evaluate()
            print(f"Final evaluation: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
            
            # Save final model
            self.save_checkpoint(self.episode_count)
            
            # Save training data
            self.logger.save_rewards()
            
            # Print summary
            summary = self.logger.get_summary_stats()
            print(f"\nTraining Summary:")
            print(f"Total episodes: {summary.get('total_episodes', 0)}")
            print(f"Best eval reward: {summary.get('best_eval_reward', 0):.2f}")
            print(f"Recent mean reward: {summary.get('recent_mean_reward', 0):.2f}")
            
            # Close environments and logger
            self.env.close()
            self.eval_env.close()
            self.logger.close()
            
            training_time = time.time() - start_time
            print(f"Total training time: {training_time/3600:.2f} hours")
    
    def plot_results(self, save_path: str = None):
        """
        Plot training results.
        
        Args:
            save_path: Path to save the plot
        """
        self.logger.plot_rewards(save_path)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return self.logger.get_summary_stats() 