import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
import random

from nets.dqn_net import DQNNetwork
from core.buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent implementation.
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Number of possible actions
        config: Configuration dictionary
        device: Device to run computations on
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Network parameters
        self.learning_rate = config['agent']['learning_rate']
        self.gamma = config['agent']['gamma']
        self.epsilon_start = config['agent']['epsilon_start']
        self.epsilon_end = config['agent']['epsilon_end']
        self.epsilon_decay = config['agent']['epsilon_decay']
        self.target_update_freq = config['agent']['target_update_freq']
        
        # Training parameters
        self.batch_size = config['training']['batch_size']
        self.buffer_size = config['training']['buffer_size']
        self.min_buffer_size = config['training']['min_buffer_size']
        
        # Advanced features
        self.double_dqn = config.get('advanced', {}).get('double_dqn', False)
        
        # Network architecture
        hidden_sizes = config['network']['hidden_sizes']
        activation = config['network']['activation']
        
        # Initialize networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_sizes, activation).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_sizes, activation).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, state_dim)
        
        # Training state
        self.epsilon = self.epsilon_start
        self.step_count = 0
        self.update_count = 0
        
        # Set random seeds for reproducibility
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config['env']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        Update the Q-network using a batch of experiences.
        
        Returns:
            Dictionary containing loss and other metrics
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return {'loss': 0.0, 'q_value': 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }
    
    def save_model(self, filepath: str):
        """Save the Q-network to a file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the Q-network from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions given a state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for all actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def evaluate(self, env, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            env: Environment to evaluate in
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.q_network.eval()
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        self.q_network.train()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        } 