import numpy as np
import torch
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Args:
        capacity: Maximum number of experiences to store
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space (1 for discrete)
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Storage for transitions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        self.position = 0
        self.size = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                               torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.BoolTensor(self.dones[indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer that samples transitions based on TD-error.
    
    Args:
        capacity: Maximum number of experiences to store
        state_dim: Dimension of the state space
        alpha: Priority exponent (0 = uniform sampling, 1 = fully prioritized)
        beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        beta_increment: Beta increment per sampling
    """
    
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Storage for transitions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        self.position = 0
        self.size = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the prioritized replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Set priority to maximum for new transitions
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions using prioritized sampling.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.BoolTensor(self.dones[indices])
        weights = torch.FloatTensor(weights)
        indices_tensor = torch.LongTensor(indices)
        
        return states, actions, rewards, next_states, dones, indices_tensor, weights
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priorities (typically TD-errors)
        """
        indices = indices.cpu().numpy()
        priorities = priorities.cpu().numpy()
        
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size 