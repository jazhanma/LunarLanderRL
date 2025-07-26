import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action spaces.
    
    Args:
        input_dim: Dimension of the observation space
        output_dim: Number of possible actions
        hidden_sizes: List of hidden layer sizes
        activation: Activation function ('relu', 'tanh', etc.)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = [128, 128], 
                 activation: str = 'relu'):
        super(DQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if activation == 'relu' else nn.Tanh()
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values for each action of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.output_dim, (1,)).item()
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all actions given a state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            return self.forward(state)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    
    Args:
        input_dim: Dimension of the observation space
        output_dim: Number of possible actions
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = [128, 128], 
                 activation: str = 'relu'):
        super(DuelingDQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        # Shared feature layers
        shared_layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_sizes[:-1]):
            shared_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if activation == 'relu' else nn.Tanh()
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden_sizes[-1], output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values for each action of shape (batch_size, output_dim)
        """
        shared_features = self.shared_layers(x)
        
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values 