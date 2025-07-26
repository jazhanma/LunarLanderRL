# LunarLander DQN: Code Walkthrough

## Overview

This document provides a comprehensive walkthrough of the LunarLander DQN implementation, designed to help you understand the codebase architecture and training pipeline for technical interviews.

## Project Structure

```
LunarLanderRL/
├── src/                    # Core implementation
│   ├── agents/            # RL agent implementations
│   ├── nets/              # Neural network architectures
│   ├── core/              # Training utilities
│   └── envs/              # Environment wrappers
├── scripts/               # Training and evaluation scripts
├── configs/               # Configuration files
└── docs/                  # Documentation
```

## Core Components

### 1. DQN Agent (`src/agents/dqn.py`)

The `DQNAgent` class implements the core DQN algorithm:

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, config, device):
        # Initialize Q-networks (main and target)
        self.q_network = DQNNetwork(...)
        self.target_network = DQNNetwork(...)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(...)
        
        # Training parameters
        self.epsilon = self.epsilon_start
        self.optimizer = optim.Adam(...)
```

**Key Methods:**
- `select_action()`: Epsilon-greedy action selection
- `store_transition()`: Add experience to replay buffer
- `update()`: Q-network training step
- `evaluate()`: Performance evaluation

### 2. Neural Network (`src/nets/dqn_net.py`)

The `DQNNetwork` implements the Q-function approximator:

```python
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        # Build sequential network
        layers = []
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(...), nn.ReLU()])
        layers.append(nn.Linear(..., output_dim))
        self.network = nn.Sequential(*layers)
```

**Features:**
- Configurable architecture (hidden layer sizes)
- Xavier weight initialization
- Support for different activation functions

### 3. Experience Replay (`src/core/buffer.py`)

The `ReplayBuffer` stores and samples experiences:

```python
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        # Pre-allocated numpy arrays for efficiency
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, 1))
        self.rewards = np.zeros((capacity, 1))
        # ... other fields
```

**Key Features:**
- Efficient circular buffer implementation
- Batch sampling for training
- Memory-efficient storage using numpy arrays

### 4. Training Pipeline (`src/core/trainer.py`)

The `Trainer` orchestrates the entire training process:

```python
class Trainer:
    def train_episode(self):
        state = self.env.reset()
        for step in range(max_steps):
            action = self.agent.select_action(state)
            next_state, reward, done = self.env.step(action)
            self.agent.store_transition(state, action, reward, next_state, done)
            
            if len(self.agent.replay_buffer) >= min_buffer_size:
                self.agent.update()
```

## Training Flow

### 1. Initialization Phase
```python
# Load configuration
config = load_config('configs/default.yaml')

# Create environment and agent
env = make_env('LunarLander-v2', seed=42)
agent = DQNAgent(state_dim, action_dim, config, device)

# Initialize trainer
trainer = Trainer(config, device)
```

### 2. Episode Execution
```python
def train_episode(self):
    state = self.env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Action selection
        action = self.agent.select_action(state, training=True)
        
        # Environment interaction
        next_state, reward, done = self.env.step(action)
        
        # Experience storage
        self.agent.store_transition(state, action, reward, next_state, done)
        
        # Network update (if buffer is ready)
        if len(self.agent.replay_buffer) >= min_buffer_size:
            self.agent.update()
```

### 3. Q-Network Update
```python
def update(self):
    # Sample batch from replay buffer
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    
    # Compute current Q-values
    current_q_values = self.q_network(states).gather(1, actions)
    
    # Compute target Q-values
    with torch.no_grad():
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * ~dones
    
    # Compute loss and update
    loss = F.mse_loss(current_q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

## Key Design Decisions

### 1. Modular Architecture
- **Separation of Concerns**: Agent, networks, and training logic are separate
- **Configurability**: All hyperparameters in YAML config files
- **Extensibility**: Easy to add new agents or environments

### 2. Reproducibility
```python
def _set_seeds(self):
    seed = self.config['env']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### 3. Efficient Implementation
- **Pre-allocated Arrays**: Replay buffer uses numpy arrays for speed
- **Batch Processing**: Network updates use batched tensors
- **Memory Management**: Proper tensor device placement

### 4. Monitoring and Logging
```python
class Logger:
    def log_episode(self, episode, reward, length, loss, q_value, epsilon):
        # Log to TensorBoard
        self.writer.add_scalar('Training/Episode_Reward', reward, episode)
        # Store for plotting
        self.episode_rewards.append(reward)
```

## Configuration System

The YAML configuration system allows easy hyperparameter tuning:

```yaml
# configs/default.yaml
agent:
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995

training:
  max_episodes: 2000
  batch_size: 64
  buffer_size: 100000
```

## Evaluation Pipeline

### 1. Model Loading
```python
def load_model(self, filepath):
    checkpoint = torch.load(filepath, map_location=self.device)
    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    self.epsilon = checkpoint['epsilon']
```

### 2. Performance Assessment
```python
def evaluate(self, env, num_episodes=100):
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        while not done:
            action = self.select_action(state, training=False)
            state, reward, done = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }
```

## Common Interview Questions

### Q: How does DQN handle the exploration-exploitation trade-off?
**A**: DQN uses epsilon-greedy exploration with decaying epsilon. Initially, epsilon=1.0 for full exploration, then decays to epsilon=0.01 for minimal exploration. The decay rate (0.995) balances sufficient exploration with convergence.

### Q: Why do we need a target network?
**A**: The target network provides stable targets for Q-value updates. Without it, the network would be chasing moving targets, leading to training instability. The target network is updated less frequently (every 1000 steps) to maintain stability.

### Q: How does experience replay improve training?
**A**: Experience replay breaks temporal correlations in sequential experiences, allowing the network to learn from diverse, uncorrelated samples. It also improves sample efficiency by reusing experiences multiple times.

### Q: What's the purpose of the replay buffer size?
**A**: The buffer size (100,000) determines how much experience is stored. Too small: limited diversity; too large: memory usage and stale experiences. 100,000 provides sufficient diversity while being memory-efficient.

### Q: How do you know when training has converged?
**A**: We monitor the moving average of episode rewards. When it consistently stays above 200 (the success threshold) for 100+ episodes, the agent has converged. We also check evaluation performance every 100 episodes.

## Performance Optimization Tips

### 1. Device Management
```python
# Efficient tensor operations
states = states.to(self.device)
actions = actions.to(self.device)
```

### 2. Gradient Clipping
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
```

### 3. Batch Processing
```python
# Efficient batch sampling
states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
```

## Extension Points

The codebase is designed for easy extension:

1. **New Agents**: Implement new RL algorithms by extending the base agent interface
2. **New Environments**: Add environment wrappers in `src/envs/`
3. **Advanced Features**: Add prioritized replay, dueling networks, etc.
4. **Multi-Agent**: Extend for multi-agent scenarios

This modular design makes the codebase interview-ready and demonstrates good software engineering practices in ML projects. 