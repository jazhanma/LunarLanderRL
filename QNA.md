# LunarLander DQN: Interview Q&A Cheat Sheet

## Algorithm Fundamentals

### Q: What is DQN and how does it work?
**A**: Deep Q-Network (DQN) is a reinforcement learning algorithm that combines Q-learning with deep neural networks. It learns to approximate Q-values (state-action values) using a neural network instead of a Q-table. The key innovations are:
- **Experience Replay**: Stores and samples past experiences to break temporal correlations
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation

### Q: What's the difference between Q-learning and DQN?
**A**: 
- **Q-learning**: Uses a Q-table to store Q-values for discrete state-action pairs
- **DQN**: Uses a neural network to approximate Q-values for continuous or large state spaces
- **Scalability**: Q-learning doesn't scale to continuous states, DQN does
- **Function Approximation**: DQN generalizes across similar states

### Q: Why do we need experience replay in DQN?
**A**: Experience replay addresses several issues:
- **Temporal Correlations**: Breaks sequential dependencies in experiences
- **Sample Efficiency**: Reuses experiences multiple times
- **Stability**: Provides diverse, uncorrelated training samples
- **Data Distribution**: Maintains stable data distribution for training

### Q: What is the target network and why is it important?
**A**: The target network is a separate copy of the Q-network that:
- **Provides Stable Targets**: Used to compute target Q-values for training
- **Reduces Instability**: Updated less frequently than the main network
- **Prevents Oscillations**: Stops the network from chasing moving targets
- **Improves Convergence**: Leads to more stable training

## Implementation Details

### Q: How do you implement epsilon-greedy exploration?
**A**: 
```python
def select_action(self, state, training=True):
    if training and random.random() < self.epsilon:
        return random.randrange(self.action_dim)  # Exploration
    else:
        q_values = self.q_network(state)
        return q_values.argmax().item()  # Exploitation
```

### Q: What's the Q-learning update rule in DQN?
**A**: 
```python
# Current Q-value
current_q = self.q_network(states).gather(1, actions)

# Target Q-value
with torch.no_grad():
    next_q = self.target_network(next_states).max(1)[0]
    target_q = rewards + gamma * next_q * ~dones

# Loss and update
loss = F.mse_loss(current_q, target_q)
```

### Q: How do you handle the replay buffer implementation?
**A**: 
```python
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        # Pre-allocated arrays for efficiency
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, 1))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
        
    def push(self, state, action, reward, next_state, done):
        # Circular buffer implementation
        self.states[self.position] = state
        # ... store other fields
        self.position = (self.position + 1) % self.capacity
```

## Hyperparameters and Tuning

### Q: What are the key hyperparameters in DQN?
**A**: 
- **Learning Rate**: 0.001 (Adam optimizer)
- **Discount Factor (γ)**: 0.99 (high for long-term planning)
- **Epsilon Decay**: 0.995 (gradual exploration reduction)
- **Batch Size**: 64 (balance stability vs efficiency)
- **Buffer Size**: 100,000 (sufficient experience diversity)
- **Target Update Frequency**: 1000 steps (stability)

### Q: How do you choose the learning rate?
**A**: 
- **Too High**: Training instability, loss oscillations
- **Too Low**: Slow convergence, stuck in local optima
- **Sweet Spot**: 0.001 for Adam optimizer
- **Rule of Thumb**: Start with 0.001, adjust based on loss curves

### Q: What's the optimal epsilon decay schedule?
**A**: 
- **Start**: 1.0 (full exploration)
- **End**: 0.01 (minimal exploration)
- **Decay**: 0.995 per episode
- **Rationale**: Gradual reduction balances exploration with convergence

## Training and Convergence

### Q: How do you know when DQN has converged?
**A**: 
- **Reward Threshold**: Moving average above 200 (success criterion)
- **Stability**: Consistent performance over 100+ episodes
- **Loss Stabilization**: Training loss plateaus
- **Epsilon**: Reaches minimum value (0.01)

### Q: What are common training issues and how do you fix them?
**A**: 
- **Diverging Loss**: Reduce learning rate, increase buffer size
- **Slow Convergence**: Increase epsilon decay, adjust network architecture
- **Overfitting**: Add regularization, reduce network size
- **Instability**: Use gradient clipping, adjust target update frequency

### Q: How do you monitor training progress?
**A**: 
- **Episode Rewards**: Track moving average
- **Training Loss**: Monitor convergence
- **Q-Values**: Check for reasonable ranges
- **Epsilon**: Ensure proper decay
- **Evaluation**: Regular performance assessment

## Environment-Specific Questions

### Q: What makes LunarLander challenging for RL?
**A**: 
- **Continuous State Space**: 8-dimensional state
- **Delayed Rewards**: Success only at episode end
- **Precision Required**: Exact landing on target
- **Multiple Objectives**: Balance fuel efficiency with success
- **Stochastic Dynamics**: Physics-based environment

### Q: How does the reward structure work in LunarLander?
**A**: 
- **Positive Rewards**: Successful landing (+100), fuel efficiency
- **Negative Rewards**: Crashes (-100), fuel consumption
- **Shaping Rewards**: Intermediate rewards for good behavior
- **Success Criterion**: 200+ average reward

### Q: What's the state and action space?
**A**: 
- **State**: 8 dimensions (position, velocity, angle, angular velocity, contact)
- **Actions**: 4 discrete (no-op, main engine, left/right orientation)
- **Observation**: Continuous values, automatically normalized

## Advanced Topics

### Q: What are the limitations of DQN?
**A**: 
- **Discrete Actions Only**: Can't handle continuous action spaces
- **Overestimation Bias**: Q-values tend to be overestimated
- **Sample Inefficiency**: Requires many experiences
- **Hyperparameter Sensitivity**: Performance depends on careful tuning

### Q: How would you improve DQN?
**A**: 
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage streams
- **Prioritized Replay**: Focus on important experiences
- **Rainbow DQN**: Combines multiple improvements

### Q: What's the difference between on-policy and off-policy methods?
**A**: 
- **On-Policy**: Learn from current policy (e.g., A3C, PPO)
- **Off-Policy**: Learn from different policy (e.g., DQN, DDPG)
- **DQN**: Off-policy - learns from replay buffer experiences
- **Advantage**: Off-policy methods are more sample efficient

## System Design Questions

### Q: How would you scale DQN to larger environments?
**A**: 
- **Distributed Training**: Multiple workers collecting experiences
- **Prioritized Replay**: Focus on important experiences
- **Larger Networks**: Deeper/wider architectures
- **Better Exploration**: Advanced exploration strategies

### Q: How do you handle memory constraints in DQN?
**A**: 
- **Efficient Buffer**: Use numpy arrays instead of lists
- **Gradient Checkpointing**: Trade computation for memory
- **Selective Storage**: Store only essential information
- **Buffer Management**: Implement efficient sampling

### Q: What's the computational complexity of DQN?
**A**: 
- **Forward Pass**: O(network_size) per action
- **Backward Pass**: O(network_size) per update
- **Memory**: O(buffer_size * state_dim)
- **Training**: O(episodes * steps_per_episode * batch_size)

## Real-World Applications

### Q: How would you apply DQN to real-world problems?
**A**: 
- **Robotics**: Autonomous navigation, manipulation
- **Gaming**: Game AI, strategy optimization
- **Finance**: Trading strategies, portfolio management
- **Healthcare**: Treatment optimization, resource allocation

### Q: What are the challenges of deploying RL in production?
**A**: 
- **Safety**: Ensuring safe exploration and operation
- **Simulation Gap**: Differences between sim and reality
- **Sample Efficiency**: Limited real-world data
- **Interpretability**: Understanding agent decisions

### Q: How do you evaluate RL agents in production?
**A**: 
- **A/B Testing**: Compare with baseline policies
- **Safety Metrics**: Monitor for unsafe behaviors
- **Performance Metrics**: Track key performance indicators
- **Robustness Testing**: Test under various conditions

## Code Quality and Best Practices

### Q: How do you ensure reproducible results in RL?
**A**: 
- **Seeding**: Set random seeds for all components
- **Deterministic Operations**: Use deterministic algorithms where possible
- **Environment Control**: Control environment randomness
- **Version Control**: Track code, data, and hyperparameters

### Q: What's your testing strategy for RL code?
**A**: 
- **Unit Tests**: Test individual components
- **Integration Tests**: Test full training pipeline
- **Regression Tests**: Ensure performance doesn't degrade
- **Environment Tests**: Test with different environments

### Q: How do you handle hyperparameter tuning?
**A**: 
- **Grid Search**: Systematic parameter exploration
- **Random Search**: More efficient than grid search
- **Bayesian Optimization**: Advanced optimization techniques
- **Automated Tuning**: Tools like Optuna, Ray Tune

This Q&A covers the most common interview questions for RL/DQN positions and demonstrates deep understanding of both theoretical concepts and practical implementation details. 