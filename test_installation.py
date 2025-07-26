#!/usr/bin/env python3
"""
Simple test script to verify LunarLander DQN installation and basic functionality.
"""

import sys
import os
import yaml
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.envs.make_env import make_env, get_env_info
        from src.agents.dqn import DQNAgent
        from src.nets.dqn_net import DQNNetwork
        from src.core.buffer import ReplayBuffer
        from src.core.logger import Logger
        from src.core.trainer import Trainer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment creation and basic interaction."""
    print("\nTesting environment...")
    
    try:
        from src.envs.make_env import make_env, get_env_info
        
        # Test environment info
        env_info = get_env_info("LunarLander-v3")
        print(f"✅ Environment info retrieved: {env_info['observation_dim']} states, {env_info['action_dim']} actions")
        
        # Test environment creation
        env = make_env("LunarLander-v3", seed=42)
        state, _ = env.reset()
        print(f"✅ Environment created, state shape: {state.shape}")
        
        # Test basic interaction
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        print(f"✅ Environment step successful, reward: {reward}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_network():
    """Test neural network creation and forward pass."""
    print("\nTesting neural network...")
    
    try:
        from src.nets.dqn_net import DQNNetwork
        
        # Create network
        network = DQNNetwork(input_dim=8, output_dim=4, hidden_sizes=[128, 128])
        print(f"✅ Network created with {sum(p.numel() for p in network.parameters())} parameters")
        
        # Test forward pass
        state = torch.randn(1, 8)
        q_values = network(state)
        print(f"✅ Forward pass successful, output shape: {q_values.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def test_replay_buffer():
    """Test replay buffer functionality."""
    print("\nTesting replay buffer...")
    
    try:
        from src.core.buffer import ReplayBuffer
        
        # Create buffer
        buffer = ReplayBuffer(capacity=1000, state_dim=8)
        print("✅ Replay buffer created")
        
        # Test storing and sampling
        for i in range(100):
            state = np.random.randn(8)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = np.random.choice([True, False])
            
            buffer.push(state, action, reward, next_state, done)
        
        # Test sampling
        if len(buffer) >= 32:
            batch = buffer.sample(32)
            print(f"✅ Buffer sampling successful, batch size: {len(batch[0])}")
        else:
            print("⚠️  Buffer not full enough for sampling")
        
        return True
    except Exception as e:
        print(f"❌ Replay buffer test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Environment: {config['env']['name']}")
        print(f"   - Learning rate: {config['agent']['learning_rate']}")
        print(f"   - Batch size: {config['training']['batch_size']}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_device():
    """Test device availability."""
    print("\nTesting device availability...")
    
    try:
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("✅ CPU only mode available")
        
        # Test tensor operations
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations successful, result shape: {z.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("LunarLander DQN Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_network,
        test_replay_buffer,
        test_config,
        test_device
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nYou can now run training with:")
        print("python scripts/train.py --config configs/default.yaml")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 