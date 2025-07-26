#!/usr/bin/env python3
"""
Simple test to check LunarLander-v3 environment.
"""

import gymnasium as gym

def test_lunarlander():
    """Test LunarLander-v3 environment creation."""
    print("Testing LunarLander-v3...")
    
    try:
        # Try to create the environment
        env = gym.make('LunarLander-v3')
        print("✅ LunarLander-v3 created successfully!")
        
        # Test basic interaction
        state, _ = env.reset()
        print(f"✅ Environment reset successful, state shape: {state.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={terminated or truncated}")
            if terminated or truncated:
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ LunarLander-v3 test failed: {e}")
        return False

def test_available_envs():
    """Test what environments are available."""
    print("\nTesting available environments...")
    
    try:
        # List all environments
        all_envs = list(gym.envs.registry.keys())
        
        # Find Box2D environments
        box2d_envs = [env for env in all_envs if 'box2d' in env.lower()]
        print(f"Found {len(box2d_envs)} Box2D environments:")
        for env in box2d_envs[:10]:  # Show first 10
            print(f"  - {env}")
        
        # Find LunarLander environments
        lunar_envs = [env for env in all_envs if 'lunar' in env.lower()]
        print(f"\nFound {len(lunar_envs)} LunarLander environments:")
        for env in lunar_envs:
            print(f"  - {env}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment listing failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LunarLander-v3 Test")
    print("=" * 50)
    
    test_available_envs()
    test_lunarlander()
    
    print("=" * 50) 