#!/usr/bin/env python3
"""
Simple test to check Box2D and LunarLander environment.
"""

import sys
import os

def test_box2d_import():
    """Test if Box2D can be imported."""
    print("Testing Box2D import...")
    try:
        import Box2D
        print(f"✅ Box2D imported successfully: {Box2D.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Box2D import failed: {e}")
        return False

def test_gymnasium_box2d():
    """Test if gymnasium can create Box2D environments."""
    print("\nTesting gymnasium Box2D environments...")
    try:
        import gymnasium as gym
        
        # Try to list Box2D environments
        box2d_envs = [env for env in gym.envs.registry.keys() if 'box2d' in env.lower()]
        print(f"✅ Found {len(box2d_envs)} Box2D environments")
        
        # Try to create LunarLander
        try:
            env = gym.make('LunarLander-v3')
            print("✅ LunarLander-v3 created successfully")
            env.close()
            return True
        except Exception as e:
            print(f"❌ LunarLander-v3 creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Gymnasium Box2D test failed: {e}")
        return False

def test_alternative_environment():
    """Test with a different environment to see if it's a specific issue."""
    print("\nTesting alternative environment...")
    try:
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        print("✅ CartPole-v1 created successfully")
        env.close()
        return True
    except Exception as e:
        print(f"❌ Alternative environment failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Box2D and LunarLander Test")
    print("=" * 50)
    
    tests = [
        test_box2d_import,
        test_gymnasium_box2d,
        test_alternative_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Box2D is working correctly.")
    else:
        print("❌ Some tests failed. There might be a compatibility issue.")
        print("\nTroubleshooting steps:")
        print("1. Try restarting your Python session")
        print("2. Check if you have conflicting packages")
        print("3. Try: pip uninstall box2d-py && pip install box2d-py")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 