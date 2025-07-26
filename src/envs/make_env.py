import gymnasium as gym
import numpy as np
from typing import Optional


def make_env(env_name: str, seed: Optional[int] = None, render_mode: Optional[str] = None):
    """
    Create and configure a gymnasium environment.
    
    Args:
        env_name: Name of the environment
        seed: Random seed for reproducibility
        render_mode: Rendering mode ('human', 'rgb_array', etc.)
    
    Returns:
        Configured environment
    """
    env = gym.make(env_name, render_mode=render_mode)
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


def get_env_info(env_name: str):
    """
    Get environment information including observation and action spaces.
    
    Args:
        env_name: Name of the environment
    
    Returns:
        Dictionary with environment information
    """
    env = gym.make(env_name)
    env_info = {
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'observation_dim': env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
        'action_dim': env.action_space.n if hasattr(env.action_space, 'n') else None,
        'is_discrete': isinstance(env.action_space, gym.spaces.Discrete),
        'is_continuous': isinstance(env.action_space, gym.spaces.Box)
    }
    env.close()
    return env_info 