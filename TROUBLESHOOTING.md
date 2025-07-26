# Troubleshooting Guide

## Box2D Installation Issues

### Problem: "Box2D is not installed" error on Windows

This is a common issue on Windows systems. Here are several solutions to try:

#### Solution 1: Install from source
```bash
pip uninstall box2d-py -y
pip install box2d-py --no-cache-dir
```

#### Solution 2: Use conda (recommended for Windows)
```bash
conda install -c conda-forge box2d-py
```

#### Solution 3: Install pre-compiled wheel
```bash
pip install box2d-py==2.3.5 --force-reinstall
```

#### Solution 4: Use gymnasium with box2d extras
```bash
pip install "gymnasium[box2d]==0.28.1" --force-reinstall
```

### Problem: ImportError for Box2D

If Box2D is installed but not importing:

1. Check if it's installed: `pip list | grep box2d`
2. Try different import names:
   ```python
   import Box2D  # Try this first
   import box2d  # Try this if first fails
   ```

### Problem: LunarLander environment not found

If you get "Environment not found" for LunarLander:

1. Make sure you're using the correct environment name:
   - `LunarLander-v3` (current)
   - `LunarLander-v2` (deprecated)

2. Check if Box2D environments are available:
   ```python
   import gymnasium as gym
   print([env for env in gym.envs.registry.keys() if 'LunarLander' in env])
   ```

## Alternative Solutions

### Use CartPole for Testing
If Box2D continues to fail, you can test the DQN implementation with CartPole:

```bash
python scripts/train.py --config configs/default.yaml
```

### Use Different Environment
You can modify the config to use other environments:
- `CartPole-v1` (simple, no Box2D required)
- `Acrobot-v1` (simple, no Box2D required)
- `MountainCar-v0` (simple, no Box2D required)

## Common Import Issues

### Problem: ModuleNotFoundError for src modules

If you get import errors when running scripts:

1. Make sure you're running from the project root directory
2. Check that all `__init__.py` files exist
3. Try running with explicit path:
   ```bash
   PYTHONPATH=src python scripts/train.py --config configs/default.yaml
   ```

### Problem: CUDA/GPU issues

If you get CUDA errors:

1. Check if CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. Try running on CPU:
   ```bash
   python scripts/train.py --config configs/default.yaml --device cpu
   ```

## Performance Issues

### Problem: Training is too slow

1. Use GPU acceleration (CUDA)
2. Reduce batch size in config
3. Reduce network size
4. Use smaller replay buffer

### Problem: Agent not learning

1. Check learning rate (too high or too low)
2. Verify epsilon decay schedule
3. Check if replay buffer is filling up
4. Monitor loss values during training

## Getting Help

If none of these solutions work:

1. Check the full error message
2. Verify your Python version (3.8+ recommended)
3. Try creating a fresh virtual environment
4. Check if all dependencies are compatible

## Environment-Specific Issues

### LunarLander-v3
- Requires Box2D
- Complex environment (8 state dimensions, 4 actions)
- Needs larger network and more training episodes
- Target reward: 200+ (solved)

### CartPole-v1
- Simple environment (4 state dimensions, 2 actions)
- No external dependencies
- Target reward: 195+ (solved)
- Good for testing DQN implementation 