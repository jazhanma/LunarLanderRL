#!/usr/bin/env python3
"""
Training script for LunarLander DQN agent.
"""

import argparse
import yaml
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent on LunarLander')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_arg: str) -> str:
    """Determine device to use."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    else:
        return device_arg


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed is not None:
        config['env']['seed'] = args.seed
    
    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Start training
    trainer.train()
    
    # Plot results
    plot_path = os.path.join(config['logging']['log_dir'], 'training_results.png')
    trainer.plot_results(plot_path)
    print(f"Training plots saved to: {plot_path}")


if __name__ == '__main__':
    main() 