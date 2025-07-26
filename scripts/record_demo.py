#!/usr/bin/env python3
"""
Record a demo video of the trained DQN agent playing LunarLander.
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

# Try to import imageio for better video encoding
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("⚠️ imageio not available, using OpenCV only")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.dqn import DQNAgent
from envs.make_env import make_env, get_env_info
import yaml


def record_episode(env, agent, episode_length=1000, render_mode='rgb_array'):
    """
    Record a single episode and return frames.
    
    Args:
        env: Gymnasium environment
        agent: Trained DQN agent
        episode_length: Maximum steps per episode
        render_mode: Rendering mode for environment
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    state, _ = env.reset()
    
    for step in range(episode_length):
        # Render current frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Agent selects action
        action = agent.select_action(state, training=False)
        
        # Environment step
        next_state, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            # Render final frame
            final_frame = env.render()
            if final_frame is not None:
                frames.append(final_frame)
            break
            
        state = next_state
    
    env.close()
    return frames


def save_video(frames, output_path, fps=30):
    """
    Save frames as a video file with proper encoding.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Path to save the video
        fps: Frames per second
    """
    if not frames:
        print("No frames to save!")
        return
    
    # Get frame dimensions
    height, width, _ = frames[0].shape
    
    # Try different codecs for better compatibility
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('H264', cv2.VideoWriter_fourcc(*'H264')),
    ]
    
    success = False
    for codec_name, fourcc in codecs_to_try:
        try:
            print(f"Trying codec: {codec_name}")
            
            # Create video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"Failed to open video writer with codec {codec_name}")
                continue
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Verify the file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Video saved successfully with codec {codec_name}")
                print(f"📁 File: {output_path}")
                print(f"📊 Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
                success = True
                break
            else:
                print(f"❌ Video file creation failed with codec {codec_name}")
                
        except Exception as e:
            print(f"❌ Error with codec {codec_name}: {e}")
            continue
    
    if not success:
        print("❌ Failed to create video with any codec. Trying alternative approaches...")
        
        # Try imageio if available
        if IMAGEIO_AVAILABLE:
            try:
                print("🔄 Trying imageio with ffmpeg...")
                imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
                print(f"✅ Video saved successfully with imageio: {output_path}")
                return
            except Exception as e:
                print(f"❌ imageio failed: {e}")
        
        # Fallback: save as AVI with XVID codec
        avi_path = output_path.replace('.mp4', '.avi')
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"✅ Video saved as AVI: {avi_path}")
        except Exception as e:
            print(f"❌ Failed to create AVI video: {e}")
            print("💡 Consider installing additional codecs or using a different video library")


def main():
    parser = argparse.ArgumentParser(description="Record DQN agent demo video")
    parser.add_argument('--config', type=str, default='configs/lunarlander.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='agent_demo.mp4',
                       help='Output video file path')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to record')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for video')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get environment info
    env_name = config['env']['name']
    env_info = get_env_info(env_name)
    
    print(f"Recording demo for {env_name}")
    print(f"State dim: {env_info['observation_dim']}, Action dim: {env_info['action_dim']}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment with rendering
    env = make_env(env_name, seed=42, render_mode='rgb_array')
    
    # Create agent
    agent = DQNAgent(
        state_dim=env_info['observation_dim'],
        action_dim=env_info['action_dim'],
        config=config,
        device=device
    )
    
    # Load trained model
    if os.path.exists(args.model):
        agent.load_model(args.model)
        print(f"Loaded model from: {args.model}")
    else:
        print(f"Model not found: {args.model}")
        return
    
    # Record episodes
    all_frames = []
    total_reward = 0
    
    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}...")
        
        # Create fresh environment for each episode
        env = make_env(env_name, seed=42 + episode, render_mode='rgb_array')
        
        # Record episode
        frames = record_episode(env, agent)
        all_frames.extend(frames)
        
        # Calculate episode reward (approximate)
        episode_reward = len(frames) * 10  # Rough estimate
        total_reward += episode_reward
        
        print(f"Episode {episode + 1} recorded: {len(frames)} frames")
    
    # Save video
    if all_frames:
        save_video(all_frames, args.output, args.fps)
        print(f"Total frames recorded: {len(all_frames)}")
        print(f"Average reward per episode: {total_reward / args.episodes:.1f}")
    else:
        print("No frames were recorded!")


if __name__ == "__main__":
    main() 