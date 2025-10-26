#!/usr/bin/env python3
"""Test script to check checkpoint compatibility with different environments."""

import torch
import gymnasium as gym
import spinkick_example  # This registers the spinkick environments

def test_checkpoint_compatibility():
    """Test which environment configuration matches the checkpoint."""
    
    # Load the checkpoint
    checkpoint_path = "/home/hoan/KunTao_Workspace/SBMP/g1_spinkick_example/logs/rsl_rl/g1_spinkick_box_pushing/2025-10-25_21-52-34/model_0.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print("Checkpoint model state dict keys:")
    for key, value in checkpoint["model_state_dict"].items():
        if "actor.0.weight" in key:
            print(f"  {key}: {value.shape}")
        elif "critic.0.weight" in key:
            print(f"  {key}: {value.shape}")
        elif "actor_obs_normalizer._mean" in key:
            print(f"  {key}: {value.shape}")
        elif "critic_obs_normalizer._mean" in key:
            print(f"  {key}: {value.shape}")
    
    print("\nTesting environment observation spaces:")
    
    # Test different environments
    env_configs = [
        "Mjlab-Spinkick-Unitree-G1",
        "Mjlab-Spinkick-Unitree-G1-Play", 
        "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Play"
    ]
    
    for env_id in env_configs:
        try:
            env = gym.make(env_id)
            obs_space = env.observation_space
            print(f"\n{env_id}:")
            print(f"  Policy obs shape: {obs_space['policy'].shape}")
            print(f"  Critic obs shape: {obs_space['critic'].shape}")
            
            # Check if dimensions match checkpoint
            policy_dim = obs_space['policy'].shape[0]
            critic_dim = obs_space['critic'].shape[0]
            
            # Get expected dimensions from checkpoint
            actor_input_dim = checkpoint["model_state_dict"]["actor.0.weight"].shape[1]
            critic_input_dim = checkpoint["model_state_dict"]["critic.0.weight"].shape[1]
            
            print(f"  Expected actor input: {actor_input_dim}")
            print(f"  Expected critic input: {critic_input_dim}")
            print(f"  Match: {policy_dim == actor_input_dim and critic_dim == critic_input_dim}")
            
            env.close()
            
        except Exception as e:
            print(f"\n{env_id}: Error - {e}")

if __name__ == "__main__":
    test_checkpoint_compatibility()
