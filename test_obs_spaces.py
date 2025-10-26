#!/usr/bin/env python3
"""Test script to check observation spaces without creating full environments."""

import sys
import os
sys.path.append('/home/hoan/KunTao_Workspace/SBMP/g1_spinkick_example/mjlab/src')

from mjlab.tasks.tracking.config.g1.flat_env_cfg import (
    G1FlatNoStateEstimationEnvCfg_PLAY,
    G1FlatEnvCfg_PLAY
)
from mjlab.tasks.tracking.tracking_env_cfg import ObservationCfg
import spinkick_example
from spinkick_example.spinkick_env_cfg import G1SpinkickCfg_PLAY

def test_observation_spaces():
    """Test observation spaces for different configurations."""
    
    print("Testing observation spaces for different configurations:")
    
    # Test G1FlatNoStateEstimationEnvCfg_PLAY
    try:
        cfg = G1FlatNoStateEstimationEnvCfg_PLAY()
        cfg.commands.motion.motion_file = "dummy.npz"  # Dummy file for testing
        cfg.scene.num_envs = 1
        
        # Calculate observation dimensions
        policy_obs = []
        critic_obs = []
        
        # Policy observations
        if cfg.observations.policy.command is not None:
            policy_obs.append(58)  # command dimension
        if cfg.observations.policy.motion_anchor_pos_b is not None:
            policy_obs.append(3)   # motion_anchor_pos_b
        if cfg.observations.policy.motion_anchor_ori_b is not None:
            policy_obs.append(6)   # motion_anchor_ori_b
        if cfg.observations.policy.base_lin_vel is not None:
            policy_obs.append(3)   # base_lin_vel
        if cfg.observations.policy.base_ang_vel is not None:
            policy_obs.append(3)   # base_ang_vel
        if cfg.observations.policy.joint_pos is not None:
            policy_obs.append(29)  # joint_pos
        if cfg.observations.policy.joint_vel is not None:
            policy_obs.append(29)  # joint_vel
        if cfg.observations.policy.actions is not None:
            policy_obs.append(29)  # actions
        if cfg.observations.policy.object_pose_w is not None:
            policy_obs.append(7)   # object_pose_w
        if cfg.observations.policy.object_pos_b is not None:
            policy_obs.append(3)   # object_pos_b
        if cfg.observations.policy.object_ori_b is not None:
            policy_obs.append(6)   # object_ori_b
            
        # Critic observations (includes all policy + additional)
        critic_obs = policy_obs.copy()
        if cfg.observations.critic.body_pos is not None:
            critic_obs.append(42)  # body_pos
        if cfg.observations.critic.body_ori is not None:
            critic_obs.append(84)  # body_ori
            
        policy_dim = sum(policy_obs)
        critic_dim = sum(critic_obs)
        
        print(f"\nG1FlatNoStateEstimationEnvCfg_PLAY:")
        print(f"  Policy obs: {policy_dim} dimensions")
        print(f"  Critic obs: {critic_dim} dimensions")
        print(f"  Policy components: {policy_obs}")
        print(f"  Critic components: {critic_obs}")
        
    except Exception as e:
        print(f"Error with G1FlatNoStateEstimationEnvCfg_PLAY: {e}")
    
    # Test G1FlatEnvCfg_PLAY (with state estimation)
    try:
        cfg = G1FlatEnvCfg_PLAY()
        cfg.commands.motion.motion_file = "dummy.npz"
        cfg.scene.num_envs = 1
        
        # Calculate observation dimensions
        policy_obs = []
        critic_obs = []
        
        # Policy observations
        if cfg.observations.policy.command is not None:
            policy_obs.append(58)  # command dimension
        if cfg.observations.policy.motion_anchor_pos_b is not None:
            policy_obs.append(3)   # motion_anchor_pos_b
        if cfg.observations.policy.motion_anchor_ori_b is not None:
            policy_obs.append(6)   # motion_anchor_ori_b
        if cfg.observations.policy.base_lin_vel is not None:
            policy_obs.append(3)   # base_lin_vel
        if cfg.observations.policy.base_ang_vel is not None:
            policy_obs.append(3)   # base_ang_vel
        if cfg.observations.policy.joint_pos is not None:
            policy_obs.append(29)  # joint_pos
        if cfg.observations.policy.joint_vel is not None:
            policy_obs.append(29)  # joint_vel
        if cfg.observations.policy.actions is not None:
            policy_obs.append(29)  # actions
        if cfg.observations.policy.object_pose_w is not None:
            policy_obs.append(7)   # object_pose_w
        if cfg.observations.policy.object_pos_b is not None:
            policy_obs.append(3)   # object_pos_b
        if cfg.observations.policy.object_ori_b is not None:
            policy_obs.append(6)   # object_ori_b
            
        # Critic observations (includes all policy + additional)
        critic_obs = policy_obs.copy()
        if cfg.observations.critic.body_pos is not None:
            critic_obs.append(42)  # body_pos
        if cfg.observations.critic.body_ori is not None:
            critic_obs.append(84)  # body_ori
            
        policy_dim = sum(policy_obs)
        critic_dim = sum(critic_obs)
        
        print(f"\nG1FlatEnvCfg_PLAY:")
        print(f"  Policy obs: {policy_dim} dimensions")
        print(f"  Critic obs: {critic_dim} dimensions")
        print(f"  Policy components: {policy_obs}")
        print(f"  Critic components: {critic_obs}")
        
    except Exception as e:
        print(f"Error with G1FlatEnvCfg_PLAY: {e}")
    
    # Test G1SpinkickCfg_PLAY
    try:
        cfg = G1SpinkickCfg_PLAY()
        cfg.commands.motion.motion_file = "dummy.npz"
        cfg.scene.num_envs = 1
        
        # Calculate observation dimensions
        policy_obs = []
        critic_obs = []
        
        # Policy observations
        if cfg.observations.policy.command is not None:
            policy_obs.append(58)  # command dimension
        if cfg.observations.policy.motion_anchor_pos_b is not None:
            policy_obs.append(3)   # motion_anchor_pos_b
        if cfg.observations.policy.motion_anchor_ori_b is not None:
            policy_obs.append(6)   # motion_anchor_ori_b
        if cfg.observations.policy.base_lin_vel is not None:
            policy_obs.append(3)   # base_lin_vel
        if cfg.observations.policy.base_ang_vel is not None:
            policy_obs.append(3)   # base_ang_vel
        if cfg.observations.policy.joint_pos is not None:
            policy_obs.append(29)  # joint_pos
        if cfg.observations.policy.joint_vel is not None:
            policy_obs.append(29)  # joint_vel
        if cfg.observations.policy.actions is not None:
            policy_obs.append(29)  # actions
        if cfg.observations.policy.object_pose_w is not None:
            policy_obs.append(7)   # object_pose_w
        if cfg.observations.policy.object_pos_b is not None:
            policy_obs.append(3)   # object_pos_b
        if cfg.observations.policy.object_ori_b is not None:
            policy_obs.append(6)   # object_ori_b
            
        # Critic observations (includes all policy + additional)
        critic_obs = policy_obs.copy()
        if cfg.observations.critic.body_pos is not None:
            critic_obs.append(42)  # body_pos
        if cfg.observations.critic.body_ori is not None:
            critic_obs.append(84)  # body_ori
            
        policy_dim = sum(policy_obs)
        critic_dim = sum(critic_obs)
        
        print(f"\nG1SpinkickCfg_PLAY:")
        print(f"  Policy obs: {policy_dim} dimensions")
        print(f"  Critic obs: {critic_dim} dimensions")
        print(f"  Policy components: {policy_obs}")
        print(f"  Critic components: {critic_obs}")
        
    except Exception as e:
        print(f"Error with G1SpinkickCfg_PLAY: {e}")

if __name__ == "__main__":
    test_observation_spaces()
