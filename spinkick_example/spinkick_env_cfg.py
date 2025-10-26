import math
from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm
from mjlab.managers.manager_term_config import term
from mjlab.tasks.tracking.config.g1.flat_env_cfg import G1FlatNoStateEstimationEnvCfg
from mjlab.tasks.tracking.tracking_env_cfg import TerminationsCfg, RewardCfg, ObservationCfg
from mjlab.tasks.tracking.mdp import rewards as tracking_rewards
from mjlab.envs.mdp import rewards as base_rewards
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]


def base_ang_vel_exceed(
  env: ManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


def motion_tracking_error_linear(
  env: ManagerBasedRlEnv, 
  command_name: str, 
  std: float
) -> torch.Tensor:
  """Linear reward for motion tracking to provide better gradients for large errors.
  
  Mathematical formulation: r = max(0, 1 - ||e||_2 / (std * sqrt(3)))
  This provides linear gradients for errors up to std*sqrt(3), then constant reward.
  """
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  
  # Position error
  pos_error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  
  # Orientation error  
  from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude
  ori_error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  
  # Combined error with different scales
  total_error = torch.sqrt(pos_error + ori_error)
  
  # Linear reward: r = max(0, 1 - error / threshold)
  threshold = std * math.sqrt(3)  # sqrt(3) for 3D position + orientation
  reward = torch.clamp(1.0 - total_error / threshold, min=0.0)
  
  return reward


def motion_velocity_tracking_error_linear(
  env: ManagerBasedRlEnv,
  command_name: str, 
  std: float
) -> torch.Tensor:
  """Linear reward for velocity tracking to encourage smooth motion."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  
  # Linear velocity error
  lin_vel_error = torch.sum(
    torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1
  )
  
  # Angular velocity error
  ang_vel_error = torch.sum(
    torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1
  )
  
  total_error = torch.sqrt(lin_vel_error + ang_vel_error)
  threshold = std * math.sqrt(6)  # sqrt(6) for 3D linear + 3D angular velocity
  reward = torch.clamp(1.0 - total_error / threshold, min=0.0)
  
  return reward


def motion_tracking_progress_reward(
  env: ManagerBasedRlEnv,
  command_name: str
) -> torch.Tensor:
  """Reward for making progress through the motion sequence.
  
  Mathematical formulation: r = progress_rate * (1 - error_penalty)
  This encourages both temporal progress and accuracy.
  """
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  
  # Progress rate: how far through the motion we are
  progress_rate = command.time_steps.float() / command.motion.time_step_total
  
  # Error penalty based on current tracking accuracy
  pos_error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  error_penalty = torch.clamp(pos_error / 0.1, max=1.0)  # Normalize to [0,1]
  
  # Combined reward
  reward = progress_rate * (1.0 - error_penalty)
  
  return reward


def actual_object_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Get the actual object position in world frame."""
  box: Entity = env.scene[asset_cfg.name]
  return box.data.body_link_pos_w[:, 0]  # (N, 3) root body position


def actual_object_quat_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Get the actual object orientation in world frame."""
  box: Entity = env.scene[asset_cfg.name]
  return box.data.body_link_quat_w[:, 0]  # (N, 4) root body quaternion


def object_position_error(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Compute the error between actual and desired object position.
  
  This is crucial for box pushing - the robot needs to know how far the box
  is from where it should be to learn to push it correctly.
  """
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # Desired object position from motion data
  desired_pos = command.object_pos_w  # (N, 3)
  
  # Actual object position from simulation
  actual_pos = box.data.body_link_pos_w[:, 0]  # (N, 3)
  
  # Position error: actual - desired
  error = actual_pos - desired_pos
  return error


def object_orientation_error(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Compute the error between actual and desired object orientation."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_error_magnitude
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # Desired object orientation from motion data
  desired_quat = command.object_quat_w  # (N, 4)
  
  # Actual object orientation from simulation
  actual_quat = box.data.body_link_quat_w[:, 0]  # (N, 4)
  
  # Orientation error magnitude
  error = quat_error_magnitude(desired_quat, actual_quat)
  return error


def object_velocity_error(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Compute the error between actual and desired object velocity."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # For now, we'll use the object's linear velocity as a proxy
  # In a more sophisticated setup, we'd compute the desired velocity from motion data
  actual_vel = box.data.body_link_lin_vel_w[:, 0]  # (N, 3)
  
  # For box pushing, we want the box to move forward, so desired velocity is positive x
  desired_vel = torch.zeros_like(actual_vel)
  desired_vel[:, 0] = 0.1  # Small forward velocity target
  
  error = actual_vel - desired_vel
  return error


def robot_to_object_distance(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Compute the distance between robot and object for contact learning."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # Robot anchor position (torso)
  robot_pos = command.robot_anchor_pos_w  # (N, 3)
  
  # Object position
  object_pos = box.data.body_link_pos_w[:, 0]  # (N, 3)
  
  # Distance vector
  distance = torch.norm(robot_pos - object_pos, dim=-1, keepdim=True)  # (N, 1)
  return distance


def box_pushing_reward(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
  """Reward for successfully pushing the box forward.
  
  Mathematical formulation: r = exp(-||actual_pos - desired_pos||_2^2 / std^2)
  This encourages the robot to push the box to the desired position.
  """
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # Desired object position from motion data
  desired_pos = command.object_pos_w  # (N, 3)
  
  # Actual object position from simulation
  actual_pos = box.data.body_link_pos_w[:, 0]  # (N, 3)
  
  # Position error
  error = torch.sum(torch.square(actual_pos - desired_pos), dim=-1)
  
  # Exponential reward
  reward = torch.exp(-error / (std ** 2))
  return reward


def box_velocity_reward(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
  """Reward for box moving in the correct direction (forward)."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # Get box velocity
  box_vel = box.data.body_link_lin_vel_w[:, 0]  # (N, 3)
  
  # We want the box to move forward (positive x direction)
  forward_velocity = box_vel[:, 0]  # (N,)
  
  # Reward positive forward velocity
  reward = torch.tanh(forward_velocity / std)  # Normalize and saturate
  return reward


def contact_force_reward(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Reward for maintaining contact with the box (encourages pushing)."""
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from typing import cast
  
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  box: Entity = env.scene[asset_cfg.name]
  
  # Get contact forces between robot and box
  # This is a simplified version - in practice, you'd need to set up contact sensors
  # For now, we'll use distance as a proxy for contact
  robot_pos = command.robot_anchor_pos_w  # (N, 3)
  object_pos = box.data.body_link_pos_w[:, 0]  # (N, 3)
  
  distance = torch.norm(robot_pos - object_pos, dim=-1)
  
  # Reward being close to the box (encourages contact)
  contact_reward = torch.exp(-distance / 0.5)  # Exponential decay with distance
  return contact_reward


@dataclass
class SpinkickObservationsCfg(ObservationCfg):
  """Enhanced observation configuration for box pushing."""
  
  def __post_init__(self):
    # Add object error observations for box pushing
    self.policy.object_pos_error: ObsTerm = term(
      ObsTerm,
      func=object_position_error,
      params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
    )
    
    self.policy.object_ori_error: ObsTerm = term(
      ObsTerm,
      func=object_orientation_error,
      params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
    )
    
    self.policy.object_vel_error: ObsTerm = term(
      ObsTerm,
      func=object_velocity_error,
      params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
    )
    
    self.policy.robot_to_object_distance: ObsTerm = term(
      ObsTerm,
      func=robot_to_object_distance,
      params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
    )
    
    # Add actual object pose observations
    self.policy.actual_object_pos_w: ObsTerm = term(
      ObsTerm,
      func=actual_object_pos_w,
      params={"asset_cfg": SceneEntityCfg("box")},
    )
    
    self.policy.actual_object_quat_w: ObsTerm = term(
      ObsTerm,
      func=actual_object_quat_w,
      params={"asset_cfg": SceneEntityCfg("box")},
    )


@dataclass
class SpinkickRewardsCfg(RewardCfg):
  """Enhanced reward configuration for spinkick motions with better gradient flow."""
  
  # Linear tracking rewards for better gradients
  motion_tracking_linear: RewTerm = term(
    RewTerm,
    func=motion_tracking_error_linear,
    weight=1.0,
    params={"command_name": "motion", "std": 0.3},
  )
  
  motion_velocity_tracking_linear: RewTerm = term(
    RewTerm,
    func=motion_velocity_tracking_error_linear,
    weight=0.5,
    params={"command_name": "motion", "std": 1.0},
  )
  
  motion_progress: RewTerm = term(
    RewTerm,
    func=motion_tracking_progress_reward,
    weight=0.3,
    params={"command_name": "motion"},
  )
  
  # Keep some exponential rewards for fine-tuning
  motion_global_root_pos: RewTerm = term(
    RewTerm,
    func=tracking_rewards.motion_global_anchor_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
  )
  
  motion_global_root_ori: RewTerm = term(
    RewTerm,
    func=tracking_rewards.motion_global_anchor_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4},
  )
  
  # Body tracking rewards
  motion_body_pos: RewTerm = term(
    RewTerm,
    func=tracking_rewards.motion_relative_body_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.3},
  )
  
  motion_body_ori: RewTerm = term(
    RewTerm,
    func=tracking_rewards.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4},
  )
  
  motion_body_lin_vel: RewTerm = term(
    RewTerm,
    func=tracking_rewards.motion_global_body_linear_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 1.0},
  )
  
  motion_body_ang_vel: RewTerm = term(
    RewTerm,
    func=tracking_rewards.motion_global_body_angular_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 3.14},
  )
  
  # Object tracking rewards (if object is present)
  object_global_pos: RewTerm | None = term(
    RewTerm,
    func=tracking_rewards.object_global_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "object_asset_cfg": SceneEntityCfg("box"), "std": 0.25},
  )
  
  object_global_ori: RewTerm | None = term(
    RewTerm,
    func=tracking_rewards.object_global_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "object_asset_cfg": SceneEntityCfg("box"), "std": 0.4},
  )
  
  ee_relative_to_object_pos: RewTerm | None = term(
    RewTerm,
    func=tracking_rewards.object_relative_ee_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "object_asset_cfg": SceneEntityCfg("box"), "std": 0.2, "body_names": [
      "left_ankle_roll_link",
      "right_ankle_roll_link", 
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ]},
  )
  
  # Penalties
  action_rate_l2: RewTerm = term(
    RewTerm, 
    func=base_rewards.action_rate_l2, 
    weight=-0.1
  )
  
  joint_limit: RewTerm = term(
    RewTerm,
    func=base_rewards.joint_pos_limits,
    weight=-10.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
  )
  
  self_collisions: RewTerm = term(
    RewTerm,
    func=tracking_rewards.self_collision_cost,
    weight=-10.0,
    params={"sensor_name": "self_collision"},
  )
  
  # Box pushing specific rewards
  box_pushing: RewTerm = term(
    RewTerm,
    func=box_pushing_reward,
    weight=2.0,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box"), "std": 0.3},
  )
  
  box_velocity: RewTerm = term(
    RewTerm,
    func=box_velocity_reward,
    weight=1.0,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box"), "std": 0.5},
  )
  
  contact_force: RewTerm = term(
    RewTerm,
    func=contact_force_reward,
    weight=0.5,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
  )


@dataclass
class SpinkickTerminationsCfg(TerminationsCfg):
  base_ang_vel_exceed: DoneTerm = term(
    DoneTerm,
    func=base_ang_vel_exceed,
    params={"threshold": _MAX_ANG_VEL},
  )


@dataclass
class G1SpinkickCfg(G1FlatNoStateEstimationEnvCfg):
  terminations: SpinkickTerminationsCfg = field(default_factory=SpinkickTerminationsCfg)
  rewards: SpinkickRewardsCfg = field(default_factory=SpinkickRewardsCfg)
  observations: SpinkickObservationsCfg = field(default_factory=SpinkickObservationsCfg)
  

@dataclass
class G1SpinkickCfg_PLAY(G1SpinkickCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None
    self.terminations.anchor_pos = None
    self.terminations.anchor_ori = None
    self.terminations.ee_body_pos = None
    self.terminations.base_ang_vel_exceed = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class G1SpinkickCfg_TRAIN(G1SpinkickCfg):
  """Enhanced training configuration with improved adaptive sampling for box pushing."""
  
  def __post_init__(self):
    super().__post_init__()
    
    # Enhanced adaptive sampling parameters based on mathematical analysis
    self.commands.motion.adaptive_alpha = 0.01  # Faster convergence of failure counts
    self.commands.motion.adaptive_lambda = 0.9  # Smoother kernel for better exploration
    self.commands.motion.adaptive_uniform_ratio = 0.2  # Higher exploration ratio
    self.commands.motion.adaptive_kernel_size = 5  # Larger kernel for smoother sampling
    
    # Enhanced observation corruption for robustness
    self.observations.policy.enable_corruption = True
    
    # Shorter episodes for faster training
    self.episode_length_s = 8.0
    
    # Enhanced domain randomization for better generalization
    # Note: push_by_setting_velocity only accepts velocity_range, not force_range/torque_range
    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
      "yaw": (-0.8, 0.8),
    }
    
    # Add velocity-based randomization
    self.commands.motion.velocity_range = {
      "x": (-0.1, 0.1),
      "y": (-0.1, 0.1), 
      "z": (-0.05, 0.05),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    }
    
    # Enhanced box pushing specific parameters
    self.commands.motion.pose_range = {
      "x": (-0.1, 0.1),  # Allow more position variation
      "y": (-0.1, 0.1),
      "z": (-0.05, 0.05),
      "roll": (-0.2, 0.2),  # Allow more orientation variation
      "pitch": (-0.2, 0.2),
      "yaw": (-0.3, 0.3),
    }
    
    # Enhanced joint position randomization for box pushing
    self.commands.motion.joint_position_range = (-0.15, 0.15)  # Slightly larger range
    
    # Enable more aggressive domain randomization for box pushing
    self.events.base_com.params["ranges"] = {
      0: (-0.05, 0.05),  # x position
      1: (-0.1, 0.1),    # y position  
      2: (-0.1, 0.1),    # z position
    }
    
    # Enhanced friction randomization for box contact
    self.events.foot_friction.params["ranges"] = (0.2, 1.5)  # Wider friction range


@dataclass
class G1SpinkickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  """Optimized PPO configuration for box pushing tasks."""
  
  policy: RslRlPpoActorCriticCfg = field(
    default_factory=lambda: RslRlPpoActorCriticCfg(
      init_noise_std=0.5,  # Lower initial noise for more stable learning
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    )
  )
  
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,  # Higher entropy for better exploration in box pushing
      num_learning_epochs=8,  # More learning epochs for complex box pushing task
      num_mini_batches=8,  # More mini-batches for better gradient estimates
      learning_rate=3.0e-4,  # Lower learning rate for more stable learning
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=0.5,  # Lower gradient clipping for stability
    )
  )
  
  experiment_name: str = "g1_spinkick_box_pushing"
  save_interval: int = 50  # Save more frequently
  num_steps_per_env: int = 32  # More steps per environment for better sample efficiency
  max_iterations: int = 50_000  # More iterations for complex task