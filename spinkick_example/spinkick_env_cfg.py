import math
from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import term
from mjlab.tasks.tracking.config.g1.flat_env_cfg import G1FlatNoStateEstimationEnvCfg
from mjlab.tasks.tracking.tracking_env_cfg import TerminationsCfg, RewardCfg, ObservationCfg
from mjlab.tasks.tracking.mdp import rewards as tracking_rewards
from mjlab.envs.mdp import rewards as base_rewards
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.entity import EntityCfg
import mujoco
from mjlab.tasks.tracking import mdp
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)



_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]


def base_ang_vel_exceed(
  env: ManagerBasedRlEnv,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.any(ang_vel.abs() > threshold, dim=-1)


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

  _, ori = subtract_frame_transforms(
    command.object_pos_w,
    command.object_quat_w,
    box.data.body_link_pos_w[:, 0],
    box.data.body_link_quat_w[:, 0],
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


@dataclass
class ObjectMotionTrackingObservationsCfg(ObservationCfg):
  """observation configuration for box pushing."""
  
  @dataclass
  class PolicyCfg(ObservationCfg.PolicyCfg):
    # Add object-specific observations to the policy observations
    object_pos_b: ObsTerm | None = term(
      ObsTerm, func=mdp.object_pos_b,
       history_length=5,
         params={"command_name": "motion"}
    )
    object_ori_b: ObsTerm | None = term(
      ObsTerm, func=mdp.object_ori_b,
       history_length=5,
         params={"command_name": "motion"}
    )
    
    object_pos_error: ObsTerm = term(
        ObsTerm,
        func=object_position_error,
        params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
        history_length=5,
        noise=Unoise(n_min=-0.05, n_max=0.05),
      )
    
    object_ori_error: ObsTerm = term(
        ObsTerm,
        func=object_orientation_error,
        params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
        history_length=5,
        noise=Unoise(n_min=-0.05, n_max=0.05),
      )

  @dataclass
  class PrivilegedCfg(ObservationCfg.PrivilegedCfg):
    # Add object-specific observations to the privileged observations
    object_pos_b: ObsTerm | None = term(
      ObsTerm, func=mdp.object_pos_b, params={"command_name": "motion"}
    )
    object_ori_b: ObsTerm | None = term(
      ObsTerm, func=mdp.object_ori_b, params={"command_name": "motion"}
    )
    
    object_pos_error: ObsTerm = term(
        ObsTerm,
        func=object_position_error,
        params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
        history_length=5,
      )
    
    object_ori_error: ObsTerm = term(
        ObsTerm,
        func=object_orientation_error,
        params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
        history_length=5,
      )

  # Override the default policy and critic configurations
  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)
  

@dataclass
class ObjectMotionTrackingRewardCfg(RewardCfg):
  """Reward configuration for box pushing."""
  # Object tracking rewards (enabled when object is present)
  object_global_pos: RewTerm | None = term(
    RewTerm,
    func=mdp.object_global_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "object_asset_cfg": SceneEntityCfg("box"), "std": 0.25},
  )
  object_global_ori: RewTerm | None = term(
    RewTerm,
    func=mdp.object_global_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "object_asset_cfg": SceneEntityCfg("box"), "std": 0.4},
  )

  bad_termination: RewTerm = term(
    RewTerm,
    func=mdp.is_terminated,
    weight=-100.0,
    # params={"termination_names": ["anchor_pos", "anchor_ori", "ee_body_pos"]},
  )
  
  
@dataclass
class MotionTrackingTerminationsCfg(TerminationsCfg):
  base_ang_vel_exceed: DoneTerm = term(
    DoneTerm,
    func=base_ang_vel_exceed,
    params={"threshold": _MAX_ANG_VEL},
  )

@dataclass
class G1MotionTrackingCfg(G1FlatNoStateEstimationEnvCfg):
  terminations: MotionTrackingTerminationsCfg = field(default_factory=MotionTrackingTerminationsCfg)

@dataclass
class G1ObjectMotionTrackingCfg(G1MotionTrackingCfg):
  rewards: ObjectMotionTrackingRewardCfg = field(default_factory=ObjectMotionTrackingRewardCfg)
  observations: ObjectMotionTrackingObservationsCfg = field(default_factory=ObjectMotionTrackingObservationsCfg)
  terminations: MotionTrackingTerminationsCfg = field(default_factory=MotionTrackingTerminationsCfg)

  def __post_init__(self):
    super().__post_init__()

    # Add largebox object as a floating body by including its MJCF
    def _make_largebox_spec() -> mujoco.MjSpec:
      from pathlib import Path
      # Resolve path to mjlab/src/mjlab/scene/largebox.xml relative to this file.
      # parents[4] points to the package directory that contains `scene/`.
      largebox_xml = "/home/hoan/KunTao_Workspace/SBMP/g1_spinkick_example/mjlab/src/mjlab/scene/largebox.xml"
      return mujoco.MjSpec.from_file(str(largebox_xml))

    self.scene.entities["box"] = EntityCfg(spec_fn=_make_largebox_spec)
  

@dataclass
class G1MotionTrackingCfg_PLAY(G1MotionTrackingCfg):
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
class G1ObjectMotionTrackingCfg_PLAY(G1ObjectMotionTrackingCfg):
  
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
class G1MotionTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  """Optimized PPO configuration for motion tracking tasks."""

  experiment_name: str = "g1_motion_tracking"
  save_interval: int = 50  # Save more frequently
  max_iterations: int = 50_000  # More iterations for complex task

@dataclass
class G1ObjectMotionTrackingPPORunnerCfg(G1MotionTrackingPPORunnerCfg):
  """Optimized PPO configuration for object motion tracking tasks."""

  experiment_name: str = "g1_object_motion_tracking"
