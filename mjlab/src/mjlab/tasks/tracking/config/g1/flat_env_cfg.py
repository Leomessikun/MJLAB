from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg
from mjlab.entity import EntityCfg
import mujoco


@dataclass
class G1FlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    self_collision_sensor = ContactSensorCfg(
      name="self_collision",
      subtree1="pelvis",
      subtree2="pelvis",
      data=("found",),
      reduce="netforce",
      num=10,  # Report up to 10 contacts.
    )
    g1_cfg = replace(G1_ROBOT_CFG, sensors=(self_collision_sensor,))

    # Add robot
    self.scene.entities = {"robot": g1_cfg}

    # Add largebox object as a floating body by including its MJCF
    def _make_largebox_spec() -> mujoco.MjSpec:
      from pathlib import Path
      # Resolve path to mjlab/src/mjlab/scene/largebox.xml relative to this file.
      # parents[4] points to the package directory that contains `scene/`.
      largebox_xml = Path(__file__).parents[4] / "scene" / "largebox.xml"
      return mujoco.MjSpec.from_file(str(largebox_xml))

    self.scene.entities["box"] = EntityCfg(spec_fn=_make_largebox_spec)

    # Add box object for locomanipulation tasks (sub3_largebox_000_original)
    # Updated from mesh bounds (meters)
    # half_extents = (
    #   0.235577,
    #   0.229365,
    #   0.203948,
    # )
    # object_center = (
    #   0.001494,
    #   -0.000715,
    #   0.005756,
    # )

    # def _make_box_spec() -> mujoco.MjSpec:
    #   spec = mujoco.MjSpec()
    #   body = spec.worldbody.add_body(name="box_body")
    #   # Place the body at the object's world center; geom at local origin
    #   body.pos = object_center
    #   # Keep box static (no joints) - it should not move or rotate
    #   body.add_geom(
    #     name="box_collision",
    #     type=mujoco.mjtGeom.mjGEOM_BOX,
    #     size=half_extents,
    #     pos=(0.0, 0.0, 0.0),
    #   )
    #   return spec

    # Add the box as a floating entity that can be kinematically driven
    #box_cfg = EntityCfg(spec_fn=_make_box_spec, replicate_physics=True)
    #self.scene.entities["box"] = box_cfg
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    self.commands.motion.anchor_body_name = "torso_link"
    self.commands.motion.body_names = [
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ]

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot[1-7]_collision$"
    ]
    self.events.base_com.params["asset_cfg"].body_names = "torso_link"

    self.terminations.ee_body_pos.params["body_names"] = [
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ]

    self.viewer.body_name = "torso_link"


@dataclass
class G1FlatNoStateEstimationEnvCfg(G1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.motion_anchor_pos_b = None
    self.observations.policy.base_lin_vel = None


@dataclass
class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class G1FlatNoStateEstimationEnvCfg_PLAY(G1FlatNoStateEstimationEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Disable termination conditions for full trajectory playback
    self.terminations.anchor_pos = None
    self.terminations.anchor_ori = None
    self.terminations.ee_body_pos = None

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)