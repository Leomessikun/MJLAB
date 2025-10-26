from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  axis_angle_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)


class OmniNpzMotionLoader:
  def __init__(
    self,
    motion_file: str,
    output_fps: int,
    device: torch.device | str,
  ):
    self.motion_file = motion_file
    self.output_fps = output_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads qpos (root[7]+joints[29]) from OmniRetarget .npz."""
    data = np.load(self.motion_file)
    if "qpos" not in data or "fps" not in data:
      raise ValueError(
        "OmniRetarget NPZ must contain 'qpos' and 'fps' keys: " + self.motion_file
      )
    qpos = torch.from_numpy(data["qpos"]).to(torch.float32).to(self.device)
    fps = float(data["fps"]) if not isinstance(data["fps"], np.ndarray) else float(data["fps"][()])

    if qpos.ndim != 2 or qpos.shape[1] < 36:
      raise ValueError(
        f"Expected qpos shape (T, >=36). Got {tuple(qpos.shape)} from {self.motion_file}"
      )

    # Parse root and joint positions. Ignore any extra dims (e.g., object DOFs).
    # OmniRetarget README states the floating base is stored as [qw, qx, qy, qz, x, y, z]
    # (quat first, then position). So slice accordingly.
    root_quat = qpos[:, 0:4]
    root_pos = qpos[:, 4:7]
    dof_pos = qpos[:, 7 : 7 + 29]

    # Optional object pose: present when D >= 43 with [qw, qx, qy, qz, x, y, z]
    self.has_object = qpos.shape[1] >= 43
    if self.has_object:
      object_quat = qpos[:, 36:40]
      object_pos = qpos[:, 40:43]
    else:
      object_quat = None
      object_pos = None

    self.input_fps = int(round(fps))
    self.input_dt = 1.0 / self.input_fps

    # Normalize quaternions to unit length to avoid downstream orientation/velocity errors.
    root_quat_norm = torch.norm(root_quat, dim=-1, keepdim=True).clamp_min(1e-8)
    self.motion_base_rots_input = root_quat / root_quat_norm
    self.motion_base_poss_input = root_pos
    self.motion_dof_poss_input = dof_pos

    if self.has_object:
      obj_quat_norm = torch.norm(object_quat, dim=-1, keepdim=True).clamp_min(1e-8)
      self.object_rots_input = object_quat / obj_quat_norm  # type: ignore[operator]
      self.object_pos_input = object_pos  # type: ignore[assignment]

    self.input_frames = qpos.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Resample motion to output fps via linear (pos) and slerp (quat) interpolation."""
    if self.input_fps == self.output_fps:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if getattr(self, "has_object", False):
        self.object_poss = self.object_pos_input  # type: ignore[attr-defined]
        self.object_rots = self.object_rots_input  # type: ignore[attr-defined]
      self.output_frames = self.input_frames
      return

    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    # Simple normalized lerp for quats to avoid dependency on slerp here.
    q0 = self.motion_base_rots_input[index_0]
    q1 = self.motion_base_rots_input[index_1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)  # ensure shortest path
    q = q0 * (1 - blend.unsqueeze(1)) + q1 * blend.unsqueeze(1)
    self.motion_base_rots = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )

    # Object interpolation if present
    if getattr(self, "has_object", False):
      # Positions linear interp
      self.object_poss = self._lerp(  # type: ignore[attr-defined]
        self.object_pos_input[index_0],  # type: ignore[attr-defined]
        self.object_pos_input[index_1],  # type: ignore[attr-defined]
        blend.unsqueeze(1),
      )
      # Quats: normalized lerp
      oq0 = self.object_rots_input[index_0]  # type: ignore[attr-defined]
      oq1 = self.object_rots_input[index_1]  # type: ignore[attr-defined]
      odot = (oq0 * oq1).sum(-1, keepdim=True)
      oq1 = torch.where(odot < 0, -oq1, oq1)
      oq = oq0 * (1 - blend.unsqueeze(1)) + oq1 * blend.unsqueeze(1)
      self.object_rots = oq / torch.norm(oq, dim=-1, keepdim=True).clamp_min(1e-8)  # type: ignore[attr-defined]

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    # Approximate angular velocity via quaternion finite difference.
    # omega ≈ 2 * axis_angle(q_{t+1} * conj(q_{t-1})) / (2*dt)
    q = self.motion_base_rots
    q_prev, q_next = q[:-2], q[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    self.motion_base_ang_vels = omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= (self.output_frames if hasattr(self, "output_frames") else self.input_frames):
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def convert(
  input_file: str,
  output_file: str,
  output_fps: float = 50.0,
  device: str = "cuda:0",
):
  """Convert OmniRetarget NPZ (qpos/fps) to mjlab motion.npz format."""
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(G1FlatEnvCfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  motion = OmniNpzMotionLoader(
    motion_file=input_file, output_fps=int(round(output_fps)), device=sim.device
  )

  robot: Entity = scene["robot"]
  box: Entity | None = scene.entities.get("box") if hasattr(scene, "entities") else None

  log: dict[str, Any] = {
    "fps": [int(round(output_fps))],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  if getattr(motion, "has_object", False):
    log["object_pos_w"] = []
    log["object_quat_w"] = []

  frames_total = getattr(motion, "output_frames", motion.input_frames)
  pbar = tqdm(total=frames_total, desc="Converting", unit="frame", ncols=100)

  scene.reset()
  file_saved = False
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = quat_apply_inverse(motion_base_rot, motion_base_ang_vel)
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    # Directly assign in order; OmniRetarget is already 29-DoF ordered like G1 here.
    joint_pos[:, :] = motion_dof_pos
    joint_vel[:, :] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # If object present, place the box freejoint pose.
    # NOTE: motion.get_next_state() increments motion.current_idx and may reset it to 0
    # when the sequence completes; avoid slicing with negative indices which yields
    # empty tensors (shape [0,7]) by computing the effective frame index here.
    if getattr(motion, "has_object", False) and box is not None and not box.data.is_fixed_base:
      # Determine the correct frame index we just consumed from motion.get_next_state().
      if reset_flag:
        frame_idx = (frames_total - 1) if frames_total > 0 else 0
      else:
        frame_idx = motion.current_idx - 1

      # Safely slice using frame_idx:frame_idx+1 to produce a (1, 3) and (1, 4) tensors.
      pos_src = getattr(motion, "object_poss", motion.object_pos_input)
      rot_src = getattr(motion, "object_rots", motion.object_rots_input)
      obj_pos_slice = pos_src[frame_idx : frame_idx + 1]
      obj_rot_slice = rot_src[frame_idx : frame_idx + 1]
      obj_pose = torch.cat([obj_pos_slice, obj_rot_slice], dim=-1)
      box.write_root_link_pose_to_sim(obj_pose)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(
      robot.data.body_link_quat_w[0, :].cpu().numpy().copy()
    )
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

    if "object_pos_w" in log and box is not None:
      # Read box world pose from sim
      # xipos/xquat are world frame positions for bodies; for root use root body id
      log["object_pos_w"].append(box.data.body_link_pos_w[0, 0].cpu().numpy().copy())
      log["object_quat_w"].append(box.data.body_link_quat_w[0, 0].cpu().numpy().copy())

    pbar.update(1)
    if reset_flag:
      file_saved = True

  pbar.close()
  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)
  if "object_pos_w" in log:
    log["object_pos_w"] = np.stack(log["object_pos_w"], axis=0)
    log["object_quat_w"] = np.stack(log["object_quat_w"], axis=0)

  np.savez(output_file, **log)  # type: ignore[arg-type]
  print(f"Saved mjlab motion to: {output_file}")


def main(
  input_file: str,
  output_file: str,
  output_fps: float = 50.0,
  device: str = "cuda:0",
):
  convert(
    input_file=input_file, output_file=output_file, output_fps=output_fps, device=device
  )


if __name__ == "__main__":
  tyro.cli(main)
