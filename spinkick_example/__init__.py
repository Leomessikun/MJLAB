import gymnasium as gym

gym.register(
  id="Mjlab-MotionTracking-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1MotionTrackingCfg",
    "rl_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1MotionTrackingPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-MotionTracking-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1MotionTrackingCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1MotionTrackingPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-ObjectMotionTracking-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1ObjectMotionTrackingCfg",
    "rl_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1ObjectMotionTrackingPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-ObjectMotionTracking-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1ObjectMotionTrackingCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.spinkick_env_cfg:G1ObjectMotionTrackingPPORunnerCfg",
  },
)