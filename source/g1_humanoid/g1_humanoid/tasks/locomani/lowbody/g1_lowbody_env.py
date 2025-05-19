from __future__ import annotations

import torch

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate
from isaaclab.sensors import ContactSensor

from .g1_lowbody_cfg import G1LowBodyEnvCfg


class G1LowBodyEnv(DirectRLEnv):
    cfg: G1LowBodyEnvCfg

    def __init__(self, cfg: G1LowBodyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        '''# action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits'''

        # DOF and key body indexes
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body) # torso
        self.upper_body_indexes = self.robot.find_joints(self.cfg.upper_body_names)[0] # arm and fingers
        self.feet_indexes = self.robot.find_joints(self.cfg.feet_names)[0]
        self.waist_indexes = self.robot.find_joints(self.cfg.waist_names)[0]
        self.hips_indexes = self.robot.find_joints(self.cfg.hips_names)[0]
        self.lower_body_indexes = self.waist_indexes + self.hips_indexes + self.feet_indexes # lower body

        # action offset and scale
        self.action_scale = self.cfg.action_scale
        self.default_joint_pos = self.robot.data.default_joint_pos[0]
        self.default_lower_joint_pos = self.default_joint_pos[:,self.lower_body_indexes]
        self.default_upper_joint_pos = self.default_joint_pos[:,self.upper_body_indexes]

    def _setup_scene(self):
        # robot
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # number of envs
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add terrain
        self.scene.terrains["terrain"] = self._terrain

        # add lights
        light_cfg = self.cfg.sky_light
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        lower_body_target = self.default_lower_joint_pos + self.action_scale * self.actions
        upper_body_target = self.default_upper_joint_pos + 0
        # set lower body
        self.robot.set_joint_position_target(lower_body_target, self.lower_body_indexes)
        # set upper body
        self.robot.set_joint_position_target(upper_body_target, self.upper_body_indexes)

    def _get_observations(self) -> dict:
        # relative joint positions and velocities
        joint_pos_rel = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel_rel = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        # build task observation
        obs = compute_obs(
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            joint_pos_rel,
            joint_vel_rel,
            self.actions.clone()
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # fall
        died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)


@torch.jit.script
def compute_obs(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel_rel: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    
    obs = torch.cat(
        (
            root_lin_vel_b,
            root_ang_vel_b,
            projected_gravity_b,
            joint_pos_rel,
            joint_vel_rel,
            actions,
        ),
        dim=-1,
    )
    return obs

