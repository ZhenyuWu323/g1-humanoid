from __future__ import annotations
import math

import torch

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate
from isaaclab.sensors import ContactSensor
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.utils.noise.noise_model import uniform_noise
from .g1_lowbody_cfg import G1LowBodyEnvCfg
from isaaclab.managers import SceneEntityCfg
from . import mdp


class G1LowBodyEnv(DirectRLEnv):
    cfg: G1LowBodyEnvCfg

    def __init__(self, cfg: G1LowBodyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # DOF and key body indexes
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body) # torso link
        self.upper_body_indexes = self.robot.find_joints(self.cfg.upper_body_names)[0] # arm and fingers
        self.feet_indexes = self.robot.find_joints(self.cfg.feet_names)[0]
        self.waist_indexes = self.robot.find_joints(self.cfg.waist_names)[0]
        self.hips_indexes = self.robot.find_joints(self.cfg.hips_names)[0]
        self.lower_body_indexes = self.waist_indexes + self.hips_indexes + self.feet_indexes # lower body

        # action offset and scale
        self.action_scale = self.cfg.action_scale
        self.default_joint_pos = self.robot.data.default_joint_pos[0]
        self.default_lower_joint_pos = self.default_joint_pos[self.lower_body_indexes]
        self.default_upper_joint_pos = self.default_joint_pos[self.upper_body_indexes]

        # noise models
        if self.cfg.obs_noise_models:
            self.obs_noise_models = {}
            for key, value in self.cfg.obs_noise_models.items():
                self.obs_noise_models[key] = value.class_type(value, self.num_envs, self.sim.device)


        # body velocity command 
        self.velocity_command = mdp.UniformVelocityCommand(self.cfg.base_velocity, self)

        # actions and previous actions (only lower body DOFs are in action space)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)

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
        self.cfg.sky_light_cfg.func("/World/Light", self.cfg.sky_light_cfg)


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

        # update previous actions
        self.prev_actions = self.actions.clone()

        # relative joint positions and velocities
        joint_pos_rel = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel_rel = self.robot.data.joint_vel - self.robot.data.default_joint_vel
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b

        # apply noise models
        if self.obs_noise_models:
            root_lin_vel_b = self.obs_noise_models["root_lin_vel_b"].apply(root_lin_vel_b)
            root_ang_vel_b = self.obs_noise_models["root_ang_vel_b"].apply(root_ang_vel_b)
            projected_gravity_b = self.obs_noise_models["projected_gravity_b"].apply(projected_gravity_b)
            joint_pos_rel = self.obs_noise_models["joint_pos_rel"].apply(joint_pos_rel)
            joint_vel_rel = self.obs_noise_models["joint_vel_rel"].apply(joint_vel_rel)

        # get command
        vel_command = self.velocity_command.command

        # build task observation
        obs = compute_obs(
            root_lin_vel_b,
            root_ang_vel_b,
            projected_gravity_b,
            joint_pos_rel,
            joint_vel_rel,
            vel_command,
            self.actions.clone()
        )

        observations = {"policy": obs} # NOTE: need to add 'critic' key for asymmetric policies
        return observations

    def _get_rewards(self) -> torch.Tensor:

        """
        Tracking Rewards
        """
        # linear velocity tracking
        lin_vel_xy_reward = mdp.track_lin_vel_xy_yaw_frame_exp(
            root_quat_w=self.robot.data.root_quat_w,
            root_lin_vel_w=self.robot.data.root_lin_vel_w,
            vel_command=self.velocity_command.command,
            std=0.5,
            weight=self.cfg.reward_scales["track_lin_vel_xy_exp"],
        )

        # angular velocity tracking
        ang_vel_z_reward = mdp.track_ang_vel_z_world_exp(
            root_ang_vel_w=self.robot.data.root_ang_vel_w,
            vel_command=self.velocity_command.command,
            std=0.5,
            weight=self.cfg.reward_scales["track_ang_vel_z_exp"],
        )

        """
        Panelty Term
        """
        # terminate when the robot falls
        died, _ = self._get_dones()
        die_penalty = mdp.termination_penalty(died, weight=self.cfg.reward_scales["termination_penalty"])

        # linear velocity z
        lin_vel_z_penalty = mdp.lin_vel_z_l2(
            root_lin_vel_b=self.robot.data.root_lin_vel_b,
            weight=self.cfg.reward_scales["lin_vel_z_l2"],
        )

        # flat orientation
        flat_orientation_penalty = mdp.flat_orientation_l2(
            projected_gravity_b=self.robot.data.projected_gravity_b,
            weight=self.cfg.reward_scales["flat_orientation_l2"],
        )

        # joint deviation waist
        joint_deviation_waist = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.waist_indexes,
            weight=self.cfg.reward_scales["joint_deviation_waist"],
        )

        # joint deviation upper body (arms and fingers)
        joint_deviation_upper_body = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.upper_body_indexes,
            weight=self.cfg.reward_scales["joint_deviation_upper_body"],
        )

        # joint deviation hips
        joint_deviation_hips = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.hips_indexes,
            weight=self.cfg.reward_scales["joint_deviation_hips"],
        )

        # joint position limits
        joint_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.lower_body_indexes + self.upper_body_indexes,
            weight=self.cfg.reward_scales["dof_pos_limits"],
        )

        # joint torques
        joint_torques_l2 = mdp.joint_torque_l2(
            joint_torque=self.robot.data.applied_torque,
            joint_idx=self.lower_body_indexes + self.upper_body_indexes,
            weight=self.cfg.reward_scales["dof_torques_l2"],
        )

        # joint accelerations
        joint_accelerations_l2 = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.lower_body_indexes + self.upper_body_indexes,
            weight=self.cfg.reward_scales["dof_acc_l2"],
        )

        # action rate
        action_rate = mdp.action_rate_l2(
            action=self.actions,
            prev_action=self.prev_actions,
            weight=self.cfg.reward_scales["action_rate_l2"],
        )

        """
        Feet Contact Rewards
        """
        # feet slides penalty
        feet_slide_penalty = mdp.feet_slide(
            env=self,
            weight=self.cfg.reward_scales["feet_slide"],
            sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            asset_cfg=SceneEntityCfg("robot", body_names=self.cfg.feet_names),
        )

        # reward
        reward = lin_vel_xy_reward + ang_vel_z_reward + die_penalty + lin_vel_z_penalty + flat_orientation_penalty + joint_deviation_waist + joint_deviation_upper_body + joint_deviation_hips + joint_pos_limits + joint_torques_l2 + joint_accelerations_l2 + action_rate + feet_slide_penalty
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # fall
        died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # reset robot
        self.robot.reset(env_ids)
        # reset command
        self.velocity_command._resample_command(env_ids)
        # reset actions
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        super()._reset_idx(env_ids)


@torch.jit.script
def compute_obs(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel_rel: torch.Tensor,
    vel_command: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    
    obs = torch.cat(
        (
            root_lin_vel_b,
            root_ang_vel_b,
            projected_gravity_b,
            joint_pos_rel,
            joint_vel_rel,
            vel_command,
            actions,
        ),
        dim=-1,
    )
    return obs

