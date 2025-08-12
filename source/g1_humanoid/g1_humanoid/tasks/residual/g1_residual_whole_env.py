from __future__ import annotations
import math
from typing import List

import torch

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.utils.noise.noise_model import uniform_noise
from .g1_residual_whole_cfg import G1ResidualWholeBodyEnvCfg
from isaaclab.managers import SceneEntityCfg
from . import mdp
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply_inverse
from .utils import compute_dof_pos_tracking_weight

class G1ResidualWholeBodyEnv(DirectRLEnv):
    cfg: G1ResidualWholeBodyEnvCfg

    def __init__(self, cfg: G1ResidualWholeBodyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        ##########################################################################################
        # DOF and key body indexes
        ##########################################################################################
        
        # body keys
        self.body_keys = self.cfg.body_keys

        # joint indexes
        self.upper_body_indexes = self.robot.find_joints(self.cfg.upper_body_names)[0] # arm and fingers
        self.feet_indexes = self.robot.find_joints(self.cfg.feet_names)[0]
        self.waist_indexes = self.robot.find_joints(self.cfg.waist_names)[0]
        self.hips_yaw_roll_indexes = self.robot.find_joints(self.cfg.hips_names[:2])[0]
        self.knee_indexes = self.robot.find_joints(self.cfg.hips_names[-1])[0]
        self.hips_indexes = self.robot.find_joints(self.cfg.hips_names)[0]
        self.lower_body_indexes = self.waist_indexes + self.hips_indexes + self.feet_indexes # lower body

        # plate body index
        self.plate_body_index = self.robot.data.body_names.index(self.cfg.plate_name)


        # body/link indexes
        self.feet_body_indexes = self.robot.find_bodies(self.cfg.feet_body_name)[0]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body) # torso link

        # action scale
        self.action_scale = self.cfg.action_scale

        # default joint positions
        self.default_joint_pos = self.robot.data.default_joint_pos
        self.default_lower_joint_pos = self.default_joint_pos[:,self.lower_body_indexes]
        self.default_upper_joint_pos = self.default_joint_pos[:,self.upper_body_indexes]

        

        # noise models
        if self.cfg.obs_noise_models:
            self.obs_noise_models = {}
            for key, value in self.cfg.obs_noise_models.items():
                self.obs_noise_models[key] = value.class_type(value, self.num_envs, self.sim.device)


        # body velocity command 
        self.velocity_command = mdp.UniformVelocityCommand(self.cfg.base_velocity, self)

        # actions and previous actions
        self.base_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_base_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.residual_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_residual_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)

        # gait phase
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.leg_phases = torch.zeros(self.num_envs, len(self.feet_body_indexes), device=self.device)

        # object/plate relative position
        self.object_plate_rel_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # history
        self.obs_history_length = getattr(self.cfg, 'obs_history_length', 5)  # t-4:t (5 steps)
        self._init_history_buffers()

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tracking_lin_vel_xy",
                "tracking_ang_vel_z",
                "gait_phase_reward",
                "feet_clearance_reward",
                "penalty_object_pos_deviation",
                "penalty_object_flat_orientation",
                "object_upright_bonus",
                "tracking_upper_body_dof_pos",
            ]
        }

    def _init_history_buffers(self):
        """Initialize history buffers for observations and actions."""
        
        # proprioceptive observations
        self.root_lin_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.root_ang_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.projected_gravity_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.dof_pos_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.dof_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.base_action_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.residual_action_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        # plate observations
        self.plate_projected_gravity_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.plate_lin_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.plate_ang_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        
        # object observations
        self.object_projected_gravity_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.object_rel_pos_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.object_lin_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self.object_ang_vel_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self._buffers_initialized = False


    def _initialize_buffers_with_current_state(self):
        # proprioceptive observations
        dof_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        dof_vel = self.robot.data.joint_vel
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        # plate observations
        plate_quat_w = self.robot.data.body_link_quat_w[:, self.plate_body_index, :]
        projected_gravity_plate = quat_apply_inverse(plate_quat_w, self.robot.data.GRAVITY_VEC_W).to(self.sim.device)
        plate_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.plate_body_index, :].to(self.sim.device)
        plate_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.plate_body_index, :].to(self.sim.device)
        # object observations
        object_rel_pos = self._object.data.body_pos_w[:, 0, :] - self.robot.data.body_pos_w[:, self.plate_body_index, :]
        object_projected_gravity = quat_apply_inverse(self._object.data.body_link_quat_w[:, 0, :], self.robot.data.GRAVITY_VEC_W).to(self.sim.device)
        object_lin_vel_w = self._object.data.body_lin_vel_w[:, 0, :].to(self.sim.device)
        object_ang_vel_w = self._object.data.body_ang_vel_w[:, 0, :].to(self.sim.device)
        
        # fill the history length
        for _ in range(self.obs_history_length):
            # proprioceptive observations
            self.root_lin_vel_buffer.append(root_lin_vel_b)
            self.root_ang_vel_buffer.append(root_ang_vel_b)
            self.projected_gravity_buffer.append(projected_gravity_b)
            self.dof_pos_buffer.append(dof_pos)
            self.dof_vel_buffer.append(dof_vel)
            self.base_action_buffer.append(self.base_actions)
            self.residual_action_buffer.append(self.residual_actions)
            # plate observations
            self.plate_projected_gravity_buffer.append(projected_gravity_plate)
            self.plate_lin_vel_buffer.append(plate_lin_vel_w)
            self.plate_ang_vel_buffer.append(plate_ang_vel_w)
            # object observations
            self.object_projected_gravity_buffer.append(object_projected_gravity)
            self.object_rel_pos_buffer.append(object_rel_pos)
            self.object_lin_vel_buffer.append(object_lin_vel_w)
            self.object_ang_vel_buffer.append(object_ang_vel_w)

    def _setup_scene(self):
        # robot
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # height scanner
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # number of envs
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.scene._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # object
        self._object = RigidObject(self.cfg.obj_cfg)
        self.scene.rigid_objects["object"] = self._object
        

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.cfg.sky_light_cfg.func("/World/Light", self.cfg.sky_light_cfg)


    def _pre_physics_step(self, base_action: torch.Tensor, residual_action: torch.Tensor):
        # update previous actions
        self.prev_base_actions = self.base_actions.clone()
        self.prev_residual_actions = self.residual_actions.clone()
        self.base_actions = base_action.clone()
        self.residual_actions = residual_action.clone()

    def _apply_action(self):
        actions = self.base_actions + self.residual_actions
        upper_actions = actions[:, :self.cfg.action_dim["upper_body"]]
        lower_actions = actions[:, self.cfg.action_dim["upper_body"]:]

        upper_body_target = self.default_upper_joint_pos + self.action_scale * upper_actions
        lower_body_target = self.default_lower_joint_pos + self.action_scale * lower_actions

        # set upper body
        self.robot.set_joint_position_target(upper_body_target, self.upper_body_indexes)
        # set lower body
        self.robot.set_joint_position_target(lower_body_target, self.lower_body_indexes)


    def _post_physics_step(self):
        # update gait phase
        current_time = self.episode_length_buf * self.step_dt
        self.phase = (current_time % self.cfg.gait_period) / self.cfg.gait_period
        self.leg_phases = torch.zeros(self.num_envs, len(self.feet_body_indexes), device=self.device)
        self.leg_phases[:, 0] = self.phase # left leg
        self.leg_phases[:, 1] = (self.phase + self.cfg.phase_offset) % 1.0 # right leg


    def _update_history_buffers(self):
        """Update history buffers for observations and actions."""
        dof_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        dof_vel = self.robot.data.joint_vel
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b

        # update history buffers
        self.root_lin_vel_buffer.append(root_lin_vel_b)
        self.root_ang_vel_buffer.append(root_ang_vel_b)
        self.projected_gravity_buffer.append(projected_gravity_b)
        self.dof_pos_buffer.append(dof_pos)
        self.dof_vel_buffer.append(dof_vel)
        self.base_action_buffer.append(self.base_actions)
        self.residual_action_buffer.append(self.residual_actions)
        # plate observations
        plate_quat_w = self.robot.data.body_link_quat_w[:, self.plate_body_index, :]
        projected_gravity_plate = quat_apply_inverse(plate_quat_w, self.robot.data.GRAVITY_VEC_W).to(self.sim.device)
        plate_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.plate_body_index, :].to(self.sim.device)
        plate_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.plate_body_index, :].to(self.sim.device)
        self.plate_projected_gravity_buffer.append(projected_gravity_plate)
        self.plate_lin_vel_buffer.append(plate_lin_vel_w)
        self.plate_ang_vel_buffer.append(plate_ang_vel_w)
        # object observations
        object_rel_pos = self._object.data.body_pos_w[:, 0, :] - self.robot.data.body_pos_w[:, self.plate_body_index, :]
        object_projected_gravity = quat_apply_inverse(self._object.data.body_link_quat_w[:, 0, :], self.robot.data.GRAVITY_VEC_W).to(self.sim.device)
        object_lin_vel_w = self._object.data.body_lin_vel_w[:, 0, :].to(self.sim.device)
        object_ang_vel_w = self._object.data.body_ang_vel_w[:, 0, :].to(self.sim.device)
        self.object_projected_gravity_buffer.append(object_projected_gravity)
        self.object_rel_pos_buffer.append(object_rel_pos)
        self.object_lin_vel_buffer.append(object_lin_vel_w)
        self.object_ang_vel_buffer.append(object_ang_vel_w)

    
    def _scale_observations(self, observations_dict: dict) -> dict:
        scaled_observations_dict = {}
        for obs_name, obs_value in observations_dict.items():
            if hasattr(self.cfg.obs_scales, obs_name):
                scale = getattr(self.cfg.obs_scales, obs_name)
                scaled_observations_dict[obs_name] = obs_value * scale
            else:
                scaled_observations_dict[obs_name] = obs_value
        return list(scaled_observations_dict.values())
    
    
    
    def _get_observations(self) -> dict:

        if not hasattr(self, '_buffers_initialized') or not self._buffers_initialized:
            self._initialize_buffers_with_current_state()
            self._buffers_initialized = True

        # update history buffers
        self._update_history_buffers()

        # get history observations
        lin_vel_buffer_flat = self.root_lin_vel_buffer.buffer.reshape(self.num_envs, -1)
        ang_vel_buffer_flat = self.root_ang_vel_buffer.buffer.reshape(self.num_envs, -1)
        projected_gravity_buffer_flat = self.projected_gravity_buffer.buffer.reshape(self.num_envs, -1)
        dof_pos_buffer_flat = self.dof_pos_buffer.buffer.reshape(self.num_envs, -1)
        dof_vel_buffer_flat = self.dof_vel_buffer.buffer.reshape(self.num_envs, -1)
        base_action_buffer_flat = self.base_action_buffer.buffer.reshape(self.num_envs, -1)
        residual_action_buffer_flat = self.residual_action_buffer.buffer.reshape(self.num_envs, -1)

        # plate observations
        plate_projected_gravity_buffer_flat = self.plate_projected_gravity_buffer.buffer.reshape(self.num_envs, -1)
        plate_lin_vel_buffer_flat = self.plate_lin_vel_buffer.buffer.reshape(self.num_envs, -1)
        plate_ang_vel_buffer_flat = self.plate_ang_vel_buffer.buffer.reshape(self.num_envs, -1)

        # object observations
        object_projected_gravity_buffer_flat = self.object_projected_gravity_buffer.buffer.reshape(self.num_envs, -1)
        object_rel_pos_buffer_flat = self.object_rel_pos_buffer.buffer.reshape(self.num_envs, -1)
        object_lin_vel_buffer_flat = self.object_lin_vel_buffer.buffer.reshape(self.num_envs, -1)
        object_ang_vel_buffer_flat = self.object_ang_vel_buffer.buffer.reshape(self.num_envs, -1)

        # get command
        vel_command = self.velocity_command.command

        # phase
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)


        # scale observations
        actor_observations_dict = {
            'root_ang_vel_b': ang_vel_buffer_flat, # 15
            'projected_gravity_b': projected_gravity_buffer_flat, # 15
            'vel_command': vel_command, # 3
            'ref_upper_body_dof_pos': self.default_upper_joint_pos, # 14
            'dof_pos': dof_pos_buffer_flat, # 145
            'dof_vel': dof_vel_buffer_flat, # 145
            'actions': base_action_buffer_flat, # 145
        }
        critic_observations_dict = {
            'root_lin_vel_b': lin_vel_buffer_flat,
            'root_ang_vel_b': ang_vel_buffer_flat,
            'projected_gravity_b': projected_gravity_buffer_flat,
            'vel_command': vel_command,
            'ref_upper_body_dof_pos': self.default_upper_joint_pos,
            'dof_pos': dof_pos_buffer_flat,
            'dof_vel': dof_vel_buffer_flat,
            'actions': base_action_buffer_flat,
        }

        residual_actor_observations_dict = {
            'root_lin_vel_b': lin_vel_buffer_flat,
            'root_ang_vel_b': ang_vel_buffer_flat,
            'projected_gravity_b': projected_gravity_buffer_flat,
            'vel_command': vel_command,
            'ref_upper_body_dof_pos': self.default_upper_joint_pos,
            'dof_pos': dof_pos_buffer_flat,
            'dof_vel': dof_vel_buffer_flat,
            'base_actions': base_action_buffer_flat,
            'residual_actions': residual_action_buffer_flat,
            'projected_gravity_plate': plate_projected_gravity_buffer_flat,
            'plate_lin_vel_w': plate_lin_vel_buffer_flat,
            'plate_ang_vel_w': plate_ang_vel_buffer_flat,
            'object_projected_gravity': object_projected_gravity_buffer_flat,
            'object_rel_pos': object_rel_pos_buffer_flat,
            'object_lin_vel_w': object_lin_vel_buffer_flat,
            'object_ang_vel_w': object_ang_vel_buffer_flat,
        }
        residual_critic_observations_dict = {
            'root_lin_vel_b': lin_vel_buffer_flat,
            'root_ang_vel_b': ang_vel_buffer_flat,
            'projected_gravity_b': projected_gravity_buffer_flat,
            'vel_command': vel_command,
            'ref_upper_body_dof_pos': self.default_upper_joint_pos,
            'dof_pos': dof_pos_buffer_flat,
            'dof_vel': dof_vel_buffer_flat,
            'base_actions': base_action_buffer_flat,
            'residual_actions': residual_action_buffer_flat,
            'projected_gravity_plate': plate_projected_gravity_buffer_flat,
            'plate_lin_vel_w': plate_lin_vel_buffer_flat,
            'plate_ang_vel_w': plate_ang_vel_buffer_flat,
            'object_projected_gravity': object_projected_gravity_buffer_flat,
            'object_rel_pos': object_rel_pos_buffer_flat,
            'object_lin_vel_w': object_lin_vel_buffer_flat,
            'object_ang_vel_w': object_ang_vel_buffer_flat,
        }

        actor_scaled_obs = self._scale_observations(actor_observations_dict)
        critic_scaled_obs = self._scale_observations(critic_observations_dict)
        residual_actor_scaled_obs = self._scale_observations(residual_actor_observations_dict)
        residual_critic_scaled_obs = self._scale_observations(residual_critic_observations_dict)

        actor_obs = compute_obs(actor_scaled_obs)
        critic_obs = compute_obs(critic_scaled_obs)
        residual_actor_obs = compute_obs(residual_actor_scaled_obs)
        residual_critic_obs = compute_obs(residual_critic_scaled_obs)

        observations = {
            "actor_obs": actor_obs, 
            "critic_obs": critic_obs, 
            "residual_actor_obs": residual_actor_obs, 
            "residual_critic_obs": residual_critic_obs}
        return observations

    def _get_rewards(self) -> dict:

        """
        Lower Body Tracking Rewards
        """
        
        tracking_lin_vel_xy = mdp.track_lin_vel_xy_yaw_frame_exp(
            root_quat_w=self.robot.data.root_quat_w,
            root_lin_vel_w=self.robot.data.root_lin_vel_w,
            vel_command=self.velocity_command.command,
            sigma=0.25,
            weight=1.0,
        )
        tracking_ang_vel_z = mdp.track_ang_vel_z_base_exp(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            vel_command=self.velocity_command.command,
            sigma=0.25,
            weight=0.5,
        )


        """
        Lower Body Penalty Terms
        """
        # terminate when the robot falls
        died, _ = self._get_dones()
      

        # linear velocity z
        penalty_lin_vel_z = mdp.lin_vel_z_l2(
            root_lin_vel_b=self.robot.data.root_lin_vel_b,
            weight=-2.0,
        )

        # angular velocity xy
        penalty_ang_vel_xy = mdp.ang_vel_xy_l2(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            weight=-0.05,
        )

        # flat orientation
        penalty_flat_orientation = mdp.flat_orientation_l2(
            projected_gravity_b=self.robot.data.projected_gravity_b,
            weight=-5.0,
        )

        # joint deviation waist
        penalty_dof_pos_waist = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.waist_indexes,
            weight=-1.0,
        )

        # joint deviation hips
        penalty_dof_pos_hips = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.hips_yaw_roll_indexes,
            weight=-1.0,
        )

        # joint position limits
        penalty_lower_body_dof_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.lower_body_indexes,
            weight=-5.0,
        )


        # joint accelerations
        penalty_lower_body_dof_acc = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.lower_body_indexes,
            weight=-2.5e-7,
        )

        # joint velocities
        penalty_lower_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.lower_body_indexes,
            weight=-0.001,
        )

        # action rate
        penalty_lower_body_action_rate = mdp.action_rate_l2(
            action=self.residual_actions[:, self.cfg.action_dim["upper_body"]:],
            prev_action=self.prev_residual_actions[:, self.cfg.action_dim["upper_body"]:],
            weight=-0.05,
        )

        # base height
        penalty_base_height = mdp.base_height(
            body_pos_w=self.robot.data.body_pos_w,
            body_idx=self.ref_body_index,
            height_scanner=self._height_scanner,
            target_height=self.cfg.target_base_height,
            weight=-10,
        )

        """
        Lower Body Feet Contact Rewards
        """
        # feet slides penalty
        penalty_feet_slide = mdp.feet_slide(
            body_lin_vel_w=self.robot.data.body_lin_vel_w,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            weight=-0.2,
        )


        # feet gait
        feet_gait_reward = mdp.feet_gait(
            env=self,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            period=0.8,
            offset=[0.0, 0.5],
            threshold=0.55,
            command=self.velocity_command.command,
            weight=0.5,
        )

        # feet clearance
        feet_clearance_reward = mdp.feet_clearance(
            body_pos_w=self.robot.data.body_pos_w,
            body_lin_vel_w=self.robot.data.body_lin_vel_w,
            feet_body_indexes=self.feet_body_indexes,
            target_feet_height=self.cfg.target_feet_height,
            sigma=0.05,
            tanh_mult=2.0,
            weight=1.0,
        )

        """
        Upper Body Rewards
        """
        # upper body tracking
        tracking_upper_body_dof_pos = mdp.joint_tracking_exp(
            joint_pos=self.robot.data.joint_pos,
            joint_idx=self.upper_body_indexes,
            joint_pos_command=self.default_upper_joint_pos,
            weight=compute_dof_pos_tracking_weight(self._object.data.body_link_quat_w[:, 0, :], self.robot.data.GRAVITY_VEC_W),
            sigma=0.1,
        )

        """
        Upper Body Penalty Terms
        """
        # upper body torques
        penalty_upper_body_dof_torques = mdp.joint_torque_l2(
            joint_torque=self.robot.data.applied_torque,
            joint_idx=self.upper_body_indexes,
            weight=0.0,
        )

        # upper body accelerations
        penalty_upper_body_dof_acc = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.upper_body_indexes,
            weight=-2.5e-7,
        )

        # upper body position limits
        penalty_upper_body_dof_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.upper_body_indexes,
            weight=-5.0,
        )

        # upper body action rate
        penalty_upper_body_action_rate = mdp.action_rate_l2(
            action=self.residual_actions[:, :self.cfg.action_dim["upper_body"]],
            prev_action=self.prev_residual_actions[:, :self.cfg.action_dim["upper_body"]],
            weight=-0.05,
        )

        # upper body velocities
        penalty_upper_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.upper_body_indexes,
            weight=-0.001,
        )

        """
        Upper Body Object Rewards
        """
        # penalty object position deviation
        penalty_object_pos_deviation = mdp.object_pos_deviation(
            object_pos_w=self._object.data.body_pos_w[:, 0, :],
            plate_pos_w=self.robot.data.body_pos_w[:, self.plate_body_index, :],
            default_rel_pos_w=self.object_plate_rel_pos,
            weight=-0.01,
        )
        
        # object flat orientation
        penalty_object_flat_orientation = mdp.body_orientation_l2(
            body_rot_w=self._object.data.body_link_quat_w,
            gravity_vec_w=self.robot.data.GRAVITY_VEC_W,
            body_idx=0,
            weight=-0.5 ,
        )
        # object upright bonus
        object_upright_bonus = mdp.cup_upright_bonus_exp(
            body_rot_w=self._object.data.body_link_quat_w,
            gravity_vec_w=self.robot.data.GRAVITY_VEC_W,
            body_idx=0,
            weight=1.0,
            sigma=0.1,
        )

        # alive reward
        alive_reward = mdp.alive_reward(terminated=died, weight=0.15)
        # locomotion reward
        residual_lower_body_reward = (tracking_lin_vel_xy + 
                             tracking_ang_vel_z + 
                             penalty_lin_vel_z + 
                             penalty_ang_vel_xy + 
                             penalty_flat_orientation + 
                             penalty_dof_pos_waist + 
                             penalty_dof_pos_hips + 
                             penalty_lower_body_dof_pos_limits + 
                             penalty_lower_body_dof_acc + 
                             penalty_lower_body_dof_vel + 
                             penalty_lower_body_action_rate + 
                             penalty_feet_slide + 
                             penalty_base_height +
                             feet_gait_reward +
                             feet_clearance_reward)
        
        # upper body reward
        residual_upper_body_reward = (
            tracking_upper_body_dof_pos + 
            penalty_upper_body_dof_torques + 
            penalty_upper_body_dof_acc + 
            penalty_upper_body_dof_pos_limits + 
            penalty_upper_body_action_rate + 
            penalty_upper_body_dof_vel + 
            penalty_object_pos_deviation +
            penalty_object_flat_orientation +
            object_upright_bonus
        )

        residual_whole_body_reward = residual_upper_body_reward + residual_lower_body_reward + alive_reward

        self._episode_sums["tracking_lin_vel_xy"] += tracking_lin_vel_xy
        self._episode_sums["tracking_ang_vel_z"] += tracking_ang_vel_z
        self._episode_sums["gait_phase_reward"] += feet_gait_reward
        self._episode_sums["feet_clearance_reward"] += feet_clearance_reward
        self._episode_sums["penalty_object_pos_deviation"] += penalty_object_pos_deviation
        self._episode_sums["penalty_object_flat_orientation"] += penalty_object_flat_orientation
        self._episode_sums["object_upright_bonus"] += object_upright_bonus
        self._episode_sums["tracking_upper_body_dof_pos"] += tracking_upper_body_dof_pos
        # reward 
        residual_whole_body_reward = residual_whole_body_reward * self.step_dt
        lower_body_reward = torch.zeros(self.num_envs, device=self.device)
        upper_body_reward = torch.zeros(self.num_envs, device=self.device)
        return {'upper_body': upper_body_reward, 
                'lower_body': lower_body_reward, 
                'residual_whole_body': residual_whole_body_reward}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # fall
        died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        extras = dict()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # apply terrain curriculum
        if self.cfg.terrain_generator_cfg.curriculum:
            avg_terrain_level = mdp.terrain_levels(env=self, env_ids=env_ids, vel_command=self.velocity_command.command)
            extras["Curriculum/terrain_level"] = avg_terrain_level.item()

        # reset robot
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # reset command
        self.velocity_command._resample_command(env_ids)
        # reset proprioceptive observations
        self.base_actions[env_ids] = 0.0
        self.prev_base_actions[env_ids] = 0.0
        self.residual_actions[env_ids] = 0.0
        self.prev_residual_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.root_ang_vel_buffer.reset(env_ids)
        self.root_lin_vel_buffer.reset(env_ids)
        self.projected_gravity_buffer.reset(env_ids)
        self.dof_pos_buffer.reset(env_ids)
        self.dof_vel_buffer.reset(env_ids)
        self.base_action_buffer.reset(env_ids)
        self.residual_action_buffer.reset(env_ids)
        self.phase[env_ids] = 0.0
        self.leg_phases[env_ids] = 0.0

        # reset plate observations
        self.plate_projected_gravity_buffer.reset(env_ids)
        self.plate_lin_vel_buffer.reset(env_ids)
        self.plate_ang_vel_buffer.reset(env_ids)
        # reset object observations
        self.object_projected_gravity_buffer.reset(env_ids)
        self.object_rel_pos_buffer.reset(env_ids)
        self.object_lin_vel_buffer.reset(env_ids)
        self.object_ang_vel_buffer.reset(env_ids)

        # reset object
        plate_pos_w = self.robot.data.body_pos_w[env_ids, self.plate_body_index, :].clone()
        plate_pos_w[:, 2] += 0.1 # offset the object to the top of the plate
        object_quat_w = self._object.data.default_root_state[env_ids, 3:7].clone()
        self._object.write_root_pose_to_sim(torch.cat([plate_pos_w, object_quat_w], dim=-1), env_ids=env_ids)
        self.object_plate_rel_pos[env_ids] = self._object.data.body_pos_w[env_ids, 0, :] - plate_pos_w
        object_vel = torch.zeros(env_ids.shape[0], 6, device=self.device)
        self._object.write_root_velocity_to_sim(object_vel, env_ids=env_ids)

        # reset logging
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)


    def step(self, action: dict) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        base_action = action['base_action'].to(self.device)
        residual_action = action['residual_action'].to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            base_action = self._action_noise_model.apply(base_action)
            residual_action = self._action_noise_model.apply(residual_action)

        # clip actions
        clip_actions = self.cfg.clip_action
        base_action = torch.clip(base_action, -clip_actions, clip_actions)
        residual_action = torch.clip(residual_action, -clip_actions, clip_actions)

        # process actions
        self._pre_physics_step(base_action, residual_action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update gait phase
        self._post_physics_step()

        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        #if self.cfg.observation_noise_model:
            #self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # clip observations
        clip_observations = self.cfg.clip_observation
        for key, value in self.obs_buf.items():
            self.obs_buf[key] = torch.clip(value, -clip_observations, clip_observations)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


@torch.jit.script
def compute_obs(obs_tensors: List[torch.Tensor]) -> torch.Tensor:
    
    return torch.cat(obs_tensors, dim=-1)

