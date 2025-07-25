from __future__ import annotations
import math
from typing import List

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
from .g1_decoupled_cfg import G1DecoupledEnvCfg
from isaaclab.managers import SceneEntityCfg
from . import mdp
from .utils import compute_projected_gravity
from isaaclab.envs.common import VecEnvStepReturn

class G1DecoupledEnv(DirectRLEnv):
    cfg: G1DecoupledEnvCfg

    def __init__(self, cfg: G1DecoupledEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)

        # gait phase
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.leg_phases = torch.zeros(self.num_envs, len(self.feet_body_indexes), device=self.device)

        # history
        self.obs_history_length = getattr(self.cfg, 'obs_history_length', 5)  # t-4:t (5 steps)
        self._init_history_buffers()

    def _init_history_buffers(self):
        """Initialize history buffers for observations and actions."""
        
        # proprioceptive observations
        self.root_ang_vel_b_history = torch.zeros((self.num_envs, self.obs_history_length, 3), device=self.sim.device)
        self.projected_gravity_b_history = torch.zeros((self.num_envs, self.obs_history_length, 3), device=self.sim.device)
        self.dof_pos_history = torch.zeros((self.num_envs, self.obs_history_length, self.cfg.action_space), device=self.sim.device)
        self.dof_vel_history = torch.zeros((self.num_envs, self.obs_history_length, self.cfg.action_space), device=self.sim.device)
        
        # action history buffer
        self.action_history = torch.zeros((self.num_envs, self.obs_history_length, self.cfg.action_space), device=self.sim.device)
        
        # step counter for history management
        self.history_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)

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
        # update previous actions
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self):
        upper_actions = self.actions[:, :self.cfg.action_dim["upper_body"]]
        lower_actions = self.actions[:, self.cfg.action_dim["upper_body"]:]

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

        # update history buffers
        self._update_history_buffers()
        self.history_step_counter += 1

    def _update_history_buffers(self):
        """Update history buffers for observations and actions."""
        dof_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        dof_vel = self.robot.data.joint_vel
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b

        # update history buffers
        self.root_ang_vel_b_history = torch.roll(self.root_ang_vel_b_history, shifts=1, dims=1)
        self.root_ang_vel_b_history[:, 0, :] = root_ang_vel_b
        
        self.projected_gravity_b_history = torch.roll(self.projected_gravity_b_history, shifts=1, dims=1)
        self.projected_gravity_b_history[:, 0, :] = projected_gravity_b
        
        self.dof_pos_history = torch.roll(self.dof_pos_history, shifts=1, dims=1)
        self.dof_pos_history[:, 0, :] = dof_pos
        
        self.dof_vel_history = torch.roll(self.dof_vel_history, shifts=1, dims=1)
        self.dof_vel_history[:, 0, :] = dof_vel

        mask = self.history_step_counter > 0
        if mask.any():
            self.action_history = torch.roll(self.action_history, shifts=1, dims=1)
            self.action_history[mask, 0, :] = self.prev_actions[mask]


    def _get_observations(self) -> dict:

        # get proprioceptive observations
        dof_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        dof_vel = self.robot.data.joint_vel
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b

        # get history observations
        root_ang_vel_b_hist_flat = self.root_ang_vel_b_history.flatten(start_dim=1)  # (num_envs, history_length * 3)
        projected_gravity_b_hist_flat = self.projected_gravity_b_history.flatten(start_dim=1)  # (num_envs, history_length * 3)
        dof_pos_hist_flat = self.dof_pos_history.flatten(start_dim=1)  # (num_envs, history_length * total_dof)
        dof_vel_hist_flat = self.dof_vel_history.flatten(start_dim=1)  # (num_envs, history_length * total_dof)
        action_hist_flat = self.action_history.flatten(start_dim=1)  # (num_envs, action_history_length * action_dim)


        # plate velocity in world frame
        plate_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.plate_body_index, :]
        plate_lin_acc_w = self.robot.data.body_lin_acc_w[:, self.plate_body_index, :]

        # plate angular velocity in world frame
        plate_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.plate_body_index, :]
        plate_ang_acc_w = self.robot.data.body_ang_acc_w[:, self.plate_body_index, :]

        # plate projected gravity
        plate_projected_gravity_b = compute_projected_gravity(self.plate_body_index, self.robot.data.body_quat_w, self.robot.data.GRAVITY_VEC_W)

        # apply noise models
        '''if self.obs_noise_models:
            root_lin_vel_b = self.obs_noise_models["root_lin_vel_b"].apply(root_lin_vel_b)
            root_ang_vel_b = self.obs_noise_models["root_ang_vel_b"].apply(root_ang_vel_b)
            projected_gravity_b = self.obs_noise_models["projected_gravity_b"].apply(projected_gravity_b)
            joint_pos_rel = self.obs_noise_models["joint_pos_rel"].apply(joint_pos_rel)
            joint_vel_rel = self.obs_noise_models["joint_vel_rel"].apply(joint_vel_rel)'''

        # get command
        vel_command = self.velocity_command.command

        # phase
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)


        # scale observations
        actor_observations_dict = {
            'root_ang_vel_b': root_ang_vel_b_hist_flat, # 15
            'projected_gravity_b': projected_gravity_b_hist_flat, # 15
            'vel_command': vel_command, # 3
            'ref_upper_body_dof_pos': self.default_upper_joint_pos, # 14
            'dof_pos': dof_pos_hist_flat, # 145
            'dof_vel': dof_vel_hist_flat, # 145
            'actions': action_hist_flat, # 145
            'sin_phase': sin_phase, # 1
            'cos_phase': cos_phase, # 1
        }
        critic_observations_dict = {
            'root_lin_vel_b': root_lin_vel_b,
            'root_ang_vel_b': root_ang_vel_b,
            'projected_gravity_b': projected_gravity_b,
            'vel_command': vel_command,
            'ref_upper_body_dof_pos': self.default_upper_joint_pos,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'actions': self.actions.clone(),
            'sin_phase': sin_phase,
            'cos_phase': cos_phase,
        }
        actor_scaled_obs = {}
        critic_scaled_obs = {}
        for obs_name, obs_value in actor_observations_dict.items():
            if hasattr(self.cfg.obs_scales, obs_name):
                scale = getattr(self.cfg.obs_scales, obs_name)
                actor_scaled_obs[obs_name] = obs_value * scale
            else:
                actor_scaled_obs[obs_name] = obs_value
        for obs_name, obs_value in critic_observations_dict.items():
            if hasattr(self.cfg.obs_scales, obs_name):
                scale = getattr(self.cfg.obs_scales, obs_name)
                critic_scaled_obs[obs_name] = obs_value * scale
            else:
                critic_scaled_obs[obs_name] = obs_value
        actor_obs_list = list(actor_scaled_obs.values())
        critic_obs_list = list(critic_scaled_obs.values())

        # build task observation
        actor_obs = compute_obs(actor_obs_list)
        critic_obs = compute_obs(critic_obs_list)

        observations = {"actor_obs": actor_obs, "critic_obs": critic_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        """
        Lower Body Tracking Rewards
        """
        # linear velocity tracking
        tracking_lin_vel_x = mdp.track_lin_vel_x_base_exp(
            root_lin_vel_b=self.robot.data.root_lin_vel_b,
            vel_command=self.velocity_command.command,
            std=0.5,
            weight=self.cfg.reward_scales["track_lin_vel_x"] if "track_lin_vel_x" in self.cfg.reward_scales else 0,
        )
        tracking_lin_vel_y = mdp.track_lin_vel_y_base_exp(
            root_lin_vel_b=self.robot.data.root_lin_vel_b,
            vel_command=self.velocity_command.command,
            std=0.5,
            weight=self.cfg.reward_scales["track_lin_vel_y"] if "track_lin_vel_y" in self.cfg.reward_scales else 0,
        )

        # angular velocity tracking
        tracking_ang_vel_z = mdp.track_ang_vel_z_base_exp(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            vel_command=self.velocity_command.command,
            std=0.5,
            weight=self.cfg.reward_scales["track_ang_vel_z"] if "track_ang_vel_z" in self.cfg.reward_scales else 0,
        )

        """
        Lower Body Penalty Terms
        """
        # terminate when the robot falls
        died, _ = self._get_dones()
        die_penalty = mdp.termination_penalty(died, weight=self.cfg.reward_scales["termination_penalty"] if "termination_penalty" in self.cfg.reward_scales else 0)

        # linear velocity z
        penalty_lin_vel_z = mdp.lin_vel_z_l2(
            root_lin_vel_b=self.robot.data.root_lin_vel_b,
            weight=self.cfg.reward_scales["penalty_lin_vel_z"] if "penalty_lin_vel_z" in self.cfg.reward_scales else 0,
        )

        # angular velocity xy
        penalty_ang_vel_xy = mdp.ang_vel_xy_l2(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            weight=self.cfg.reward_scales["penalty_ang_vel_xy"] if "penalty_ang_vel_xy" in self.cfg.reward_scales else 0,
        )

        # flat orientation
        penalty_flat_orientation = mdp.flat_orientation_l2(
            projected_gravity_b=self.robot.data.projected_gravity_b,
            weight=self.cfg.reward_scales["penalty_flat_orientation"] if "penalty_flat_orientation" in self.cfg.reward_scales else 0,
        )

        # joint deviation waist
        penalty_dof_pos_waist = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.waist_indexes,
            weight=self.cfg.reward_scales["penalty_dof_pos_waist"] if "penalty_dof_pos_waist" in self.cfg.reward_scales else 0,
        )

        # joint deviation hips
        penalty_dof_pos_hips = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.hips_yaw_roll_indexes,
            weight=self.cfg.reward_scales["penalty_dof_pos_hips"] if "penalty_dof_pos_hips" in self.cfg.reward_scales else 0,
        )

        # joint position limits
        penalty_lower_body_dof_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.lower_body_indexes,
            weight=self.cfg.reward_scales["penalty_lower_body_dof_pos_limits"] if "penalty_lower_body_dof_pos_limits" in self.cfg.reward_scales else 0,
        )

        # joint torques
        penalty_lower_body_dof_torques = mdp.joint_torque_l2(
            joint_torque=self.robot.data.applied_torque,
            joint_idx=self.hips_indexes,
            weight=self.cfg.reward_scales["penalty_lower_body_dof_torques"] if "penalty_lower_body_dof_torques" in self.cfg.reward_scales else 0,
        )

        # joint accelerations
        penalty_lower_body_dof_acc = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.hips_indexes,
            weight=self.cfg.reward_scales["penalty_lower_body_dof_acc"] if "penalty_lower_body_dof_acc" in self.cfg.reward_scales else 0,
        )

        # joint velocities
        penalty_lower_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.hips_indexes,
            weight=self.cfg.reward_scales["penalty_lower_body_dof_vel"] if "penalty_lower_body_dof_vel" in self.cfg.reward_scales else 0,
        )

        # action rate
        penalty_lower_body_action_rate = mdp.action_rate_l2(
            action=self.actions[:, self.cfg.action_dim["upper_body"]:],
            prev_action=self.prev_actions[:, self.cfg.action_dim["upper_body"]:],
            weight=self.cfg.reward_scales["penalty_lower_body_action_rate"] if "penalty_lower_body_action_rate" in self.cfg.reward_scales else 0,
        )

        # base height
        penalty_base_height = mdp.base_height(
            body_pos_w=self.robot.data.body_pos_w,
            body_idx=self.ref_body_index,
            target_height=self.cfg.target_base_height,
            weight=self.cfg.reward_scales["penalty_base_height"] if "penalty_base_height" in self.cfg.reward_scales else 0,
        )

        # stand still
        penalty_stand_still = mdp.stand_still(
            joint_pos=self.robot.data.joint_pos,
            joint_idx=self.lower_body_indexes,
            default_joint_pos=self.robot.data.default_joint_pos,
            vel_command=self.velocity_command.command,
            weight=self.cfg.reward_scales["penalty_stand_still"] if "penalty_stand_still" in self.cfg.reward_scales else 0,
        )

        """
        Lower Body Feet Contact Rewards
        """
        # feet slides penalty
        penalty_feet_slide = mdp.feet_slide(
            body_lin_vel_w=self.robot.data.body_lin_vel_w,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            weight=self.cfg.reward_scales["feet_slide"] if "feet_slide" in self.cfg.reward_scales else 0,
        )

        # feet air time
        feet_air_time = mdp.feet_air_time_positive_biped(
            vel_command=self.velocity_command.command,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            threshold=0.5,
            weight=self.cfg.reward_scales["feet_air_time"] if "feet_air_time" in self.cfg.reward_scales else 0,
        )

        # feet swing height
        penalty_feet_swing_height = mdp.feet_swing_height(
            body_pos_w=self.robot.data.body_pos_w,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            weight=self.cfg.reward_scales["feet_swing_height"] if "feet_swing_height" in self.cfg.reward_scales else 0,
            target_height=0.1,
        )

        # gait phase reward
        gait_phase_reward = mdp.gait_phase_reward(
            env=self,
            contact_sensor=self._contact_sensor,
            leg_phases=self.leg_phases,
            feet_body_indexes=self.feet_body_indexes,
            stance_phase_threshold=self.cfg.stance_phase_threshold,
            weight=self.cfg.reward_scales["gait_phase_reward"] if "gait_phase_reward" in self.cfg.reward_scales else 0,
        )

        """
        Upper Body Rewards
        """
        # plate flat orientation
        plate_projected_gravity_b = compute_projected_gravity(self.plate_body_index, self.robot.data.body_quat_w, self.robot.data.GRAVITY_VEC_W)
        plate_flat_orientation_penalty = mdp.flat_orientation_l2(
            projected_gravity_b=plate_projected_gravity_b,
            weight=self.cfg.reward_scales["plate_flat_orientation_l2"] if "plate_flat_orientation_l2" in self.cfg.reward_scales else 0,
        )
        
        # upper body tracking
        tracking_upper_body_dof_pos = mdp.joint_tracking_exp(
            joint_pos=self.robot.data.joint_pos,
            joint_idx=self.upper_body_indexes,
            joint_pos_command=self.default_upper_joint_pos,
            weight=self.cfg.reward_scales["tracking_upper_body_dof_pos"] if "tracking_upper_body_dof_pos" in self.cfg.reward_scales else 0,
            std=0.5,
        )

        """
        Upper Body Penalty Terms
        """
        # upper body torques
        penalty_upper_body_dof_torques = mdp.joint_torque_l2(
            joint_torque=self.robot.data.applied_torque,
            joint_idx=self.upper_body_indexes,
            weight=self.cfg.reward_scales["penalty_upper_body_dof_torques"] if "penalty_upper_body_dof_torques" in self.cfg.reward_scales else 0,
        )

        # upper body accelerations
        penalty_upper_body_dof_acc = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.upper_body_indexes,
            weight=self.cfg.reward_scales["penalty_upper_body_dof_acc"] if "penalty_upper_body_dof_acc" in self.cfg.reward_scales else 0,
        )

        # upper body position limits
        penalty_upper_body_dof_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.upper_body_indexes,
            weight=self.cfg.reward_scales["penalty_upper_body_dof_pos_limits"] if "penalty_upper_body_dof_pos_limits" in self.cfg.reward_scales else 0,
        )

        # upper body action rate
        penalty_upper_body_action_rate = mdp.action_rate_l2(
            action=self.actions[:, :self.cfg.action_dim["upper_body"]],
            prev_action=self.prev_actions[:, :self.cfg.action_dim["upper_body"]],
            weight=self.cfg.reward_scales["penalty_upper_body_action_rate"] if "penalty_upper_body_action_rate" in self.cfg.reward_scales else 0,
        )

        # upper body velocities
        penalty_upper_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.upper_body_indexes,
            weight=self.cfg.reward_scales["penalty_upper_body_dof_vel"] if "penalty_upper_body_dof_vel" in self.cfg.reward_scales else 0,
        )

        # upper body termination
        penalty_upper_body_termination = mdp.termination_penalty(
            terminated=died,
            weight=self.cfg.reward_scales["penalty_upper_body_termination"] if "penalty_upper_body_termination" in self.cfg.reward_scales else 0,
        )


		# locomotion reward
        locomotion_reward = (tracking_lin_vel_x + 
                             tracking_lin_vel_y + 
                             tracking_ang_vel_z + 
                             die_penalty + 
                             penalty_lin_vel_z + 
                             penalty_ang_vel_xy + 
                             penalty_flat_orientation + 
                             penalty_dof_pos_waist + 
                             penalty_dof_pos_hips + 
                             penalty_lower_body_dof_pos_limits + 
                             penalty_lower_body_dof_torques + 
                             penalty_lower_body_dof_acc + 
                             penalty_lower_body_dof_vel + 
                             penalty_lower_body_action_rate + 
                             penalty_stand_still + 
                             penalty_feet_slide + 
                             feet_air_time + 
                             penalty_feet_swing_height + 
                             gait_phase_reward + 
                             penalty_base_height)
        
		# upper body reward
        upper_body_reward = (
            tracking_upper_body_dof_pos + 
            penalty_upper_body_dof_torques + 
            penalty_upper_body_dof_acc + 
            penalty_upper_body_dof_pos_limits + 
            penalty_upper_body_action_rate + 
            penalty_upper_body_dof_vel + 
            penalty_upper_body_termination
        )

        # reward 
        lower_body_reward = locomotion_reward #NOTE: removed * self.step_dt
        upper_body_reward = upper_body_reward 
        return {'upper_body': upper_body_reward, 'lower_body': lower_body_reward}

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
        self.episode_length_buf[env_ids] = 0
        self.history_step_counter[env_ids] = 0
        self.root_ang_vel_b_history[env_ids] = 0.0
        self.projected_gravity_b_history[env_ids] = 0.0
        self.dof_pos_history[env_ids] = 0.0
        self.dof_vel_history[env_ids] = 0.0
        self.action_history[env_ids] = 0.0
        self.phase[env_ids] = 0.0
        self.leg_phases[env_ids] = 0.0
    
        super()._reset_idx(env_ids)


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
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
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # clip actions
        clip_actions = self.cfg.clip_action
        action = torch.clip(action, -clip_actions, clip_actions)

        # process actions
        self._pre_physics_step(action)

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

