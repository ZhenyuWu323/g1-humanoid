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
from .joint_test_cfg import G1JointTestEnvCfg
from isaaclab.managers import SceneEntityCfg
from . import mdp
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.managers import CommandManager
from isaaclab.utils.string import resolve_matching_names

class G1JointTestEnv(DirectRLEnv):
    cfg: G1JointTestEnvCfg

    def __init__(self, cfg: G1JointTestEnvCfg, render_mode: str | None = None, **kwargs):
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

        # sdk joint sequence
        joint_ids_map, _ = resolve_matching_names(self.robot.joint_names, self.cfg.sdk_joint_sequence, preserve_order=True)
        print("[INFO] Joint to motor index: ", joint_ids_map)

        print("[INFO] upper body default joint positions: ", self.default_upper_joint_pos)

        print("[INFO] lower body default joint positions: ", self.default_lower_joint_pos)

        print("[INFO] whole body default joint positions: ", self.default_joint_pos)

        

        

        # noise models
        if self.cfg.obs_noise_models:
            self.obs_noise_models = {}
            for key, value in self.cfg.obs_noise_models.items():
                self.obs_noise_models[key] = value.class_type(value, self.num_envs, self.sim.device)


        # body velocity command 
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # actions and previous actions
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)

        # gait phase
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.leg_phases = torch.zeros(self.num_envs, len(self.feet_body_indexes), device=self.device)


        # history
        self.obs_history_length = getattr(self.cfg, 'obs_history_length', 5)  # t-4:t (5 steps)
        self._init_history_buffers()

        # observation noise models
        if self.cfg.obs_noise_models:
            self.obs_noise_models = {}
            for key, value in self.cfg.obs_noise_models.items():
                self.obs_noise_models[key] = value.class_type(value, self.num_envs, self.sim.device)

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tracking_lin_vel_xy",
                "tracking_ang_vel_z",
                "gait_phase_reward",
                "feet_clearance_reward",
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
        self.action_buffer = CircularBuffer(max_len=self.obs_history_length, batch_size=self.num_envs, device=self.sim.device)
        self._buffers_initialized = False


    def _initialize_buffers_with_current_state(self):
        dof_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        dof_vel = self.robot.data.joint_vel
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        projected_gravity_b = self.robot.data.projected_gravity_b
        
            
        # fill the history length
        for _ in range(self.obs_history_length):
            self.root_lin_vel_buffer.append(root_lin_vel_b)
            self.root_ang_vel_buffer.append(root_ang_vel_b)
            self.projected_gravity_buffer.append(projected_gravity_b)
            self.dof_pos_buffer.append(dof_pos)
            self.dof_vel_buffer.append(dof_vel)
            self.action_buffer.append(self.actions)
           

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
        self.action_buffer.append(self.actions)


    def _apply_observation_noise(self, observations_dict: dict) -> dict:
        noisy_observations_dict = {}
        for obs_name, obs_value in observations_dict.items():
            if obs_name in self.cfg.obs_noise_models:
                noisy_observations_dict[obs_name] = self.obs_noise_models[obs_name].apply(obs_value)
            else:
                noisy_observations_dict[obs_name] = obs_value
        return noisy_observations_dict
        

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
        action_buffer_flat = self.action_buffer.buffer.reshape(self.num_envs, -1)

        # get command
        vel_command = self.command_manager.get_command("base_velocity")

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
            'actions': action_buffer_flat, # 145
            
        }
        critic_observations_dict = {
            'root_lin_vel_b': lin_vel_buffer_flat,
            'root_ang_vel_b': ang_vel_buffer_flat,
            'projected_gravity_b': projected_gravity_buffer_flat,
            'vel_command': vel_command,
            'ref_upper_body_dof_pos': self.default_upper_joint_pos,
            'dof_pos': dof_pos_buffer_flat,
            'dof_vel': dof_vel_buffer_flat,
            'actions': action_buffer_flat,
            
        }
        # scale observations
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

        # apply observation noise
        actor_noisy_obs = self._apply_observation_noise(actor_scaled_obs)

        # compute observations
        actor_obs_list = list(actor_noisy_obs.values())
        critic_obs_list = list(critic_scaled_obs.values())

        # build task observation
        actor_obs = compute_obs(actor_obs_list)
        real_robot_obs = torch.tensor([0.00000000e+00,0.00000000e+00,-8.72664561e-04,8.72664561e-04
,8.72664561e-04,0.00000000e+00,1.30899681e-03,0.00000000e+00
,0.00000000e+00,8.72664561e-04,4.36332281e-04,-4.36332281e-04
,4.36332281e-04,1.74532912e-03,-8.72664561e-04,5.78239672e-02
,-2.94654649e-02,-9.97891843e-01,5.78207187e-02,-2.94813327e-02
,-9.97891843e-01,5.78279458e-02,-2.95037273e-02,-9.97890770e-01
,5.77910095e-02,-2.95223724e-02,-9.97892082e-01,5.77707849e-02
,-2.95147169e-02,-9.97893751e-01,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00
,0.00000000e+00,0.00000000e+00,0.00000000e+00,-2.50733763e-01
,-1.47187620e-01,-9.10648901e-04,-2.60776728e-02,-8.53915978e-03
,0.00000000e+00,-2.82568997e-03,-8.03513758e-05,0.00000000e+00
,1.24878585e-01,1.44368649e-01,2.59698153e-01,-2.72820890e-01
,6.65278137e-02,-9.44629461e-02,1.17313579e-01,-6.64669096e-01
,3.91286239e-02,-4.69853021e-02,-7.45059252e-02,3.10870800e-02
,2.94991702e-01,-2.79639900e-01,3.75453800e-01,-9.50828433e-01
,5.66851974e-01,-5.52325130e-01,1.94209650e-01,-2.43642181e-01
,-2.50733763e-01,-1.47187620e-01,-9.24040796e-04,-2.60776728e-02
,-8.54768138e-03,0.00000000e+00,-2.82568997e-03,-1.20527060e-04
,0.00000000e+00,1.24878585e-01,1.44377172e-01,2.59674191e-01
,-2.72820890e-01,6.65321946e-02,-9.44587737e-02,1.17313579e-01
,-6.64693058e-01,3.91215682e-02,-4.69780862e-02,-7.44939446e-02
,3.11350171e-02,2.94991702e-01,-2.79639900e-01,3.75381887e-01
,-9.50852394e-01,5.66859603e-01,-5.52332819e-01,1.94209650e-01
,-2.43642181e-01,-2.50747144e-01,-1.47187620e-01,-9.24040796e-04
,-2.60691512e-02,-8.54768138e-03,0.00000000e+00,-2.83908192e-03
,-1.20527060e-04,0.00000000e+00,1.24870062e-01,1.44368649e-01
,2.59686172e-01,-2.72820890e-01,6.65234476e-02,-9.44546610e-02
,1.17313579e-01,-6.64669096e-01,3.91356684e-02,-4.69853245e-02
,-7.45059252e-02,3.10870800e-02,2.95003682e-01,-2.79639900e-01
,3.75405848e-01,-9.50840414e-01,5.66859603e-01,-5.52317500e-01
,1.94201976e-01,-2.43634507e-01,-2.50760525e-01,-1.47214383e-01
,-8.83865112e-04,-2.60776728e-02,-8.53915978e-03,0.00000000e+00
,-2.83908192e-03,-9.37432706e-05,0.00000000e+00,1.24878585e-01
,1.44360125e-01,2.59686172e-01,-2.72820890e-01,6.65234476e-02
,-9.44629461e-02,1.17313579e-01,-6.64657116e-01,3.91356684e-02
,-4.69853021e-02,-7.44220391e-02,3.11110485e-02,2.95003682e-01
,-2.79639900e-01,3.75393867e-01,-9.50840414e-01,5.66859603e-01
,-5.52332819e-01,1.94209650e-01,-2.43611500e-01,-2.50747144e-01
,-1.47187620e-01,-8.97257007e-04,-2.60691512e-02,-8.54768138e-03
,0.00000000e+00,-2.82568997e-03,-1.20527060e-04,0.00000000e+00
,1.24887109e-01,1.44385695e-01,2.59686172e-01,-2.72844851e-01
,6.65234476e-02,-9.44671184e-02,1.17313579e-01,-6.64681077e-01
,3.91356684e-02,-4.69925180e-02,-7.44939446e-02,3.10870800e-02
,2.95003682e-01,-2.79639900e-01,3.75417829e-01,-9.50816453e-01
,5.66859603e-01,-5.52317500e-01,1.94209650e-01,-2.43634507e-01
,8.57081250e-05,5.14248793e-04,5.99956955e-04,1.09083077e-04
,-1.63624631e-04,0.00000000e+00,5.99956955e-04,1.97128719e-03
,0.00000000e+00,-1.79987086e-03,-1.63624631e-04,4.60194249e-04
,-2.30097125e-04,8.07123142e-05,7.80390255e-05,-5.36893262e-04
,-9.20388498e-04,1.33323410e-04,-2.31575017e-04,-3.83495208e-04
,-2.30097125e-04,-7.66990415e-05,0.00000000e+00,-6.13592332e-04
,0.00000000e+00,-5.39961213e-04,0.00000000e+00,2.45436939e-04
,-7.36310787e-04,8.57081250e-05,3.42832500e-04,-5.14248793e-04
,1.03628926e-03,2.18166155e-04,0.00000000e+00,-3.42832500e-04
,-2.14270339e-03,0.00000000e+00,-3.81790771e-04,1.63624631e-04
,-3.83495208e-04,9.20388498e-04,2.51713820e-04,3.45817243e-04
,0.00000000e+00,2.30097125e-04,-4.05902654e-04,2.30411402e-04
,6.13592332e-04,-3.06796166e-04,7.66990415e-04,1.53398083e-04
,-7.66990415e-05,0.00000000e+00,0.00000000e+00,-6.38136000e-04
,4.41786513e-04,2.94524332e-04,-2.57124397e-04,5.14248793e-04
,-7.71373219e-04,9.81747755e-04,3.27249261e-04,0.00000000e+00
,-1.71416250e-04,-1.71416250e-04,0.00000000e+00,-1.52716308e-03
,7.63581542e-04,1.53398083e-04,0.00000000e+00,-2.69041047e-05
,1.57225266e-04,-3.06796166e-04,3.83495208e-04,-4.44411380e-05
,-2.78064545e-04,-1.53398083e-04,-5.36893262e-04,7.66990415e-05
,0.00000000e+00,-2.30097125e-04,-1.53398083e-04,0.00000000e+00
,9.81747653e-05,-1.47262166e-04,5.39961213e-04,-5.99956955e-04
,-1.71416250e-04,1.71416250e-04,-5.45415387e-04,3.27249261e-04
,0.00000000e+00,-5.99956955e-04,9.42789484e-04,0.00000000e+00
,-1.25445554e-03,-2.72707694e-04,-1.53398083e-04,-3.06796166e-04
,-2.47759144e-05,4.22135432e-04,0.00000000e+00,-9.20388498e-04
,-2.23524010e-04,-2.78791791e-04,0.00000000e+00,4.60194249e-04
,4.60194249e-04,0.00000000e+00,-7.66990415e-04,-4.60194249e-04
,-1.47262166e-04,-5.39961213e-04,2.94524332e-04,4.90873877e-04
,-8.57081381e-04,8.57081250e-05,7.71373219e-04,5.45415387e-04
,1.09083077e-04,0.00000000e+00,2.57124397e-04,8.57081381e-04
,0.00000000e+00,-5.45415387e-04,1.14537240e-03,-1.53398083e-04
,2.30097125e-04,-1.61424628e-04,-1.88591977e-04,1.53398083e-04
,7.66990415e-05,-2.66646821e-04,-5.08475991e-04,5.36893262e-04
,-1.53398083e-04,-2.30097125e-04,-3.06796166e-04,0.00000000e+00
,0.00000000e+00,4.90873826e-05,-2.45436939e-04,1.96349531e-04
,-1.07992243e-03,1.10942614e+00,-1.40530920e+00,6.34809852e-01
,-2.88750339e+00,-2.67866731e-01,-3.81216407e-04,1.12625134e+00
,-1.26006484e+00,1.53712559e+00,-3.79819989e+00,2.27399039e+00
,-2.22818303e+00,8.02367866e-01,-9.64832246e-01,-6.39850460e-03
,9.17331278e-02,1.67601779e-02,-8.58673334e-01,-4.17536378e-01
,2.98389971e-01,-1.58839300e-01,-1.18424647e-01,1.06865391e-01
,3.56848300e-01,4.35102522e-01,1.15972126e+00,-2.68467963e-02
,-1.43005431e-01,-2.79927440e-02,1.10945106e+00,-1.40541470e+00
,6.34858489e-01,-2.88735676e+00,-2.67766476e-01,-2.90051103e-04
,1.12631297e+00,-1.26012290e+00,1.53702271e+00,-3.79827762e+00
,2.27397799e+00,-2.22812176e+00,8.02397072e-01,-9.64770854e-01
,-6.39907829e-03,9.17294472e-02,1.67166516e-02,-8.58802021e-01
,-4.17451203e-01,2.98477471e-01,-1.58718690e-01,-1.18527465e-01
,1.06767833e-01,3.56735259e-01,4.35340405e-01,1.15934503e+00
,-2.72609293e-02,-1.43031523e-01,-2.79393084e-02,1.10946381e+00
,-1.40540099e+00,6.34816706e-01,-2.88715458e+00,-2.67716676e-01
,-3.62291932e-04,1.12637627e+00,-1.26014042e+00,1.53705323e+00
,-3.79837751e+00,2.27400589e+00,-2.22809196e+00,8.02375793e-01
,-9.64603245e-01,-6.36622868e-03,9.18224901e-02,1.67326853e-02
,-8.58846903e-01,-4.17451262e-01,2.98556089e-01,-1.58635765e-01
,-1.18517689e-01,1.06787331e-01,3.56687039e-01,4.35305357e-01
,1.15913939e+00,-2.73883343e-02,-1.43096402e-01,-2.78789662e-02
,1.10946441e+00,-1.40545273e+00,6.34795308e-01,-2.88700342e+00
,-2.67596453e-01,-3.47867608e-04,1.12638688e+00,-1.26017344e+00
,1.53706896e+00,-3.79844999e+00,2.27401233e+00,-2.22807908e+00
,8.02341163e-01,-9.64531720e-01,-6.33736886e-03,9.18165743e-02
,1.67507827e-02,-8.58989835e-01,-4.17551458e-01,2.98425019e-01
,-1.58657119e-01,-1.18518732e-01,1.06758766e-01,3.56561184e-01
,4.35294181e-01,1.15843213e+00,-2.76846290e-02,-1.43160194e-01
,-2.79376544e-02,1.10938036e+00,-1.40540028e+00,6.34774566e-01
,-2.88697052e+00,-2.67684430e-01,-3.70278955e-04,1.12643898e+00
,-1.26028728e+00,1.53711104e+00,-3.79833555e+00,2.27404118e+00
,-2.22807860e+00,8.02404761e-01,-9.64507878e-01,-6.38886355e-03
,9.17696804e-02,1.67526379e-02,-8.59034121e-01,-4.17738080e-01
,2.98423767e-01,-1.58750921e-01,-1.18494801e-01,1.06802464e-01
,3.56473982e-01,4.35318649e-01,1.15753829e+00,-2.81538665e-02
,-1.43164426e-01,-2.80194767e-02]).unsqueeze(0).to(self.sim.device)
        #actor_obs[:, :15] = real_robot_obs[:, :15] # ang vel
        # actor_obs[:, 15:30] = real_robot_obs[:, 15:30] # projected gravity
        # actor_obs[:, 30:33] = real_robot_obs[:, 30:33] # vel command
        # actor_obs[:, 33:47] = real_robot_obs[:, 33:47] # upper body joint pos
        # actor_obs[:, 47:192] = real_robot_obs[:, 47:192] # dof pos
        # actor_obs[:, 192:337] = real_robot_obs[:, 192:337] # dof vel
        # actor_obs[:, 337:482] = real_robot_obs[:, 337:482] # action

        critic_obs = compute_obs(critic_obs_list)

        observations = {"upper_body_actor_obs": actor_obs, "upper_body_critic_obs": critic_obs, "lower_body_actor_obs": actor_obs, "lower_body_critic_obs": critic_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        """
        Lower Body Tracking Rewards
        """
        
        tracking_lin_vel_xy = mdp.track_lin_vel_xy_yaw_frame_exp(
            root_quat_w=self.robot.data.root_quat_w,
            root_lin_vel_w=self.robot.data.root_lin_vel_w,
            vel_command=self.command_manager.get_command("base_velocity"),
            sigma=0.25,
            weight=1.0,
        )
        tracking_ang_vel_z = mdp.track_ang_vel_z_base_exp(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            vel_command=self.command_manager.get_command("base_velocity"),
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
            action=self.actions[:, self.cfg.action_dim["upper_body"]:],
            prev_action=self.prev_actions[:, self.cfg.action_dim["upper_body"]:],
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
            command=self.command_manager.get_command("base_velocity"),
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
            weight=1.0,
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
            action=self.actions[:, :self.cfg.action_dim["upper_body"]],
            prev_action=self.prev_actions[:, :self.cfg.action_dim["upper_body"]],
            weight=-0.05,
        )

        # upper body velocities
        penalty_upper_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.upper_body_indexes,
            weight=-0.001,
        )

            
        # alive reward
        alive_reward = mdp.alive_reward(terminated=died, weight=0.15)

		# locomotion reward
        locomotion_reward = (tracking_lin_vel_xy + 
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
                             feet_clearance_reward +
                             alive_reward)
        
		# upper body reward
        upper_body_reward = (
            tracking_upper_body_dof_pos + 
            penalty_upper_body_dof_torques + 
            penalty_upper_body_dof_acc + 
            penalty_upper_body_dof_pos_limits + 
            penalty_upper_body_action_rate + 
            penalty_upper_body_dof_vel + 
            alive_reward
        )
        
        self._episode_sums["tracking_lin_vel_xy"] += tracking_lin_vel_xy
        self._episode_sums["tracking_ang_vel_z"] += tracking_ang_vel_z
        self._episode_sums["gait_phase_reward"] += feet_gait_reward
        self._episode_sums["feet_clearance_reward"] += feet_clearance_reward
        self._episode_sums["tracking_upper_body_dof_pos"] += tracking_upper_body_dof_pos

        # reward 
        lower_body_reward = locomotion_reward * self.step_dt
        upper_body_reward = upper_body_reward * self.step_dt
        return {'upper_body': upper_body_reward, 'lower_body': lower_body_reward}

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
            avg_terrain_level = mdp.terrain_levels(env=self, env_ids=env_ids, vel_command=self.command_manager.get_command("base_velocity"))
            extras["Curriculum/terrain_level"] = avg_terrain_level.item()

        # reset robot
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # reset command
        self.command_manager.reset(env_ids)
        self.event_manager.reset(env_ids)
        # reset actions
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.root_ang_vel_buffer.reset(env_ids)
        self.root_lin_vel_buffer.reset(env_ids)
        self.projected_gravity_buffer.reset(env_ids)
        self.dof_pos_buffer.reset(env_ids)
        self.dof_vel_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.phase[env_ids] = 0.0
        self.leg_phases[env_ids] = 0.0
        

        # reset logging
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)


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

        # -- update command
        if self.cfg.commands:
            self.command_manager.compute(dt=self.step_dt)
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

    def close(self):
        if not self._is_closed:
            if self.cfg.commands:
                del self.command_manager
        super().close()

@torch.jit.script
def compute_obs(obs_tensors: List[torch.Tensor]) -> torch.Tensor:
    
    return torch.cat(obs_tensors, dim=-1)

