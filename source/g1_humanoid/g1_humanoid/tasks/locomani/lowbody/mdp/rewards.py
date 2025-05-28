# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv


def track_lin_vel_xy_yaw_frame_exp(
    root_quat_w: torch.Tensor, root_lin_vel_w: torch.Tensor, vel_command: torch.Tensor, std: float, weight: float
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    vel_yaw = quat_rotate_inverse(yaw_quat(root_quat_w), root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(vel_command[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2) * weight



def track_ang_vel_z_world_exp(
    root_ang_vel_w: torch.Tensor, vel_command: torch.Tensor, std: float, weight: float
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    ang_vel_error = torch.square(vel_command[:, 2] - root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2) * weight


def track_lin_vel_xy_base_exp(root_lin_vel_b: torch.Tensor, vel_command: torch.Tensor, weight: float, std: float) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) base frame frame using exponential kernel."""
    
    lin_vel_error = torch.sum(
        torch.square(vel_command[:, :2] - root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2) * weight


def track_ang_vel_z_base_exp(root_ang_vel_b: torch.Tensor, vel_command: torch.Tensor, weight: float, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) base frame using exponential kernel."""

    ang_vel_error = torch.square(vel_command[:, 2] - root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2) * weight


def lin_vel_z_l2(root_lin_vel_b: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""

    return torch.square(root_lin_vel_b[:, 2]) * weight


def flat_orientation_l2(projected_gravity_b: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * weight


def action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""

    return torch.sum(torch.square(action - prev_action), dim=1) * weight


def joint_accel_l2(joint_accel: torch.Tensor, joint_idx: Sequence[int], weight: float) -> torch.Tensor:
    """Penalize the rate of change of the joint accelerations using L2 squared kernel."""

    return torch.sum(torch.square(joint_accel[:, joint_idx]), dim=1) * weight


def joint_torque_l2(joint_torque: torch.Tensor, joint_idx: Sequence[int], weight: float) -> torch.Tensor:
    """Penalize the rate of change of the joint torques using L2 squared kernel."""

    return torch.sum(torch.square(joint_torque[:, joint_idx]), dim=1) * weight


def joint_deviation_l1(joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, joint_idx: Sequence[int], weight: float) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # compute out of limits constraints
    angle = joint_pos[:, joint_idx] - default_joint_pos[:, joint_idx]
    return torch.sum(torch.abs(angle), dim=1) * weight


def joint_pos_limits(joint_pos: torch.Tensor, soft_joint_pos_limits: torch.Tensor, joint_idx: Sequence[int], weight: float) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # compute out of limits constraints
    out_of_limits = -(
        joint_pos[:, joint_idx] - soft_joint_pos_limits[:, joint_idx, 0]
    ).clip(max=0.0)
    out_of_limits += (
        joint_pos[:, joint_idx] - soft_joint_pos_limits[:, joint_idx, 1]
    ).clip(min=0.0)

    return torch.sum(out_of_limits, dim=1) * weight


def termination_penalty(terminated: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize termination."""

    return terminated * weight


def feet_air_time(
    env: DirectRLEnv, vel_command: torch.Tensor, sensor_cfg: SceneEntityCfg, threshold: float, weight: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(vel_command[:, :2], dim=1) > 0.1
    return reward * weight


def feet_air_time_positive_biped(env, vel_command: torch.Tensor, threshold: float, sensor_cfg: SceneEntityCfg, weight: float) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(vel_command[:, :2], dim=1) > 0.1
    return reward * weight


def feet_slide(env, weight: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward * weight

