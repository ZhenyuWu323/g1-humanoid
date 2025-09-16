from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms, quat_mul, quat_inv
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



def plate_projected_gravity(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate")) -> torch.Tensor:
    """Projected gravity on the plate."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    plate_body_index = env.plate_body_index

    plate_rot_w = robot_asset.data.body_quat_w[:, plate_body_index, :]
    gravity_vec_w = robot_asset.data.GRAVITY_VEC_W
    
    plate_orientation = quat_apply_inverse(plate_rot_w, gravity_vec_w)
    return plate_orientation

def plate_lin_acc_w(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate")) -> torch.Tensor:
    """Linear acceleration of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    plate_body_index = env.plate_body_index
    return robot_asset.data.body_lin_acc_w[:, plate_body_index, :]


def plate_ang_acc_w(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate")) -> torch.Tensor:
    """Angular acceleration of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    plate_body_index = env.plate_body_index
    return robot_asset.data.body_ang_acc_w[:, plate_body_index, :]



def object_physics(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data._root_physx_view.get_material_properties()[:, 0, [0,2]]


def object_mass(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data._root_physx_view.get_masses()


def object_projected_gravity(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data.projected_gravity_b

def object_com(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data.com_pos_b[:, 0, :]


def base_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    return env.base_actions

def residual_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    return env.residual_actions


def plate_lin_vel_w(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Linear velocity of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    return plate_asset.data.body_link_lin_vel_w


def plate_ang_vel_w(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Angular velocity of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    return plate_asset.data.body_link_ang_vel_w


def object_lin_vel_w(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Linear velocity of the object in the world frame."""
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data.body_link_lin_vel_w


def object_ang_vel_w(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Angular velocity of the object in the world frame."""
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data.body_link_ang_vel_w


def object_position_w(env: ManagerBasedEnv, object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Position of the object in the world frame."""
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    return object_asset.data.root_pos_w


def plate_position_w(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Position of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    return plate_asset.data.root_pos_w


def plate_pose_robot_frame(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]

    robot_quat_w = robot_asset.data.root_quat_w
    pos_in_robot_frame = quat_apply_inverse(robot_quat_w, plate_asset.data.root_pos_w - robot_asset.data.root_pos_w)
    quat_in_robot_frame = quat_mul(quat_inv(robot_quat_w), plate_asset.data.root_quat_w)

    return torch.cat([pos_in_robot_frame, quat_in_robot_frame], dim=-1)


def plate_twist_robot_frame(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]

    robot_quat_w = robot_asset.data.root_quat_w
    lin_vel_in_robot_frame =  quat_apply_inverse(robot_quat_w, plate_asset.data.root_lin_vel_w - robot_asset.data.root_lin_vel_w)
    ang_vel_in_robot_frame = quat_apply_inverse(robot_quat_w, plate_asset.data.root_ang_vel_w - robot_asset.data.root_ang_vel_w)
    return torch.cat([lin_vel_in_robot_frame, ang_vel_in_robot_frame], dim=-1)


def object_pose_in_plate_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    plate_body_index = env.plate_body_index

    # compute the object pose in the plate frame
    plate_quat_w = robot_asset.data.body_quat_w[:, plate_body_index, :]
    plate_pos_w = robot_asset.data.body_pos_w[:, plate_body_index, :]
    pos_in_plate_frame = quat_apply_inverse(plate_quat_w, object_asset.data.root_pos_w - plate_pos_w)
    quat_in_plate_frame = quat_mul(quat_inv(plate_quat_w), object_asset.data.root_quat_w)

    return torch.cat([pos_in_plate_frame, quat_in_plate_frame], dim=-1)


def object_position_in_plate_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    plate_body_index = env.plate_body_index

    # compute the object pose in the plate frame
    plate_quat_w = robot_asset.data.body_quat_w[:, plate_body_index, :]
    plate_pos_w = robot_asset.data.body_pos_w[:, plate_body_index, :]
    pos_in_plate_frame = quat_apply_inverse(plate_quat_w, object_asset.data.root_pos_w - plate_pos_w)
    
    return pos_in_plate_frame


def object_orientation_in_plate_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    plate_body_index = env.plate_body_index

    # compute the object pose in the plate frame
    plate_quat_w = robot_asset.data.body_quat_w[:, plate_body_index, :]
    quat_in_plate_frame = quat_mul(quat_inv(plate_quat_w), object_asset.data.root_quat_w)

    return quat_in_plate_frame



def object_twist_in_plate_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    plate_body_index = env.plate_body_index


    plate_quat_w = robot_asset.data.body_quat_w[:, plate_body_index, :]
    plate_lin_vel_w = robot_asset.data.body_lin_vel_w[:, plate_body_index, :]
    plate_ang_vel_w = robot_asset.data.body_ang_vel_w[:, plate_body_index, :]
    lin_vel_in_plate_frame =  quat_apply_inverse(plate_quat_w, object_asset.data.root_lin_vel_w - plate_lin_vel_w)
    ang_vel_in_plate_frame = quat_apply_inverse(plate_quat_w, object_asset.data.root_ang_vel_w - plate_ang_vel_w)
    return torch.cat([lin_vel_in_plate_frame, ang_vel_in_plate_frame], dim=-1)



def object_pose_in_camera_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="d435_link"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    camera_body_index = env.camera_body_index

    # compute the object pose in the camera frame
    camera_quat_w = robot_asset.data.body_quat_w[:, camera_body_index, :]
    camera_pos_w = robot_asset.data.body_pos_w[:, camera_body_index, :]
    pos_in_camera_frame = quat_apply_inverse(camera_quat_w, object_asset.data.root_pos_w - camera_pos_w)
    quat_in_camera_frame = quat_mul(quat_inv(camera_quat_w), object_asset.data.root_quat_w)

    return torch.cat([pos_in_camera_frame, quat_in_camera_frame], dim=-1)

def object_position_in_camera_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="d435_link"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    camera_body_index = env.camera_body_index

    # compute the object pose in the camera frame
    camera_quat_w = robot_asset.data.body_quat_w[:, camera_body_index, :]
    camera_pos_w = robot_asset.data.body_pos_w[:, camera_body_index, :]
    pos_in_camera_frame = quat_apply_inverse(camera_quat_w, object_asset.data.root_pos_w - camera_pos_w)

    return pos_in_camera_frame


def object_orientation_in_camera_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="d435_link"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    camera_body_index = env.camera_body_index

    # compute the object pose in the camera frame
    camera_quat_w = robot_asset.data.body_quat_w[:, camera_body_index, :]
    quat_in_camera_frame = quat_mul(quat_inv(camera_quat_w), object_asset.data.root_quat_w)

    return quat_in_camera_frame



def object_pose_in_plate_frame_test(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]

    # compute the object pose in the plate frame
    plate_quat_w = plate_asset.data.root_quat_w
    plate_pos_w = plate_asset.data.root_pos_w
    pos_in_plate_frame = quat_apply_inverse(plate_quat_w, object_asset.data.root_pos_w - plate_pos_w)
    quat_in_plate_frame = quat_mul(quat_inv(plate_quat_w), object_asset.data.root_quat_w)

    return torch.cat([pos_in_plate_frame, quat_in_plate_frame], dim=-1)


def object_twist_in_plate_frame_test(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]


    plate_quat_w = plate_asset.data.root_quat_w
    plate_lin_vel_w = plate_asset.data.root_lin_vel_w
    plate_ang_vel_w = plate_asset.data.root_ang_vel_w
    lin_vel_in_plate_frame =  quat_apply_inverse(plate_quat_w, object_asset.data.root_lin_vel_w - plate_lin_vel_w)
    ang_vel_in_plate_frame = quat_apply_inverse(plate_quat_w, object_asset.data.root_ang_vel_w - plate_ang_vel_w)
    return torch.cat([lin_vel_in_plate_frame, ang_vel_in_plate_frame], dim=-1)