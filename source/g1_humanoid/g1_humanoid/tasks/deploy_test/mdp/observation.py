from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



def plate_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Projected gravity on the plate."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b

def plate_lin_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Linear acceleration of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_acc_w


def plate_ang_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Angular acceleration of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_ang_acc_w


def object_pose_in_plate_frame(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]

    # compute the object pose in the plate frame
    object_pos_in_plate = compute_object_pos_in_plate_frame(
        object_quat_w=object_asset.data.body_link_quat_w[:, 0, :],
        object_pos_w=object_asset.data.body_pos_w[:, 0, :],
        plate_quat_w=plate_asset.data.body_link_quat_w[:, 0, :],
        plate_pos_w=plate_asset.data.body_pos_w[:, 0, :],
    )
    return object_pos_in_plate


def object_twist_in_plate_frame(env: ManagerBasedEnv, plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]

    object_lin_vel_plate, object_ang_vel_plate = compute_object_twist_in_plate_frame(
        object_lin_vel_w=object_asset.data.body_lin_vel_w[:, 0, :],
        object_ang_vel_w=object_asset.data.body_ang_vel_w[:, 0, :],
        plate_lin_vel_w=plate_asset.data.body_lin_vel_w[:, 0, :],
        plate_ang_vel_w=plate_asset.data.body_ang_vel_w[:, 0, :],
        plate_quat_w=plate_asset.data.body_link_quat_w[:, 0, :],
        plate_pos_w=plate_asset.data.body_pos_w[:, 0, :],
        object_pos_w=object_asset.data.body_pos_w[:, 0, :],
    )
    return torch.cat([object_lin_vel_plate, object_ang_vel_plate], dim=-1)


def object_pose_in_camera_frame(env: ManagerBasedEnv, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="d435_link"), object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    camera_body_index = robot_asset.data.body_names.index(robot_asset_cfg.body_names)

    # compute the object pose in the camera frame
    object_pos_in_camera = compute_object_pose_in_camera_frame(
        object_quat_w=object_asset.data.body_link_quat_w[:, 0, :],
        object_pos_w=object_asset.data.body_pos_w[:, 0, :],
        camera_quat_w=robot_asset.data.body_link_quat_w[:, camera_body_index, :],
        camera_pos_w=robot_asset.data.body_pos_w[:, camera_body_index, :],
    )
    return object_pos_in_camera




@torch.jit.script
def compute_dof_pos_tracking_weight(object_quat, gravity_vec):

    object_up = quat_apply_inverse(object_quat, gravity_vec)
    tilt_angle = torch.acos(torch.clamp(object_up[:, 2], -1, 1))
    
    
    tilt_threshold = 0.15  
    weight_multiplier = torch.where(
        tilt_angle > tilt_threshold,
        torch.exp(-(tilt_angle - tilt_threshold) * 10),  
        torch.ones_like(tilt_angle)
    )
    
    return 0.5 * torch.clamp(weight_multiplier, 0.02, 1.0)  


@torch.jit.script
def compute_object_pose_in_camera_frame(object_quat_w, object_pos_w, camera_quat_w, camera_pos_w):
    object_pos_camera, object_quat_camera = subtract_frame_transforms(
        t01=camera_pos_w,      # camera pose in world
        q01=camera_quat_w,     # camera orientation in world  
        t02=object_pos_w,      # object pose in world
        q02=object_quat_w      # object orientation in world
    )
    object_pose_camera = torch.cat([object_pos_camera, object_quat_camera], dim=-1) # (N, 7)
    
    return object_pose_camera # x, y, z, qw, qx, qy, qz

@torch.jit.script
def compute_object_pos_in_plate_frame(object_quat_w, object_pos_w, plate_quat_w, plate_pos_w):
    object_pos_plate, object_quat_plate = subtract_frame_transforms(
        t01=plate_pos_w,      # plate pose in world
        q01=plate_quat_w,     # plate orientation in world  
        t02=object_pos_w,      # object pose in world
        q02=object_quat_w      # object orientation in world
    )
    object_pose_plate = torch.cat([object_pos_plate, object_quat_plate], dim=-1) # (N, 7)
    return object_pose_plate # x, y, z, qw, qx, qy, qz


@torch.jit.script
def compute_object_twist_in_plate_frame(
    object_lin_vel_w: torch.Tensor,   # (N,3)
    object_ang_vel_w: torch.Tensor,   # (N,3)
    plate_lin_vel_w: torch.Tensor,    # (N,3)
    plate_ang_vel_w: torch.Tensor,    # (N,3)
    plate_quat_w: torch.Tensor,       # (N,4) (w,x,y,z)
    plate_pos_w: torch.Tensor,        # (N,3)
    object_pos_w: torch.Tensor,       # (N,3)
) -> tuple[torch.Tensor, torch.Tensor]:
    # relative velocity in world frame
    r_op_w = object_pos_w - plate_pos_w
    rel_ang_w = object_ang_vel_w - plate_ang_vel_w
    rel_lin_w = object_lin_vel_w - plate_lin_vel_w - torch.cross(plate_ang_vel_w, r_op_w, dim=-1)
    # transform to plate frame
    rel_ang_p = quat_apply_inverse(plate_quat_w, rel_ang_w)
    rel_lin_p = quat_apply_inverse(plate_quat_w, rel_lin_w)
    return rel_lin_p, rel_ang_p


@torch.jit.script
def is_object_on_plate(object_pos_in_plate):
    # check position
    z_condition = torch.abs(object_pos_in_plate[:, 2]) < 0.1  # 10cm
    xy_condition = torch.norm(object_pos_in_plate[:, :2], dim=1) < 0.3

    return z_condition & xy_condition 