
import torch
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
import numpy as np





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
def compute_object_pos_in_plate_frame(object_quat_w, object_pos_w, plate_quat_w, plate_pos_w):
    object_pos_plate, object_quat_plate = subtract_frame_transforms(
        t01=plate_pos_w,      # plate pose in world
        q01=plate_quat_w,     # plate orientation in world  
        t02=object_pos_w,      # object pose in world
        q02=object_quat_w      # object orientation in world
    )
    object_pose_plate = torch.cat([object_pos_plate, object_quat_plate], dim=-1) # (N, 7)
    return object_pose_plate # x, y, z, qw, qx, qy, qz