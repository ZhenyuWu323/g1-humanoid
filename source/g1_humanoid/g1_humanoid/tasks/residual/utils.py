import torch
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms




def compute_dof_pos_tracking_weight(object_quat, gravity_vec):

    object_up = quat_apply_inverse(object_quat, gravity_vec)
    tilt_angle = torch.acos(torch.clamp(object_up[:, 2], -1, 1))
    
    
    tilt_threshold = 0.15  # ~8.6度
    weight_multiplier = torch.where(
        tilt_angle > tilt_threshold,
        torch.exp(-(tilt_angle - tilt_threshold) * 10),  
        torch.ones_like(tilt_angle)
    )
    
    return 0.5 * torch.clamp(weight_multiplier, 0.02, 1.0)  




def compute_object_pose_in_camera_frame(object_quat_w, object_pos_w, camera_quat_w, camera_pos_w):
    object_pos_camera, object_quat_camera = subtract_frame_transforms(
        t01=camera_pos_w,      # camera pose in world
        q01=camera_quat_w,     # camera orientation in world  
        t02=object_pos_w,      # object pose in world
        q02=object_quat_w      # object orientation in world
    )
    object_pose_camera = torch.cat([object_pos_camera, object_quat_camera], dim=-1) # (N, 7)
    
    return object_pose_camera # x, y, z, qw, qx, qy, qz