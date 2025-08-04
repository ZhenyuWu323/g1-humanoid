import torch
from isaaclab.utils.math import quat_apply_inverse




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