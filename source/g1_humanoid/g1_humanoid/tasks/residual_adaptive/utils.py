import torch
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms




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