import torch
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.sensors import ContactSensor




@torch.jit.script
def compute_dof_pos_tracking_weight(object_projected_gravity):

    tilt_angle = torch.acos(torch.clamp(object_projected_gravity[:, 2], -1, 1))
    
    
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




def generate_cylinder_variants(num_variants=24):

    radius_range = (0.02, 0.05)      
    height_range = (0.08, 0.15)      
    mass_range = (0.05, 0.3)         
    friction_range = (0.1, 1.0)     
    
    variants = {}
    
    for i in range(num_variants):
        radius = np.random.uniform(*radius_range)
        height = np.random.uniform(*height_range)
        mass = np.random.uniform(*mass_range)
        static_friction = np.random.uniform(*friction_range)
        dynamic_friction = static_friction * np.random.uniform(0.8, 1.0)  
        
        
        variants[f"object_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Object_{i:02d}",
            spawn=sim_utils.CylinderCfg(
                radius=radius,
                height=height,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=0.0,
                ),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 0.0)),
        )
    
    return variants



def is_lost_contact(contact_sensor: ContactSensor):
    # check if just lost contact
    last_contact_time = contact_sensor.data.last_contact_time[:, 0] 
    current_air_time = contact_sensor.data.current_air_time[:, 0] 
    return torch.logical_and(last_contact_time > 0.0, current_air_time > 0.0)
