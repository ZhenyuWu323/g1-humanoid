import math
from typing import Optional
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.envs import DirectRLEnv
from pxr import Gf, Sdf, UsdGeom, Vt
import omni
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
_all_forces = torch.tensor([])

def randomize_cylinder_scale(
    env: DirectRLEnv,
    env_ids: torch.Tensor | None,
    radius_scale_range: tuple[float, float],
    height_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression "/World/envs/env_.*/Object" has a child
    with the path "/World/envs/env_.*/Object/mesh", then the relative child path should be "mesh" or
    "/mesh".

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)


    # sample scale values
    radius_samples = math_utils.sample_uniform(
        radius_scale_range[0], radius_scale_range[1], (len(env_ids),), device="cpu"
    )
    height_samples = math_utils.sample_uniform(
        height_scale_range[0], height_scale_range[1], (len(env_ids),), device="cpu"
    )
    # convert to list for the for loop
    rand_samples = torch.stack([radius_samples, radius_samples, height_samples], dim=1)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


def apply_external_force_torque_custom(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    force_range: list[float | tuple[float, float]] | torch.Tensor | None = None,
    torque_range: list[float | tuple[float, float]] | torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Apply constant or randomized external forces and torques to specified bodies.

    This function applies forces and torques to the bodies of the asset. It supports both constant values
    and randomized ranges for each axis. The forces and torques are applied to the bodies by calling
    ``asset.set_external_force_and_torque``. The forces and torques are only applied when
    ``asset.write_data_to_sim()`` is called in the environment.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply forces and torques to. If None, applies to all environments.
        force_range: The force specification for each axis [x, y, z]. Each axis can be:
               - A constant float value
               - A tuple (min, max) for randomization within the range
               - None for no force applied
        torque_range: The torque specification for each axis [x, y, z]. Each axis can be:
                - A constant float value
                - A tuple (min, max) for randomization within the range
                - None for no torque applied
        asset_cfg: The asset configuration specifying the asset and body IDs.

    Returns:
        torch.Tensor: The applied forces for all environments. 
                     Shape: [num_envs, 3]. Contains current force values for all environments,
                     with updated values for environments specified in env_ids.

    Example:
        Apply constant forces and torques:
        >>> forces = apply_external_force_torque_custom(env, env_ids, force=[10.0, 0.0, 0.0], torque=[0.0, 0.0, 5.0])
        
        Apply randomized forces with ranges:
        >>> forces = apply_external_force_torque_custom(env, env_ids, force=[(-5.0, 5.0), 0.0, (-2.0, 2.0)])
        
        Mix constant and randomized values:
        >>> forces = apply_external_force_torque_custom(env, env_ids, 
        ...                           force=[(-10.0, 10.0), 0.0, 5.0],
        ...                           torque=[0.0, (-3.0, 3.0), 2.0])
    """
    global _all_forces
    
    # Initialize _all_forces if not already done or resize if needed
    if _all_forces.numel() == 0 or _all_forces.shape[0] < env.num_envs:
        _all_forces = torch.zeros((env.num_envs, 3), device=env.device)
    
    # Ensure _all_forces is on the correct device
    if _all_forces.device != env.device:
        _all_forces = _all_forces.to(env.device)

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # resolve asset configuration
    asset_cfg.resolve(env.scene)
    
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # prepare force tensor
    if force_range is not None:
        if isinstance(force_range, torch.Tensor):
            # Direct tensor input
            forces = force_range.to(device=asset.device, dtype=torch.float32)
            if forces.dim() == 1:
                forces = forces.unsqueeze(0).unsqueeze(0).expand(len(env_ids), num_bodies, 3).contiguous()
            elif forces.dim() == 2:
                forces = forces.unsqueeze(1).expand(-1, num_bodies, -1).contiguous()
        else:
            # List input - handle constant values and ranges per axis
            if len(force_range) != 3:
                raise ValueError(f"Force list must have exactly 3 elements [x, y, z], got {len(force_range)} elements")

            force_components = []
            for i, axis_spec in enumerate(force_range):
                if isinstance(axis_spec, (tuple, list)) and len(axis_spec) == 2:
                    # Randomize within range for this axis
                    min_val, max_val = axis_spec
                    if min_val > max_val:
                        raise ValueError(f"Force axis {i}: min_val ({min_val}) must be <= max_val ({max_val})")
                    axis_values = math_utils.sample_uniform(
                        min_val, max_val, 
                        (len(env_ids), num_bodies), 
                        device=asset.device
                    )
                elif isinstance(axis_spec, (int, float)):
                    # Constant value for this axis
                    axis_values = torch.full(
                        (len(env_ids), num_bodies), 
                        float(axis_spec), 
                        device=asset.device, 
                        dtype=torch.float32
                    )
                else:
                    raise ValueError(f"Force axis {i} must be a number or a tuple (min, max), got {type(axis_spec)}: {axis_spec}")
                force_components.append(axis_values)
            
            # Stack components to form [num_envs, num_bodies, 3]
            forces = torch.stack(force_components, dim=-1)
    else:
        # create zero forces if none specified
        forces = torch.zeros((len(env_ids), num_bodies, 3), dtype=torch.float32, device=asset.device)

    # prepare torque tensor
    if torque_range is not None:
        if isinstance(torque_range, torch.Tensor):
            # Direct tensor input
            torques = torque_range.to(device=asset.device, dtype=torch.float32)
            if torques.dim() == 1:
                torques = torques.unsqueeze(0).unsqueeze(0).expand(len(env_ids), num_bodies, 3).contiguous()
            elif torques.dim() == 2:
                torques = torques.unsqueeze(1).expand(-1, num_bodies, -1).contiguous()
        else:
            # List input - handle constant values and ranges per axis
            if len(torque_range) != 3:
                raise ValueError(f"Torque list must have exactly 3 elements [x, y, z], got {len(torque_range)} elements")

            torque_components = []
            for i, axis_spec in enumerate(torque_range):
                if isinstance(axis_spec, (tuple, list)) and len(axis_spec) == 2:
                    # Randomize within range for this axis
                    min_val, max_val = axis_spec
                    if min_val > max_val:
                        raise ValueError(f"Torque axis {i}: min_val ({min_val}) must be <= max_val ({max_val})")
                    axis_values = math_utils.sample_uniform(
                        min_val, max_val, 
                        (len(env_ids), num_bodies), 
                        device=asset.device
                    )
                elif isinstance(axis_spec, (int, float)):
                    # Constant value for this axis
                    axis_values = torch.full(
                        (len(env_ids), num_bodies), 
                        float(axis_spec), 
                        device=asset.device, 
                        dtype=torch.float32
                    )
                else:
                    raise ValueError(f"Torque axis {i} must be a number or a tuple (min, max), got {type(axis_spec)}: {axis_spec}")
                torque_components.append(axis_values)
            
            # Stack components to form [num_envs, num_bodies, 3]
            torques = torch.stack(torque_components, dim=-1)
    else:
        # create zero torques if none specified
        torques = torch.zeros((len(env_ids), num_bodies, 3), dtype=torch.float32, device=asset.device)

    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)
    
    # Store the applied forces for the specified environments (take first body for simplicity)
    # Sum forces across all bodies if multiple bodies are affected
    if num_bodies > 1:
        applied_forces = forces.sum(dim=1)  # Sum across bodies: [num_envs, 3]
    else:
        applied_forces = forces.squeeze(1)  # Remove body dimension: [num_envs, 3]
    
    # Update the global forces array for the specified environments
    _all_forces[env_ids] = applied_forces
    
    return _all_forces

def clear_external_wrenches(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Clear any external forces and torques applied to the specified asset.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to clear forces for.
        asset_cfg: The asset configuration specifying the asset and body IDs.
        
    Returns:
        torch.Tensor: The updated forces for all environments after clearing.
                     Shape: [num_envs, 3]. Contains zero forces for specified environments.
    """
    global _all_forces
    
    # Initialize _all_forces if not already done or resize if needed
    if _all_forces.numel() == 0 or _all_forces.shape[0] < env.num_envs:
        _all_forces = torch.zeros((env.num_envs, 3), device=env.device)
    
    # Ensure _all_forces is on the correct device
    if _all_forces.device != env.device:
        _all_forces = _all_forces.to(env.device)

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset_cfg.resolve(env.scene)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # Passing zero tensors disables external wrenches
    asset.set_external_force_and_torque(
        forces=torch.zeros((len(env_ids), num_bodies, 3), device=asset.device),
        torques=torch.zeros((len(env_ids), num_bodies, 3), device=asset.device),
        env_ids=env_ids,
        body_ids=asset_cfg.body_ids
    )
    
    # Update the global forces array to zero for the specified environments
    _all_forces[env_ids] = torch.zeros((len(env_ids), 3), device=env.device)
    
    return _all_forces



def randomize_rigid_body_com_fixed(
    env: DirectRLEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:,:3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)



def reset_plate_state(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"),
    plate_asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    object_asset_cfg: Optional[SceneEntityCfg] = None,
    plate_offset: list[float] = [0.42, 0.0, 0.12],
    object_z_up: float = 0.1,
):
    """Reset the state of the plate."""
    # extract the used quantities (to enable type-hinting)
    plate_asset: RigidObject = env.scene[plate_asset_cfg.name]
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]

    reference_frame_states = robot_asset.data.body_state_w[env_ids, robot_asset_cfg.body_ids, :].clone()

    # reset plate
    plate_offset_tensor = torch.tensor(
        plate_offset, device=env.device
    ).unsqueeze(0).expand(len(env_ids), -1)
    offset_world = quat_apply(reference_frame_states[:, 3:7], plate_offset_tensor)
    plate_pos_world = reference_frame_states[:, :3] + offset_world
    plate_asset.write_root_link_pose_to_sim(torch.cat([plate_pos_world, plate_asset.data.default_root_state[env_ids, 3:7]], dim=-1), env_ids=env_ids)
    plate_asset.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)

    # reset object
    if object_asset_cfg is not None:
        object_asset: RigidObject = env.scene[object_asset_cfg.name]
        if isinstance(object_asset.cfg.spawn, sim_utils.CylinderCfg):
            object_z_up = object_asset.cfg.spawn.height / 2
        object_pos_world = plate_pos_world.clone()
        object_pos_world[:, 2] += object_z_up
        object_asset.write_root_link_pose_to_sim(torch.cat([object_pos_world, object_asset.data.default_root_state[env_ids, 3:7]], dim=-1), env_ids=env_ids)
        object_asset.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def reset_object_state(
    env: DirectRLEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="plate"),
    object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_z_up: float = 0.1,
    random_position_radius: float = 0.08
):
    """Reset the state of the object."""
    # extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_asset_cfg.name]
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    plate_body_index = robot_asset_cfg.body_ids

    reference_frame_states = robot_asset.data.body_state_w[env_ids, plate_body_index, :].clone()
    num_envs = len(env_ids)
    device = env.device
    object_pos_world = reference_frame_states[:, :3].clone()

    if random_position_radius > 0.0:

        random_radius = torch.sqrt(torch.rand(num_envs, device=device)) * random_position_radius
        random_angle = torch.rand(num_envs, device=device) * 2 * math.pi
        
        # Convert to cartesian coordinates in plate frame
        random_x = random_radius * torch.cos(random_angle)
        random_y = random_radius * torch.sin(random_angle)
        object_pos_world[:, 0] += random_x  # x offset
        object_pos_world[:, 1] += random_y  # y offset
    

    # reset object
    if isinstance(object_asset.cfg.spawn, sim_utils.CylinderCfg):
            object_z_up = object_asset.cfg.spawn.height / 2
    object_pos_world[:, 2] += object_z_up # z offset
    object_asset.write_root_link_pose_to_sim(torch.cat([object_pos_world, object_asset.data.default_root_state[env_ids, 3:7]], dim=-1), env_ids=env_ids)
    object_asset.write_root_com_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)