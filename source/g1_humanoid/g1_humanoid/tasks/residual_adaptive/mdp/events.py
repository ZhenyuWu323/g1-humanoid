import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.envs import DirectRLEnv
from pxr import Gf, Sdf, UsdGeom, Vt
import omni
import isaaclab.sim as sim_utils



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
    force: list[float | tuple[float, float]] | torch.Tensor | None = None,
    torque: list[float | tuple[float, float]] | torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply constant or randomized external forces and torques to specified bodies.

    This function applies forces and torques to the bodies of the asset. It supports both constant values
    and randomized ranges for each axis. The forces and torques are applied to the bodies by calling
    ``asset.set_external_force_and_torque``. The forces and torques are only applied when
    ``asset.write_data_to_sim()`` is called in the environment.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply forces and torques to. If None, applies to all environments.
        force: The force specification for each axis [x, y, z]. Each axis can be:
               - A constant float value
               - A tuple (min, max) for randomization within the range
               - None for no force applied
        torque: The torque specification for each axis [x, y, z]. Each axis can be:
                - A constant float value  
                - A tuple (min, max) for randomization within the range
                - None for no torque applied
        asset_cfg: The asset configuration specifying the asset and body IDs.

    Example:
        Apply constant forces and torques:
        >>> apply_constant_force_torque(env, env_ids, force=[10.0, 0.0, 0.0], torque=[0.0, 0.0, 5.0])
        
        Apply randomized forces with ranges:
        >>> apply_constant_force_torque(env, env_ids, force=[(-5.0, 5.0), 0.0, (-2.0, 2.0)])
        
        Mix constant and randomized values:
        >>> apply_constant_force_torque(env, env_ids, 
        ...                           force=[(-10.0, 10.0), 0.0, 5.0],
        ...                           torque=[0.0, (-3.0, 3.0), 2.0])
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # prepare force tensor
    if force is not None:
        if isinstance(force, torch.Tensor):
            # Direct tensor input
            forces = force.to(device=asset.device, dtype=torch.float32)
            if forces.dim() == 1:
                forces = forces.unsqueeze(0).unsqueeze(0).expand(len(env_ids), num_bodies, 3).contiguous()
            elif forces.dim() == 2:
                forces = forces.unsqueeze(1).expand(-1, num_bodies, -1).contiguous()
        else:
            # List input - handle constant values and ranges per axis
            if len(force) != 3:
                raise ValueError(f"Force list must have exactly 3 elements [x, y, z], got {len(force)} elements")
            
            force_components = []
            for i, axis_spec in enumerate(force):
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
    if torque is not None:
        if isinstance(torque, torch.Tensor):
            # Direct tensor input
            torques = torque.to(device=asset.device, dtype=torch.float32)
            if torques.dim() == 1:
                torques = torques.unsqueeze(0).unsqueeze(0).expand(len(env_ids), num_bodies, 3).contiguous()
            elif torques.dim() == 2:
                torques = torques.unsqueeze(1).expand(-1, num_bodies, -1).contiguous()
        else:
            # List input - handle constant values and ranges per axis
            if len(torque) != 3:
                raise ValueError(f"Torque list must have exactly 3 elements [x, y, z], got {len(torque)} elements")
            
            torque_components = []
            for i, axis_spec in enumerate(torque):
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