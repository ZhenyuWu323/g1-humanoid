import math
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from . import mdp
from g1_humanoid.assets import G1_WITH_PLATE, G1_CFG
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.envs.common import ViewerCfg

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    scale_control_gain = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            'distribution': 'uniform',
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    add_plate_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="plate"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )




@configclass
class G1DecoupledEnvCfg(DirectRLEnvCfg):
    """ G1 Decoupled Locomanipulation Environment Configuration """


    # simulation configuration
    episode_length_s = 20.0
    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count = 10 * 2**15
        ),
    )
    body_keys = ['upper_body', 'lower_body']

    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        eye=[2.5, -2.0, 1.8],
        lookat=[0.0, 0.0, 0.0],
    )


    # MDP configuration
    # NOTE: Remember to update these if any updates are made to env
    observation_space = {
        "actor_obs": 482,
        "critic_obs": 497,
    }
    action_dim= {
        "upper_body": 14,
        "lower_body": 15,
    }
    action_space = 29
    action_scale = 0.25
    state_space = 0
    obs_history_length = 5

    # obs noise
    obs_noise_models: dict[str, NoiseModelCfg] = {
        "root_lin_vel_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        "root_ang_vel_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2)),
        "projected_gravity_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
        "dof_pos": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),
        "dof_vel": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
    }


    # terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # lights
    sky_light_cfg = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",)

    # robot configuration
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )
    reference_body = "torso_link"

    arm_names = [".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",]
    
    
    waist_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    hips_names = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"]

    feet_names = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]

    lower_body_names = waist_names + hips_names + feet_names
    upper_body_names = arm_names 
    feet_body_name = ".*_ankle_roll_link"

    # gait phase
    gait_period = 0.8
    phase_offset = 0.5
    stance_phase_threshold = 0.55

    # events
    events: EventCfg = EventCfg()
    #events.push_robot = None
    events.add_plate_mass = None

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)

    # reward scales
    reward_scales = {
        "track_line_vel_xy":1.0,
        "track_ang_vel_z":0.5,
        "alive": 0.15,
        "penalty_lin_vel_z":-2.0,
        "penalty_ang_vel_xy":-0.05,
        "penalty_lower_body_dof_vel":-0.001,
        "penalty_lower_body_dof_acc": -2.5e-7,
        "penalty_lower_body_action_rate": -0.05,
        "penalty_lower_body_dof_pos_limits": -5.0,
        "penalty_dof_pos_waist": -1.0,
        "penalty_dof_pos_hips": -1.0,
        "penalty_flat_orientation": -5.0,
        "penalty_base_height": -10.0,
        "gait_phase_reward": 0.5,
        "feet_slide": -0.2,
        "feet_clearance": 1.0,


        # upper body
        "tracking_upper_body_dof_pos": 0.5,
        "penalty_upper_body_dof_torques": -1e-5,
        "penalty_upper_body_dof_acc": -2.5e-7,
        "penalty_upper_body_dof_pos_limits": -5.0,
        "penalty_upper_body_dof_action_rate": -0.05,
        "penalty_upper_body_dof_vel": -1e-3,
        #"penalty_upper_body_termination": -100.0,
    }

    # observation scales
    obs_scales = {
        "root_lin_vel_b": 2.0,
        "root_ang_vel_b": 0.25,
        "projected_gravity_b": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    }

    # clips
    clip_action = 100
    clip_observation = 100

    # terminations
    termination_height = 0.5
    

    # command
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )
    # target base height
    target_base_height = 0.78

    # target feet height
    target_feet_height = 0.12

    # knee joint threshold
    knee_joint_threshold = 0.2


@configclass
class G1DecoupledPlateEnvCfg(G1DecoupledEnvCfg):
    """ G1 Decoupled Plate Locomanipulation Environment Configuration """

    # robot configuration
    robot: ArticulationCfg = G1_WITH_PLATE.replace(prim_path="/World/envs/env_.*/Robot")

    plate_name = "plate"

    events: EventCfg = EventCfg()

    observation_space = {
        "actor_obs": 482,
        "critic_obs": 497 + 45,
    }


@configclass
class G1DecoupledPlateObjectEnvCfg(G1DecoupledPlateEnvCfg):
    """ G1 Decoupled Plate Object Locomanipulation Environment Configuration """

    # robot configuration
    robot: ArticulationCfg = G1_WITH_PLATE.replace(prim_path="/World/envs/env_.*/Robot")

    events: EventCfg = EventCfg()

    # object configuration
    obj_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    