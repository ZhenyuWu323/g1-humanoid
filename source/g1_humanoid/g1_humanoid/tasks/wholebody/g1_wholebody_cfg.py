import math
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from g1_humanoid.assets import G1_WITH_PLATE
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
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
class G1WholeBodyEnvCfg(DirectRLEnvCfg):
    """ G1 Whole Body Locomanipulation Environment Configuration """


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


    # MDP configuration # TODO: NEED add more attribute for upper and lower Actor/Critics
    observation_space = 114
    action_space = 14 + 15 # NOTE: upper body DOFs + lower body DOFs
    action_scale = 0.5 # NOTE: Upper body DOFS need to be scaled by at least 1.0
    state_space = 0

    # obs noise
    '''observation_noise_model: NoiseModelCfg = NoiseModelCfg(
        noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)
    )'''
    obs_noise_models: dict[str, NoiseModelCfg] = {
        "root_lin_vel_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        "root_ang_vel_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2)),
        "projected_gravity_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
        "joint_pos_rel": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),
        "joint_vel_rel": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
        "plate_lin_vel_w": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        "plate_lin_acc_w": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        "plate_ang_vel_w": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        "plate_ang_acc_w": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
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
    robot: ArticulationCfg = G1_WITH_PLATE.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )
    reference_body = "torso_link"

    plate_name = "plate"
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



    # events
    events: EventCfg = EventCfg()
    events.push_robot = None
    events.add_base_mass = None
    events.add_plate_mass = None

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # reward scales
    reward_scales = {
        "lin_vel_z_l2": -0.2,
        "flat_orientation_l2": -1.0, 
        "action_rate_l2": -0.005,
        "dof_acc_l2": -1.0e-7,
        "dof_torques_l2": -2.0e-6,
        "track_ang_vel_z_exp": 1.0,
        "feet_air_time": 0.75,
        "joint_deviation_waist": -0.5, # unitree offcial use -1.0
        "joint_deviation_upper_body": -0.5, # unitree offcial use -1.0
        "joint_deviation_hips": -0.5, # unitree offcial use -1.0
        "dof_pos_limits": -1.0,
        "feet_slide": -0.1,
        "termination_penalty": -200.0,
        "track_lin_vel_xy_exp": 1.0,
        "ang_vel_xy_l2": -0.05,
        "base_height": -10.0,
        "gait_phase_reward": 0.18,
        "feet_swing_height": -20.0,
        "feet_slide": -0.2,
        "plate_flat_orientation_l2": -0.05,
        "plate_lin_acc_l2": -0.01,
        "plate_ang_acc_l2": -0.01,
        "plate_lin_acc_exp": 0,
        "plate_ang_acc_exp": 0,
    }

    # terminations
    termination_height = 0.5
    

    # command
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    # target base height
    target_base_height = 0.78