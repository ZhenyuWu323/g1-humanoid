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
from g1_humanoid.assets import G1_WITH_HAND, G1_CFG, G1_WITH_TRAY
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen

@configclass
class EventCfg:
    """Configuration for events."""

   # startup
    robot_material = EventTerm(
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

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": (0.05, 0.15),
            "operation": "abs",
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
            "mass_distribution_params": (0.3, 1.5),
            "operation": "abs",
        },
    )

    set_object_com = EventTerm(
        func=mdp.randomize_rigid_body_com_fixed,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
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
            "velocity_range": (-1.0, 1.0),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_object_state,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot", body_names="plate"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "random_position_radius": 0.00,
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    upper_body_actor_obs: PolicyCfg = PolicyCfg()
    lower_body_actor_obs: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        #plate_pose_robot_frame = ObsTerm(func=mdp.plate_pose_robot_frame)
        #plate_twist_robot_frame = ObsTerm(func=mdp.plate_twist_robot_frame)
        plate_projected_gravity = ObsTerm(func=mdp.plate_projected_gravity)
        plate_lin_acc_w = ObsTerm(func=mdp.plate_lin_acc_w)
        plate_ang_acc_w = ObsTerm(func=mdp.plate_ang_acc_w)
        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    upper_body_critic_obs: CriticCfg = CriticCfg()
    lower_body_critic_obs: CriticCfg = CriticCfg()



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    UpperBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=['left_shoulder_pitch_joint', 
                     'right_shoulder_pitch_joint', 
                     'left_shoulder_roll_joint', 
                     'right_shoulder_roll_joint', 
                     'left_shoulder_yaw_joint', 
                     'right_shoulder_yaw_joint', 
                     'left_elbow_joint', 
                     'right_elbow_joint', 
                     'left_wrist_roll_joint', 
                     'right_wrist_roll_joint', 
                     'left_wrist_pitch_joint', 
                     'right_wrist_pitch_joint', 
                     'left_wrist_yaw_joint', 
                     'right_wrist_yaw_joint'], 
        scale=0.25, 
        use_default_offset=True,
        preserve_order=True
    )

    LowerBodyAction = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            'waist_yaw_joint', 
            'waist_roll_joint', 
            'waist_pitch_joint',
            'left_hip_pitch_joint', 
            'right_hip_pitch_joint', 
            'left_hip_roll_joint', 
            'right_hip_roll_joint', 
            'left_hip_yaw_joint', 
            'right_hip_yaw_joint', 
            'left_knee_joint', 
            'right_knee_joint',
            'left_ankle_pitch_joint', 
            'right_ankle_pitch_joint', 
            'left_ankle_roll_joint', 
            'right_ankle_roll_joint'], 
        scale=0.25, 
        use_default_offset=True,
        preserve_order=True
    )



@configclass
class G1JointBaselineEnvCfg(DirectRLEnvCfg):
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
        # upper body
        "upper_body_actor_obs": 480,
        "upper_body_critic_obs": 480 + 15 + 45,
        # lower body
        "lower_body_actor_obs": 480,
        "lower_body_critic_obs": 480 + 15 + 45,
    }
    action_dim= {
        "upper_body": 14,
        "lower_body": 15,
    }
    action_space = 29
    action_scale = 0.25
    state_space = 0
    obs_history_length = 5

    

    # terrain configuration
    terrain_generator_cfg = terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=9,
        num_cols=21,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        },
        curriculum=True,
    )


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=terrain_generator_cfg.num_rows - 1,
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
    robot: ArticulationCfg = G1_WITH_HAND.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, update_period=sim.dt
    )
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=sim.dt * decimation,
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
    pelvis_names = "pelvis"
    hand_names = "^(left|right)_(hand|palm|base|thumb|index|middle|ring|little).*$"
    not_hand_names = "^(?!.*_(hand|palm|base|thumb|index|middle|ring|little)).*$"
    plate_name = "plate"

    # gait phase
    gait_period = 0.8
    phase_offset = 0.5
    stance_phase_threshold = 0.55

    # events
    events: EventCfg = EventCfg()

    # command
    commands: CommandsCfg = CommandsCfg()

    # actions
    actions: ActionsCfg = ActionsCfg()

    # observations
    observations: ObservationsCfg = ObservationsCfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)


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
    

    # target base height
    target_base_height = 0.78

    # target feet height
    target_feet_height = 0.12

    # knee joint threshold
    knee_joint_threshold = 0.2

    # sdk joint sequence
    sdk_joint_sequence = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # object configuration
    plate_offset = [0.42, 0.0, 0.12] # x, y, z offset from pelvis
    plate_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.19,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    plate_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Plate", 
        update_period=sim.dt, 
        track_air_time=True,
        history_length=3,
    )

    # object configuration
    obj_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

