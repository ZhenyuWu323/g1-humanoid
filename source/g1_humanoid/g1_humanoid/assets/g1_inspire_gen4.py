import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from g1_humanoid.assets import ASSETS_DATA_DIR

G1_INSPIRE_GEN4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSETS_DATA_DIR}/g1_inspire_gen4.usd",
        usd_path=f"{ASSETS_DATA_DIR}/g1_inspire_gen4_update.usd", # fix mimic joint issue of left_middle_2_joint
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
            fix_root_link=True,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.01, rest_offset=0.0
        # ),
        # joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        # joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit=120,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        ),
        "arm_forearm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=60,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
            armature={
                ".*_wrist_.*": 0.01,
            },
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_1_joint",
                ".*_middle_1_joint",
                ".*_ring_1_joint",
                ".*_little_1_joint",
                ".*_thumb_1_joint",
                ".*_thumb_2_joint",
            ],
            # velocity_limit=3.0,
            # effort_limit={
            #     ".*_index_1_joint": 5.0,
            #     ".*_middle_1_joint": 5.0,
            #     ".*_ring_1_joint": 5.0,
            #     ".*_little_1_joint": 5.0,
            #     ".*_thumb_1_joint": 5.0,
            #     ".*_thumb_2_joint": 5.0,
            # },
            # stiffness={
            #     ".*_index_1_joint": 5.0,
            #     ".*_middle_1_joint": 5.0,
            #     ".*_ring_1_joint": 5.0,
            #     ".*_little_1_joint": 5.0,
            #     ".*_thumb_1_joint": 5.0,
            #     ".*_thumb_2_joint": 5.0,
            # },
            # damping={
            #     ".*_index_1_joint": 0.5,
            #     ".*_middle_1_joint": 0.5,
            #     ".*_ring_1_joint": 0.5,
            #     ".*_little_1_joint": 0.5,
            #     ".*_thumb_1_joint": 0.5,
            #     ".*_thumb_2_joint": 0.5,
            # },
            # armature={
            #     ".*_index_1_joint": 0.001,
            #     ".*_middle_1_joint": 0.001,
            #     ".*_ring_1_joint": 0.001,
            #     ".*_little_1_joint": 0.001,
            #     ".*_thumb_1_joint": 0.001,
            #     ".*_thumb_2_joint": 0.001,
            # },
            velocity_limit=10.0,
            effort_limit=3.0,
            stiffness=1.5,
            damping=0.5,
            friction=0.01,
        ),
    },
)
"""Configuration for the Unitree G1 29 DoF Humanoid robot with Inspire Gen4 Hands."""
