import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from g1_humanoid.assets import ASSETS_DATA_DIR

G1_INSPIRE_GEN4_CFG_2 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/g1_inspire_gen4_2.0.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.84),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
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
            effort_limit=300,
            velocity_limit=100,
            stiffness=40.0,
            damping=10.0,
        ),
        "arm_forearm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=40.0,
            damping=10.0,
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
            velocity_limit_sim=5.0,
            effort_limit_sim=5.0,
            stiffness=1.0,
            damping=0.1,
        ),
    },
)
"""Configuration for the Unitree G1 29 DoF Humanoid robot with Inspire Gen4 Hands."""




G1_INSPIRE_GEN4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/g1_inspire_gen4_update.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
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
            velocity_limit_sim=5.0,
            effort_limit_sim=5.0,
            stiffness=1.0,
            damping=0.1,
        ),
    },
)