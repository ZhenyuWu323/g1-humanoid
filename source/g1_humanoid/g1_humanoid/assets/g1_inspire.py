import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from g1_humanoid.assets import ASSETS_DATA_DIR

G1_INSPIRE_FTP = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/g1-inspire-ftp/g1_ftp.usd",
        activate_contact_sensors=True,
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
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.2,
        },
        #joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                "waist_yaw_joint": 300.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 300.0
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
                "waist_yaw_joint": 0.03,
                "waist_roll_joint": 0.03,
                "waist_pitch_joint": 0.03
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
        ),
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness={
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_pitch_joint": 90.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 60.0,
            },
            damping={
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1.0,
            },
            armature={
                ".*_shoulder_roll_joint": 0.03,
                ".*_shoulder_pitch_joint": 0.03,
                ".*_shoulder_yaw_joint": 0.03,
                ".*_elbow_joint": 0.03,
            },
        ),
        "arm_forearm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=4.0,
            damping=0.2,
            armature=0.03
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_.*",
                ".*_middle_.*",
                ".*_ring_.*",
                ".*_little_.*",
                ".*_thumb_.*"
            ],
            velocity_limit_sim=5.0,
            effort_limit_sim=5.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.001
        ),
    },
)
"""Configuration for the Unitree G1 29 DoF Humanoid robot with Inspire FTP Hands."""


G1_INSPIRE_DFQ = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/g1-inspire-dfq/g1_dfq.usd",
        activate_contact_sensors=True,
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
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
        },
        #joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_yaw_joint": 0.01,
                "waist_roll_joint": 0.01,
                "waist_pitch_joint": 0.01
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=40.0,
            damping=10.0,
            armature=0.01
        ),
        "arm_forearm": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=40.0,
            damping=10.0,
            armature=0.01
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_.*",
                ".*_middle_.*",
                ".*_ring_.*",
                ".*_pinky_.*",
                ".*_thumb_.*"
            ],
            velocity_limit_sim=5.0,
            effort_limit_sim=5.0,
            stiffness=1.0,
            damping=0.1,
            armature=0.001
        ),
    },
)
"""Configuration for the Unitree G1 29 DoF Humanoid robot with Inspire DFQ Hands."""