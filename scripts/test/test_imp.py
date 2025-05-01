# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from g1_humanoid.assets import G1_INSPIRE_DFQ
from isaaclab_assets.robots.unitree import G1_CFG
from g1_humanoid.assets import G1_INSPIRE_FTP


"""
Describe robot
"""

# Minimal to describe robot: spawn and actuators
# USD check https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots.html



class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # object
    cone = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/Cone',
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (-0.2, 1.0, 0.0),
            rot = (0.5, 0.0, 0.5, 0.0)
        )
    )

    # robot
    g1bot = G1_INSPIRE_FTP.replace(prim_path="{ENV_REGEX_NS}/G1bot")
    '''g2bot = G1_INSPIRE_FTP.replace(
        prim_path="{ENV_REGEX_NS}/G2bot",
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (-0.2, -1.0, 0.74)
        )
    )'''


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    '''robot = scene["g1bot"]
    print(robot.num_joints)
    print(robot.joint_names)'''

    while simulation_app.is_running():
        scene.reset()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()