
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

""" G1 Decoupled Locomanipulation"""

gym.register(
    id="Template-G1-Decoupled-Locomanipulation",
    entry_point=f"{__name__}.g1_decoupled_env:G1DecoupledEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_decoupled_cfg:G1DecoupledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DecoupledPPORunnerCfg",
    },
)

""" G1 Decoupled Plate Locomanipulation"""

gym.register(
    id="Template-G1-Decoupled-Plate-Locomanipulation",
    entry_point=f"{__name__}.g1_decoupled_env:G1DecoupledEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_decoupled_cfg:G1DecoupledPlateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DecoupledPPORunnerCfg",
    },
)

""" G1 Decoupled Plate Object Locomanipulation"""

gym.register(
    id="Template-G1-Decoupled-Plate-Object-Locomanipulation",
    entry_point=f"{__name__}.g1_decoupled_env:G1DecoupledEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_decoupled_cfg:G1DecoupledPlateObjectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DecoupledPPORunnerCfg",
    },
)
