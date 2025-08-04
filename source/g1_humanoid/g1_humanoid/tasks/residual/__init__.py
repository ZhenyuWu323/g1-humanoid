
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

""" G1 Residual Locomanipulation"""

gym.register(
    id="G1-Residual-Locomanipulation",
    entry_point=f"{__name__}.g1_residual_env:G1ResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_residual_cfg:G1ResidualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualPPORunnerCfg",
    },
)


""" G1 Residual Locomanipulation Pretrain"""

gym.register(
    id="G1-Residual-Locomanipulation-Pretrain",
    entry_point=f"{__name__}.g1_residual_pre_env:G1ResidualPreEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_residual_pre_cfg:G1ResidualPreEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualPPORunnerCfg",
    },
)
