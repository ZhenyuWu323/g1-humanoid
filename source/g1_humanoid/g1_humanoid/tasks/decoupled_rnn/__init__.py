
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


""" G1 Decoupled Plate Object Locomanipulation with RNN"""

gym.register(
    id="G1-Decoupled-Plate-Object-RNN-Locomanipulation",
    entry_point=f"{__name__}.g1_decoupled_rnn_env:G1DecoupledRNNEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_decoupled_rnn_cfg:G1DecoupledRNNEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DecoupledPPORunnerCfg",
    },
)
