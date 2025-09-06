
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

""" G1 Residual Locomanipulation Whole Body"""

gym.register(
    id="G1-Residual-Locomanipulation-Adaptive",
    entry_point=f"{__name__}.g1_adaptive_env:G1ResidualAdaptiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_adaptive_cfg:G1ResidualAdaptiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptivePPORunnerCfg",
    },
)




""" G1 Residual Locomanipulation Adaptive Distill """

gym.register(
    id="G1-Residual-Locomanipulation-Adaptive-Distill",
    entry_point=f"{__name__}.g1_adaptive_distill_env:G1ResidualAdaptiveDistillEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_adaptive_cfg:G1ResidualAdaptiveDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptiveDistillRunnerCfg",
    },
)



""" G1 Residual Locomanipulation Adaptive Fine-Tune """

gym.register(
    id="G1-Residual-Locomanipulation-Adaptive-Fine-Tune",
    entry_point=f"{__name__}.g1_adaptive_env:G1ResidualAdaptiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_adaptive_cfg:G1ResidualAdaptiveFineTuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptivePPORunnerCfg",
    },
)


""" G1 Residual Locomanipulation Adaptive Evaluate """

gym.register(
    id="G1-Residual-Locomanipulation-Adaptive-Evaluate",
    entry_point=f"{__name__}.g1_adaptive_eval_env:G1ResidualAdaptiveEvaluateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_adaptive_eval_cfg:G1ResidualAdaptiveEvaluateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptiveEvaluateRunnerCfg",
    },
)