
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
    id="G1-Joint-Test",
    entry_point=f"{__name__}.joint_test_env:G1JointTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_test_cfg:G1JointTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JointTestRunnerCfg",
    },
)

gym.register(
    id="G1-Joint-Baseline",
    entry_point=f"{__name__}.joint_baseline_env:G1JointBaselineEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_baseline_cfg:G1JointBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1JointBaselineRunnerCfg",
    },
)

gym.register(
    id="G1-Residual-Priv",
    entry_point=f"{__name__}.residual_env:G1ResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_cfg:G1ResidualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptivePPORunnerCfg",
    },
)


gym.register(
    id="G1-Residual-Test",
    entry_point=f"{__name__}.residual_test_env:G1ResidualTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_test_cfg:G1ResidualTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptivePPORunnerCfg",
    },
)


gym.register(
    id="G1-Residual-Stu",
    entry_point=f"{__name__}.residual_env:G1ResidualDistillEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_cfg:G1ResidualDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptiveDistillRunnerCfg",
    },
)

gym.register(
    id="G1-Residual-Stu-Test",
    entry_point=f"{__name__}.residual_test_env:G1ResidualDistillTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.residual_test_cfg:G1ResidualDistillTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ResidualAdaptiveDistillRunnerCfg",
    },
)