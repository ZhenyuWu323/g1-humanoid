# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg
from .encoder_cfg import RslRlActorCriticEncoderCfg, RslRlStudentTeacherEncoderCfg

@configclass
class G1ResidualAdaptivePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_residual_adaptive"
    empirical_normalization = False
    # residual upper bodypolicy
    residual_whole_body_policy = RslRlActorCriticEncoderCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_d_model=32,
        encoder_nhead=2,
        encoder_num_layers=1,
        activation="elu",
    )
    # upper body policy
    upper_body_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # lower body policy
    lower_body_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1ResidualAdaptiveDistillRunnerCfg(G1ResidualAdaptivePPORunnerCfg):
    experiment_name = "g1_residual_adaptive_distillation"

    residual_whole_body_policy = RslRlStudentTeacherEncoderCfg(
        actor_hidden_dims=[512, 256, 128],
        teacher_encoder_dim=[32,2,1],
        student_encoder_dim=[32,2,2],
        activation="elu",
    )

    ppo_algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    distillation_algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5, #5 is better
        learning_rate=1e-3,
        gradient_length=15,
    )


@configclass
class G1ResidualAdaptiveEvaluateRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "g1_residual_adaptive_evaluate"
    empirical_normalization = False
    # residual upper bodypolicy
    residual_whole_body_policy = RslRlStudentTeacherEncoderCfg(
        actor_hidden_dims=[512, 256, 128],
        teacher_encoder_dim=[32,2,1],
        student_encoder_dim=[32,2,2],
        activation="elu",
    )
    # upper body policy
    upper_body_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # lower body policy
    lower_body_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )




