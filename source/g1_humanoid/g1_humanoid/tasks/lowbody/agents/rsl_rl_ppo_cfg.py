# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg

@configclass
class G1LocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_lowbody_locomotion"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
class G1LowBodyLocomotionPPORunnerCfg(G1LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "g1_lowbody_locomotion"
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=0.8,
            actor_hidden_dims=[32],
            critic_hidden_dims=[32],
            activation="elu",
            rnn_hidden_dim=64,
            rnn_num_layers=1,
            rnn_type='lstm',
        )
        

@configclass
class G1LowBodyPlateLocomotionPPORunnerCfg(G1LocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "g1_lowbody_plate_locomotion"
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=0.8,
            actor_hidden_dims=[32],
            critic_hidden_dims=[32],
            activation="elu",
            rnn_hidden_dim=64,
            rnn_num_layers=1,
            rnn_type='lstm',
        )
        
