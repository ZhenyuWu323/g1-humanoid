# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state
from ..residual_adaptive_env_wrapper import ResidualAdaptiveVecEnvWrapper
from .adaptive_model_eval import AdaptiveModelEval

class ResidualAdaptiveEvalRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: ResidualAdaptiveVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.upper_body_policy_cfg = train_cfg["upper_body_policy"]
        self.lower_body_policy_cfg = train_cfg["lower_body_policy"]
        self.residual_policy_cfg = train_cfg["residual_whole_body_policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.num_privileged_obs = self.env.num_privileged_obs

        # keys
        self.body_keys = ['upper_body', 'lower_body', 'residual_whole_body']
        self.obs_keys = ['actor_obs', 'critic_obs', 'residual_actor_obs', 'residual_critic_obs', 'encoder_obs']
        
        # setup policies and algorithms
        self.__setup_policy()

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    


    def __setup_policy(self):
        # initialize policies
        self.policies = {}
        

        self.policies["upper_body"] = ActorCritic(
            num_actor_obs=self.num_obs["actor_obs"],
            num_critic_obs=self.num_obs["critic_obs"],
            num_actions=self.num_actions["upper_body"],
            **self.upper_body_policy_cfg
        ).to(self.device)
        self.policies["lower_body"] = ActorCritic(
            num_actor_obs=self.num_obs["actor_obs"],
            num_critic_obs=self.num_obs["critic_obs"],
            num_actions=self.num_actions["lower_body"],
            **self.lower_body_policy_cfg
        ).to(self.device)
        self.policies["residual_whole_body"] = AdaptiveModelEval(
            num_actor_obs=self.num_obs["residual_actor_obs"],
            num_actions=self.num_actions["upper_body"] + self.num_actions["lower_body"],
            num_encoder_obs=int(self.num_obs["encoder_obs"]/self.env.history_length),
            num_time_steps=self.env.history_length,
            num_encoder_output=self.env.encoder_output_dim,
            **self.residual_policy_cfg
        ).to(self.device)
        # NOTE: disable gradient for lower and upper body policies
        for body_key in self.body_keys:
            if body_key == "upper_body" or body_key == "lower_body":
                for param in self.policies[body_key].parameters():
                    param.requires_grad = False
    

        self.actor_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

       

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {}
        # save model
        saved_dict.update({
            f"model_state_dict_{key}": self.policies[key].state_dict()
            for key in self.body_keys
        })
        # save infos
        saved_dict["infos"] = infos
        # save model
        torch.save(saved_dict, path)


    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        
        # -- Load model
        resumed_training = True
        for body_key in self.body_keys:
            try:
                self.policies[body_key].load_state_dict(loaded_dict[f"model_state_dict_{body_key}"])
            except KeyError:
                if body_key == "residual_whole_body":
                    print(f"Warning: {body_key} not found in checkpoint, skipping...")
                    continue
                else:
                    resumed_training = False
                    break
            except Exception as e:
                print(f"Error loading {body_key}: {e}")
                resumed_training = False
                break
    
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode
        
        # Move all policies to device if specified
        if device is not None:
            for body_key in self.body_keys:
                self.policies[body_key].to(device)
        
        def multi_actor_inference_policy(obs, residual_obs, encoder_obs):
            
            # Get actions from each body part
            actions_list = []
            #actions_list.append(torch.zeros(self.env.num_envs, 14, device=self.device)) # TODO: remove this
            actions_list.append(self.policies["upper_body"].act_inference(obs))
            actions_list.append(self.policies["lower_body"].act_inference(obs))
            
            # Concatenate actions (same order as training)
            combined_actions = torch.cat(actions_list, dim=1)

            # residual actions
            residual_actions = self.policies["residual_whole_body"].act_inference(residual_obs, encoder_obs)
            action_dict = {
                "base_action": combined_actions,
                "residual_action": residual_actions
            }
            return action_dict
        
        return multi_actor_inference_policy

    def eval_mode(self):
        # -- PPO
        for body_key in self.body_keys:
            self.policies[body_key].eval()
        # -- Normalization
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
