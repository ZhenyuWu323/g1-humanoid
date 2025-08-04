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
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state
from .joint_env_wrapper import JointRslRlVecEnvWrapper

class JointOnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: JointRslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.upper_body_policy_cfg = train_cfg["upper_body_policy"]
        self.lower_body_policy_cfg = train_cfg["lower_body_policy"]
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
        #self.num_privileged_obs = self.env.num_privileged_obs

        # keys
        self.body_keys = ['upper_body', 'lower_body']
        self.obs_keys = ['lower_body_actor_obs', 'lower_body_critic_obs', 'upper_body_actor_obs', 'upper_body_critic_obs']
        
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
        upper_body_policy_class = eval(self.upper_body_policy_cfg.pop("class_name"))
        assert upper_body_policy_class in [ActorCritic, ActorCriticRecurrent], "Upper body policy class is expected to be ActorCritic or ActorCriticRecurrent."
        lower_body_policy_class = eval(self.lower_body_policy_cfg.pop("class_name"))
        assert lower_body_policy_class in [ActorCritic, ActorCriticRecurrent], "Lower body policy class is expected to be ActorCritic or ActorCriticRecurrent."

        
        self.policies["upper_body"] = upper_body_policy_class(
            num_actor_obs=self.num_obs["upper_body_actor_obs"],
            num_critic_obs=self.num_obs["upper_body_critic_obs"],
            num_actions=self.num_actions["upper_body"],
            **self.upper_body_policy_cfg
        ).to(self.device)
        self.policies["lower_body"] = lower_body_policy_class(
            num_actor_obs=self.num_obs["lower_body_actor_obs"],
            num_critic_obs=self.num_obs["lower_body_critic_obs"],
            num_actions=self.num_actions["lower_body"],
            **self.lower_body_policy_cfg
        ).to(self.device)
        
        # initialize algorithm
        self.algs = {}
        self.alg_cfg.pop("class_name")
        for body_key in self.body_keys:
            self.algs[body_key] = PPO(
                policy=self.policies[body_key],
                device=self.device,
                **self.alg_cfg,
                multi_gpu_cfg=self.multi_gpu_cfg
            )

        # initialize storage
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=[self.num_obs["actor_obs"]], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.num_obs["critic_obs"]], until=1.0e8).to(
                self.device
            )
        else:
            self.actor_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        for body_key in self.body_keys:
            if body_key == "upper_body":
                self.algs[body_key].init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [self.num_obs["upper_body_actor_obs"]],
                [self.num_obs["upper_body_critic_obs"]],
                [self.num_actions[body_key]],
            )
            else:
                self.algs[body_key].init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [self.num_obs["lower_body_actor_obs"]],
                [self.num_obs["lower_body_critic_obs"]],
                [self.num_actions[body_key]],
            )

    def __rollout_step(self,
                       upper_body_actor_obs,
                       upper_body_critic_obs,
                       lower_body_actor_obs,
                       lower_body_critic_obs,
                       ep_infos, 
                       cur_episode_length, 
                       cur_reward_sum, 
                       cur_reward_sum_dict, 
                       rewbuffer, 
                       rewbuffer_dict, 
                       lenbuffer):
        action_dict = {}
        for key in self.body_keys:
            if key == "upper_body":
                action_dict[key] = self.algs[key].act(upper_body_actor_obs, upper_body_critic_obs)
            else:
                action_dict[key] = self.algs[key].act(lower_body_actor_obs, lower_body_critic_obs)
        # Step the environment
        #action_dict["upper_body"] = torch.zeros(self.env.num_envs, 14, device=self.device) # TODO: remove this
        action = torch.cat([action_dict["upper_body"], action_dict["lower_body"]], dim=1)
        assert action_dict["upper_body"].shape[1] == 14, "Upper body should have 14 actions"
        assert action_dict["lower_body"].shape[1] == 15, "Lower body should have 15 actions"
        assert action.shape[1] == 29, "Total actions should be 29"

        obs, rewards, dones, infos = self.env.step(action.to(self.env.device))
        upper_body_actor_obs, upper_body_critic_obs = obs["upper_body_actor_obs"], obs["upper_body_critic_obs"]
        lower_body_actor_obs, lower_body_critic_obs = obs["lower_body_actor_obs"], obs["lower_body_critic_obs"]
        # Move to device
        upper_body_actor_obs, upper_body_critic_obs = (upper_body_actor_obs.to(self.device), upper_body_critic_obs.to(self.device))
        lower_body_actor_obs, lower_body_critic_obs = (lower_body_actor_obs.to(self.device), lower_body_critic_obs.to(self.device))
        dones = dones.to(self.device)
        rewards = {key: rewards[key].to(self.device) for key in self.body_keys}
        # perform normalization
        upper_body_actor_obs = self.actor_obs_normalizer(upper_body_actor_obs)
        upper_body_critic_obs = self.critic_obs_normalizer(upper_body_critic_obs)
        lower_body_actor_obs = self.actor_obs_normalizer(lower_body_actor_obs)
        lower_body_critic_obs = self.critic_obs_normalizer(lower_body_critic_obs)
        # process the step
        for key in self.body_keys:
            self.algs[key].process_env_step(rewards[key], dones, infos)
        
        # book keeping
        if self.log_dir is not None:
            if "episode" in infos:
                ep_infos.append(infos["episode"])
            elif "log" in infos:
                ep_infos.append(infos["log"])
            # Update rewards
            for key in self.body_keys:
                cur_reward_sum_dict[key] += rewards[key]
                cur_reward_sum += rewards[key]
            # Update episode length
            cur_episode_length += 1
            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            for key in self.body_keys:
                rewbuffer_dict[key].extend(cur_reward_sum_dict[key][new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0
            for key in self.body_keys:
                cur_reward_sum_dict[key][new_ids] = 0

        return upper_body_actor_obs, upper_body_critic_obs, lower_body_actor_obs, lower_body_critic_obs
    


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, _ = self.env.get_observations()
        upper_body_actor_obs, upper_body_critic_obs = obs["upper_body_actor_obs"], obs["upper_body_critic_obs"]
        lower_body_actor_obs, lower_body_critic_obs = obs["lower_body_actor_obs"], obs["lower_body_critic_obs"]
        upper_body_actor_obs, upper_body_critic_obs = upper_body_actor_obs.to(self.device), upper_body_critic_obs.to(self.device)
        lower_body_actor_obs, lower_body_critic_obs = lower_body_actor_obs.to(self.device), lower_body_critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rewbuffer_dict = {key: deque(maxlen=100) for key in self.body_keys}
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum_dict = {key: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) for key in self.body_keys}
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)


        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    upper_body_actor_obs, upper_body_critic_obs, lower_body_actor_obs, lower_body_critic_obs = self.__rollout_step(upper_body_actor_obs, upper_body_critic_obs, lower_body_actor_obs, lower_body_critic_obs, ep_infos, cur_episode_length, cur_reward_sum, cur_reward_sum_dict, rewbuffer, rewbuffer_dict, lenbuffer)

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    for key in self.body_keys:
                        if key == "upper_body":
                            self.algs[key].compute_returns(upper_body_critic_obs)
                        else:
                            self.algs[key].compute_returns(lower_body_critic_obs)

            # update policy
            loss_dict = {}
            for key in self.body_keys:
                loss_dict[key] = self.algs[key].update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        
        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses (Multi-Actor-Critic)
        for body_key in self.body_keys:
            body_loss_dict = locs["loss_dict"][body_key]
            for loss_name, loss_value in body_loss_dict.items():
                self.writer.add_scalar(f"Loss/{loss_name}_{body_key}", loss_value, locs["it"])
            # Learning rate for each body part
            self.writer.add_scalar(f"Loss/learning_rate_{body_key}", self.algs[body_key].learning_rate, locs["it"])

        # -- Policy (Multi-Actor-Critic)
        for body_key in self.body_keys:
            mean_std = self.policies[body_key].action_std.mean()
            self.writer.add_scalar(f"Policy/mean_noise_std_{body_key}", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training (Total and Per Body Part)
        if len(locs["rewbuffer"]) > 0:
            # Total rewards
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)
            
            # Per body part rewards
            for body_key in self.body_keys:
                if len(locs["rewbuffer_dict"][body_key]) > 0:
                    mean_reward = statistics.mean(locs["rewbuffer_dict"][body_key])
                    self.writer.add_scalar(f"Train/mean_reward_{body_key}", mean_reward, locs["it"])
                    if self.logger_type != "wandb":
                        self.writer.add_scalar(f"Train/mean_reward_{body_key}/time", mean_reward, self.tot_time)

        # -- Console logging
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            
            # -- Action noise std for each body part
            for body_key in self.body_keys:
                mean_std = self.policies[body_key].action_std.mean()
                log_string += f"""{'Mean action noise std (' + body_key + '):':>{pad}} {mean_std.item():.2f}\n"""
            
            # -- Losses for each body part
            for body_key in self.body_keys:
                body_loss_dict = locs["loss_dict"][body_key]
                for loss_name, loss_value in body_loss_dict.items():
                    log_string += f"""{f'Mean {loss_name} loss ({body_key}):':>{pad}} {loss_value:.4f}\n"""
            
            # -- Total reward
            log_string += f"""{'Mean total reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            
            # -- Per body part rewards
            for body_key in self.body_keys:
                if len(locs["rewbuffer_dict"][body_key]) > 0:
                    mean_reward = statistics.mean(locs["rewbuffer_dict"][body_key])
                    log_string += f"""{'Mean reward (' + body_key + '):':>{pad}} {mean_reward:.2f}\n"""
            
            # -- Episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            
            # -- Action noise std for each body part
            for body_key in self.body_keys:
                mean_std = self.policies[body_key].action_std.mean()
                log_string += f"""{'Mean action noise std (' + body_key + '):':>{pad}} {mean_std.item():.2f}\n"""
            
            # -- Losses for each body part
            for body_key in self.body_keys:
                body_loss_dict = locs["loss_dict"][body_key]
                for loss_name, loss_value in body_loss_dict.items():
                    log_string += f"""{f'{loss_name} ({body_key}):':>{pad}} {loss_value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                                locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {}
        # save model
        saved_dict.update({
            f"model_state_dict_{key}": self.algs[key].policy.state_dict()
            for key in self.body_keys
        })
        # save optimizer
        saved_dict.update({
            f"optimizer_state_dict_{key}": self.algs[key].optimizer.state_dict()
            for key in self.body_keys
        })
        # save iteration
        saved_dict["iter"] = self.current_learning_iteration
        # save infos
        saved_dict["infos"] = infos
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["actor_obs_norm_state_dict"] = self.actor_obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = True
        for body_key in self.body_keys:
            try:
                self.algs[body_key].policy.load_state_dict(loaded_dict[f"model_state_dict_{body_key}"])
            except:
                resumed_training = False
                break
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.actor_obs_normalizer.load_state_dict(loaded_dict["actor_obs_norm_state_dict"])
                self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                try:
                    self.critic_obs_normalizer.load_state_dict(loaded_dict["actor_obs_norm_state_dict"])
                except:
                    pass
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            for body_key in self.body_keys:
                self.algs[body_key].optimizer.load_state_dict(loaded_dict[f"optimizer_state_dict_{body_key}"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode
        
        # Move all policies to device if specified
        if device is not None:
            for body_key in self.body_keys:
                self.algs[body_key].policy.to(device)
            if self.empirical_normalization:
                self.actor_obs_normalizer.to(device)
        
        def multi_actor_inference_policy(upper_body_actor_obs, lower_body_actor_obs):
            # Apply normalization if enabled
            if self.empirical_normalization:
                upper_body_actor_obs = self.actor_obs_normalizer(upper_body_actor_obs)
                lower_body_actor_obs = self.actor_obs_normalizer(lower_body_actor_obs)
            
            # Get actions from each body part
            actions_list = []
            #actions_list.append(torch.zeros(self.env.num_envs, 14, device=self.device)) # TODO: remove this
            actions_list.append(self.algs["upper_body"].policy.act_inference(upper_body_actor_obs))
            actions_list.append(self.algs["lower_body"].policy.act_inference(lower_body_actor_obs))
            
            # Concatenate actions (same order as training)
            combined_actions = torch.cat(actions_list, dim=1)
            return combined_actions
        
        return multi_actor_inference_policy

    def train_mode(self):
        # -- PPO
        for body_key in self.body_keys:
            self.algs[body_key].policy.train()
        # -- Normalization
        if self.empirical_normalization:
            self.actor_obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        for body_key in self.body_keys:
            self.algs[body_key].policy.eval()
        # -- Normalization
        if self.empirical_normalization:
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
