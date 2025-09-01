# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.storage import RolloutStorage
from .student_teacher_encode import StudentTeacherEncoder
from .distill_storage import DistillationRolloutStorage


class DistillationEncoder:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacherEncoder
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        gradient_length=1,
        learning_rate=1e-4,
        loss_type="mse",
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.student_encoder.parameters(), lr=learning_rate)
        self.transition = DistillationRolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape, encoded_obs_shape
    ):
        # create rollout storage
        base_storage = RolloutStorage(
            training_type="distillation",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=student_obs_shape,
            privileged_obs_shape=teacher_obs_shape,
            actions_shape=actions_shape,
            rnd_state_shape=None,
            device=self.device,
        )
        self.storage = DistillationRolloutStorage(
            base_storage=base_storage,
            encoded_obs_shape=encoded_obs_shape,
            device=self.device,
        )

    def act(self, obs, teacher_obs):
        # compute the actions and encoded observations
        student_actions, student_encoded_obs = self.policy.act_inference(obs)
        teacher_actions, teacher_encoded_obs = self.policy.evaluate(teacher_obs)

        # record
        self.transition.actions = student_actions.detach()
        self.transition.privileged_actions = teacher_actions.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        self.transition.encoded_observations = student_encoded_obs.detach()
        self.transition.privileged_encoded_observations = teacher_encoded_obs.detach()
        return self.transition.actions


    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, encoded_obs, privileged_encoded_obs, dones in self.storage.generator():

                # inference the student for gradient computation
                actions, student_encoded_obs = self.policy.act_inference(obs)
                

                # behavior cloning loss NOTE: also include l2 of actions
                behavior_loss = self.loss_fn(actions, privileged_actions) + self.loss_fn(student_encoded_obs, privileged_encoded_obs)

                # total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
