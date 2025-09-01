# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage

class DistillationRolloutStorage:
    """
    Wrapper around RolloutStorage specifically designed for distillation training.
    Adds support for storing and retrieving encoded observations from both student and teacher.
    """
    
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            # New fields for distillation
            self.encoded_observations = None
            self.privileged_encoded_observations = None
            # Other fields that might be needed
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        base_storage: RolloutStorage,
        encoded_obs_shape: int,
        device="cpu",
    ):
        """
        Initialize the distillation storage wrapper.
        
        Args:
            base_storage: The original RolloutStorage instance
            encoded_obs_shape: Shape of the encoded observations (e.g., (32,) for 32-dim encoder output)
            device: Device to store tensors on
        """
        self.base_storage = base_storage
        self.device = device
        self.encoded_obs_shape = encoded_obs_shape
        
        # Create additional storage for encoded observations
        num_transitions = base_storage.num_transitions_per_env
        num_envs = base_storage.num_envs
        
        self.encoded_observations = torch.zeros(
            num_transitions, num_envs, *encoded_obs_shape, device=self.device
        )
        self.privileged_encoded_observations = torch.zeros(
            num_transitions, num_envs, *encoded_obs_shape, device=self.device
        )
        
        # Create transition object
        self.transition = self.Transition()

    @property
    def step(self):
        return self.base_storage.step

    @property
    def num_transitions_per_env(self):
        return self.base_storage.num_transitions_per_env
    
    @property
    def num_envs(self):
        return self.base_storage.num_envs

    def add_transitions(self, transition: Transition):
        """Add transition to both base storage and encoded observation storage."""
        # Add to base storage (this handles all the original fields)
        self.base_storage.add_transitions(transition)
        
        # Add encoded observations to our extended storage
        current_step = self.base_storage.step - 1  # step is incremented in base_storage
        if transition.encoded_observations is not None:
            self.encoded_observations[current_step].copy_(transition.encoded_observations)
        if transition.privileged_encoded_observations is not None:
            self.privileged_encoded_observations[current_step].copy_(transition.privileged_encoded_observations)

    def clear(self):
        """Clear both base storage and encoded observation storage."""
        self.base_storage.clear()

    def generator(self):
        """
        Generator for distillation training that yields:
        (observations, privileged_observations, actions, privileged_actions, 
         encoded_observations, privileged_encoded_observations, dones)
        """
        if self.base_storage.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.base_storage.privileged_observations is not None:
                privileged_observations = self.base_storage.privileged_observations[i]
            else:
                privileged_observations = self.base_storage.observations[i]
            
            yield (
                self.base_storage.observations[i],           # student observations
                privileged_observations,                     # teacher observations  
                self.base_storage.actions[i],               # student actions
                self.base_storage.privileged_actions[i],    # teacher actions
                self.encoded_observations[i],               # student encoded observations
                self.privileged_encoded_observations[i],    # teacher encoded observations
                self.base_storage.dones[i]                  # dones
            )

    def batch_generator(self, batch_size=None):
        """
        Alternative generator that yields batches instead of individual timesteps.
        Useful for more efficient training.
        """
        if batch_size is None:
            batch_size = self.num_envs * self.num_transitions_per_env
            
        # Flatten all data
        observations = self.base_storage.observations.flatten(0, 1)
        if self.base_storage.privileged_observations is not None:
            privileged_observations = self.base_storage.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations
            
        actions = self.base_storage.actions.flatten(0, 1)
        privileged_actions = self.base_storage.privileged_actions.flatten(0, 1)
        encoded_observations = self.encoded_observations.flatten(0, 1)
        privileged_encoded_observations = self.privileged_encoded_observations.flatten(0, 1)
        dones = self.base_storage.dones.flatten(0, 1)
        
        # Create random indices
        total_samples = observations.shape[0]
        indices = torch.randperm(total_samples, device=self.device)
        
        # Yield batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                observations[batch_indices],
                privileged_observations[batch_indices],
                actions[batch_indices],
                privileged_actions[batch_indices],
                encoded_observations[batch_indices],
                privileged_encoded_observations[batch_indices],
                dones[batch_indices]
            )

    # Delegate other methods to base storage
    def __getattr__(self, name):
        """Delegate any missing methods/attributes to the base storage."""
        return getattr(self.base_storage, name)


def create_distillation_storage(
    num_envs,
    num_transitions_per_env,
    student_obs_shape,
    teacher_obs_shape,
    actions_shape,
    encoded_obs_shape,
    device="cpu",
):
    """
    Convenience function to create a distillation storage setup.
    
    Args:
        num_envs: Number of environments
        num_transitions_per_env: Number of transitions per environment
        student_obs_shape: Shape of student observations
        teacher_obs_shape: Shape of teacher observations (privileged)
        actions_shape: Shape of actions
        encoded_obs_shape: Shape of encoded observations (encoder output)
        device: Device to store tensors on
    
    Returns:
        DistillationRolloutStorage instance
    """
    
    # Create base storage
    base_storage = RolloutStorage(
        training_type="distillation",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions_per_env,
        obs_shape=student_obs_shape,
        privileged_obs_shape=teacher_obs_shape,
        actions_shape=actions_shape,
        device=device,
    )
    
    # Wrap it with distillation storage
    distillation_storage = DistillationRolloutStorage(
        base_storage=base_storage,
        encoded_obs_shape=encoded_obs_shape,
        device=device,
    )
    
    return distillation_storage