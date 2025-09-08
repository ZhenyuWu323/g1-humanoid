# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from ..encoder import TransformerEncoder

class StudentTeacherEncoder(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        num_student_encoder_obs,
        num_teacher_encoder_obs,
        num_time_steps,
        num_encoder_output,
        actor_hidden_dims=[256, 256, 256],
        teacher_encoder_dim=[128,8,1], # d_model, nhead, num_layers
        student_encoder_dim=[128,8,1], # d_model, nhead, num_layers
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded
        self.num_actor_obs = num_actor_obs
        self.num_time_steps = num_time_steps
        self.num_student_encoder_obs = num_student_encoder_obs
        self.num_teacher_encoder_obs = num_teacher_encoder_obs
        
        
        # Policy
        mlp_input_dim_a = num_actor_obs + num_encoder_output
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        self.actor.eval()
        for param in self.actor.parameters():
            param.requires_grad = False


        # Teacher Enco
        teacher_encoder_d_model, teacher_encoder_nhead, teacher_encoder_num_layers = teacher_encoder_dim
        self.teacher_encoder = TransformerEncoder(num_teacher_encoder_obs, teacher_encoder_d_model, teacher_encoder_nhead, teacher_encoder_num_layers, num_encoder_output, num_time_steps)
        self.teacher_encoder.eval()
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

        # Student Encoder
        student_encoder_d_model, student_encoder_nhead, student_encoder_num_layers = student_encoder_dim
        self.student_encoder = TransformerEncoder(num_student_encoder_obs, student_encoder_d_model, student_encoder_nhead, student_encoder_num_layers, num_encoder_output, num_time_steps)

        print(f"Actor MLP: {self.actor}")
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student Encoder: {self.student_encoder}")



    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    
    
    def _prepare_student_input(self, observations):

        # extract encoder observations
        batch_size = observations.shape[0]
        student_encoder_obs_dim = self.num_student_encoder_obs * self.num_time_steps
        encoder_obs = observations[:, -student_encoder_obs_dim:]  # (B, 105)
        encoder_obs = encoder_obs.view(batch_size, self.num_time_steps, self.num_student_encoder_obs)
        encoded_obs = self.student_encoder(encoder_obs)  # (B, 32)
        
        # extract regular observations
        regular_obs = observations[:, :-student_encoder_obs_dim]  # (B, regular_dim)
        
        # concatenate
        return regular_obs, encoded_obs

    def _prepare_teacher_input(self, observations):
        # extract encoder observations
        batch_size = observations.shape[0]
        teacher_encoder_obs_dim = self.num_teacher_encoder_obs * self.num_time_steps
        encoder_obs = observations[:, -teacher_encoder_obs_dim:]  # (B, 105)
        encoder_obs = encoder_obs.view(batch_size, self.num_time_steps, self.num_teacher_encoder_obs)
        encoded_obs = self.teacher_encoder(encoder_obs)  # (B, 32)
        
        # extract regular observations
        regular_obs = observations[:, :-teacher_encoder_obs_dim]  # (B, regular_dim)
        
        # concatenate
        return regular_obs, encoded_obs


    def act_inference(self, observations):
        #self.update_distribution(observations)
        regular_obs, encoded_obs = self._prepare_student_input(observations)
        actions = self.actor(torch.cat([regular_obs, encoded_obs], dim=-1))
        return actions, encoded_obs

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            regular_obs, encoded_obs = self._prepare_teacher_input(teacher_observations)
            actions = self.actor(torch.cat([regular_obs, encoded_obs], dim=-1))
        return actions, encoded_obs

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                        module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("student_encoder" in key for key in state_dict.keys()):  # loading parameters from distillation training
            # Resume distillation training - load everything
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher_encoder.eval()
            return True
            
        elif any("encoder" in key for key in state_dict.keys()):  # loading parameters from rl training
            # Start distillation training - load teacher encoder + actor
            
            # 1. Load teacher encoder
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "encoder." in key:
                    teacher_state_dict[key.replace("encoder.", "")] = value
            self.teacher_encoder.load_state_dict(teacher_state_dict, strict=strict)
            
            # 2. Load actor
            actor_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    actor_state_dict[key] = value
            
            # Load actor parameters into current model
            current_state_dict = self.state_dict()
            current_state_dict.update(actor_state_dict)
            super().load_state_dict(current_state_dict, strict=False)
            
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher_encoder.eval()
            return False
            
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
