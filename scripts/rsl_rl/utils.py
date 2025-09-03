
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

def compute_metrics(env):
    """Compute metrics for the plate.
    Metrics:
    - Projected gravity on the xy plane
    - Linear acceleration
    - Angular acceleration
    """
    plate_index = env.unwrapped.robot.data.body_names.index(env.unwrapped.cfg.plate_name)

    # Use the observation functions for consistency
    # compute projected gravity on the xy plane
    plate_body_quat_w = env.unwrapped.robot.data.body_link_quat_w[:, plate_index, :]
    gravity_vec_w = env.unwrapped.robot.data.GRAVITY_VEC_W
    projected_gravity = quat_apply_inverse(plate_body_quat_w, gravity_vec_w)
    projected_gravity_xy = projected_gravity[:, :2]
    projected_gravity_xy_norm = torch.norm(projected_gravity_xy, dim=-1)

    # Compute linear velocity
    lin_vel = env.unwrapped.robot.data.body_lin_vel_w[:, plate_index, :]
    lin_vel_norm = torch.norm(lin_vel, dim=-1)

    # compute angular velocity
    ang_vel = env.unwrapped.robot.data.body_ang_vel_w[:, plate_index, :]
    ang_vel_norm = torch.norm(ang_vel, dim=-1)

    # compute linear acceleration
    linear_acc_vec = env.unwrapped.robot.data.body_lin_acc_w[:, plate_index, :]
    linear_acc = torch.norm(linear_acc_vec, dim=-1)

    # compute angular acceleration
    angular_acc_vec = env.unwrapped.robot.data.body_ang_acc_w[:, plate_index, :]
    angular_acc = torch.norm(angular_acc_vec, dim=-1)


    # Use the observation functions for consistency
    # compute rigid body linear velocity
    rigid_lin_vel = env.unwrapped._object.data.body_lin_vel_w[:, 0, :]
    rigid_lin_vel_norm = torch.norm(rigid_lin_vel, dim=-1)

    # compute rigid body angular velocity
    rigid_ang_vel = env.unwrapped._object.data.body_ang_vel_w[:, 0, :]
    rigid_ang_vel_norm = torch.norm(rigid_ang_vel, dim=-1)

    # compute rigid body projected gravity
    object_body_quat_w = env.unwrapped._object.data.body_link_quat_w[:, 0, :]
    gravity_vec_w = env.unwrapped.robot.data.GRAVITY_VEC_W
    rigid_projected_gravity = quat_apply_inverse(object_body_quat_w, gravity_vec_w)
    rigid_projected_gravity_xy = rigid_projected_gravity[:, :2]
    rigid_projected_gravity_xy_norm = torch.norm(rigid_projected_gravity_xy, dim=-1)

    # create a dictionary to store the metrics
    metrics = {
        "Plate_Projected_Gravity_XY": projected_gravity_xy_norm,
        "Plate_Linear_Velocity": lin_vel_norm,
        "Plate_Angular_Velocity": ang_vel_norm,
        "Plate_Linear_Acceleration": linear_acc,
        "Plate_Angular_Acceleration": angular_acc,
        "Object_Linear_Velocity": rigid_lin_vel_norm,
        "Object_Angular_Velocity": rigid_ang_vel_norm,
        "Object_Projected_Gravity_XY": rigid_projected_gravity_xy_norm,
    }

    return metrics





#!/usr/bin/env python3

"""
Comprehensive robot data logger for Isaac Lab environments.
This module provides logging capabilities for robot state data during play sessions.
"""

import os
import json
import pickle
import numpy as np
import torch
import datetime
from datetime import datetime
from typing import Optional, Dict, Any


from isaaclab.managers import SceneEntityCfg
from g1_humanoid.tasks.residual_adaptive import mdp


class RobotDataLogger:
    """Comprehensive data logger for robot state during play."""

    def __init__(self, log_dir: str, experiment_name: str, save_interval: int = 1):
        """
        Initialize the robot data logger.
        
        Args:
            log_dir: Directory where logs will be saved
            experiment_name: Name of the experiment
            save_interval: Save data every N steps
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name + f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.save_interval = save_interval
        self.data_log_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.data_log_dir, exist_ok=True)
        
        # Initialize data storage
        self.reset_data_storage()
        
        # Create metadata
        self.metadata = {
            "creation_time": datetime.now().isoformat(),
            "save_interval": save_interval,
            "description": "Comprehensive robot state data during play",
            "sim_dt": 0.005,  # Default simulation timestep
            "decimation": 4,  # Default decimation factor
        }
        
        print(f"[INFO] Robot data will be saved to: {self.data_log_dir}")
    
    def reset_data_storage(self):
        """Reset all data storage containers."""
        self.data = {
            # Basic state data
            "timesteps": [],
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            
            # Robot state data
            "robot_states": {
                "base_position": [],
                "base_orientation": [],
                "base_lin_vel": [],
                "base_ang_vel": [],
                "base_lin_acc": [],
                "base_ang_acc": [],
                "joint_positions": [],
                "joint_velocities": [],
                "joint_efforts": [],
            },
            
            # Plate-specific data
            "plate_data": {
                "position": [],
                "orientation": [],
                "lin_vel": [],
                "ang_vel": [],
                "lin_acc": [],
                "ang_acc": [],
                "projected_gravity": [],
            },
            
            # Object data (cylinder)
            "object_data": {
                "position": [],
                "orientation": [],
                "lin_vel": [],
                "ang_vel": [],
                "lin_acc": [],
                "ang_acc": [],
                "projected_gravity": [],
                "deviation_from_plate": [],
            },
            
            # Command data
            "commands": {
                "base_velocity": [],
            },
            
            # Event data
            "events": {},
            
            # Computed metrics
            "metrics": {},

            # Termination conditions
            "termination": {}
        }
    
    def log_step_data(self, env, obs, actions, rewards, dones, timestep, metrics=None):
        """
        Log comprehensive data for a single step.
        
        Args:
            env: The environment instance
            obs: Observations from the environment
            actions: Actions taken by the agent
            rewards: Rewards received
            dones: Done flags
            timestep: Current timestep
            metrics: Optional computed metrics dictionary
        """
        try:
            # Basic data
            self.data["timesteps"].append(timestep)
            # self.data["observations"].append(obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs)
            self.data["actions"].append(actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions)
            self.data["rewards"].append(rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards)
            self.data["dones"].append(dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones)
            
            # Robot state data
            robot = env.scene["robot"]
            
            # Base state
            self.data["robot_states"]["base_position"].append(robot.data.root_pos_w.cpu().numpy())
            self.data["robot_states"]["base_orientation"].append(robot.data.root_quat_w.cpu().numpy())
            self.data["robot_states"]["base_lin_vel"].append(robot.data.root_lin_vel_w.cpu().numpy())
            self.data["robot_states"]["base_ang_vel"].append(robot.data.root_ang_vel_w.cpu().numpy())
            
            # Base acceleration (from root center of mass)
            root_com_acc = robot.data.body_com_acc_w[:, 0, :]  # First body is typically the root/torso
            self.data["robot_states"]["base_lin_acc"].append(root_com_acc[:, :3].cpu().numpy())  # Linear acceleration
            self.data["robot_states"]["base_ang_acc"].append(root_com_acc[:, 3:].cpu().numpy())  # Angular acceleration

            # Joint state
            self.data["robot_states"]["joint_positions"].append(robot.data.joint_pos.cpu().numpy())
            self.data["robot_states"]["joint_velocities"].append(robot.data.joint_vel.cpu().numpy())
            if hasattr(robot.data, 'joint_effort'):
                self.data["robot_states"]["joint_efforts"].append(robot.data.joint_effort.cpu().numpy())
            
            # Plate-specific data
            self._log_plate_data(env)
            
            # Object data (if cylinder exists)
            if "cylinder" in env.scene.keys():
                self._log_object_data(env)
            
            # Command data
            self._log_command_data(env)
            
            # Metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.data["metrics"]:
                        self.data["metrics"][metric_name] = []
                    self.data["metrics"][metric_name].append(
                        metric_value.cpu().numpy() if isinstance(metric_value, torch.Tensor) else metric_value
                    )

            # Check termination conditions
            self._log_termination_conditions(env)

            # Event data
            self._log_event(env)

        except Exception as e:
            print(f"[WARNING] Failed to log step data: {e}")
    
    def _log_plate_data(self, env):
        """Log plate-specific data."""
        try:
            asset_cfg = SceneEntityCfg("robot", body_names="plate")
            asset_cfg.resolve(env.scene)
            
            # Get plate body data
            robot = env.scene["robot"]
            
            # Position and orientation (use body index from resolved config)
            try:
                if hasattr(asset_cfg, 'body_ids'):
                    plate_body_idx = asset_cfg.body_ids[0]
                    self.data["plate_data"]["position"].append(
                        robot.data.body_pos_w[:, plate_body_idx].cpu().numpy()
                    )
                    self.data["plate_data"]["orientation"].append(
                        robot.data.body_quat_w[:, plate_body_idx].cpu().numpy()
                    )
            except Exception as e:
                print(f"[WARNING] Could not log plate position/orientation: {e}")
            
            # Velocities
            plate_lin_vel = mdp.body_lin_vel(env, asset_cfg)
            plate_ang_vel = mdp.body_ang_vel(env, asset_cfg)
            self.data["plate_data"]["lin_vel"].append(plate_lin_vel.cpu().numpy())
            self.data["plate_data"]["ang_vel"].append(plate_ang_vel.cpu().numpy())
            
            # Accelerations
            plate_lin_acc = mdp.body_lin_acc(env, asset_cfg)
            plate_ang_acc = mdp.body_ang_acc(env, asset_cfg)
            self.data["plate_data"]["lin_acc"].append(plate_lin_acc.cpu().numpy())
            self.data["plate_data"]["ang_acc"].append(plate_ang_acc.cpu().numpy())
            
            # Projected gravity
            plate_proj_grav = mdp.body_projected_gravity(env, asset_cfg)
            self.data["plate_data"]["projected_gravity"].append(plate_proj_grav.cpu().numpy())
            
        except Exception as e:
            print(f"[WARNING] Failed to log plate data: {e}")
    
    def _log_object_data(self, env):
        """Log object (cylinder) data."""
        try:
            object_cfg = SceneEntityCfg("cylinder")
            object_cfg.resolve(env.scene)
            
            obj = env.scene["cylinder"]
            
            # Position and orientation
            self.data["object_data"]["position"].append(obj.data.root_pos_w.cpu().numpy())
            self.data["object_data"]["orientation"].append(obj.data.root_quat_w.cpu().numpy())
            
            # Velocities
            obj_lin_vel = mdp.rigid_body_lin_vel(env, object_cfg)
            obj_ang_vel = mdp.rigid_body_ang_vel(env, object_cfg)
            self.data["object_data"]["lin_vel"].append(obj_lin_vel.cpu().numpy())
            self.data["object_data"]["ang_vel"].append(obj_ang_vel.cpu().numpy())

            # Accelerations
            obj_lin_acc = mdp.rigid_body_lin_acc(env, object_cfg)
            obj_ang_acc = mdp.rigid_body_ang_acc(env, object_cfg)
            self.data["object_data"]["lin_acc"].append(obj_lin_acc.cpu().numpy())
            self.data["object_data"]["ang_acc"].append(obj_ang_acc.cpu().numpy())

            # Projected gravity
            obj_proj_grav = mdp.rigid_body_projected_gravity(env, object_cfg)
            self.data["object_data"]["projected_gravity"].append(obj_proj_grav.cpu().numpy())
            
            # Deviation from plate
            robot_cfg = SceneEntityCfg("robot", body_names="plate")
            robot_cfg.resolve(env.scene)
            obj_deviation = mdp.object_deviration_pos(env, object_cfg, robot_cfg)
            self.data["object_data"]["deviation_from_plate"].append(obj_deviation.cpu().numpy())
            
        except Exception as e:
            print(f"[WARNING] Failed to log object data: {e}")
    
    def _log_command_data(self, env):
        """Log command data."""
        try:
            if hasattr(env, 'command_manager') and env.command_manager is not None:
                commands = env.command_manager.get_command("base_velocity")
                self.data["commands"]["base_velocity"].append(commands.cpu().numpy())
        except Exception as e:
            print(f"[WARNING] Failed to log command data: {e}")

    def _log_termination_conditions(self, env):
        """Check and log any termination conditions.
        To make sure the experiment is fair, we do not add any termination into the environment. So we need to check the termination conditions manually here.
        """
        try:
            # Initialize termination lists if they don't exist
            if "bad_orientation" not in self.data["termination"]:
                self.data["termination"]["bad_orientation"] = []
            if "root_height_below_minimum" not in self.data["termination"]:
                self.data["termination"]["root_height_below_minimum"] = []
            if "link_height_below_minimum" not in self.data["termination"]:
                self.data["termination"]["link_height_below_minimum"] = []
            if "object_bad_orientation" not in self.data["termination"]:
                self.data["termination"]["object_bad_orientation"] = []
            if "failed" not in self.data["termination"]:
                self.data["termination"]["failed"] = []

            # Check for bad orientation
            self.data["termination"]["bad_orientation"].append(
                mdp.bad_orientation(env, limit_angle=0.8, asset_cfg=SceneEntityCfg("robot")).cpu().numpy()
            )
            
            # Check for root height below minimum
            self.data["termination"]["root_height_below_minimum"].append(
                mdp.root_height_below_minimum(env, minimum_height=0.2).cpu().numpy()
            )

            # Check object height below minimum
            self.data["termination"]["link_height_below_minimum"].append(
                mdp.link_height_below_minimum(env, minimum_height=0.5, asset_cfg=SceneEntityCfg("cylinder")).cpu().numpy()
            )

            # Check object bad orientation
            self.data["termination"]["object_bad_orientation"].append(
                mdp.bad_orientation(env, limit_angle=0.8, asset_cfg=SceneEntityCfg("cylinder")).cpu().numpy()
            )

            # Check if any termination condition is met for current step (per environment)
            current_terminations_per_env = (
                self.data["termination"]["bad_orientation"][-1] |
                self.data["termination"]["root_height_below_minimum"][-1] |
                self.data["termination"]["link_height_below_minimum"][-1] |
                self.data["termination"]["object_bad_orientation"][-1]
            )

            self.data["termination"]["failed"].append(current_terminations_per_env)

        except Exception as e:
            print(f"[WARNING] Failed to check termination conditions: {e}")
    
    def _log_event(self, env):
        """Log any events that occur during the play session."""
        try:
            if hasattr(env, 'event_manager') and env.event_manager is not None:
                events = env.event_manager.term_return_values
                if events:
                    # events is a dict[str, tensor], so we need to process it properly
                    for event_name, event_tensor in events.items():
                        if event_name not in self.data["events"]:
                            self.data["events"][event_name] = []
                        self.data["events"][event_name].append(
                            event_tensor.cpu().numpy() if isinstance(event_tensor, torch.Tensor) else event_tensor
                        )
        except Exception as e:
            print(f"[WARNING] Failed to log events: {e}")


    def save_data(self, timestep):
        """Save data to files at specified intervals."""
        if timestep % self.save_interval == 0:
            try:
                # Save as pickle for fast loading
                pickle_file = os.path.join(self.data_log_dir, f"robot_data_step_{timestep}.pkl")
                with open(pickle_file, 'wb') as f:
                    pickle.dump(self.data, f)
                
                # Save metadata as JSON
                metadata_file = os.path.join(self.data_log_dir, "metadata.json")
                self.metadata["last_timestep"] = timestep
                self.metadata["total_steps"] = len(self.data["timesteps"])
                with open(metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                    
            except Exception as e:
                print(f"[WARNING] Failed to save data at timestep {timestep}: {e}")
    
    def save_final_data(self):
        """Save final comprehensive data."""
        try:
            # Save complete dataset
            final_file = os.path.join(self.data_log_dir, "complete_robot_data.pkl")
            with open(final_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Save as compressed numpy arrays for easier analysis
            npz_file = os.path.join(self.data_log_dir, "robot_data_arrays.npz")
            np_data = {}
            
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    elif isinstance(v, list) and len(v) > 0:
                        try:
                            items.append((new_key, np.array(v)))
                        except:
                            pass  # Skip non-numeric data
                return dict(items)
            
            np_data = flatten_dict(self.data)
            np.savez_compressed(npz_file, **np_data)
            
            print(f"[INFO] Final robot data saved to {self.data_log_dir}")
            print(f"[INFO] Total steps logged: {len(self.data['timesteps'])}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save final data: {e}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the logged data."""
        summary = {
            "total_timesteps": len(self.data["timesteps"]),
            "data_categories": list(self.data.keys()),
            "metrics_available": list(self.data["metrics"].keys()) if self.data["metrics"] else [],
            "has_plate_data": bool(self.data["plate_data"]["lin_vel"]),
            "has_object_data": bool(self.data["object_data"]["lin_vel"]),
            "has_robot_states": bool(self.data["robot_states"]["base_position"]),
            "save_interval": self.save_interval,
            "log_directory": self.data_log_dir
        }
        return summary
    
    def export_to_csv(self, output_dir: Optional[str] = None):
        """Export basic data to CSV format for easy analysis."""
        import pandas as pd
        
        if output_dir is None:
            output_dir = self.data_log_dir
        
        try:
            # Basic timestep data
            basic_df = pd.DataFrame({"timestep": self.data["timesteps"]})
            basic_df.to_csv(os.path.join(output_dir, "timesteps.csv"), index=False)
            
            # Metrics data (if available)
            if self.data["metrics"]:
                metrics_data = {"timestep": self.data["timesteps"]}
                for metric_name, values in self.data["metrics"].items():
                    if len(values) == len(self.data["timesteps"]):
                        if isinstance(values[0], np.ndarray) and values[0].ndim > 0:
                            metrics_data[metric_name] = [np.mean(v) for v in values]
                        else:
                            metrics_data[metric_name] = values
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
            
            print(f"[INFO] CSV data exported to {output_dir}")
            
        except Exception as e:
            print(f"[ERROR] Failed to export CSV data: {e}")