#!/usr/bin/env python3

"""
Script to analyze and visualize robot state data collected during play sessions.
This script processes the comprehensive robot data logs and generates various plots and analysis.

Commands:
python scripts/rsl_rl/data_processing/analyze_robot_data.py --data_path <path_to_data> [--full]
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib
    # Set backend to Agg for non-interactive plotting (fixes display warnings)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Set style for better plots
    plt.style.use('default')  # Use default style instead of deprecated seaborn style
    PLOTTING_AVAILABLE = True
except ImportError:
    print("[WARNING] Matplotlib not available. Plotting functions will be disabled.")
    PLOTTING_AVAILABLE = False

class RobotDataAnalyzer:
    """Analyzer for comprehensive robot state data."""
    
    def __init__(self, data_path: str, env_index: int = 0):
        self.data_path = data_path
        self.env_index = env_index
        self.data = None
        self.metadata = None
        self.dataframes = None
        # Cache for termination analysis to avoid multiple calculations
        self._termination_mask = None
        self._termination_info = None
        self.load_data()
    
    def load_data(self):
        """Load robot data from files."""
        # Try to load complete data first
        complete_file = os.path.join(self.data_path, "complete_robot_data.pkl")
        if os.path.exists(complete_file):
            print(f"[INFO] Loading complete data from: {complete_file}")
            with open(complete_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            # Load latest partial data
            data_files = [f for f in os.listdir(self.data_path) if f.startswith("robot_data_step_") and f.endswith(".pkl")]
            if data_files:
                latest_file = max(data_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                print(f"[INFO] Loading latest data from: {latest_file}")
                with open(os.path.join(self.data_path, latest_file), 'rb') as f:
                    self.data = pickle.load(f)
            else:
                raise FileNotFoundError(f"No robot data files found in {self.data_path}")
        
        # Load metadata
        metadata_file = os.path.join(self.data_path, "metadata.json")
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"[INFO] Loaded data with {len(self.data['timesteps'])} timesteps")
        
        # Check available environments
        if self.data['robot_states']['base_position']:
            positions = np.array(self.data['robot_states']['base_position'])
            if positions.ndim >= 3:
                num_envs = positions.shape[1]
                print(f"[INFO] Data contains {num_envs} environments")
                if self.env_index >= num_envs:
                    print(f"[WARNING] Environment index {self.env_index} out of range (0-{num_envs-1}). Using environment 0.")
                    self.env_index = 0
                else:
                    print(f"[INFO] Using environment {self.env_index} for analysis")
                
    def get_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Get processed dataframes. Convert if not already done."""
        if self.dataframes is None:
            self.dataframes = self.convert_to_dataframes()
        return self.dataframes
        
    def convert_to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert data to pandas DataFrames for easier analysis."""
        if not self.data:
            return {}
            
        dfs = {}
        
        # Calculate real time from timesteps
        # Default values from common Isaac Lab configurations
        sim_dt = 0.005  # Default simulation timestep
        decimation = 4  # Default decimation factor
        real_dt = sim_dt * decimation  # Real time per logged step
        
        # Try to get actual values from metadata if available
        if self.metadata:
            sim_dt = self.metadata.get('sim_dt', sim_dt)
            decimation = self.metadata.get('decimation', decimation)
            real_dt = sim_dt * decimation
        
        timesteps = np.array(self.data['timesteps'])
        real_time = timesteps * real_dt  # Convert timesteps to real time in seconds
        
        # Basic data
        basic_data = {
            'timestep': self.data['timesteps'],
            'time': real_time,
        }
                
        # Robot state data
        if self.data['robot_states']['base_position']:
            robot_data = {'timestep': self.data['timesteps'], 'time': real_time}
            
            # Base position (take selected environment, get x, y, z)
            positions = np.array(self.data['robot_states']['base_position'])
            if positions.ndim >= 3:  # [timestep, env, xyz]
                robot_data['base_pos_x'] = positions[:, self.env_index, 0]
                robot_data['base_pos_y'] = positions[:, self.env_index, 1] 
                robot_data['base_pos_z'] = positions[:, self.env_index, 2]
            
            # Base velocities
            if self.data['robot_states']['base_lin_vel']:
                velocities = np.array(self.data['robot_states']['base_lin_vel'])
                if velocities.ndim >= 3:
                    robot_data['base_vel_x'] = velocities[:, self.env_index, 0]
                    robot_data['base_vel_y'] = velocities[:, self.env_index, 1]
                    robot_data['base_vel_z'] = velocities[:, self.env_index, 2]
                    robot_data['base_speed'] = np.linalg.norm(velocities[:, self.env_index, :], axis=1)
            
            # Angular velocities  
            if self.data['robot_states']['base_ang_vel']:
                ang_vels = np.array(self.data['robot_states']['base_ang_vel'])
                if ang_vels.ndim >= 3:
                    robot_data['base_ang_vel_x'] = ang_vels[:, self.env_index, 0]
                    robot_data['base_ang_vel_y'] = ang_vels[:, self.env_index, 1]
                    robot_data['base_ang_vel_z'] = ang_vels[:, self.env_index, 2]
                    robot_data['base_ang_speed'] = np.linalg.norm(ang_vels[:, self.env_index, :], axis=1)
            
            # Base accelerations
            if self.data['robot_states']['base_lin_acc']:
                accelerations = np.array(self.data['robot_states']['base_lin_acc'])
                if accelerations.ndim >= 3:
                    robot_data['base_acc_x'] = accelerations[:, self.env_index, 0]
                    robot_data['base_acc_y'] = accelerations[:, self.env_index, 1]
                    robot_data['base_acc_z'] = accelerations[:, self.env_index, 2]
                    robot_data['base_acceleration'] = np.linalg.norm(accelerations[:, self.env_index, :], axis=1)
            
            # Angular accelerations
            if self.data['robot_states']['base_ang_acc']:
                ang_accs = np.array(self.data['robot_states']['base_ang_acc'])
                if ang_accs.ndim >= 3:
                    robot_data['base_ang_acc_x'] = ang_accs[:, self.env_index, 0]
                    robot_data['base_ang_acc_y'] = ang_accs[:, self.env_index, 1]
                    robot_data['base_ang_acc_z'] = ang_accs[:, self.env_index, 2]
                    robot_data['base_ang_acceleration'] = np.linalg.norm(ang_accs[:, self.env_index, :], axis=1)
            
            dfs['robot_states'] = pd.DataFrame(robot_data)
        
        # Plate data
        if self.data['plate_data']['lin_vel']:
            plate_data = {'timestep': self.data['timesteps'], 'time': real_time}
            
            # Linear velocities
            lin_vels = np.array(self.data['plate_data']['lin_vel'])
            if lin_vels.ndim >= 3:
                plate_data['plate_vel_x'] = lin_vels[:, self.env_index, 0]
                plate_data['plate_vel_y'] = lin_vels[:, self.env_index, 1]
                plate_data['plate_vel_z'] = lin_vels[:, self.env_index, 2]
                plate_data['plate_speed'] = np.linalg.norm(lin_vels[:, self.env_index, :], axis=1)
            
            # Angular velocities
            if self.data['plate_data']['ang_vel']:
                ang_vels = np.array(self.data['plate_data']['ang_vel'])
                if ang_vels.ndim >= 3:
                    plate_data['plate_ang_vel_x'] = ang_vels[:, self.env_index, 0]
                    plate_data['plate_ang_vel_y'] = ang_vels[:, self.env_index, 1]
                    plate_data['plate_ang_vel_z'] = ang_vels[:, self.env_index, 2]
                    plate_data['plate_ang_speed'] = np.linalg.norm(ang_vels[:, self.env_index, :], axis=1)
            
            # Accelerations
            if self.data['plate_data']['lin_acc']:
                lin_accs = np.array(self.data['plate_data']['lin_acc'])
                if lin_accs.ndim >= 3:
                    plate_data['plate_acc_x'] = lin_accs[:, self.env_index, 0]
                    plate_data['plate_acc_y'] = lin_accs[:, self.env_index, 1]
                    plate_data['plate_acc_z'] = lin_accs[:, self.env_index, 2]
                    plate_data['plate_acceleration'] = np.linalg.norm(lin_accs[:, self.env_index, :], axis=1)
            
            dfs['plate_data'] = pd.DataFrame(plate_data)
        
        # Object data
        if self.data['object_data'] and self.data['object_data']['lin_vel']:
            object_data = {'timestep': self.data['timesteps'], 'time': real_time}
            
            # Object positions
            if self.data['object_data']['position']:
                positions = np.array(self.data['object_data']['position'])
                if positions.ndim >= 3:
                    object_data['object_pos_x'] = positions[:, self.env_index, 0]
                    object_data['object_pos_y'] = positions[:, self.env_index, 1]
                    object_data['object_pos_z'] = positions[:, self.env_index, 2]
            
            # Object velocities
            lin_vels = np.array(self.data['object_data']['lin_vel'])
            if lin_vels.ndim >= 3:
                object_data['object_vel_x'] = lin_vels[:, self.env_index, 0]
                object_data['object_vel_y'] = lin_vels[:, self.env_index, 1]
                object_data['object_vel_z'] = lin_vels[:, self.env_index, 2]
                object_data['object_speed'] = np.linalg.norm(lin_vels[:, self.env_index, :], axis=1)
            
            # Object deviation from plate
            if self.data['object_data']['deviation_from_plate']:
                deviations = np.array(self.data['object_data']['deviation_from_plate'])
                if deviations.ndim >= 3:
                    object_data['object_dev_x'] = deviations[:, self.env_index, 0]
                    object_data['object_dev_y'] = deviations[:, self.env_index, 1]
                    object_data['object_deviation'] = np.linalg.norm(deviations[:, self.env_index, :], axis=1)
            
            # Object linear accelerations
            if self.data['object_data'].get('lin_acc'):
                lin_accs = np.array(self.data['object_data']['lin_acc'])
                if lin_accs.ndim >= 3:
                    object_data['object_acc_x'] = lin_accs[:, self.env_index, 0]
                    object_data['object_acc_y'] = lin_accs[:, self.env_index, 1]
                    object_data['object_acc_z'] = lin_accs[:, self.env_index, 2]
                    object_data['object_acceleration'] = np.linalg.norm(lin_accs[:, self.env_index, :], axis=1)
            
            # Object angular accelerations
            if self.data['object_data'].get('ang_acc'):
                ang_accs = np.array(self.data['object_data']['ang_acc'])
                if ang_accs.ndim >= 3:
                    object_data['object_ang_acc_x'] = ang_accs[:, self.env_index, 0]
                    object_data['object_ang_acc_y'] = ang_accs[:, self.env_index, 1]
                    object_data['object_ang_acc_z'] = ang_accs[:, self.env_index, 2]
                    object_data['object_ang_acceleration'] = np.linalg.norm(ang_accs[:, self.env_index, :], axis=1)
            
            dfs['object_data'] = pd.DataFrame(object_data)
        
        # Command data
        if self.data and 'commands' in self.data and self.data['commands'] and 'base_velocity' in self.data['commands']:
            command_data = {'timestep': self.data['timesteps'], 'time': real_time}
            
            # Base velocity commands (linear x, linear y, angular z)
            base_vel_cmds = np.array(self.data['commands']['base_velocity'])
            if base_vel_cmds.ndim >= 3:
                command_data['cmd_vel_x'] = base_vel_cmds[:, self.env_index, 0]      # Linear velocity X
                command_data['cmd_vel_y'] = base_vel_cmds[:, self.env_index, 1]      # Linear velocity Y
                command_data['cmd_ang_vel_z'] = base_vel_cmds[:, self.env_index, 2]  # Angular velocity Z (yaw)
                command_data['cmd_lin_speed'] = np.linalg.norm(base_vel_cmds[:, self.env_index, :2], axis=1)  # Linear speed magnitude

            dfs['command_data'] = pd.DataFrame(command_data)
        
        # Termination data
        if self.data and 'termination' in self.data and self.data['termination']:
            termination_data = {'timestep': self.data['timesteps'], 'time': real_time}
            
            for term_name, term_values in self.data['termination'].items():
                if term_values:
                    term_array = np.array(term_values)
                    if term_array.ndim >= 2:
                        # Take selected environment for plotting
                        termination_data[term_name] = term_array[:, self.env_index].astype(float)
                    else:
                        termination_data[term_name] = term_array.astype(float)
            
            dfs['termination_data'] = pd.DataFrame(termination_data)
        
        # Event data
        if self.data and 'events' in self.data and self.data['events']:
            event_data = {'timestep': self.data['timesteps'], 'time': real_time}
            
            for event_name, event_values in self.data['events'].items():
                if event_values:
                    try:
                        event_array = np.array(event_values)
                        n_timesteps = len(self.data['timesteps'])
                        
                        if event_array.ndim >= 2:
                            # For clear events, calculate magnitude
                            if 'clear' in event_name.lower() and event_array.shape[-1] >= 3:
                                # Create arrays filled with zeros for all timesteps
                                magnitude_data = np.zeros(n_timesteps)
                                x_data = np.zeros(n_timesteps)
                                y_data = np.zeros(n_timesteps)
                                z_data = np.zeros(n_timesteps)
                                
                                # Fill the data where events occurred (assuming events start from some timestep)
                                n_events = event_array.shape[0]
                                start_idx = n_timesteps - n_events  # Events start from this timestep
                                
                                magnitude_data[start_idx:] = np.linalg.norm(event_array[:, self.env_index, :3], axis=1)
                                x_data[start_idx:] = event_array[:, self.env_index, 0]
                                y_data[start_idx:] = event_array[:, self.env_index, 1] 
                                z_data[start_idx:] = event_array[:, self.env_index, 2]
                                
                                event_data[f'force_magnitude'] = magnitude_data
                                event_data[f'force_x'] = x_data
                                event_data[f'force_y'] = y_data
                                event_data[f'force_z'] = z_data
                        else:
                            # For 1D events, also pad with zeros
                            scalar_data = np.zeros(n_timesteps)
                            n_events = len(event_array)
                            start_idx = n_timesteps - n_events
                            scalar_data[start_idx:] = event_array
                            event_data[event_name] = scalar_data
                            
                    except Exception as e:
                        print(f"[WARNING] Could not process event {event_name}: {e}")

            if len(event_data) > 2:  # More than just timestep and time
                dfs['event_data'] = pd.DataFrame(event_data)
                
        return dfs    
    
    def get_all_env_termination_data(self) -> Dict[str, pd.DataFrame]:
        """Get termination data for all environments as percentages."""
        if not self.data or 'termination' not in self.data or not self.data['termination']:
            return {}
        
        # Calculate real time from timesteps
        sim_dt = 0.005
        decimation = 4
        real_dt = sim_dt * decimation
        
        if self.metadata:
            sim_dt = self.metadata.get('sim_dt', sim_dt)
            decimation = self.metadata.get('decimation', decimation)
            real_dt = sim_dt * decimation
        
        timesteps = np.array(self.data['timesteps'])
        real_time = timesteps * real_dt
        
        # Get number of environments
        first_term_data = next(iter(self.data['termination'].values()))
        if not first_term_data:
            return {}
        
        term_array = np.array(first_term_data)
        if term_array.ndim < 2:
            return {}
        
        num_envs = term_array.shape[1]
        print(f"[INFO] Processing termination data for {num_envs} environments")
        
        # Create data structure for all environments
        all_env_data = {}
        for term_name, term_values in self.data['termination'].items():
            if term_values:
                term_array = np.array(term_values)
                if term_array.ndim >= 2:
                    # Calculate percentage of environments that have this termination at each timestep
                    env_percentages = np.mean(term_array.astype(float), axis=1) * 100.0
                    all_env_data[term_name] = env_percentages
        
        if all_env_data:
            all_env_data['timestep'] = self.data['timesteps']
            all_env_data['time'] = real_time
            return {'all_env_termination': pd.DataFrame(all_env_data)}
        
        return {}

    def get_termination_analysis(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get both the pre-termination mask and termination information for all environments.
        
        Only checks the 'failed' termination condition to determine when environments terminate.
        Includes data up to and including the first timestep where termination occurs.
        Results are cached to avoid multiple calculations.
        
        Returns:
            Tuple containing:
            - np.ndarray: Boolean mask of shape [timestep, env] where True indicates
                         the timestep is before or at the first termination for that environment.
            - Dict: Termination information containing:
                - num_environments: Total number of environments
                - terminated_envs: Number of environments that terminated
                - termination_timesteps: List of first termination timestep for each env (or -1 if never terminated)
                - termination_percentage: Percentage of environments that terminated
        """
        # Return cached results if available
        if self._termination_mask is not None and self._termination_info is not None:
            return self._termination_mask, self._termination_info
        
        # Default empty returns
        default_mask = np.array([[True]])
        default_info = {}
        
        if not self.data or 'termination' not in self.data or not self.data['termination']:
            # If no termination data, return all True
            if self.data and 'robot_states' in self.data and self.data['robot_states'].get('base_position'):
                positions = np.array(self.data['robot_states']['base_position'])
                if positions.ndim >= 3:
                    mask = np.ones((positions.shape[0], positions.shape[1]), dtype=bool)
                    self._termination_mask = mask
                    self._termination_info = default_info
                    return mask, default_info
            self._termination_mask = default_mask
            self._termination_info = default_info
            return default_mask, default_info
        
        # Only check for 'failed' termination condition
        if 'failed' not in self.data['termination'] or not self.data['termination']['failed']:
            # If no 'failed' data, return all True (no terminations detected)
            if self.data and 'robot_states' in self.data and self.data['robot_states'].get('base_position'):
                positions = np.array(self.data['robot_states']['base_position'])
                if positions.ndim >= 3:
                    mask = np.ones((positions.shape[0], positions.shape[1]), dtype=bool)
                    self._termination_mask = mask
                    self._termination_info = default_info
                    return mask, default_info
            self._termination_mask = default_mask
            self._termination_info = default_info
            return default_mask, default_info
        
        # Get the failed termination data
        terminated_data = np.array(self.data['termination']['failed'], dtype=bool)
        
        if terminated_data.ndim < 2:
            # Handle case where data doesn't have proper dimensions
            if self.data and 'robot_states' in self.data and self.data['robot_states'].get('base_position'):
                positions = np.array(self.data['robot_states']['base_position'])
                if positions.ndim >= 3:
                    mask = np.ones((positions.shape[0], positions.shape[1]), dtype=bool)
                    self._termination_mask = mask
                    self._termination_info = default_info
                    return mask, default_info
            self._termination_mask = default_mask
            self._termination_info = default_info
            return default_mask, default_info
        
        # For each environment, find the first timestep where termination occurs
        mask = np.ones_like(terminated_data, dtype=bool)
        num_envs = terminated_data.shape[1]
        termination_timesteps = []
        terminated_count = 0
        
        for env_idx in range(num_envs):
            env_terminations = terminated_data[:, env_idx]
            first_termination = np.where(env_terminations)[0]
            
            if len(first_termination) > 0:
                # Include the first termination timestep in the evaluation
                # Set all timesteps AFTER first termination to False
                first_termination_idx = first_termination[0]
                mask[first_termination_idx + 1:, env_idx] = False
                
                # Record termination info
                termination_timesteps.append(int(first_termination_idx))
                terminated_count += 1
            else:
                termination_timesteps.append(-1)  # Never terminated
        
        termination_percentage = (terminated_count / num_envs) * 100.0
        
        termination_info = {
            'num_environments': num_envs,
            'terminated_envs': terminated_count,
            'termination_timesteps': termination_timesteps,
            'termination_percentage': float(termination_percentage)
        }
        
        # Cache the results
        self._termination_mask = mask
        self._termination_info = termination_info
        
        return mask, termination_info
    
    def get_pre_termination_mask(self) -> np.ndarray:
        """Get a boolean mask for data points up to and including first termination in each environment.
        
        This is a convenience method that calls get_termination_analysis() and returns only the mask.
        
        Returns:
            np.ndarray: Boolean mask of shape [timestep, env] where True indicates
                       the timestep is before or at the first termination for that environment.
        """
        mask, _ = self.get_termination_analysis()
        return mask

    def get_termination_info(self) -> Dict[str, Any]:
        """Get information about which environments terminated and when.
        
        This is a convenience method that calls get_termination_analysis() and returns only the info.
        Only considers the 'failed' termination condition.
        
        Returns:
            Dict containing:
                - num_environments: Total number of environments
                - terminated_envs: Number of environments that terminated
                - termination_timesteps: List of first termination timestep for each env (or -1 if never terminated)
                - termination_percentage: Percentage of environments that terminated
        """
        _, info = self.get_termination_analysis()
        return info

    def get_all_env_robot_stats(self) -> Dict[str, float]:
        """Get robot state statistics across all environments, including data up to and including the first termination."""
        if not self.data or 'robot_states' not in self.data or not self.data['robot_states']:
            return {}
        
        stats = {}
        # Get mask for data up to and including termination
        pre_term_mask = self.get_pre_termination_mask()
        
        # Base speed stats
        if self.data['robot_states'].get('base_lin_vel'):
            velocities = np.array(self.data['robot_states']['base_lin_vel'])
            if velocities.ndim >= 3:
                # Calculate speed magnitude for each timestep and environment
                speeds = np.linalg.norm(velocities, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_speeds = speeds[pre_term_mask]
                if len(valid_speeds) > 0:
                    stats['base_speed_mean'] = float(np.mean(valid_speeds))
                    stats['base_speed_max'] = float(np.max(valid_speeds))
        
        # Angular speed stats
        if self.data['robot_states'].get('base_ang_vel'):
            ang_vels = np.array(self.data['robot_states']['base_ang_vel'])
            if ang_vels.ndim >= 3:
                # Calculate angular speed magnitude for each timestep and environment
                ang_speeds = np.linalg.norm(ang_vels, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_ang_speeds = ang_speeds[pre_term_mask]
                if len(valid_ang_speeds) > 0:
                    stats['base_ang_speed_mean'] = float(np.mean(valid_ang_speeds))
                    stats['base_ang_speed_max'] = float(np.max(valid_ang_speeds))
        
        # Linear acceleration stats
        if self.data['robot_states'].get('base_lin_acc'):
            accelerations = np.array(self.data['robot_states']['base_lin_acc'])
            if accelerations.ndim >= 3:
                # Calculate acceleration magnitude for each timestep and environment
                acc_magnitudes = np.linalg.norm(accelerations, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_acc_magnitudes = acc_magnitudes[pre_term_mask]
                if len(valid_acc_magnitudes) > 0:
                    stats['base_acceleration_mean'] = float(np.mean(valid_acc_magnitudes))
                    stats['base_acceleration_max'] = float(np.max(valid_acc_magnitudes))
        
        # Angular acceleration stats
        if self.data['robot_states'].get('base_ang_acc'):
            ang_accs = np.array(self.data['robot_states']['base_ang_acc'])
            if ang_accs.ndim >= 3:
                # Calculate angular acceleration magnitude for each timestep and environment
                ang_acc_magnitudes = np.linalg.norm(ang_accs, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_ang_acc_magnitudes = ang_acc_magnitudes[pre_term_mask]
                if len(valid_ang_acc_magnitudes) > 0:
                    stats['base_ang_acceleration_mean'] = float(np.mean(valid_ang_acc_magnitudes))
                    stats['base_ang_acceleration_max'] = float(np.max(valid_ang_acc_magnitudes))
        
        return stats
    
    def get_all_env_plate_stats(self) -> Dict[str, float]:
        """Get plate statistics across all environments, including data up to and including the first termination."""
        if not self.data or 'plate_data' not in self.data or not self.data['plate_data']:
            return {}
        
        stats = {}
        
        # Get mask for data up to and including termination
        pre_term_mask = self.get_pre_termination_mask()
        
        # Plate speed stats
        if self.data['plate_data'].get('lin_vel'):
            lin_vels = np.array(self.data['plate_data']['lin_vel'])
            if lin_vels.ndim >= 3:
                # Calculate speed magnitude for each timestep and environment
                speeds = np.linalg.norm(lin_vels, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_speeds = speeds[pre_term_mask]
                if len(valid_speeds) > 0:
                    stats['plate_speed_mean'] = float(np.mean(valid_speeds))
                    stats['plate_speed_max'] = float(np.max(valid_speeds))
        
        # Plate angular speed stats
        if self.data['plate_data'].get('ang_vel'):
            ang_vels = np.array(self.data['plate_data']['ang_vel'])
            if ang_vels.ndim >= 3:
                # Calculate angular speed magnitude for each timestep and environment
                ang_speeds = np.linalg.norm(ang_vels, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_ang_speeds = ang_speeds[pre_term_mask]
                if len(valid_ang_speeds) > 0:
                    stats['plate_ang_speed_mean'] = float(np.mean(valid_ang_speeds))
                    stats['plate_ang_speed_max'] = float(np.max(valid_ang_speeds))
        
        # Plate acceleration stats
        if self.data['plate_data'].get('lin_acc'):
            lin_accs = np.array(self.data['plate_data']['lin_acc'])
            if lin_accs.ndim >= 3:
                # Calculate acceleration magnitude for each timestep and environment
                acc_magnitudes = np.linalg.norm(lin_accs, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_acc_magnitudes = acc_magnitudes[pre_term_mask]
                if len(valid_acc_magnitudes) > 0:
                    stats['plate_acceleration_mean'] = float(np.mean(valid_acc_magnitudes))
                    stats['plate_acceleration_max'] = float(np.max(valid_acc_magnitudes))
        
        return stats
    
    def get_all_env_object_stats(self) -> Dict[str, float]:
        """Get comprehensive object statistics across all environments, including data up to and including the first termination."""
        if not self.data or 'object_data' not in self.data or not self.data['object_data']:
            return {}
        
        stats = {}
        
        # Get mask for data up to and including termination
        pre_term_mask = self.get_pre_termination_mask()

        # Object speed stats
        if self.data['object_data'].get('lin_vel'):
            lin_vels = np.array(self.data['object_data']['lin_vel'])
            if lin_vels.ndim >= 3:
                # Calculate speed magnitude for each timestep and environment
                speeds = np.linalg.norm(lin_vels, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_speeds = speeds[pre_term_mask]
                if len(valid_speeds) > 0:
                    stats['object_speed_mean'] = float(np.mean(valid_speeds))
                    stats['object_speed_max'] = float(np.max(valid_speeds))
        
        # Object deviation stats - only compute at final timestep
        if self.data['object_data'].get('deviation_from_plate'):
            deviations = np.array(self.data['object_data']['deviation_from_plate'])
            if deviations.ndim >= 3:
                # Get final timestep deviation for each environment
                final_deviations = deviations[-1, :, :]  # Shape: [env, xyz]
                # Calculate deviation magnitude for each environment at final timestep
                final_dev_magnitudes = np.linalg.norm(final_deviations, axis=1)  # Shape: [env]
                
                # Apply mask to exclude environments that terminated
                pre_term_mask = self.get_pre_termination_mask()
                final_mask = pre_term_mask[-1, :]  # Final timestep mask for each environment
                valid_final_dev_magnitudes = final_dev_magnitudes[final_mask]
                
                if len(valid_final_dev_magnitudes) > 0:
                    stats['object_deviation_mean'] = float(np.mean(valid_final_dev_magnitudes))
                    stats['object_deviation_max'] = float(np.max(valid_final_dev_magnitudes))
                    stats['object_deviation_min'] = float(np.min(valid_final_dev_magnitudes))
        
        # Object linear acceleration stats
        if self.data['object_data'].get('lin_acc'):
            lin_accs = np.array(self.data['object_data']['lin_acc'])
            if lin_accs.ndim >= 3:
                # Calculate magnitude for each timestep and environment
                lin_acc_magnitudes = np.linalg.norm(lin_accs, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_lin_acc_magnitudes = lin_acc_magnitudes[pre_term_mask]
                if len(valid_lin_acc_magnitudes) > 0:
                    stats['object_lin_acc_mean'] = float(np.mean(valid_lin_acc_magnitudes))
                    stats['object_lin_acc_max'] = float(np.max(valid_lin_acc_magnitudes))
        
        # Object angular acceleration stats
        if self.data['object_data'].get('ang_acc'):
            ang_accs = np.array(self.data['object_data']['ang_acc'])
            if ang_accs.ndim >= 3:
                # Calculate magnitude for each timestep and environment
                ang_acc_magnitudes = np.linalg.norm(ang_accs, axis=2)  # Shape: [timestep, env]
                # Apply mask to exclude post-termination data
                valid_ang_acc_magnitudes = ang_acc_magnitudes[pre_term_mask]
                if len(valid_ang_acc_magnitudes) > 0:
                    stats['object_ang_acc_mean'] = float(np.mean(valid_ang_acc_magnitudes))
                    stats['object_ang_acc_max'] = float(np.max(valid_ang_acc_magnitudes))
        
        return stats
    
    def get_success_rate(self) -> Dict[str, float]:
        """Get success rate (1 - failure rate) across all environments."""
        if not self.data or 'termination' not in self.data or not self.data['termination']:
            return {}
        
        # Check if 'failed' termination condition exists
        if 'failed' not in self.data['termination'] or not self.data['termination']['failed']:
            return {}
        
        failed_data = np.array(self.data['termination']['failed'])
        if failed_data.ndim < 2:
            return {}
        
        # Calculate final failure rate (percentage of environments that failed at the end)
        final_timestep_failures = failed_data[-1, :]  # Last timestep for all environments
        failure_rate = np.mean(final_timestep_failures.astype(float)) * 100.0
        success_rate = 100.0 - failure_rate
        
        # Also calculate overall failure rate (any environment that failed at any time)
        ever_failed = np.any(failed_data.astype(bool), axis=0)  # Environments that ever failed
        overall_failure_rate = np.mean(ever_failed.astype(float)) * 100.0
        overall_success_rate = 100.0 - overall_failure_rate
        
        return {
            'final_success_rate': float(success_rate),
            'final_failure_rate': float(failure_rate),
            'overall_success_rate': float(overall_success_rate),
            'overall_failure_rate': float(overall_failure_rate)
        }

        
    def plot_robot_analysis(self, save_path: Optional[str] = None):
        """Plot comprehensive robot analysis including trajectory and accelerations."""
        if not PLOTTING_AVAILABLE:
            print("[WARNING] Plotting not available. Install matplotlib and seaborn to enable plotting.")
            return
            
        dfs = self.get_dataframes()
        if 'robot_states' not in dfs:
            print("[WARNING] No robot state data available")
            return
        
        robot_df = dfs['robot_states']
        
        # Create a larger figure with more subplots for comprehensive analysis
        fig = plt.figure(figsize=(20, 15))
        
        # 2D trajectory (top view)
        ax1 = fig.add_subplot(331)
        scatter = ax1.scatter(robot_df['base_pos_x'], robot_df['base_pos_y'], 
                            c=robot_df['time'], s=20, alpha=0.7, cmap='viridis')
        ax1.set_title('Robot Trajectory (Top View)')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        plt.colorbar(scatter, ax=ax1, label='Time (s)')
        
        # Height over time
        ax2 = fig.add_subplot(332)
        ax2.plot(robot_df['time'], robot_df['base_pos_z'])
        ax2.set_title('Robot Height Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Z Position (m)')
        ax2.grid(True, alpha=0.3)
        
        # Speed over time
        ax3 = fig.add_subplot(333)
        if 'base_speed' in robot_df.columns:
            ax3.plot(robot_df['time'], robot_df['base_speed'])
            ax3.set_title('Robot Speed Over Time')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Speed (m/s)')
            ax3.grid(True, alpha=0.3)
        
        # Linear acceleration magnitude
        ax4 = fig.add_subplot(334)
        if 'base_acceleration' in robot_df.columns:
            ax4.plot(robot_df['time'], robot_df['base_acceleration'])
            ax4.set_title('Robot Linear Acceleration')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Acceleration (m/s²)')
            ax4.grid(True, alpha=0.3)
        
        # Linear acceleration components
        ax5 = fig.add_subplot(335)
        if all(col in robot_df.columns for col in ['base_acc_x', 'base_acc_y', 'base_acc_z']):
            ax5.plot(robot_df['time'], robot_df['base_acc_x'], label='X', alpha=0.8)
            ax5.plot(robot_df['time'], robot_df['base_acc_y'], label='Y', alpha=0.8)
            ax5.plot(robot_df['time'], robot_df['base_acc_z'], label='Z', alpha=0.8)
            ax5.set_title('Linear Acceleration Components')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Acceleration (m/s²)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Angular velocity magnitude
        ax6 = fig.add_subplot(336)
        if 'base_ang_speed' in robot_df.columns:
            ax6.plot(robot_df['time'], robot_df['base_ang_speed'])
            ax6.set_title('Robot Angular Speed')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Angular Speed (rad/s)')
            ax6.grid(True, alpha=0.3)
        
        # Angular acceleration magnitude
        ax7 = fig.add_subplot(337)
        if 'base_ang_acceleration' in robot_df.columns:
            ax7.plot(robot_df['time'], robot_df['base_ang_acceleration'])
            ax7.set_title('Robot Angular Acceleration')
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Angular Acceleration (rad/s²)')
            ax7.grid(True, alpha=0.3)
        
        # Angular acceleration components
        ax8 = fig.add_subplot(338)
        if all(col in robot_df.columns for col in ['base_ang_acc_x', 'base_ang_acc_y', 'base_ang_acc_z']):
            ax8.plot(robot_df['time'], robot_df['base_ang_acc_x'], label='X', alpha=0.8)
            ax8.plot(robot_df['time'], robot_df['base_ang_acc_y'], label='Y', alpha=0.8)
            ax8.plot(robot_df['time'], robot_df['base_ang_acc_z'], label='Z', alpha=0.8)
            ax8.set_title('Angular Acceleration Components')
            ax8.set_xlabel('Time (s)')
            ax8.set_ylabel('Angular Acceleration (rad/s²)')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # Velocity components
        ax9 = fig.add_subplot(339)
        if all(col in robot_df.columns for col in ['base_vel_x', 'base_vel_y', 'base_vel_z']):
            ax9.plot(robot_df['time'], robot_df['base_vel_x'], label='X', alpha=0.8)
            ax9.plot(robot_df['time'], robot_df['base_vel_y'], label='Y', alpha=0.8)
            ax9.plot(robot_df['time'], robot_df['base_vel_z'], label='Z', alpha=0.8)
            ax9.set_title('Linear Velocity Components')
            ax9.set_xlabel('Time (s)')
            ax9.set_ylabel('Velocity (m/s)')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'robot_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"[INFO] Robot analysis plot saved to: {os.path.join(save_path, 'robot_analysis.png')}")
        else:
            # Auto-save when no save path provided
            default_save_path = os.path.join(os.path.dirname(self.data_path), 'robot_analysis.png')
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Robot analysis plot saved to: {default_save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_plate_analysis(self, save_path: Optional[str] = None):
        """Plot plate-specific analysis."""
        if not PLOTTING_AVAILABLE:
            print("[WARNING] Plotting not available. Install matplotlib and seaborn to enable plotting.")
            return
            
        dfs = self.get_dataframes()
        if 'plate_data' not in dfs:
            print("[WARNING] No plate data available")
            return
        
        plate_df = dfs['plate_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plate speed over time
        if 'plate_speed' in plate_df.columns:
            axes[0, 0].plot(plate_df['time'], plate_df['plate_speed'])
            axes[0, 0].set_title('Plate Linear Speed')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Speed (m/s)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plate angular speed
        if 'plate_ang_speed' in plate_df.columns:
            axes[0, 1].plot(plate_df['time'], plate_df['plate_ang_speed'])
            axes[0, 1].set_title('Plate Angular Speed')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Angular Speed (rad/s)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plate acceleration
        if 'plate_acceleration' in plate_df.columns:
            axes[1, 0].plot(plate_df['time'], plate_df['plate_acceleration'])
            axes[1, 0].set_title('Plate Acceleration')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Acceleration (m/s²)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plate velocity components
        if all(col in plate_df.columns for col in ['plate_vel_x', 'plate_vel_y', 'plate_vel_z']):
            axes[1, 1].plot(plate_df['time'], plate_df['plate_vel_x'], label='X')
            axes[1, 1].plot(plate_df['time'], plate_df['plate_vel_y'], label='Y') 
            axes[1, 1].plot(plate_df['time'], plate_df['plate_vel_z'], label='Z')
            axes[1, 1].set_title('Plate Velocity Components')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Velocity (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'plate_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"[INFO] Plate analysis plot saved to: {os.path.join(save_path, 'plate_analysis.png')}")
        else:
            # Auto-save when no save path provided
            default_save_path = os.path.join(os.path.dirname(self.data_path), 'plate_analysis.png')
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Plate analysis plot saved to: {default_save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_object_analysis(self, save_path: Optional[str] = None):
        """Plot object (cylinder) analysis."""
        if not PLOTTING_AVAILABLE:
            print("[WARNING] Plotting not available. Install matplotlib and seaborn to enable plotting.")
            return
            
        dfs = self.get_dataframes()
        if 'object_data' not in dfs:
            print("[WARNING] No object data available")
            return
        
        object_df = dfs['object_data']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Object position over time
        if all(col in object_df.columns for col in ['object_pos_x', 'object_pos_y', 'object_pos_z']):
            axes[0, 0].plot(object_df['time'], object_df['object_pos_x'], label='X')
            axes[0, 0].plot(object_df['time'], object_df['object_pos_y'], label='Y')
            axes[0, 0].plot(object_df['time'], object_df['object_pos_z'], label='Z')
            axes[0, 0].set_title('Object Position')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Object speed
        if 'object_speed' in object_df.columns:
            axes[0, 1].plot(object_df['time'], object_df['object_speed'])
            axes[0, 1].set_title('Object Speed')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Speed (m/s)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Object deviation from plate
        if 'object_deviation' in object_df.columns:
            axes[1, 0].plot(object_df['time'], object_df['object_deviation'])
            axes[1, 0].set_title('Object Deviation from Plate')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Deviation (m)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Object trajectory (top view)
        if all(col in object_df.columns for col in ['object_pos_x', 'object_pos_y']):
            scatter = axes[1, 1].scatter(object_df['object_pos_x'], object_df['object_pos_y'], 
                                       c=object_df['time'], s=20, alpha=0.7, cmap='plasma')
            axes[1, 1].set_title('Object Trajectory (Top View)')
            axes[1, 1].set_xlabel('X Position (m)')
            axes[1, 1].set_ylabel('Y Position (m)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axis('equal')
            plt.colorbar(scatter, ax=axes[1, 1], label='Time (s)')
        
        # Object linear acceleration magnitude
        if 'object_acceleration' in object_df.columns:
            axes[2, 0].plot(object_df['time'], object_df['object_acceleration'])
            axes[2, 0].set_title('Object Linear Acceleration Magnitude')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_ylabel('Acceleration (m/s²)')
            axes[2, 0].grid(True, alpha=0.3)
        
        # Object acceleration components
        if all(col in object_df.columns for col in ['object_acc_x', 'object_acc_y', 'object_acc_z']):
            axes[2, 1].plot(object_df['time'], object_df['object_acc_x'], label='X', alpha=0.8)
            axes[2, 1].plot(object_df['time'], object_df['object_acc_y'], label='Y', alpha=0.8)
            axes[2, 1].plot(object_df['time'], object_df['object_acc_z'], label='Z', alpha=0.8)
            axes[2, 1].set_title('Object Linear Acceleration Components')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Acceleration (m/s²)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        elif 'object_acceleration' in object_df.columns:
            # Fallback: plot acceleration magnitude again if components not available
            axes[2, 1].plot(object_df['time'], object_df['object_acceleration'])
            axes[2, 1].set_title('Object Linear Acceleration (Fallback)')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_ylabel('Acceleration (m/s²)')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'object_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"[INFO] Object analysis plot saved to: {os.path.join(save_path, 'object_analysis.png')}")
        else:
            # Auto-save when no save path provided
            default_save_path = os.path.join(os.path.dirname(self.data_path), 'object_analysis.png')
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Object analysis plot saved to: {default_save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_command_analysis(self, save_path: Optional[str] = None):
        """Plot command data analysis."""
        if not PLOTTING_AVAILABLE:
            print("[WARNING] Plotting not available. Install matplotlib to enable plotting.")
            return
            
        dfs = self.get_dataframes()
        if 'command_data' not in dfs:
            print("[WARNING] No command data available")
            return
        
        command_df = dfs['command_data']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Command speed over time
        if 'cmd_lin_speed' in command_df.columns:
            axes[0].plot(command_df['time'], command_df['cmd_lin_speed'])
            axes[0].set_title('Commanded Linear Speed')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Linear Speed (m/s)')
            axes[0].grid(True, alpha=0.3)
        
        # Command velocity components (linear x, y and angular z)
        if all(col in command_df.columns for col in ['cmd_vel_x', 'cmd_vel_y']):
            axes[1].plot(command_df['time'], command_df['cmd_vel_x'], label='Linear X', alpha=0.8)
            axes[1].plot(command_df['time'], command_df['cmd_vel_y'], label='Linear Y', alpha=0.8)
            if 'cmd_ang_vel_z' in command_df.columns:
                # Plot angular velocity on secondary y-axis
                ax_twin = axes[1].twinx()
                ax_twin.plot(command_df['time'], command_df['cmd_ang_vel_z'], label='Angular Z', 
                           alpha=0.8, color='red', linestyle='--')
                ax_twin.set_ylabel('Angular Velocity (rad/s)', color='red')
                ax_twin.tick_params(axis='y', labelcolor='red')
            axes[1].set_title('Command Velocity Components')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Linear Velocity (m/s)')
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        # Angular velocity over time
        if 'cmd_ang_vel_z' in command_df.columns:
            axes[2].plot(command_df['time'], command_df['cmd_ang_vel_z'])
            axes[2].set_title('Commanded Angular Velocity (Yaw)')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Angular Velocity (rad/s)')
            axes[2].grid(True, alpha=0.3)
        elif 'cmd_lin_speed' in command_df.columns:
            # Fallback to speed histogram if no angular velocity
            axes[2].hist(command_df['cmd_lin_speed'], bins=30, alpha=0.7, edgecolor='black')
            axes[2].set_title('Command Linear Speed Distribution')
            axes[2].set_xlabel('Speed (m/s)')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'command_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"[INFO] Command analysis plot saved to: {os.path.join(save_path, 'command_analysis.png')}")
        else:
            # Auto-save when no save path provided
            default_save_path = os.path.join(os.path.dirname(self.data_path), 'command_analysis.png')
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Command analysis plot saved to: {default_save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_termination_analysis(self, save_path: Optional[str] = None):
        """Plot termination conditions analysis using data from all environments."""
        if not PLOTTING_AVAILABLE:
            print("[WARNING] Plotting not available. Install matplotlib to enable plotting.")
            return
            
        # Get all environment termination data
        all_env_dfs = self.get_all_env_termination_data()
        if 'all_env_termination' not in all_env_dfs:
            print("[WARNING] No termination data available for all environments")
            return
        
        termination_df = all_env_dfs['all_env_termination']
        
        # Count termination conditions (excluding 'timestep', 'time', 'failed')
        term_columns = [col for col in termination_df.columns if col not in ['timestep', 'time', 'failed']]
        if not term_columns:
            print("[WARNING] No termination conditions found")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Termination conditions over time
        ax1 = axes[0]
        for col in term_columns:
            ax1.plot(termination_df['time'], termination_df[col], label=col.replace('_', ' ').title(), alpha=0.8)
        ax1.set_title('Termination Conditions Over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Percentage of Environments (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Termination frequency (average percentage of environments over time)
        ax2 = axes[1]
        term_percentages = []
        for col in term_columns:
            # Average percentage across all timesteps
            avg_percentage = np.mean(termination_df[col])
            term_percentages.append(avg_percentage)
        
        bars = ax2.bar(range(len(term_columns)), term_percentages)
        ax2.set_title('Average Termination Condition Frequency')
        ax2.set_xlabel('Termination Condition')
        ax2.set_ylabel('Average Percentage of Environments (%)')
        ax2.set_xticks(range(len(term_columns)))
        ax2.set_xticklabels([col.replace('_', '\n').title() for col in term_columns], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, term_percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Failed environments over time (percentage)
        ax3 = axes[2]
        if 'failed' in termination_df.columns:
            ax3.plot(termination_df['time'], termination_df['failed'], color='red', linewidth=2)
            ax3.fill_between(termination_df['time'], termination_df['failed'], alpha=0.3, color='red')
            ax3.set_title('Failed Environments Over Time')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Percentage of Failed Environments (%)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'termination_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"[INFO] Termination analysis plot saved to: {os.path.join(save_path, 'termination_analysis.png')}")
        else:
            # Auto-save when no save path provided
            default_save_path = os.path.join(os.path.dirname(self.data_path), 'termination_analysis.png')
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Termination analysis plot saved to: {default_save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def plot_event_analysis(self, save_path: Optional[str] = None):
        """Plot event data analysis."""
        if not PLOTTING_AVAILABLE:
            print("[WARNING] Plotting not available. Install matplotlib to enable plotting.")
            return
            
        dfs = self.get_dataframes()
        if 'event_data' not in dfs:
            print("[WARNING] No event data available")
            return
        
        event_df = dfs['event_data']
        
        # Get event columns (excluding 'timestep' and 'time')
        event_columns = [col for col in event_df.columns if col not in ['timestep', 'time']]
        if not event_columns:
            print("[WARNING] No event data found")
            return
        
        # Separate force magnitude columns from component columns
        force_magnitude_cols = [col for col in event_columns if 'magnitude' in col]
        force_component_cols = [col for col in event_columns if any(axis in col for axis in ['_x', '_y', '_z']) and 'magnitude' not in col]
        other_event_cols = [col for col in event_columns if col not in force_magnitude_cols + force_component_cols]
        
        # Calculate number of subplots needed
        n_plots = len(force_magnitude_cols) + (1 if force_component_cols else 0) + len(other_event_cols)
        if n_plots == 0:
            print("[WARNING] No plottable event data found")
            return
        
        # Create subplots
        n_rows = min(3, (n_plots + 1) // 2)
        n_cols = min(2, n_plots)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        # Handle different axes structures
        if n_plots == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten()
        plot_idx = 0
        
        # Plot force magnitudes
        for col in force_magnitude_cols:
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                ax.plot(event_df['time'], event_df[col])
                ax.set_title(f'{col.replace("_", " ").title()}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Force Magnitude (N)')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Plot force components (all in one subplot)
        if force_component_cols and plot_idx < len(axes_flat):
            ax = axes_flat[plot_idx]
            
            # Group components by base name
            force_groups = {}
            for col in force_component_cols:
                base_name = col.replace('_x', '').replace('_y', '').replace('_z', '')
                if base_name not in force_groups:
                    force_groups[base_name] = []
                force_groups[base_name].append(col)
            
            import matplotlib.cm as cm
            colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(force_groups)))
            for (base_name, components), color in zip(force_groups.items(), colors):
                for comp in components:
                    if comp.endswith('_x'):
                        linestyle = '-'
                        label = f'{base_name} X'
                    elif comp.endswith('_y'):
                        linestyle = '--'
                        label = f'{base_name} Y'
                    elif comp.endswith('_z'):
                        linestyle = ':'
                        label = f'{base_name} Z'
                    else:
                        linestyle = '-'
                        label = comp
                    
                    ax.plot(event_df['time'], event_df[comp], color=color, linestyle=linestyle, 
                           label=label, alpha=0.8)
            
            ax.set_title('Force Components')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Force (N)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot other events
        for col in other_event_cols:
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                ax.plot(event_df['time'], event_df[col])
                ax.set_title(f'{col.replace("_", " ").title()}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'event_analysis.png'), dpi=300, bbox_inches='tight')
            print(f"[INFO] Event analysis plot saved to: {os.path.join(save_path, 'event_analysis.png')}")
        else:
            # Auto-save when no save path provided
            default_save_path = os.path.join(os.path.dirname(self.data_path), 'event_analysis.png')
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Event analysis plot saved to: {default_save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def generate_statistics_report(self, save_path: Optional[str] = None):
        """Generate a comprehensive statistics report using data from all environments."""        
        report = []
        report.append("="*60)
        report.append("ROBOT DATA ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data path: {self.data_path}")
        
        if self.metadata:
            report.append(f"Data creation time: {self.metadata.get('creation_time', 'Unknown')}")
            report.append(f"Save interval: {self.metadata.get('save_interval', 'Unknown')}")
        
        if self.data and 'timesteps' in self.data:
            report.append(f"Total timesteps: {len(self.data['timesteps'])}")
            
        # Get number of environments
        if self.data and 'robot_states' in self.data and self.data['robot_states'].get('base_position'):
            positions = np.array(self.data['robot_states']['base_position'])
            if positions.ndim >= 3:
                num_envs = positions.shape[1]
                report.append(f"Number of environments: {num_envs}")
        
        report.append("")
        report.append("NOTE: All statistics computed across ALL environments")
        report.append("NOTE: Data is filtered to exclude timesteps AFTER first termination in each environment")
        report.append("")
        
        # Get termination info
        term_info = self.get_termination_info()
        if term_info:
            report.append("TERMINATION SUMMARY:")
            report.append("-" * 20)
            report.append(f"Total environments: {term_info['num_environments']}")
            report.append(f"Environments that terminated: {term_info['terminated_envs']}")
            report.append(f"Termination rate: {term_info['termination_percentage']:.1f}%")
            report.append("")
                
        # Robot state statistics (all environments)
        robot_stats = self.get_all_env_robot_stats()
        if robot_stats:
            report.append("ROBOT STATE STATISTICS (PRE-TERMINATION DATA):")
            report.append("-" * 50)
            
            if 'base_speed_mean' in robot_stats:
                report.append(f"Base Speed (m/s):")
                report.append(f"  Mean: {robot_stats['base_speed_mean']:.4f}")
                report.append(f"  Max:  {robot_stats['base_speed_max']:.4f}")
                report.append("")
            
            if 'base_ang_speed_mean' in robot_stats:
                report.append(f"Angular Speed (rad/s):")
                report.append(f"  Mean: {robot_stats['base_ang_speed_mean']:.4f}")
                report.append(f"  Max:  {robot_stats['base_ang_speed_max']:.4f}")
                report.append("")
            
            if 'base_acceleration_mean' in robot_stats:
                report.append(f"Base Acceleration (m/s²):")
                report.append(f"  Mean: {robot_stats['base_acceleration_mean']:.4f}")
                report.append(f"  Max:  {robot_stats['base_acceleration_max']:.4f}")
                report.append("")
            
            if 'base_ang_acceleration_mean' in robot_stats:
                report.append(f"Base Angular Acceleration (rad/s²):")
                report.append(f"  Mean: {robot_stats['base_ang_acceleration_mean']:.4f}")
                report.append(f"  Max:  {robot_stats['base_ang_acceleration_max']:.4f}")
                report.append("")
        
        # Plate statistics (all environments)
        plate_stats = self.get_all_env_plate_stats()
        if plate_stats:
            report.append("PLATE STATISTICS (PRE-TERMINATION DATA):")
            report.append("-" * 45)
            
            if 'plate_speed_mean' in plate_stats:
                report.append(f"Plate Speed (m/s):")
                report.append(f"  Mean: {plate_stats['plate_speed_mean']:.4f}")
                report.append(f"  Max:  {plate_stats['plate_speed_max']:.4f}")
                report.append("")
            
            if 'plate_ang_speed_mean' in plate_stats:
                report.append(f"Plate Angular Speed (rad/s):")
                report.append(f"  Mean: {plate_stats['plate_ang_speed_mean']:.4f}")
                report.append(f"  Max:  {plate_stats['plate_ang_speed_max']:.4f}")
                report.append("")
            
            if 'plate_acceleration_mean' in plate_stats:
                report.append(f"Plate Acceleration (m/s²):")
                report.append(f"  Mean: {plate_stats['plate_acceleration_mean']:.4f}")
                report.append(f"  Max:  {plate_stats['plate_acceleration_max']:.4f}")
                report.append("")
        
        # Object statistics (all environments)
        object_stats = self.get_all_env_object_stats()
        if object_stats:
            report.append("OBJECT STATISTICS (PRE-TERMINATION DATA):")
            report.append("-" * 46)
            
            if 'object_speed_mean' in object_stats:
                report.append(f"Object Speed (m/s):")
                report.append(f"  Mean: {object_stats['object_speed_mean']:.4f}")
                report.append(f"  Max:  {object_stats['object_speed_max']:.4f}")
                report.append("")
            
            if 'object_deviation_mean' in object_stats:
                report.append(f"Object Deviation (m):")
                report.append(f"  Mean: {object_stats['object_deviation_mean']:.4f}")
                report.append(f"  Max:  {object_stats['object_deviation_max']:.4f}")
                report.append("")
            
            if 'object_lin_acc_mean' in object_stats:
                report.append(f"Object Linear Acceleration (m/s²):")
                report.append(f"  Mean: {object_stats['object_lin_acc_mean']:.4f}")
                report.append(f"  Max:  {object_stats['object_lin_acc_max']:.4f}")
                report.append("")
            
            if 'object_ang_acc_mean' in object_stats:
                report.append(f"Object Angular Acceleration (rad/s²):")
                report.append(f"  Mean: {object_stats['object_ang_acc_mean']:.4f}")
                report.append(f"  Max:  {object_stats['object_ang_acc_max']:.4f}")
                report.append("")
        
        # Success rate statistics (all environments)
        success_stats = self.get_success_rate()
        if success_stats:
            report.append("SUCCESS RATE STATISTICS (ALL ENVIRONMENTS):")
            report.append("-" * 45)
            if 'final_success_rate' in success_stats:
                report.append(f"Final Success Rate: {success_stats['final_success_rate']:.2f}%")
                report.append(f"Final Failure Rate: {success_stats['final_failure_rate']:.2f}%")
                report.append("")
            
            if 'overall_success_rate' in success_stats:
                report.append(f"Overall Success Rate: {success_stats['overall_success_rate']:.2f}%")
                report.append(f"Overall Failure Rate: {success_stats['overall_failure_rate']:.2f}%")
                report.append("(Overall = environments that never terminated during the entire episode)")
                report.append("")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(os.path.join(save_path, 'statistics_report.txt'), 'w') as f:
                f.write(report_text)
        
        return report_text
        
    def create_full_analysis(self, output_dir: str):
        """Create a full analysis with all plots and reports."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] Creating full analysis in: {output_dir}")
        
        # Generate all plots
        self.plot_robot_analysis(output_dir)
        self.plot_plate_analysis(output_dir)
        self.plot_object_analysis(output_dir)
        self.plot_command_analysis(output_dir)
        self.plot_termination_analysis(output_dir)
        self.plot_event_analysis(output_dir)
        
        # Generate statistics report
        self.generate_statistics_report(output_dir)
                
        print(f"[INFO] Analysis complete. Results saved to: {output_dir}")


def main():
    """Main function to parse arguments and run analysis.
    
    Usage example:
    python scripts/rsl_rl/data_processing/analyze_robot_data.py --data_path <path_to_data> [--full]
    """
    parser = argparse.ArgumentParser(description="Analyze robot state data from play sessions")
    parser.add_argument("--data_path", default="logs/rsl_rl/unitree_g1_29dof_plate/2025-07-29_23-31-24_plate_v2_resume/push_2025-09-03_01-28-54", help="Path to robot data directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory for analysis results")
    parser.add_argument("--env_id", type=int, default=0, help="Environment index to analyze (default: 0)")
    parser.add_argument("--trajectory", action="store_true", help="Show robot trajectory plot")
    parser.add_argument("--plate", action="store_true", help="Show plate analysis")
    parser.add_argument("--object", action="store_true", help="Show object analysis")
    parser.add_argument("--commands", action="store_true", help="Show command analysis")
    parser.add_argument("--termination", action="store_true", help="Show termination analysis")
    parser.add_argument("--events", action="store_true", help="Show event analysis")
    parser.add_argument("--stats", action="store_true", help="Generate statistics report")
    parser.add_argument("--full", action="store_true", help="Generate full analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"[ERROR] Data path does not exist: {args.data_path}")
        return
    
    # Create analyzer
    analyzer = RobotDataAnalyzer(args.data_path, args.env_id)

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.data_path, "analysis")
    
    # Perform requested analysis
    if args.full:
        analyzer.create_full_analysis(output_dir)
    else:
        # Individual analyses
        save_path = output_dir if args.output else None
        if save_path:
            os.makedirs(save_path, exist_ok=True)
                
        if args.trajectory:
            analyzer.plot_robot_analysis(save_path)
        
        if args.plate:
            analyzer.plot_plate_analysis(save_path)
        
        if args.object:
            analyzer.plot_object_analysis(save_path)
        
        if args.commands:
            analyzer.plot_command_analysis(save_path)
        
        if args.termination:
            analyzer.plot_termination_analysis(save_path)
        
        if args.events:
            analyzer.plot_event_analysis(save_path)
        
        if args.stats:
            analyzer.generate_statistics_report(save_path)
                
        # If no specific analysis requested, show basic info
        if not any([args.trajectory, args.plate, args.object, args.commands, 
                   args.termination, args.events, args.stats]):
            print("[INFO] No specific analysis requested. Use --help to see options.")
        if analyzer.data:
            print(f"[INFO] Data contains {len(analyzer.data['timesteps'])} timesteps")
            print("[INFO] Available analysis options:")
            print("  --env_id <N> : Select environment index to analyze (default: 0)")
            print("  --trajectory : Plot robot trajectory")
            print("  --plate      : Plot plate analysis")
            print("  --object     : Plot object analysis")
            print("  --commands   : Plot command analysis")
            print("  --termination: Plot termination analysis")
            print("  --events     : Plot event analysis")
            print("  --stats      : Generate statistics report")
            print("  --full       : Generate complete analysis")


if __name__ == "__main__":
    main()