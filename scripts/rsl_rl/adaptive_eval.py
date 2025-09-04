# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--renderer",
    type=str,
    default="PathTracing",
    choices=["RayTracedLighting", "PathTracing"],
    help="Renderer to use.",
)
parser.add_argument(
    "--samples_per_pixel_per_frame",
    type=int,
    default=1,
    help="Number of samples per pixel per frame.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--teacher", action="store_true", default=False, help="Run in teacher mode.")
parser.add_argument("--log_interval", type=int, default=10, help="Log metrics to TensorBoard every N steps.")
parser.add_argument("--save_data", action="store_true", default=False, help="Save comprehensive robot state data to files.")
parser.add_argument("--data_name", type=str, default="robot_data", help="Name of the data file to save.")
parser.add_argument("--data_save_interval", type=int, default=1, help="Save data every N steps.")
parser.add_argument("--episode_num", type=int, default=None, help="Number of episodes to run (default: unlimited).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from torch.utils.tensorboard.writer import SummaryWriter

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import g1_humanoid.tasks  # noqa: F401
from residual_adaptive import ResidualAdaptiveRunner, ResidualAdaptiveVecEnvWrapper, ResidualAdaptiveDistillRunner, ResidualAdaptiveEvalRunner
from utils import compute_metrics
from data_processing import RobotDataLogger

def main():


    """
    Use CMD: python joint_play.py --task=Template-G1-Plate-Decoupled-Direct-v0 --num_envs=2 --load_run=2025-07-06_10-40-06
    """


    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        print(f"[INFO] Loading checkpoint from: {log_root_path}/{agent_cfg.load_run}/{agent_cfg.load_checkpoint}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ResidualAdaptiveVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["teacher_mode"] = args_cli.teacher
    runner = ResidualAdaptiveEvalRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_type = "Student" if not args_cli.teacher else "Teacher"

    # Setup TensorBoard logging
    tensorboard_dir = os.path.join(log_dir, "tensorboard", "play")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    print(f"[INFO] TensorBoard logs will be saved to: {tensorboard_dir}")
    print(f"[INFO] To view logs, run: tensorboard --logdir {tensorboard_dir}")
    print(f"Environment type: {type(env)}")
    print(f"Unwrapped environment type: {type(env.unwrapped)}")
    print(f"Policy type: {policy_type}")

    dt = env.unwrapped.step_dt
    max_episode_steps = env.unwrapped.max_episode_length

    data_logger = None
    if args_cli.save_data:
        data_logger = RobotDataLogger(log_dir, args_cli.data_name, args_cli.data_save_interval)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    log_step = 0

    # Initialize tracking for maximum values over all time
    max_values_all_time = {}
    # Initialize tracking for mean values over all time
    mean_values_all_time = {}
    value_counts = {}


    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actor_obs, critic_obs, residual_student_obs, residual_teacher_obs = obs["actor_obs"], obs["critic_obs"], obs["residual_student_obs"], obs["residual_teacher_obs"]
            actions = policy(actor_obs, residual_student_obs, residual_teacher_obs)
            # env stepping
            obs, rewards, dones, info = env.step(actions)
            metrics = compute_metrics(env)

            # log data
            if data_logger is not None:
                    data_logger.log_step_data(env.unwrapped, obs, actions, rewards, dones, timestep, metrics)

            if timestep % max_episode_steps == 0 and data_logger is not None:
                    data_logger.save_data(timestep)

            # Update metric values over all time
            for metric_name, metric_value in metrics.items():
                current_max = torch.max(metric_value).item()
                current_mean = torch.mean(metric_value).item()
                
                # Update max values
                if metric_name not in max_values_all_time:
                    max_values_all_time[metric_name] = current_max
                else:
                    max_values_all_time[metric_name] = max(max_values_all_time[metric_name], current_max)
                
                # Update running mean values
                if metric_name not in mean_values_all_time:
                    mean_values_all_time[metric_name] = current_mean
                    value_counts[metric_name] = 1
                else:
                    # Running average formula: new_avg = (old_avg * count + new_value) / (count + 1)
                    old_count = value_counts[metric_name]
                    mean_values_all_time[metric_name] = (mean_values_all_time[metric_name] * old_count + current_mean) / (old_count + 1)
                    value_counts[metric_name] += 1
            
            # Log metrics to TensorBoard 
            if timestep % args_cli.log_interval == 0:
                print(f"\nStep {timestep}: ", end="")
                for metric_name, metric_value in metrics.items():
                    # Log mean values across all environments (current step)
                    mean_value = torch.mean(metric_value).item()
                    writer.add_scalar(f"metrics/{metric_name}", mean_value, log_step)
                                        
                    # Log maximum value over all time
                    writer.add_scalar(f"metrics/{metric_name}_max_all_time", max_values_all_time[metric_name], log_step)
                    
                    # Log mean value over all time
                    writer.add_scalar(f"metrics/{metric_name}_mean_all_time", mean_values_all_time[metric_name], log_step)
                    
                    # Print to console
                    print(f"{metric_name}: {mean_value:.4f} (max: {max_values_all_time[metric_name]:.4f}, avg: {mean_values_all_time[metric_name]:.4f}) ", end="")
                
                # Log step timing
                step_time = time.time() - start_time
                writer.add_scalar("timing/step_time", step_time, log_step)
                print(f"| Step time: {step_time:.4f}s")
                log_step += 1

        timestep += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # Check for episode limit
        if args_cli.episode_num is not None and timestep >= args_cli.episode_num * max_episode_steps - 1:
            print(f"[INFO] Reached maximum episodes ({args_cli.episode_num}). Stopping simulation.")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Save final data if logging was enabled
    if data_logger is not None:
        data_logger.save_final_data()

    # Close TensorBoard writer
    writer.close()
    print(f"[INFO] TensorBoard logging completed. {log_step} steps logged.")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
