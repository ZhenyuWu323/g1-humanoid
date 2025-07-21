# 完全仿照UniformVelocityCommand的可视化模式

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

def push_by_setting_velocity_with_vis(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    debug_vis: bool = False,
    vis_duration: float = 2.0,
):
    """Push the asset by setting velocity with visualization following velocity command pattern.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to apply push
        velocity_range: Velocity ranges for each axis
        asset_cfg: Asset configuration
        debug_vis: Whether to show visualization (creates markers like velocity command)
        vis_duration: How long to show visualization (seconds)
    """
    # ========== 原有的push逻辑 ==========
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # get current velocities
    vel_w = asset.data.root_vel_w[env_ids]
    
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    
    # apply velocities
    vel_w += sampled_vel
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
    
    # ========== 仿照velocity command的可视化模式 ==========
    if debug_vis:
        _setup_push_visualization_like_velocity_command(env, env_ids, sampled_vel, asset, vis_duration)


def _setup_push_visualization_like_velocity_command(env, env_ids: torch.Tensor, applied_vel: torch.Tensor, asset, duration: float):
    """仿照velocity command的方式设置push可视化"""
    
    # 1. 检查并创建可视化器（仿照velocity command的模式）
    if not hasattr(env, 'push_visualizer'):
        _create_push_visualizer_like_velocity_command(env)
    
    # 2. 存储push数据（仿照velocity command存储command的方式）
    if not hasattr(env, '_push_data'):
        env._push_data = {
            'active_pushes': torch.zeros(env.num_envs, 3, device=env.device),  # 存储当前push速度
            'push_timers': torch.zeros(env.num_envs, device=env.device),       # 存储显示计时器
            'push_active_mask': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # 哪些env有active push
        }
    
    # 3. 更新push数据
    linear_vel = applied_vel[:, :3]  # 只取线性速度
    env._push_data['active_pushes'][env_ids] = linear_vel
    env._push_data['push_timers'][env_ids] = duration
    env._push_data['push_active_mask'][env_ids] = torch.norm(linear_vel, dim=1) > 0.05  # 只显示有意义的push
    
    # 4. 设置可视化回调（仿照velocity command的debug callback）
    if not hasattr(env, '_push_vis_callback_registered'):
        _register_push_debug_callback(env, asset)


def _create_push_visualizer_like_velocity_command(env):


    # 仿照velocity command的goal_vel_visualizer_cfg和current_vel_visualizer_cfg
    push_arrow_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PushCommand",  # 仿照velocity command的路径命名
        markers={
            "arrow": {
                "func": "arrow",
                "scale": [1.0, 1.0, 1.0],  # 默认scale，会动态调整
                "color": (1.0, 0.2, 0.2),  # 红色箭头表示push
            }
        }
    )
    
    push_point_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PushPoint",
        markers={
            "sphere": {
                "func": "sphere", 
                "radius": 0.08,
                "color": (1.0, 0.5, 0.0),  # 橙色球体表示push点
            }
        }
    )
    
    # 创建可视化器（仿照velocity command的模式）
    try:
        env.push_visualizer = VisualizationMarkers(push_arrow_cfg)
        env.push_point_visualizer = VisualizationMarkers(push_point_cfg)
        
        # 设置可见性（仿照velocity command的set_visibility）
        env.push_visualizer.set_visibility(True)
        env.push_point_visualizer.set_visibility(True)
        
        print("[INFO] Created push visualizers following velocity command pattern")
        
    except Exception as e:
        print(f"[WARNING] Could not create push visualizers: {e}")
        env.push_visualizer = None
        env.push_point_visualizer = None


def _register_push_debug_callback(env, asset):
    """注册push可视化回调,完全仿照velocity command的_debug_vis_callback"""
    
    # 定义debug callback函数（仿照velocity command的_debug_vis_callback）
    def _push_debug_vis_callback(event):
        # 检查robot是否初始化（仿照velocity command的检查）
        if not asset.is_initialized:
            return
            
        # 检查是否有可视化器
        if not hasattr(env, 'push_visualizer') or env.push_visualizer is None:
            return
            
        # 获取有active push的环境
        active_mask = env._push_data['push_active_mask']
        if not active_mask.any():
            return
            
        active_env_ids = torch.where(active_mask)[0]
        
        # 获取marker位置（仿照velocity command的base_pos_w计算）
        base_pos_w = asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.8  # 在机器人上方80cm（仿照velocity command的+0.5）
        
        # 获取active push的位置和速度
        active_positions = base_pos_w[active_env_ids]
        active_push_vels = env._push_data['active_pushes'][active_env_ids, :2]  # 只取XY方向
        
        # 计算箭头scale和朝向（完全仿照velocity command的_resolve_xy_velocity_to_arrow）
        arrow_scale, arrow_quat = _resolve_push_velocity_to_arrow(env, active_push_vels, asset, active_env_ids)
        
        # 显示可视化（仿照velocity command的visualize调用）
        env.push_visualizer.visualize(active_positions, arrow_quat, arrow_scale)
        env.push_point_visualizer.visualize(active_positions)
        
        # 更新计时器
        env._push_data['push_timers'] -= env.step_dt
        expired_mask = env._push_data['push_timers'] <= 0.0
        env._push_data['push_active_mask'][expired_mask] = False
    
    # 注册回调（仿照velocity command的callback注册方式）
    import omni.kit.app
    app_interface = omni.kit.app.get_app_interface()
    env._push_debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
        _push_debug_vis_callback
    )
    
    env._push_vis_callback_registered = True
    print("[INFO] Registered push debug visualization callback")


def _resolve_push_velocity_to_arrow(env, xy_velocity: torch.Tensor, asset, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """完全仿照velocity command的_resolve_xy_velocity_to_arrow方法"""
    
    # 获取默认scale（仿照velocity command的方式）
    default_scale = env.push_visualizer.cfg.markers["arrow"].scale
    
    # 计算arrow scale（仿照velocity command的算法）
    arrow_scale = torch.tensor(default_scale, device=env.device).repeat(xy_velocity.shape[0], 1)
    arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 2.0  # push的scale稍微小一点
    
    # 计算arrow方向（完全仿照velocity command的算法）
    heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    zeros = torch.zeros_like(heading_angle)
    arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
    
    # 转换到世界坐标系（完全仿照velocity command的转换）
    base_quat_w = asset.data.root_quat_w[env_ids]
    arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
    
    return arrow_scale, arrow_quat


