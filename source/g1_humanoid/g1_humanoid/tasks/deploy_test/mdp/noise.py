from isaaclab.utils.noise import NoiseCfg
from isaaclab.utils.noise import noise_model
import torch
from isaaclab.utils.math import quat_from_euler_xyz,quat_mul
from isaaclab.utils import configclass



def uniform_noise_quat(quat: torch.Tensor, cfg: NoiseCfg) -> torch.Tensor:
    num_env = quat.shape[0]
    device = quat.device
    euler_angle_min = cfg.min
    euler_angle_max = cfg.max

    delta_euler_xyz = torch.rand(size=(num_env, 3), device=device) * (euler_angle_max - euler_angle_min) + euler_angle_min
    noisy_quat = quat_from_euler_xyz(delta_euler_xyz[:, 0], delta_euler_xyz[:, 1], delta_euler_xyz[:, 2])

    obs = quat_mul(quat, noisy_quat)
    return obs

def gaussian_noise_quat(quat: torch.Tensor, cfg: NoiseCfg) -> torch.Tensor:
    num_env = quat.shape[0]
    device = quat.device
    mean = cfg.mean
    std = cfg.std

    delta_euler_xyz = torch.randn(size=(num_env, 3), device=device) * std + mean
    noisy_quat = quat_from_euler_xyz(delta_euler_xyz[:, 0], delta_euler_xyz[:, 1], delta_euler_xyz[:, 2])
    obs = quat_mul(quat, noisy_quat)
    return obs



@configclass
class GaussianNoiseQuatCfg(NoiseCfg):
    func = gaussian_noise_quat
    mean: float = 0.0
    std: float = 0.005

@configclass
class UniformNoiseQuatCfg(NoiseCfg):
    func = uniform_noise_quat
    min: float = -0.005
    max: float = 0.005
