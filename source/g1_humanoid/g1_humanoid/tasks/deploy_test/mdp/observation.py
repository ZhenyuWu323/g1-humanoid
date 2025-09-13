from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



def plate_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Projected gravity on the plate."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b

def plate_lin_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Linear acceleration of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_acc_w


def plate_ang_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Angular acceleration of the plate in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_ang_acc_w

