import torch
import isaaclab.utils.math as math_utils







def compute_projected_gravity(body_index: int, body_quat_w: torch.Tensor, gravity_vec_w: torch.Tensor) -> torch.Tensor:
    """
    Compute the projected gravity vector in the body frame.
    """
    body_quat_w = body_quat_w[:, body_index, :]
    projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_vec_w)
    return projected_gravity