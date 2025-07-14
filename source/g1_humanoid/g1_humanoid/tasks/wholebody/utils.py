import torch
import isaaclab.utils.math as math_utils

def compute_projected_gravity(body_index: int, body_quat_w: torch.Tensor, gravity_vec_w: torch.Tensor) -> torch.Tensor:
    """
    Compute the projected gravity vector in the body frame.
    """
    body_quat_w = body_quat_w[:, body_index, :]
    try:
        # Use quat_apply_inverse for quaternion rotation
        projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_vec_w)
    except AttributeError:
        # Fallback to quat_rotate_inverse if quat_apply_inverse is not available
        print("Using quat_rotate_inverse as fallback.")
        projected_gravity = math_utils.quat_rotate_inverse(body_quat_w, gravity_vec_w)
    return projected_gravity