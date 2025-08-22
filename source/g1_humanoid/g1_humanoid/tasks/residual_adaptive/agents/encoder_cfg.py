from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg
from dataclasses import MISSING
from typing import Literal


@configclass
class RslRlActorCriticEncoderCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticEncoder"
    """The policy class name. Default is ActorCriticEncoder."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    encoder_d_model: int = MISSING
    """The dimension of the encoder."""

    encoder_nhead: int = MISSING
    """The number of attention heads in the encoder."""

    encoder_num_layers: int = MISSING
    """The number of layers in the encoder."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""