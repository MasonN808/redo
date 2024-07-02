import math
import typing

import sys, os
import gymnasium as gym
# sys.path.insert(0, os.path.abspath(os.path.join('gym-minigrid')))

sys.path.insert(0, os.path.abspath(os.path.join('libs')))
print(sys.path)
import Minigrid
import torch
import torch.nn as nn

from wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from Minigrid.minigrid.wrappers import *

ATARI_ENVS = [
    "ALE/DemonAttack-v5",
    "ALE/Asterix-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Pong-v5",
]

MINIGRID_ENVS = [
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-MultiSkill-N2-v0',
]

CUSTOM_MINIGRID = [
    'GridWorld-v0'
]

def make_env(env_params: typing.Dict[str, Any], seed: int, idx: int, capture_video: bool, run_name: str):
    """Helper function to create an environment with some standard wrappers."""

    def thunk():
        env_name = env_params.pop("env_name")
        
        if capture_video and idx == 0:
            env = gym.make(env_name, render_mode="rgb_array", **env_params)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name, **env_params)

        if env_name in ATARI_ENVS:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)

        if env_name in MINIGRID_ENVS:
            env = ImgObsWrapper(env)
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


def set_cuda_configuration(gpu: typing.Any) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""

    if gpu is None or gpu == -1 or gpu is False:
        device = torch.device("cpu")
    elif isinstance(gpu, int):
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda")

    return device


@torch.no_grad()
def lecun_normal_initializer(layer: nn.Module) -> None:
    """
    Initialization according to LeCun et al. (1998).
    See here https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.initializers.lecun_normal.html
    and here https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L460 .
    Initializes bias terms to 0.
    """

    # Catch case where the whole network is passed
    if not isinstance(layer, nn.Linear | nn.Conv2d):
        return

    # For a conv layer, this is num_channels*kernel_height*kernel_width
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)

    # This implementation follows the jax one
    # https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L260
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    torch.nn.init.trunc_normal_(layer.weight)
    layer.weight *= stddev
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
