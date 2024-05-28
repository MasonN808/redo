from dataclasses import dataclass
from typing import Union, Tuple, Optional # Only needed in Python < 3.10
from agent import QNetworkBase, QNetworkNature

@dataclass
class ConfigDemonAttack:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN Atari Environemnts"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = 0
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Trying to Reproduce ReDo Results"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "ALE/DemonAttack-v5"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature] = QNetworkNature
    buffer_size: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4, theirs: 6.25 * 1e-5
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = True  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8000  # cleanRL default: 8000, 4 freq -> 8000, 1 -> 2000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000  # cleanRL default: 80000, theirs 20000
    train_frequency: int = 4  # cleanRL default: 4, theirs 1

    # ReDo settings
    enable_redo: bool = True
    redo_tau: float = 0.025  # 0.025 for default, else 0.1
    redo_check_interval: int = 1000
    redo_bs: int = 64
    wandb_group: str = f"{env_id}-ReDo={enable_redo}"