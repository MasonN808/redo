from dataclasses import dataclass
from typing import Union, Tuple, Optional # Only needed in Python < 3.10
from src.agent import QNetworkBase, QNetworkNature, QNetworkKAN

@dataclass
class ConfigDemon:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN DemonAttack"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = 0
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "running for 30million time steps to see if double descent occurs and increasing memory"
    wandb_group: str = "DemonAttack-2"
    capture_video: bool = True
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "ALE/DemonAttack-v5"
    total_timesteps: int = 30_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature] = QNetworkNature
    buffer_size: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4, theirs: 6.25 * 1e-5
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8000  # cleanRL default: 8000, 4 freq -> 8000, 1 -> 2000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000  # cleanRL default: 80000, theirs 20000
    train_frequency: int = 4  # cleanRL default: 4, theirs 1

    # ReDo settings
    enable_redo: bool = False
    redo_tau: float = 0.025  # 0.025 for default, else 0.1
    redo_check_interval: int = 1000
    redo_bs: int = 64

@dataclass
class ConfigLunar:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN LunarLander"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = 0
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Testing"
    wandb_group: str = ""
    capture_video: bool = True
    save_model: bool = False

    # Buffer settings
    use_per: bool = True

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature] = QNetworkBase
    buffer_size: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4, theirs: 6.25 * 1e-5
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8000  # cleanRL default: 8000, 4 freq -> 8000, 1 -> 2000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000  # cleanRL default: 80000, theirs 20000
    train_frequency: int = 4  # cleanRL default: 4, theirs 1

    # ReDo settings
    enable_redo: bool = False
    redo_tau: float = 0.025  # 0.025 for default, else 0.1
    redo_check_interval: int = 1000
    redo_bs: int = 64

@dataclass
class ConfigLunarKAN:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN LunarLander"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = 0
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Testing KAN Networks as simple Q-network for DQN"
    wandb_group: str = ""
    capture_video: bool = True
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkKAN] = QNetworkKAN
    buffer_size: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4, theirs: 6.25 * 1e-5
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8000  # cleanRL default: 8000, 4 freq -> 8000, 1 -> 2000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80_000  # cleanRL default: 80000, theirs 20000
    train_frequency: int = 4  # cleanRL default: 4, theirs 1

    # ReDo settings
    enable_redo: bool = False
    redo_tau: float = 0.025  # 0.025 for default, else 0.1
    redo_check_interval: int = 1000
    redo_bs: int = 64

# @dataclass
# class ConfigMujoco:
#     """Configuration for a ReDo DQN agent."""

#     # Experiment settings
#     exp_name: str = "ReDo DQN LunarLander"
#     tags: Union[Tuple[str, ...], str, None] = None
#     seed: int = 0
#     torch_deterministic: bool = True
#     gpu: Optional[int] = 0
#     track: bool = True
#     wandb_project_name: str = "ReDo"
#     wandb_entity: Optional[str] = "mason-nakamura1"
#     wandb_notes: str = "Testing"
#     wandb_group: str = ""
#     capture_video: bool = True
#     save_model: bool = False

#     # Environment settings
#     env_id: str = "LunarLander-v2"
#     total_timesteps: int = 10_000_000
#     num_envs: int = 1

#     # DQN settings
#     QNetwork: Union[QNetworkBase, QNetworkNature] = QNetworkBase
#     buffer_size: int = 1_000_000
#     batch_size: int = 32
#     learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4, theirs: 6.25 * 1e-5
#     adam_eps: float = 1.5 * 1e-4
#     use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
#     gamma: float = 0.99
#     tau: float = 1.0
#     target_network_frequency: int = 8000  # cleanRL default: 8000, 4 freq -> 8000, 1 -> 2000
#     start_e: float = 1.0
#     end_e: float = 0.01
#     exploration_fraction: float = 0.10
#     learning_starts: int = 80_000  # cleanRL default: 80000, theirs 20000
#     train_frequency: int = 4  # cleanRL default: 4, theirs 1

#     # ReDo settings
#     enable_redo: bool = False
#     redo_tau: float = 0.025  # 0.025 for default, else 0.1
#     redo_check_interval: int = 1000
#     redo_bs: int = 64
