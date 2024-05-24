from dataclasses import dataclass
from typing import Union, Tuple, Optional # Only needed in Python < 3.10
from agent import QNetworkBase, QNetworkNature, QNetworkKAN, QNetworkCustom1, QNetworkCustom2, QNetworkCustom3, QNetworkCustom4, QNetworkCustom5, QNetworkCustom6


@dataclass
class ConfigLunarKAN_custom1:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "KAN-OR"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = None
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Running Lunar lander with fast-KAN networks and base buffer"
    wandb_group: str = "Lunarlander-KAN"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkKAN, QNetworkCustom1, QNetworkCustom2, QNetworkCustom3] = QNetworkCustom1
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

    # For logging KAN neurons; chooses type of binary mask operation
    use_or = True



@dataclass
class ConfigLunarKAN_custom2:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "KAN-OR"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = None
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Running Lunar lander with fast-KAN networks and base buffer"
    wandb_group: str = "Lunarlander-KAN"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkKAN, QNetworkCustom1, QNetworkCustom2, QNetworkCustom3] = QNetworkCustom2
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

    # For logging KAN neurons; chooses type of binary mask operation
    use_or = True




@dataclass
class ConfigLunarKAN_custom3:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "KAN-OR"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = None
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Running Lunar lander with fast-KAN networks and base buffer"
    wandb_group: str = "Lunarlander-KAN"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkKAN, QNetworkCustom1, QNetworkCustom2, QNetworkCustom3] = QNetworkCustom3
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

    # For logging KAN neurons; chooses type of binary mask operation
    use_or = True

@dataclass
class ConfigLunarKAN_custom4:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "KAN-OR"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = None
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Running Lunar lander with fast-KAN networks and base buffer"
    wandb_group: str = "Lunarlander-KAN"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkKAN, QNetworkCustom4] = QNetworkCustom4
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

    # For logging KAN neurons; chooses type of binary mask operation
    use_or = True

@dataclass
class ConfigLunarKAN_custom5:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "KAN-OR"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = None
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Running Lunar lander with fast-KAN networks and base buffer"
    wandb_group: str = "Lunarlander-KAN"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkCustom5] = QNetworkCustom5
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

    # For logging KAN neurons; chooses type of binary mask operation
    use_or = True

@dataclass
class ConfigLunarKAN_custom6:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "KAN-OR"
    tags: Union[Tuple[str, ...], str, None] = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: Optional[int] = None
    track: bool = True
    wandb_project_name: str = "ReDo"
    wandb_entity: Optional[str] = "mason-nakamura1"
    wandb_notes: str = "Running Lunar lander with fast-KAN networks and base buffer"
    wandb_group: str = "Lunarlander-KAN"
    capture_video: bool = False
    save_model: bool = False

    # Buffer settings
    use_per: bool = False

    # Environment settings
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    QNetwork: Union[QNetworkBase, QNetworkNature, QNetworkCustom6] = QNetworkCustom6
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

    # For logging KAN neurons; chooses type of binary mask operation
    use_or = True