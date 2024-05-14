import torch
import torch.nn as nn
import torch.nn.functional as F
from tfkan.layers import DenseKAN
import tensorflow as tf

class QNetwork(nn.Module):
    """Base class for different QNetwork configurations."""

    def __init__(self, env):
        super().__init__()
        self.n_actions = int(env.single_action_space.n)

    def forward(self, x):
        raise NotImplementedError("Each model must implement its own forward method.")

class QNetworkNature(QNetwork):
    """Basic nature DQN agent."""

    def __init__(self, env):
        super().__init__(env)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.q = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.q(x)
        return x

# TODO: Finish. This is utilizing keras but not sure if this will change the entire code base.
class QNetworkKAN(QNetwork):
    """KAN DQN agent."""
    # See https://github.com/ZPZhou-lab/tfkan?tab=readme-ov-file#how-to-use for KAN tensorflow implementation
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.model = tf.keras.models.Sequential([
            # DenseKAN(n_input_channels),
            DenseKAN(64),
            DenseKAN(64),
            DenseKAN(self.n_actions)
        ])

        self.model.build(input_shape=(None, n_input_channels))

    def forward(self, x):
        x = self.model(x)
        return x

    
class QNetworkKANConv(QNetwork):
    """Basic nature DQN agent."""

    def __init__(self, env):
        super().__init__(env)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.q = nn.Linear(512, self.n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.q(x)
        return x

class QNetworkBase(QNetwork):
    """Base Agent with no preprocessing"""

    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]
        self.fc1 = nn.Linear(n_input_channels, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q = nn.Linear(64,  self.n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
