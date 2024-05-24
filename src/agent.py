import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKAN as KAN

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


class QNetworkKAN(QNetwork):
    """KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.model = KAN([n_input_channels, 64, 64, self.n_actions])

    def forward(self, x):
        x = self.model(x)
        return x
    
class QNetworkCustom1(QNetwork):
    """CUSTOM KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.kan1 = KAN([n_input_channels, 64])
        self.mlp1 = nn.Linear(64, 64)
        self.mlp2 = nn.Linear(64, self.n_actions)

    def forward(self, x):
        x = self.mlp2(F.relu(self.mlp1(self.kan1(x))))
        return x
    
class QNetworkCustom2(QNetwork):
    """CUSTOM KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.kan1 = KAN([n_input_channels, 64])
        self.mlp1 = nn.Linear(64, 64)
        self.kan2 = KAN([64, self.n_actions])

    def forward(self, x):
        x = self.kan2(F.relu(self.mlp1(self.kan1(x))))
        return x
    
class QNetworkCustom3(QNetwork):
    """CUSTOM KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.kan1 = KAN([n_input_channels, 64])
        self.kan2 = KAN([64, 64])
        self.mlp1 = nn.Linear(64, self.n_actions)

    def forward(self, x):
        x = self.mlp1(self.kan2(self.kan1(x)))
        return x

class QNetworkCustom4(QNetwork):
    """CUSTOM KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.mlp1 = nn.Linear(n_input_channels, 64)
        self.kan1 = KAN([64, 64])
        self.mlp2 = nn.Linear(64, self.n_actions)

    def forward(self, x):
        x = self.mlp2(self.kan1(F.relu(self.mlp1(x))))
        return x

class QNetworkCustom5(QNetwork):
    """CUSTOM KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.mlp1 = nn.Linear(n_input_channels, 64)
        self.mlp2 = nn.Linear(64, 64)
        self.kan1 = KAN([64, self.n_actions])

    def forward(self, x):
        x = self.kan1(F.relu(self.mlp2(F.relu(self.mlp1(x)))))
        return x
    
class QNetworkCustom6(QNetwork):
    """CUSTOM KAN DQN agent."""
    def __init__(self, env):
        super().__init__(env)
        n_input_channels = env.observation_space.shape[1]

        self.mlp1 = nn.Linear(n_input_channels, 64)
        self.kan1 = KAN([64, 64])
        self.kan2 = KAN([64, self.n_actions])

    def forward(self, x):
        x = self.kan2(self.kan1(F.relu(self.mlp1(x))))
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
