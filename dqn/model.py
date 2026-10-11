"""Deep Q-Network architecture from Mnih et al. (2013), Section 5.

input 84x84x4
 -> conv 16 filters 8x8 stride 4 + ReLU
 -> conv 32 filters 4x4 stride 2 + ReLU
 -> fully connected 256 + ReLU
 -> linear output, one Q-value per valid action (4..18 depending on the game)
"""
import torch
import torch.nn as nn


class DQN2013(nn.Module):
    def __init__(self, n_actions: int, in_frames: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_frames, 16, kernel_size=8, stride=4), nn.ReLU(),   # -> 16x20x20
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),          # -> 32x9x9
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: uint8 (B,4,84,84) -> scale to [0,1]
        return self.head(self.features(x.float() / 255.0))


class MLPQ(nn.Module):
    """Small fully-connected Q-network used for the fast CartPole sanity check
    (same algorithm, low-dimensional state instead of pixels)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
