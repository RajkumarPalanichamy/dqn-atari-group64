"""Unit tests: shapes from the paper, replay stacking, terminal-state target."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.agent import DQNAgent, linear_epsilon  # noqa: E402
from dqn.model import DQN2013  # noqa: E402
from dqn.preprocess import preprocess  # noqa: E402
from dqn.replay import FrameReplay  # noqa: E402


def test_preprocess_shape():
    frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    assert preprocess(frame).shape == (84, 84)


def test_network_matches_paper():
    net = DQN2013(n_actions=4)
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [(c.out_channels, c.kernel_size, c.stride) for c in convs] == [(16, (8, 8), (4, 4)), (32, (4, 4), (2, 2))]
    out = net(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
    assert out.shape == (2, 4)
    assert net.head[1].out_features == 256


def test_epsilon_schedule():
    assert linear_epsilon(0) == 1.0
    assert abs(linear_epsilon(500_000) - 0.55) < 1e-9
    assert abs(linear_epsilon(5_000_000) - 0.1) < 1e-9


def test_replay_zero_pads_across_episodes():
    m = FrameReplay(100)
    for t in range(10):
        m.add(np.full((84, 84), t, np.uint8), 0, 0.0, done=(t == 4))
    st = m._stack(6)                   # episode 2 started at t=5
    assert st[0].max() == 0 and st[1].max() == 0 and st[2].max() == 5 and st[3].max() == 6


def test_terminal_target_is_reward():
    torch.manual_seed(0)
    agent = DQNAgent(DQN2013(4), 4, lr=0.0)
    s = np.zeros((1, 4, 84, 84), np.uint8)
    with torch.no_grad():
        q_before = agent.q(torch.as_tensor(s))[0, 0].item()
    loss, _ = agent.update((s, np.array([0]), np.array([1.0], np.float32), s, np.array([1.0], np.float32)))
    assert abs(loss - (1.0 - q_before) ** 2) < 1e-5   # y = r, no bootstrap
