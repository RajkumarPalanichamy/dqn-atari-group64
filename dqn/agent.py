"""Deep Q-learning with experience replay -- Algorithm 1, Mnih et al. (2013).

    y_j = r_j                                   if s_{j+1} is terminal
    y_j = r_j + gamma * max_a' Q(s_{j+1}, a'; theta)   otherwise
    gradient step on (y_j - Q(s_j, a_j; theta))^2

In the 2013 paper the target is computed with the *current* parameters
(theta_{i-1}, i.e. the weights before this update) -- there is NO separate,
periodically-synced target network; that was introduced in Nature (2015).
`target_update` > 0 switches the 2015 target network on, for comparison only.
"""
import copy

import numpy as np
import torch
import torch.nn as nn


def linear_epsilon(step, start=1.0, end=0.1, anneal_steps=1_000_000):
    """epsilon annealed linearly 1.0 -> 0.1 over the first million frames, then fixed."""
    frac = min(1.0, step / anneal_steps)
    return start + frac * (end - start)


class DQNAgent:
    def __init__(self, qnet: nn.Module, n_actions: int, gamma=0.99, lr=2.5e-4,
                 rms_alpha=0.95, rms_eps=0.01, loss="mse", target_update=0,
                 device="cpu", seed=0):
        self.q = qnet.to(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        # RMSProp, as stated in the paper (values of lr/alpha/eps are not given
        # in 2013; these are the published DeepMind defaults).
        self.opt = torch.optim.RMSprop(self.q.parameters(), lr=lr, alpha=rms_alpha, eps=rms_eps)
        self.loss_fn = nn.MSELoss() if loss == "mse" else nn.SmoothL1Loss()
        self.target_update = target_update
        self.q_target = copy.deepcopy(self.q) if target_update > 0 else None
        self.updates = 0
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        s = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return int(self.q(s).argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, states):
        return self.q(torch.as_tensor(states, device=self.device)).cpu().numpy()

    def update(self, batch):
        s, a, r, s2, done = [torch.as_tensor(x, device=self.device) for x in batch]
        q_sa = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            net = self.q_target if self.q_target is not None else self.q   # 2013: same theta
            max_next = net(s2).max(dim=1).values
            y = r + self.gamma * (1.0 - done) * max_next                   # y = r at terminal
        loss = self.loss_fn(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.q_target is not None and self.updates % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item()), float(q_sa.mean().item())
