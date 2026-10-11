"""Fast sanity check of the same Deep Q-learning algorithm on CartPole-v1.

Runs in a few minutes on a laptop CPU, so every group member can reproduce it.
It also runs the ablation that justifies the paper's key idea: training WITH
experience replay vs. WITHOUT it (online, consecutive samples only).

    python train_cartpole.py                # 3 seeds x {replay, no-replay}
"""
import argparse
import csv
import os

import gymnasium as gym
import numpy as np
import torch

from dqn.agent import DQNAgent, linear_epsilon
from dqn.model import MLPQ
from dqn.replay import VectorReplay


def run(seed, use_replay, steps, out):
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    agent = DQNAgent(MLPQ(obs_dim, n_act), n_act, gamma=0.99, lr=1e-3, seed=seed)
    mem = VectorReplay(50_000 if use_replay else 1, obs_dim, seed=seed)
    s, _ = env.reset(seed=seed)
    ep_ret, rows, ep = 0.0, [], 0
    for t in range(1, steps + 1):
        eps = linear_epsilon(t, 1.0, 0.05, steps // 2)
        a = agent.act(s, eps)
        s2, r, term, trunc, _ = env.step(a)
        mem.add(s, a, r, s2, float(term))          # terminal -> y = r
        ep_ret += r
        s = s2
        if term or trunc:
            rows.append([t, ep, ep_ret])
            ep += 1
            ep_ret = 0.0
            s, _ = env.reset()
        if use_replay and len(mem) >= 1_000:
            agent.update(mem.sample(32))
        elif not use_replay:
            agent.update(mem.sample(1))           # most recent transition only
    env.close()
    name = f"cartpole_{'replay' if use_replay else 'noreplay'}_seed{seed}.csv"
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "episode", "return"])
        w.writerows(rows)
    last = np.mean([r[2] for r in rows[-20:]])
    print(f"replay={use_replay!s:5}  seed={seed}  episodes={len(rows):4d}  mean return (last 20) = {last:6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for use_replay in (True, False):
        for sd in a.seeds:
            run(sd, use_replay, a.steps, a.out)
