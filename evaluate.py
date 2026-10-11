"""Evaluate a trained checkpoint with the paper's protocol (epsilon = 0.05) and
save (a) a GIF of one episode and (b) a frame with the network's Q-values.

    python evaluate.py --game Breakout --ckpt runs/Breakout_2013_seed1_best.pt
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dqn.agent import DQNAgent
from dqn.atari_env import AtariEnv
from dqn.model import DQN2013
from dqn.preprocess import FrameStack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Breakout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default=None, help="prefix for output files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = a.name or a.game.lower()

    env, fs = AtariEnv(a.game, seed=2024), FrameStack()
    net = DQN2013(env.n_actions); net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    agent = DQNAgent(net, env.n_actions, seed=7)
    names = env.action_meanings()
    scores, gif, q_shot = [], [], None
    for ep in range(a.episodes):
        s = fs.reset(env.reset()); done, total, t = False, 0.0, 0
        while not done and t < 5000:
            q = agent.q_values(s[None])[0]
            if ep == 0 and t == 60:
                q_shot = (env.last_rgb.copy(), q.copy())   # frame the Q-values were computed on
            act = agent.act(s, a.epsilon)
            f, r, done = env.step(act)
            s = fs.push(f); total += r; t += 1
            if ep == 0 and t % 2 == 0 and len(gif) < 400:
                gif.append(Image.fromarray(env.last_rgb).resize((320, 420), Image.NEAREST))
        scores.append(total)
        print(f"episode {ep+1:2d}: score = {total:.0f}")
    print(f"\n{a.game}: average score over {a.episodes} episodes (eps={a.epsilon}) = {np.mean(scores):.2f}"
          f"  (min {np.min(scores):.0f}, max {np.max(scores):.0f})")
    if gif:
        gif[0].save(os.path.join(a.out, f"{name}_agent.gif"), save_all=True, append_images=gif[1:], duration=50, loop=0)
    if q_shot:
        img, q = q_shot
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
        ax[0].imshow(img); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_title("game state")
        cols = ["#F2A01D" if i == q.argmax() else "#9AA2BE" for i in range(len(q))]
        ax[1].barh(names, q, color=cols); ax[1].invert_yaxis()
        ax[1].set_title("Q(s, a; theta) - highlighted = greedy action")
        ax[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, f"{name}_q_values.png"), dpi=150)


if __name__ == "__main__":
    main()
