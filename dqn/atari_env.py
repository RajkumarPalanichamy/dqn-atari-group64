"""Atari environment set up like the 2013 paper.

* Arcade Learning Environment through Gymnasium (ALE/<Game>-v5)
* deterministic emulator (no sticky actions) and the minimal valid action set
* frame-skipping: the agent picks an action every k-th frame and the action is
  repeated on skipped frames; k = 4, except k = 3 for Space Invaders
* reward clipping to {-1, 0, +1} is done in the training loop (training only)
"""
import ale_py
import gymnasium as gym
import numpy as np

from .preprocess import preprocess

gym.register_envs(ale_py)


def frame_skip_for(game: str) -> int:
    return 3 if game.lower().replace("_", "") in ("spaceinvaders",) else 4


class AtariEnv:
    def __init__(self, game="Breakout", seed=0, render_mode=None):
        self.game = game
        self.k = frame_skip_for(game)
        self.env = gym.make(f"ALE/{game}-v5", frameskip=1, repeat_action_probability=0.0,
                            full_action_space=False, render_mode=render_mode)
        self.n_actions = self.env.action_space.n
        self.seed = seed
        self.last_rgb = None

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed += 1
        self.last_rgb = obs
        return preprocess(obs)

    def step(self, action):
        total, done = 0.0, False
        for _ in range(self.k):
            obs, r, term, trunc, _ = self.env.step(action)
            total += r
            if term or trunc:
                done = True
                break
        self.last_rgb = obs
        return preprocess(obs), total, done

    def action_meanings(self):
        return self.env.unwrapped.get_action_meanings()

    def close(self):
        self.env.close()
