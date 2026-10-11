"""Experience replay memory D (Algorithm 1 of Mnih et al., 2013).

The paper stores the last N = 1,000,000 frames. Storing every stacked state
(4x84x84) twice would need ~56 GB, so -- as in DeepMind's own code -- we store
each processed frame once and rebuild the 4-frame stacks at sampling time.
Frames from a previous episode are zero-padded out of a stack.
"""
import numpy as np


class FrameReplay:
    def __init__(self, capacity: int, frame_shape=(84, 84), stack: int = 4, seed: int = 0):
        self.N, self.k = capacity, stack
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.ptr, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, frame, action, reward, done):
        """frame = newest frame of s_t; (a_t, r_t, done_t) is what followed it."""
        i = self.ptr
        self.frames[i], self.actions[i], self.rewards[i], self.dones[i] = frame, action, reward, done
        self.ptr = (self.ptr + 1) % self.N
        self.size = min(self.size + 1, self.N)

    def _stack(self, idx):
        out = np.zeros((self.k, *self.frames.shape[1:]), dtype=np.uint8)
        out[-1] = self.frames[idx]
        for j in range(1, self.k):
            prev = (idx - j) % self.N
            if self.dones[prev]:          # previous episode ended -> keep zeros
                break
            out[-1 - j] = self.frames[prev]
        return out

    def _valid(self, idx):
        newest = (self.ptr - 1) % self.N
        d = (newest - idx) % self.N        # how many steps back from newest
        return 1 <= d and d + self.k - 1 <= self.size - 1

    def sample(self, batch: int):
        idxs = []
        while len(idxs) < batch:
            i = int(self.rng.integers(0, self.size))
            if self._valid(i):
                idxs.append(i)
        s = np.stack([self._stack(i) for i in idxs])
        s2 = np.stack([self._stack((i + 1) % self.N) for i in idxs])
        idxs = np.array(idxs)
        return s, self.actions[idxs], self.rewards[idxs], s2, self.dones[idxs].astype(np.float32)

    def __len__(self):
        return self.size


class VectorReplay:
    """Plain replay memory for low-dimensional states (CartPole check)."""

    def __init__(self, capacity, obs_dim, seed=0):
        self.N = capacity
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.ptr = self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.N
        self.size = min(self.size + 1, self.N)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.size
