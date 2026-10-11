"""Input pre-processing exactly as described in Mnih et al. (2013), Section 4.1.

210x160 RGB frame -> grey-scale -> down-sample to 110x84 -> crop an 84x84
region that roughly covers the playing area. The last 4 processed frames are
stacked to form the network input  phi(s)  of shape (4, 84, 84).
"""
from collections import deque

import cv2
import numpy as np

FRAME_H, FRAME_W = 84, 84
STACK = 4
# Rows kept from the 110x84 image. The paper only says "roughly captures the
# playing area"; 18..102 removes the score bar at the top and the bottom border.
CROP_TOP = 18


def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    """(210,160,3) uint8 RGB -> (84,84) uint8 grey."""
    grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (84, 110), interpolation=cv2.INTER_AREA)  # (w, h)
    return small[CROP_TOP:CROP_TOP + FRAME_H, :]


class FrameStack:
    """Keeps the last 4 processed frames. At the start of an episode the
    missing history is zero-padded (same convention as the replay memory)."""

    def __init__(self, k: int = STACK):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.k - 1):
            self.frames.append(np.zeros_like(frame))
        self.frames.append(frame)
        return self.state()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return self.state()

    def state(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)
