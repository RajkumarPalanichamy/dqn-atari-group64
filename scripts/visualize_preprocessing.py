"""Figure: the preprocessing pipeline on real Breakout frames from the emulator."""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.atari_env import AtariEnv  # noqa: E402
from dqn.preprocess import CROP_TOP, FrameStack  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)

env = AtariEnv("Breakout", seed=3)
fs = FrameStack()
f = env.reset()
fs.reset(f)
env.step(1)  # FIRE to launch the ball
rng = np.random.default_rng(0)
for _ in range(20):
    f, _, _ = env.step(int(rng.choice([2, 3])))
    fs.push(f)
rgb = env.last_rgb
grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
small = cv2.resize(grey, (84, 110), interpolation=cv2.INTER_AREA)
crop = small[CROP_TOP:CROP_TOP + 84]
stack = fs.state()

fig, ax = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw={"width_ratios": [160, 160, 84, 84]})
ax[0].imshow(rgb); ax[0].set_title("1. Raw frame\n210 x 160 RGB")
ax[1].imshow(grey, cmap="gray"); ax[1].set_title("2. Grey-scale\n210 x 160")
ax[2].imshow(small, cmap="gray"); ax[2].set_title("3. Down-sampled\n110 x 84")
ax[2].add_patch(plt.Rectangle((-0.5, CROP_TOP - 0.5), 84, 84, fill=False, ec="#F2A01D", lw=2))
ax[3].imshow(crop, cmap="gray"); ax[3].set_title("4. Cropped input\n84 x 84")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "preprocessing_pipeline.png"), dpi=150)

fig, ax = plt.subplots(1, 4, figsize=(10, 3))
for i in range(4):
    ax[i].imshow(stack[i], cmap="gray")
    ax[i].set_title(f"t-{3-i}" if i < 3 else "t (newest)")
    ax[i].set_xticks([]); ax[i].set_yticks([])
plt.suptitle("Network input phi(s): last 4 preprocessed frames, shape (4, 84, 84)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "frame_stack.png"), dpi=150)
print("saved preprocessing figures to", OUT)
