"""Capture the game screenshots used in the presentation directly from the
Arcade Learning Environment (so every image on the slides has a known source).

    python scripts/make_slide_images.py figures
"""
import os
import sys

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

gym.register_envs(ale_py)
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
NAVY, NAVY2, WHITE, GOLD = (20, 26, 51), (34, 43, 82), (255, 255, 255), (242, 160, 29)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GAMES = [("BeamRider", "Beam Rider"), ("Breakout", "Breakout"), ("Enduro", "Enduro"), ("Pong", "Pong"),
         ("Qbert", "Q*bert"), ("Seaquest", "Seaquest"), ("SpaceInvaders", "Space Invaders")]


def grab(game, steps=180, seed=0):
    env = gym.make(f"ALE/{game}-v5", frameskip=4, repeat_action_probability=0.0)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for t in range(steps):
        obs, _, term, trunc, _ = env.step(1 if t < 3 else int(rng.integers(env.action_space.n)))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return Image.fromarray(obs)


# 1) seven-game montage (4 x 2 grid, last tile "...")
tw, th, lab, gap = 300, 210, 22, 9
W, H = 4 * tw + 3 * gap, 2 * (th + lab) + gap
img = Image.new("RGB", (W, H), WHITE)
d = ImageDraw.Draw(img)
font = ImageFont.truetype(FONT, 15)
for i in range(8):
    r, c = divmod(i, 4)
    x, y = c * (tw + gap), r * (th + lab + gap)
    d.rectangle([x, y, x + tw - 1, y + th + lab - 1], fill=NAVY)
    if i < 7:
        shot = grab(GAMES[i][0]).resize((int(160 * th / 210 * 1.2), th - 8), Image.NEAREST)
        img.paste(shot, (x + (tw - shot.width) // 2, y + 4))
        name = GAMES[i][1]
        wbox = d.textlength(name, font=font)
        d.text((x + (tw - wbox) / 2, y + th + 1), name, font=font, fill=WHITE)
    else:
        for k in range(3):
            cx = x + tw / 2 - 24 + k * 24
            d.ellipse([cx - 5, y + th / 2 + 6, cx + 5, y + th / 2 + 16], fill=(154, 162, 190))
img.save(os.path.join(OUT, "slide_seven_games.png"))

# 2) four consecutive Breakout frames with numbered markers
env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0)
obs, _ = env.reset(seed=5)
frames, t = [], 0
while len(frames) < 4 and t < 2000:
    obs, _, term, trunc, _ = env.step(1 if t % 50 == 0 else 0)     # FIRE launches the ball
    t += 1
    ball_visible = obs[100:185, 8:152].max() > 0                    # ball in the empty play area
    if ball_visible and t % 3 == 0:
        frames.append(Image.fromarray(obs))
    elif not ball_visible:
        frames = []
env.close()
fw, fh, g = 180, 236, 26
strip = Image.new("RGB", (4 * fw + 3 * g + 20, fh + 20), WHITE)
d = ImageDraw.Draw(strip)
f2 = ImageFont.truetype(FONT, 14)
for i, fr in enumerate(frames):
    x = 10 + i * (fw + g)
    strip.paste(fr.resize((fw, fh), Image.NEAREST), (x, 10))
    d.ellipse([x - 8, 2, x + 14, 24], fill=(30, 136, 229))
    d.text((x - 1, 4), str(i + 1), font=f2, fill=WHITE)
strip.save(os.path.join(OUT, "slide_breakout_4frames.png"))
print("saved slide images to", OUT)
