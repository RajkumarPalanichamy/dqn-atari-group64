"""Plots: (a) our Breakout run -- eval score + average max Q (cf. paper Fig. 2),
(b) CartPole ablation -- with vs. without experience replay,
(c) Table 1 of the paper as a chart (DQN vs. baselines, % of human score)."""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
NAVY, GOLD, TEAL, RED, GREY = "#141A33", "#F2A01D", "#127C77", "#C8462B", "#5A6072"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": NAVY, "xtick.color": GREY, "ytick.color": GREY})

# (a) Breakout runs
ev_paths = sorted(glob.glob(os.path.join(RUNS, "*_eval.csv")))
if ev_paths:
    STYLE = {"2013_seed0": (GREY, "2013 algorithm, run 1"), "2013_seed1": (GOLD, "2013 algorithm, run 2"),
             "target2500_seed1": (TEAL, "+ 2015 target network (C=2500)")}
    game = os.path.basename(ev_paths[0]).split("_")[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for p in ev_paths:
        key = os.path.basename(p)[len(game) + 1:].replace("_eval.csv", "")
        col, label = STYLE.get(key, (NAVY, key))
        tr = pd.read_csv(p.replace("_eval", "_train"))
        ev = pd.read_csv(p)
        ax[0].plot(tr.frames / 1e3, tr.train_score.rolling(30, min_periods=1).mean(), color=col, lw=2, label=label)
        ax[1].plot(ev.frames / 1e3, ev.avg_max_q_holdout, "o-", color=col, lw=2, ms=4, label=label)
    ax[0].axhline(1.2, ls="--", color=RED, lw=1, label="random play (paper): 1.2")
    ax[0].set_xlabel("training frames (thousands)"); ax[0].set_ylabel("score per training episode\n(30-episode mean)")
    ax[0].set_title(f"{game}: average reward per episode", color=NAVY); ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_xlabel("training frames (thousands)"); ax[1].set_ylabel("average max Q\non held-out states")
    ax[1].set_title(f"{game}: average action value Q (cf. paper Fig. 2)", color=NAVY)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{game.lower()}_training.png"), dpi=150); plt.close()

    # single-panel version for the slides
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for p in ev_paths:
        key = os.path.basename(p)[len(game) + 1:].replace("_eval.csv", "")
        col, label = STYLE.get(key, (NAVY, key))
        tr = pd.read_csv(p.replace("_eval", "_train"))
        ax.plot(tr.frames / 1e3, tr.train_score.rolling(30, min_periods=1).mean(), color=col, lw=2, label=label)
    ax.axhline(1.2, ls="--", color=RED, lw=1, label="random play (paper): 1.2")
    ax.set_xlabel("training frames (thousands)"); ax.set_ylabel("score per episode (30-episode mean)")
    ax.set_title(f"{game}: our 400k-frame runs", color=NAVY); ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{game.lower()}_training_slide.png"), dpi=150); plt.close()

# (b) CartPole ablation
files = sorted(glob.glob(os.path.join(RUNS, "cartpole_*_seed*.csv")))
if files:
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for variant, col, label in [("replay", TEAL, "with experience replay"), ("noreplay", RED, "without replay (online)")]:
        curves = []
        grid = np.linspace(0, 30_000, 61)
        for f in [x for x in files if f"_{variant}_" in x]:
            d = pd.read_csv(f)
            sm = d["return"].rolling(10, min_periods=1).mean()
            curves.append(np.interp(grid, d.step, sm))
        if curves:
            c = np.array(curves)
            ax.plot(grid / 1e3, c.mean(0), color=col, lw=2, label=f"{label} (mean of {len(c)} seeds)")
            ax.fill_between(grid / 1e3, c.min(0), c.max(0), color=col, alpha=0.15)
    ax.set_xlabel("environment steps (thousands)"); ax.set_ylabel("episode return (10-ep mean)")
    ax.set_title("CartPole-v1: effect of experience replay", color=NAVY); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "cartpole_replay_ablation.png"), dpi=150); plt.close()

# (c) Paper Table 1
games = ["Beam Rider", "Breakout", "Enduro", "Pong", "Q*bert", "Seaquest", "Space Inv."]
T = {"Random": [354, 1.2, 0, -20.4, 157, 110, 179], "Sarsa": [996, 5.2, 129, -19, 614, 665, 271],
     "Contingency": [1743, 6, 159, -17, 960, 723, 268], "DQN": [4092, 168, 470, 20, 1952, 1705, 581],
     "Human": [7456, 31, 368, -3, 18900, 28010, 3690]}
rnd, hum = np.array(T["Random"]), np.array(T["Human"])
fig, ax = plt.subplots(figsize=(11, 3.9))
x = np.arange(len(games)); w = 0.26
for i, (m, col) in enumerate([("Sarsa", "#9AA2BE"), ("Contingency", GREY), ("DQN", GOLD)]):
    norm = 100 * (np.array(T[m]) - rnd) / (hum - rnd)
    ax.bar(x + (i - 1) * w, norm, w, color=col, label=m)
ax.axhline(100, color=RED, ls="--", lw=1.2); ax.text(6.45, 103, "human = 100%", color=RED, ha="right", fontsize=9)
ax.set_xticks(x, games); ax.set_ylabel("% of human score\n(random = 0%)")
ax.set_title("Table 1 of Mnih et al. (2013), normalised: (score - random) / (human - random)", color=NAVY)
ax.legend(frameon=False, ncol=3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "paper_table1_normalised.png"), dpi=150); plt.close()
print("figures written to", OUT)
