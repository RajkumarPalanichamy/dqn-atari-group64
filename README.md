# DQN — *Playing Atari with Deep Reinforcement Learning* (Mnih et al., 2013)

**BITS Pilani WILP · Deep Reinforcement Learning · Assignment 1 · Group 64**

| Member | BITS ID |
|---|---|
| Rajkumar | 2025AG05592 |
| Nandhakumar | 2025AG05827 |
| Sriram Aditya | 2025AG05617 |
| Abhishek Kumar Srivastava | 2025AG05839 |

A from-scratch PyTorch implementation of the algorithm in
[arXiv:1312.5602](https://arxiv.org/abs/1312.5602), written to check our reading of the
paper. It follows the 2013 paper, **not** the later 2015 Nature version.

## What is implemented (paper section → file)

| Paper | Implementation |
|---|---|
| §4.1 preprocessing: 210×160 RGB → grey → 110×84 → 84×84 crop, stack last 4 frames | `dqn/preprocess.py` |
| §5 network: conv 16@8×8 s4 → conv 32@4×4 s2 → FC 256 → one output per action | `dqn/model.py` (`DQN2013`) |
| Algorithm 1: replay memory of the last N frames, random minibatches of 32 | `dqn/replay.py` (`FrameReplay`) |
| Algorithm 1: target `y = r` if terminal else `r + γ max Q(s',a';θ)`; RMSProp | `dqn/agent.py` |
| §5 ε-greedy, ε 1.0 → 0.1 over the first 1M frames, then fixed | `dqn/agent.py` (`linear_epsilon`) |
| §5 frame-skip k = 4 (k = 3 for Space Invaders), reward clipping to −1/0/+1 | `dqn/atari_env.py`, `train_atari.py` |
| §5 evaluation with ε = 0.05; average max Q on a fixed set of held-out states (Fig. 2) | `train_atari.py` (`evaluate`, `collect_holdout`) |

**No separate target network.** The 2013 paper computes the target with the parameters
from the previous iteration θ<sub>i−1</sub>, i.e. the current network before the update.
The periodically-synced target network θ⁻ comes from the 2015 Nature paper. It is
available here only as an option (`--target-update C`) for comparison.

**Values the 2013 paper does not state** (we used the standard DeepMind values):
γ = 0.99, RMSProp lr = 2.5e-4, α = 0.95, ε = 0.01. The 84×84 crop offset (rows 18–101
of the 110×84 image) is our choice of "a region that roughly captures the playing area".

## Setup

```bash
pip install -r requirements.txt
python -m pytest -v tests/            # 5 unit tests: shapes, epsilon schedule, replay, terminal target
```

## Experiments we ran (CPU, 2 cores)

### 1. CartPole check with experience replay vs. without it — `train_cartpole.py`
The same algorithm on a small control task (MLP instead of CNN), 30k steps, 3 seeds.

```bash
python train_cartpole.py --steps 30000
```

Mean return over the last 20 episodes (CartPole-v1 caps an episode at 500):

| Variant | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| with experience replay (memory 50k, batch 32) | 267.8 | 233.8 | 245.0 | **248.8** |
| without replay (online, latest transition only) | 84.0 | 74.7 | 41.5 | **66.7** |

The same learner without replay learns much less, which supports the paper's argument that
random sampling from the replay memory breaks the correlation between consecutive samples
(`figures/cartpole_replay_ablation.png`, log in `logs/cartpole.log`).

### 2. Short Atari Breakout run — `train_atari.py`
The paper trains for **10M frames** with a 1M-frame replay memory, which takes days on a GPU.
On CPU we ran a **400k-frame** run (4% of the paper's budget), with a 100k-frame memory and
ε annealed over 200k frames. Everything else follows the paper.

```bash
python train_atari.py --game Breakout --frames 400000 --replay 100000 \
    --anneal-frames 200000 --learn-start 10000 --eval-every 40000 --eval-episodes 3
# same run with the 2015 target network, for comparison
python train_atari.py --game Breakout --seed 1 --target-update 2500 --frames 400000 --replay 100000 \
    --anneal-frames 200000 --learn-start 10000 --eval-every 40000 --eval-episodes 5
python evaluate.py --game Breakout --ckpt runs/Breakout_2013_seed1_best.pt --episodes 10
```

Three runs of 400k frames each (logs in `logs/`, CSVs and checkpoints in `runs/`):

| Run | Training score: first 40k frames → last 50 episodes | Best evaluation (ε = 0.05) | Average max Q on held-out states |
|---|---|---|---|
| 2013 algorithm, run 1 (seed 0) | 1.0 → 1.8 | 11.0 at 200k frames, collapsed to 0 by 400k frames | jumps between 0.9 and 1.7 |
| 2013 algorithm, run 2 (seed 1) | 1.1 → 2.5 | 11.0 | jumps between 0.7 and 1.6 |
| + 2015 target network, C = 2500 (seed 1) | 1.1 → 1.9 | 10.6 | rises smoothly from 0.05 to 0.44 |

Evaluating the best checkpoint of run 2 over 10 episodes at ε = 0.05 gives an
**average score of 10.3** (min 7, max 11; `logs/breakout_eval_best.log`). For comparison,
the paper's Table 1 reports 1.2 for random play, 5.2 for Sarsa, 6 for Contingency and 168 for
DQN after 10M frames.

What we observed:
* The agent clearly learns within 4% of the paper's training budget. However, the policy is a
  simple pattern: it keeps the paddle near the left wall, which scores 11 points per game.
* **Without a target network, training is unstable.** In run 1 the final policy collapsed to
  always pressing RIGHT (`figures/run1_collapsed_q_values.png`), and in both 2013 runs the Q
  estimates oscillate. With the 2015 target network the Q estimates grow smoothly
  (`figures/breakout_training.png`). This is the instability the Nature paper later fixed.
* Run 1 was trained before best-checkpoint saving was added, so only its last checkpoint
  is stored.

Full paper setting (GPU recommended):
```bash
python train_atari.py --game Breakout --frames 10000000 --replay 1000000
```

### 3. Figures
```bash
python scripts/visualize_preprocessing.py figures
python scripts/plot_results.py runs figures
python scripts/make_slide_images.py figures      # game screenshots used on the slides
```

| File | Shows |
|---|---|
| `figures/preprocessing_pipeline.png`, `figures/frame_stack.png` | the §4.1 pipeline on real emulator frames |
| `figures/breakout_training.png` | our Breakout run: score and average max Q (cf. paper Fig. 2) |
| `figures/cartpole_replay_ablation.png` | effect of experience replay |
| `figures/paper_table1_normalised.png` | the paper's Table 1 as % of human score |
| `figures/breakout_q_values.png`, `figures/breakout_agent.gif` | trained agent: Q-values and gameplay |
| `figures/screenshot_*.png` | captured console output of the runs and tests |

## Repository layout
```
dqn/            preprocess.py  model.py  replay.py  agent.py  atari_env.py
train_atari.py  train_cartpole.py  evaluate.py
scripts/        visualize_preprocessing.py  plot_results.py  render_terminal.py
tests/          test_dqn.py
runs/           CSV logs + checkpoint of our runs
logs/           console output of our runs
figures/        all plots and screenshots used in the presentation
```

## References
1. V. Mnih et al. *Playing Atari with Deep Reinforcement Learning.* NIPS Deep Learning Workshop, 2013. arXiv:1312.5602
2. V. Mnih et al. *Human-level control through deep reinforcement learning.* Nature 518, 529–533, 2015
3. M. G. Bellemare et al. *The Arcade Learning Environment.* JAIR 47, 253–279, 2013
