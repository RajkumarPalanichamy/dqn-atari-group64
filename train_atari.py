"""Train the 2013 DQN on an Atari game.

Paper settings (Mnih et al., 2013): 10M frames, replay memory 1M frames,
minibatch 32, RMSProp, epsilon 1.0 -> 0.1 over the first 1M frames,
frame-skip 4, reward clipping, evaluation with epsilon = 0.05.

Full reproduction needs a GPU and ~1-2 days per game. Example:
    python train_atari.py --game Breakout --frames 10000000 --replay 1000000
Short CPU sanity run used for our screenshots:
    python train_atari.py --game Breakout --frames 400000 --replay 100000 \
        --anneal-frames 200000 --learn-start 20000 --eval-every 50000
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from dqn.agent import DQNAgent, linear_epsilon
from dqn.atari_env import AtariEnv
from dqn.model import DQN2013
from dqn.preprocess import FrameStack
from dqn.replay import FrameReplay


def evaluate(agent, game, episodes=5, epsilon=0.05, seed=10_000, max_steps=4_500):
    """Paper protocol: epsilon-greedy with epsilon = 0.05, average total (unclipped) reward."""
    env, fs = AtariEnv(game, seed=seed), FrameStack()
    scores = []
    for _ in range(episodes):
        s = fs.reset(env.reset())
        total, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = agent.act(s, epsilon)
            f, r, done = env.step(a)
            s = fs.push(f)
            total += r
            t += 1
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def collect_holdout(game, n=500, seed=123):
    """Fixed set of states from a random policy, used to track the average
    max Q-value during training (Figure 2, right, in the paper)."""
    env, fs = AtariEnv(game, seed=seed), FrameStack()
    rng = np.random.default_rng(seed)
    s, states = fs.reset(env.reset()), []
    while len(states) < n:
        f, _, done = env.step(int(rng.integers(env.n_actions)))
        s = fs.reset(env.reset()) if done else fs.push(f)
        states.append(s)
    env.close()
    return np.stack(states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Breakout")
    ap.add_argument("--frames", type=int, default=10_000_000, help="emulator frames (agent steps x k)")
    ap.add_argument("--replay", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--anneal-frames", type=int, default=1_000_000)
    ap.add_argument("--learn-start", type=int, default=50_000, help="agent steps of random play before learning")
    ap.add_argument("--eval-every", type=int, default=250_000, help="frames between evaluations")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--target-update", type=int, default=0, help="0 = 2013 paper (no target net)")
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    variant = "2013" if args.target_update == 0 else f"target{args.target_update}"
    tag = f"{args.game}_{variant}_seed{args.seed}"
    env = AtariEnv(args.game, seed=args.seed)
    k = env.k
    agent = DQNAgent(DQN2013(env.n_actions), env.n_actions, gamma=args.gamma, lr=args.lr,
                     loss=args.loss, target_update=args.target_update, seed=args.seed)
    memory = FrameReplay(args.replay, seed=args.seed)
    fs = FrameStack()
    holdout = collect_holdout(args.game)

    log_f = open(os.path.join(args.out, f"{tag}_train.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["frames", "episode", "train_score", "epsilon", "loss", "wall_s"])
    ev_f = open(os.path.join(args.out, f"{tag}_eval.csv"), "w", newline="")
    ev = csv.writer(ev_f)
    ev.writerow(["frames", "eval_score_eps0.05", "avg_max_q_holdout"])

    frame = env.reset()
    s = fs.reset(frame)
    ep, ep_score, losses = 0, 0.0, []
    t0 = time.time()
    steps = args.frames // k
    next_eval = args.eval_every
    best = -float("inf")
    for step in range(1, steps + 1):
        frames_done = step * k
        eps = 1.0 if step < args.learn_start else linear_epsilon(frames_done, anneal_steps=args.anneal_frames)
        a = agent.act(s, eps)
        f2, r, done = env.step(a)
        memory.add(frame, a, float(np.sign(r)), done)        # reward clipping (training only)
        ep_score += r
        if done:
            log.writerow([frames_done, ep, ep_score, round(eps, 4),
                          round(float(np.mean(losses)) if losses else 0.0, 5), round(time.time() - t0, 1)])
            log_f.flush()
            ep += 1
            ep_score, losses = 0.0, []
            frame = env.reset()
            s = fs.reset(frame)
        else:
            frame = f2
            s = fs.push(f2)
        if step >= args.learn_start and len(memory) > args.batch + 4:
            loss, _ = agent.update(memory.sample(args.batch))
            losses.append(loss)
        if frames_done >= next_eval or step == steps:
            score = evaluate(agent, args.game, args.eval_episodes)
            avg_q = float(agent.q_values(holdout).max(axis=1).mean())
            ev.writerow([frames_done, score, avg_q])
            ev_f.flush()
            print(f"[{args.game}] frames={frames_done:>9,}  episodes={ep:>5}  eps={eps:.3f}  "
                  f"eval_score={score:7.2f}  avg_maxQ={avg_q:.4f}  {time.time()-t0:7.0f}s", flush=True)
            torch.save(agent.q.state_dict(), os.path.join(args.out, f"{tag}_last.pt"))
            if score >= best:
                best = score
                torch.save(agent.q.state_dict(), os.path.join(args.out, f"{tag}_best.pt"))
            next_eval += args.eval_every
    log_f.close(); ev_f.close(); env.close()


if __name__ == "__main__":
    main()
