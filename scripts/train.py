"""Train a collision-avoidance policy with SAC.

Examples:
  python scripts/train.py --controls full --steps 1500000
  python scripts/train.py --controls pedal --steps 1500000
  python scripts/train.py --controls brakes_only --steps 1500000
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from crashai.controllers import SB3Policy
from crashai.env import CrashAvoidEnv
from crashai.evaluation import Entry, evaluate, summarise

VAL_SEED_BASE = 2_000_000  # validation seeds (test seeds start at 1_000_000, training uses random seeds)


def make_env(controls, rank, seed):
    def _f():
        env = Monitor(CrashAvoidEnv(controls))
        env.reset(seed=seed + 1000 * rank)
        return env
    return _f


class ValidationCallback(BaseCallback):
    """Every `every` steps: run the policy on a fixed validation set, log it, keep the best model."""

    def __init__(self, controls, out_dir, every, n_per_scenario):
        super().__init__()
        self.controls = controls
        self.out_dir = out_dir
        self.every = every
        self.n = n_per_scenario
        self.best = np.inf
        self.next_at = every
        self.t0 = time.time()
        self.log_path = os.path.join(out_dir, "validation.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["steps", "minutes", "crash_rate", "mean_impact_kmh",
                                    "stopped", "lead_brake", "cut_in", "pedestrian"])

    def _on_step(self):
        if self.num_timesteps < self.next_at:
            return True
        self.next_at += self.every
        entry = Entry("val", dict(controls=self.controls), SB3Policy(self.model, "val"))
        s = summarise(evaluate([entry], self.n, seed_base=VAL_SEED_BASE, progress=False))["val"]
        crash = s["all"]["crash_rate"]
        row = [self.num_timesteps, round((time.time() - self.t0) / 60, 1), round(crash, 3),
               round(s["all"]["mean_impact_kmh"], 2)] + [round(s[k]["crash_rate"], 3)
                                                         for k in ("stopped", "lead_brake", "cut_in", "pedestrian")]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[val] steps={self.num_timesteps} crash={crash:.1%} impact={s['all']['mean_impact_kmh']:.1f} km/h", flush=True)
        # score: crash rate, ties broken by impact speed
        score = crash + 0.001 * s["all"]["mean_impact_kmh"]
        if score < self.best:
            self.best = score
            self.model.save(os.path.join(self.out_dir, "best_model"))
        self.model.save(os.path.join(self.out_dir, "last_model"))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", default="full", choices=["full", "pedal", "brakes_only"])
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-every", type=int, default=100_000)
    ap.add_argument("--val-n", type=int, default=50, help="validation episodes per scenario type")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = args.out or os.path.join("models", args.controls)
    os.makedirs(out, exist_ok=True)

    venv = SubprocVecEnv([make_env(args.controls, i, args.seed) for i in range(args.n_envs)])
    model = SAC(
        "MlpPolicy", venv,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=20_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=max(1, args.n_envs // 2),
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    cb = ValidationCallback(args.controls, out, args.val_every, args.val_n)
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(os.path.join(out, "last_model"))
    venv.close()


if __name__ == "__main__":
    main()
