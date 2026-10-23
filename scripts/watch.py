"""Record one emergency and animate it, one row per controller, all on the same seed.

Examples:
  python scripts/watch.py --scenario stopped --seed 3
  python scripts/watch.py --scenario cut_in --seed 7 --controllers driver_only brake_abs learned_pedal learned_full
"""

import argparse
import os

from crashai import controllers as ctl
from crashai.env import CrashAvoidEnv
from crashai.evaluation import EVAL_SEED_BASE, rule_based_entries, learned_entry
from crashai.render import animate, outcome_text, road_dict
from crashai.scenarios import SCENARIOS


def get_entry(name):
    for e in rule_based_entries():
        if e.name == name:
            return e
    if name.startswith("learned_"):
        controls = name[len("learned_"):]
        path = os.path.join("models", controls, "best_model.zip")
        return learned_entry(controls, path, name)
    raise SystemExit(f"unknown controller {name}")


def record(entry, seed, scenario):
    env = CrashAvoidEnv(record=True, **entry.env_kwargs)
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    entry.controller.reset(env)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(entry.controller.act(env, obs))
        done = term or trunc
    return {"name": entry.name, "frames": env.frames, "road": road_dict(env.road),
            "outcome_text": outcome_text(info), "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="stopped", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=0, help="index into the test set")
    ap.add_argument("--controllers", nargs="+", default=["driver_only", "brake_abs", "learned_pedal", "learned_full"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    seed = EVAL_SEED_BASE + 10_000 * SCENARIOS.index(args.scenario) + args.seed
    eps = []
    for name in args.controllers:
        ep = record(get_entry(name), seed, args.scenario)
        info = ep["info"]
        print(f"{name:>18}: {ep['outcome_text']}   ({info['surface']}, mu={info['mu']:.2f}, {info['weather']})")
        eps.append(ep)
    os.makedirs("media", exist_ok=True)
    out = args.out or os.path.join("media", f"{args.scenario}_{args.seed}.gif")
    animate(eps, out, fps=args.fps)
    print("saved", out)


if __name__ == "__main__":
    main()
