"""Run controllers on a fixed, seeded set of emergencies and summarise the results.

Every controller sees exactly the same takeover moments: the scenario and the driver phase
before the trigger depend only on the seed.
"""

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from crashai import controllers as ctl
from crashai.env import CrashAvoidEnv
from crashai.scenarios import SCENARIOS

EVAL_SEED_BASE = 1_000_000  # training never uses seeds this high


@dataclass
class Entry:
    name: str
    env_kwargs: dict
    controller: ctl.Controller
    description: str = ""


def rule_based_entries():
    return [
        Entry("driver_only", dict(controls="driver"), ctl.DriverOnly(), "No intervention: the driver reacts late or not at all"),
        Entry("brake_no_abs", dict(controls="full", use_abs=False, use_esc=False), ctl.BrakeStraight(), "Full braking, wheels can lock"),
        Entry("brake_abs", dict(controls="full", use_abs=True, use_esc=False), ctl.BrakeStraight(), "Full braking with ABS (like today's AEB)"),
        Entry("brake_swerve", dict(controls="full", use_abs=True, use_esc=False), ctl.BrakeSwerve(), "Full braking + scripted swerve, ABS"),
        Entry("brake_swerve_esc", dict(controls="full", use_abs=True, use_esc=True), ctl.BrakeSwerve(), "Full braking + scripted swerve, ABS + ESC"),
    ]


def learned_entry(controls, model_path, name=None):
    from stable_baselines3 import SAC

    model = SAC.load(model_path, device="cpu")
    name = name or f"learned_{controls}"
    return Entry(name, dict(controls=controls), ctl.SB3Policy(model, name), f"Learned policy, controls={controls}")


def run_episode(env, controller, seed, scenario):
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    controller.reset(env)
    done = False
    while not done:
        action = controller.act(env, obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info


def evaluate(entries, n_per_scenario=100, scenarios=SCENARIOS, seed_base=EVAL_SEED_BASE, progress=True):
    records = []
    for e in entries:
        env = CrashAvoidEnv(**e.env_kwargs)
        it = [(s, i) for s in scenarios for i in range(n_per_scenario)]
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc=e.name, leave=False)
        for scen, i in it:
            seed = seed_base + 10_000 * SCENARIOS.index(scen) + i
            info = run_episode(env, e.controller, seed, scen)
            records.append({
                "controller": e.name, "seed": seed, "scenario": scen,
                "surface": info["surface"], "split_mu": info["split_mu"], "weather": info["weather"],
                "outcome": info["outcome"], "impact_speed": info["impact_speed"], "after_handback": info["crashed_after_handback"],
                "speed_at_takeover": info["speed_at_takeover"], "t_intervention": info["t_intervention"],
            })
    return records


def summarise(records):
    """Per controller: overall and per-scenario crash rate and mean impact speed."""
    by = defaultdict(list)
    for r in records:
        by[(r["controller"], "all")].append(r)
        by[(r["controller"], r["scenario"])].append(r)
        by[(r["controller"], "surface:" + r["surface"])].append(r)
    out = {}
    for (name, group), rs in by.items():
        crashed = np.array([r["outcome"] in ("crash", "guardrail") for r in rs])
        impact = np.array([r["impact_speed"] if r["outcome"] in ("crash", "guardrail") else 0.0 for r in rs])
        out.setdefault(name, {})[group] = {
            "n": len(rs),
            "crash_rate": float(crashed.mean()),
            "guardrail_rate": float(np.mean([r["outcome"] == "guardrail" for r in rs])),
            "mean_impact_kmh": float(impact.mean() * 3.6),  # averaged over all episodes (0 if avoided)
            "mean_impact_kmh_when_crashed": float(impact[crashed].mean() * 3.6) if crashed.any() else 0.0,
        }
    return out


def markdown_table(summary, groups=("all",) + SCENARIOS):
    head = "| Controller | " + " | ".join("Overall" if g == "all" else g for g in groups) + " |"
    sep = "|---|" + "---|" * len(groups)
    lines = [head, sep]
    for name, s in summary.items():
        cells = []
        for g in groups:
            if g in s:
                cells.append(f"{100 * s[g]['crash_rate']:.0f}% ({s[g]['mean_impact_kmh']:.0f} km/h)")
            else:
                cells.append("–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(records, summary, path_prefix):
    with open(path_prefix + "_records.json", "w") as f:
        json.dump(records, f)
    with open(path_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
