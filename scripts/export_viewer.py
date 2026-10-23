"""Record showcase emergencies for the 3D viewer and write viewer/episodes.js.

Every scene is one emergency from the test set, replayed by each controller on the same seed.
Scenes are picked from results/test_records.json (run scripts/evaluate.py first).

  python scripts/export_viewer.py
"""

import collections
import json
import math
import os

import numpy as np

from crashai.env import CrashAvoidEnv
from crashai.evaluation import EVAL_SEED_BASE, learned_entry, rule_based_entries
from crashai.render import outcome_text
from crashai.scenarios import SCENARIOS

CONTROLLERS = [
    ("driver_only", "Driver only", "No system. The driver is distracted or reacts late."),
    ("brake_abs", "Emergency braking", "Full braking with ABS, wheel held straight. Like today's AEB."),
    ("brake_swerve_esc", "Scripted swerve", "Full braking plus a scripted lane change, ABS and ESC."),
    ("learned_pedal", "AI · steering + 1 pedal", "Neural network with steering, one brake pedal, throttle. ESC on."),
    ("learned_full", "AI · steering + 4 brakes", "Neural network with steering, four independent brakes, throttle."),
    ("learned_brakes_only", "AI · 4 brakes, no steering", "Neural network with four independent brakes and throttle. Steering frozen."),
]

SCENARIO_NAMES = {
    "stopped": "Stopped vehicle ahead",
    "lead_brake": "Lead car brakes hard",
    "cut_in": "Car cuts in",
    "pedestrian": "Pedestrian steps out",
}


def entries():
    by_name = {e.name: e for e in rule_based_entries()}
    for controls in ("pedal", "full", "brakes_only"):
        by_name[f"learned_{controls}"] = learned_entry(controls, os.path.join("models", controls, "best_model.zip"))
    return by_name


def pick_scenes(records):
    """Pick emergencies where the controllers disagree in an instructive way."""
    by = collections.defaultdict(dict)
    for r in records:
        by[r["controller"]][r["seed"]] = r
    crash = lambda r: r["outcome"] in ("crash", "guardrail")  # noqa: E731
    seeds = sorted(by["learned_full"])
    picked = []

    def first(cond, n=1, exclude=()):
        out = []
        for s in seeds:
            if s in exclude or s in [p[0] for p in picked]:
                continue
            if cond(s):
                out.append(s)
                if len(out) == n:
                    break
        return out

    R = lambda c, s: by[c][s]  # noqa: E731
    # 1. a stopped car where every AI makes it and every rule fails
    for s in first(lambda s: R("learned_full", s)["scenario"] == "stopped"
                   and not crash(R("learned_brakes_only", s)) and not crash(R("learned_full", s))
                   and not crash(R("learned_pedal", s)) and crash(R("brake_abs", s))
                   and crash(R("brake_swerve_esc", s)) and R("learned_full", s)["surface"] == "wet"):
        picked.append((s, "Every AI makes it. Braking-only steering changes lanes too."))
    # 2. four brakes beat one pedal on split grip
    for s in first(lambda s: R("learned_full", s)["split_mu"] and not crash(R("learned_full", s))
                   and crash(R("learned_pedal", s))):
        picked.append((s, "Slippery patch under one side. Four brakes make it, one pedal doesn't."))
    # 3-5. one per remaining scenario type where the AI avoids and AEB doesn't
    for scen in ("cut_in", "lead_brake", "pedestrian"):
        for s in first(lambda s: R("learned_full", s)["scenario"] == scen and not crash(R("learned_full", s))
                       and crash(R("brake_abs", s)) and crash(R("driver_only", s))):
            picked.append((s, None))
    # 6. ice: where grip runs out for almost everyone
    for s in first(lambda s: R("learned_full", s)["surface"] == "ice" and not crash(R("learned_full", s))
                   and crash(R("brake_swerve_esc", s))):
        picked.append((s, "Ice. Grip is so low that every move has to be gentle."))
    return [(s, by["learned_full"][s]["scenario"], note) for s, note in picked]


def r2(a, d=2):
    return [round(float(v), d) for v in a]


def record(entry, seed, scenario):
    env = CrashAvoidEnv(record=True, **entry.env_kwargs)
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    entry.controller.reset(env)
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(entry.controller.act(env, obs))
        done = term or trunc
    return env, info


def pack_run(env, info, name, label, blurb):
    fr = env.frames
    p = env.car.p
    cols = collections.defaultdict(list)
    skids = []  # [t, wheel, x, y] points while a tire slides
    for f in fr:
        x, y, psi = f["ego"]
        cols["t"].append(round(f["t"], 3))
        cols["x"].append(round(x, 3))
        cols["y"].append(round(y, 3))
        cols["psi"].append(round(psi, 4))
        cols["v"].append(round(f["speed"], 2))
        cols["steer"].append(round(f["steer"], 4))
        cols["b"].append(r2(f["brakes"]))
        cols["thr"].append(round(f["throttle"], 2))
        cols["esc"].append(int(f["esc"]))
        cols["ai"].append(int(f["phase"] == "ai"))
        cols["ttc"].append(None if math.isinf(f["ttc"]) else round(f["ttc"], 2))
        cols["ws"].append(r2(f["wheel_speed"]))
        grip = np.maximum(f["mu"] * f["fz"], 1.0)
        cols["fx"].append(r2(f["fx"] / grip))
        cols["fy"].append(r2(f["fy"] / grip))
        cols["per"].append([[round(o[0], 1), round(o[1], 1), round(o[2], 2), round(o[3], 1), round(o[4], 1), int(o[5])]
                            for o in f["perceived"][:10]])
        # skid marks: wheel positions while sliding
        c, s = math.cos(psi), math.sin(psi)
        pos = [(p.a, p.track / 2), (p.a, -p.track / 2), (-p.b, p.track / 2), (-p.b, -p.track / 2)]
        for i, (bx, by) in enumerate(pos):
            if (abs(f["slip"][i]) > 0.18 or abs(f["alpha"][i]) > 0.14) and f["speed"] > 2:
                skids.append([round(f["t"], 3), i, round(x + bx * c - by * s, 2), round(y + bx * s + by * c, 2)])
    return {
        "id": name, "label": label, "blurb": blurb,
        "outcome": info["outcome"], "outcomeText": outcome_text(info),
        "impactKmh": round(info["impact_speed"] * 3.6, 1) if info["outcome"] in ("crash", "guardrail") else 0,
        "afterHandback": bool(info["crashed_after_handback"]),
        "tTakeover": round(env.t_takeover, 3),
        "frames": cols, "skids": skids,
    }


def main():
    with open("results/test_records.json") as f:
        records = json.load(f)
    ents = entries()
    scenes = []
    for seed, scen, note in pick_scenes(records):
        runs = []
        longest = None
        for name, label, blurb in CONTROLLERS:
            env, info = record(ents[name], seed, scen)
            runs.append(pack_run(env, info, name, label, blurb))
            if longest is None or env.frames[-1]["t"] > longest.frames[-1]["t"]:
                longest = env
            print(f"  {seed} {scen:>10} {name:>20}: {outcome_text(info)}")
        road = longest.road
        obstacles = [{"kind": o.kind, "l": round(o.length, 2), "w": round(o.width, 2)} for o in longest.obstacles]
        track = {"t": [round(f["t"], 3) for f in longest.frames],
                 "pose": [[[round(o[0], 2), round(o[1], 2), round(o[2], 3)] for o in f["obstacles"]] for f in longest.frames]}
        meta = longest.scen.meta
        scenes.append({
            "id": f"{scen}-{seed - EVAL_SEED_BASE - 10_000 * SCENARIOS.index(scen)}",
            "scenario": scen, "title": SCENARIO_NAMES[scen], "note": note,
            "surface": meta["surface"], "mu": round(meta["mu"], 2), "splitMu": meta["split_mu"],
            "weather": longest.scen.weather, "speedKmh": round(longest.scen.ego_speed * 3.6),
            "road": {"yMin": road.y_min, "yMax": road.y_max, "lanes": road.n_lanes, "laneWidth": 3.6,
                     "patches": [r2(pt) for pt in road.patches]},
            "obstacles": obstacles, "obstacleTrack": track, "runs": runs,
        })
    car = ents["learned_full"]
    p = CrashAvoidEnv().car.p
    data = {"car": {"length": p.length, "width": p.width, "wheelbase": p.wheelbase, "track": p.track,
                    "a": p.a, "b": p.b, "wheelRadius": p.wheel_radius, "maxSteer": p.max_steer},
            "scenes": scenes}
    os.makedirs("viewer", exist_ok=True)
    with open("viewer/episodes.js", "w") as f:
        f.write("window.CRASHAI = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print("wrote viewer/episodes.js", os.path.getsize("viewer/episodes.js") // 1024, "KB")


if __name__ == "__main__":
    main()
