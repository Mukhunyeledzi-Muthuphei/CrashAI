"""Evaluate all controllers on the fixed test set and write results/RESULTS.md.

  python scripts/evaluate.py --n 250
"""

import argparse
import os

from crashai.evaluation import (
    evaluate, learned_entry, markdown_table, rule_based_entries, save, summarise,
)
from crashai.scenarios import SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="test episodes per scenario type")
    ap.add_argument("--model", default="best_model.zip")
    ap.add_argument("--only-learned", action="store_true")
    args = ap.parse_args()

    entries = [] if args.only_learned else rule_based_entries()
    for controls in ("pedal", "full", "brakes_only"):
        path = os.path.join("models", controls, args.model)
        if os.path.exists(path):
            entries.append(learned_entry(controls, path))
        else:
            print("no model at", path)

    records = evaluate(entries, n_per_scenario=args.n)
    summary = summarise(records)
    os.makedirs("results", exist_ok=True)
    save(records, summary, "results/test")

    surfaces = tuple(f"surface:{s}" for s in ("dry", "wet", "snow", "ice"))
    text = [
        f"# Test results\n\n{args.n} emergencies per scenario type ({args.n * len(SCENARIOS)} per controller), "
        "identical for every controller.\n",
        "Each cell: **crash rate** (mean impact speed over all episodes, 0 when avoided).\n",
        "## By scenario\n", markdown_table(summary), "\n",
        "## By road surface\n", markdown_table(summary, groups=surfaces), "\n",
    ]
    with open("results/RESULTS.md", "w") as f:
        f.write("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
