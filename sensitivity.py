# sensitivity.py
#
# What it does: shows which rankings are solid and which depend on one
#   assumption the data cannot settle: how much the general level of F1
#   drivers changed from one decade to the next (ERA_DRIFT in model.py).
#   It refits the model three times:
#     "hardly changed"  0.03% per decade: every decade's newcomers about equal
#     "default"         0.10% per decade: the setting the site uses
#     "changed a lot"   0.25% per decade: decades free to differ a lot
#   and compares the top 20 under each. A driver whose rank barely moves is
#   solid; one who jumps is resting on the assumption.
# What it reads: the same tables as model.py, and data/ratings.parquet (for
#   names and race starts).
# What it produces: web/public/data/sensitivity.json and a table printed to
#   the screen.
# Which files use it: web/app/about/page.tsx.
#
# Run:  .venv\Scripts\python.exe sensitivity.py     (about 2 minutes)

import json
from pathlib import Path

import polars as pl

import fit_ratings
import model

OUTPUT_FILE = Path("web/public/data/sensitivity.json")
ASSUMPTIONS = {"hardly changed": 0.03, "default": 0.10, "changed a lot": 0.25}
MIN_STARTS = 30  # ranked among the same drivers as the ranking page
TOP = 20
SOLID_WITHIN = 3  # a rank that moves by at most this many places counts as solid


def ranks_under(era_drift: float, pairs, pace, eligible: set) -> dict:
    """Each eligible driver's rank and peak skill when the model assumes `era_drift`."""
    saved = model.ERA_DRIFT
    model.ERA_DRIFT = era_drift
    try:
        fit = model.fit_model(pairs, pace, pairs["Year"].max(), wet_races=model.load_wet_races())
    finally:
        model.ERA_DRIFT = saved
    peaks, _, _ = fit_ratings.peak_ratings(fit)
    ranked = peaks.filter(pl.col("DriverId").is_in(list(eligible))).sort("PeakSkill", descending=True)
    return {driver: (place, skill) for place, (driver, skill) in
            enumerate(ranked.select("DriverId", "PeakSkill").iter_rows(), start=1)}


def main():
    pairs, pace = model.load_tables()
    ratings = pl.read_parquet("data/ratings.parquet")
    eligible = set(ratings.filter(pl.col("Starts") >= MIN_STARTS)["DriverId"])
    names = dict(ratings.select("DriverId", "DriverName").iter_rows())
    results = {label: ranks_under(drift, pairs, pace, eligible) for label, drift in ASSUMPTIONS.items()}

    default = results["default"]
    top = sorted(default, key=lambda driver: default[driver][0])[:TOP]
    rows = []
    print(f"{'#':>3} {'Driver':<24}" + "".join(f"{label:>16}" for label in ASSUMPTIONS) + "   verdict")
    for driver in top:
        places = [results[label][driver][0] for label in ASSUMPTIONS]
        solid = max(places) - min(places) <= SOLID_WITHIN
        rows.append({"id": driver, "name": names[driver],
                     "ranks": dict(zip(ASSUMPTIONS, places)),
                     "skills": {label: round(results[label][driver][1], 3) for label in ASSUMPTIONS},
                     "solid": solid})
        print(f"{default[driver][0]:>3} {names[driver]:<24}" + "".join(f"{p:>16}" for p in places)
              + f"   {'solid' if solid else 'depends on the assumption'}")
    # Drivers who enter the top 20 only under another assumption.
    newcomers = {label: [names[d] for d in sorted(r, key=lambda d: r[d][0])[:TOP] if d not in top]
                 for label, r in results.items() if label != "default"}
    print("\nIn the top 20 only under another assumption:", newcomers)
    OUTPUT_FILE.write_text(json.dumps({
        "assumptions": ASSUMPTIONS, "solidWithin": SOLID_WITHIN, "minStarts": MIN_STARTS,
        "top": rows, "onlyUnderOther": newcomers}, indent=1), encoding="utf-8")
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
