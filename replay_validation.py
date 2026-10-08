# replay_validation.py
#
# What it does: the main honesty check of Phase 1. It replays real races in
#   the simulator, with every driver using the strategy their team really
#   used and the real safety cars, and measures how close the simulated
#   finishing order and gaps are to what really happened. If the simulator
#   cannot reproduce races that did happen, its "what if" answers about
#   races that did not happen are worth nothing.
# What it reads: data/laps.parquet, models/lap_time_network.pt,
#   models/race_stats.json (all through simulator.py).
# What it produces: tables on the screen and docs/replay_validation.png.
# Which files use it: final_test.py calls replay() and summarise() on 2025.
#
# Run:  .venv\Scripts\python.exe replay_validation.py
#       .venv\Scripts\python.exe replay_validation.py --tune   (tries values
#           of the overtaking rule; this is how the value in simulator.py
#           was chosen)
#
# The simulator is given: the grid, each driver's qualifying gap, the real
# strategies and the real safety car laps. It is NOT given anything about
# how fast each car actually was in the race.

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

import simulator
from simulator import NEVER, LapModel, RaceSet, RaceSim

SAMPLES = 200  # how many times each race is simulated
CHART_FILE = Path("docs/replay_validation.png")


def rank(values: np.ndarray) -> np.ndarray:
    """1 for the smallest value, 2 for the next, and so on."""
    return values.argsort().argsort() + 1


def replay(race_set: RaceSet, lap_model: LapModel, stats: dict, samples: int = SAMPLES):
    """Simulate every race `samples` times. Returns (positions, times), each
    of shape (races, samples, cars)."""
    races = len(race_set.races)
    index = torch.arange(races, device=race_set.device).repeat_interleave(samples)
    sim = RaceSim(race_set, lap_model, stats, index).run_to_end()
    shape = (races, samples, simulator.CARS)
    return (sim.final_positions().reshape(shape).cpu().numpy(),
            sim.time.reshape(shape).cpu().numpy())


def score_race(race, positions: np.ndarray, times: np.ndarray) -> dict:
    """Compare one race's simulations with what really happened.

    Only drivers who really finished are compared: the simulator makes no
    attempt to predict crashes or breakdowns.
    """
    cars = len(race.drivers)
    finished = (race.retire_lap == NEVER) & (race.finish_position < 99)
    actual = rank(race.finish_position[finished])
    grid = rank(race.grid[finished])
    # Each simulation's order among the finishers, then the average of those.
    simulated = np.stack([rank(sample[:cars][finished]) for sample in positions])
    predicted = rank(simulated.mean(axis=0))

    low, high = np.percentile(simulated, [5, 95], axis=0)
    on_lead_lap = ~np.isnan(race.gap_to_winner[finished])
    finisher_times = times[:, :cars][:, finished]
    simulated_gap = np.median(finisher_times - finisher_times.min(axis=1, keepdims=True), axis=0)
    gap_error = np.abs(simulated_gap - race.gap_to_winner[finished])[on_lead_lap]
    return {
        "race": race.label,
        "wet_qualifying": race.wet_qualifying,
        "finishers": int(finished.sum()),
        "position_miss": float(np.abs(predicted - actual).mean()),
        "grid_order_miss": float(np.abs(grid - actual).mean()),
        "order_match": float(np.corrcoef(predicted, actual)[0, 1]),
        "winner_right": bool(predicted[actual == 1][0] == 1),
        "podium_right": int(((predicted <= 3) & (actual <= 3)).sum()),
        "inside_range": float(((actual >= low) & (actual <= high)).mean()),
        "gap_miss_s": float(np.median(gap_error)) if on_lead_lap.sum() > 1 else float("nan"),
        "_predicted": predicted, "_actual": actual,
    }


def score_all(race_set, positions, times) -> list[dict]:
    return [score_race(race, positions[r], times[r]) for r, race in enumerate(race_set.races)]


def summarise(rows: list[dict], name: str) -> dict:
    """Averages over a set of races."""
    mean = lambda key: float(np.nanmean([row[key] for row in rows]))
    return {
        "races": name,
        "count": len(rows),
        "position_miss": round(mean("position_miss"), 2),
        "grid_order_miss": round(mean("grid_order_miss"), 2),
        "order_match": round(mean("order_match"), 2),
        "winner_right": f"{sum(row['winner_right'] for row in rows)}/{len(rows)}",
        "podium_right": f"{sum(row['podium_right'] for row in rows)}/{3 * len(rows)}",
        "inside_range": round(mean("inside_range"), 2),
        "median_gap_miss_s": round(float(np.nanmedian([row["gap_miss_s"] for row in rows])), 1),
    }


def summary_table(rows: list[dict]) -> pl.DataFrame:
    dry_qualifying = [row for row in rows if not row["wet_qualifying"]]
    return pl.DataFrame([
        summarise(dry_qualifying, "dry race, dry qualifying"),
        summarise(rows, "all dry races"),
    ])


def save_chart(rows: list[dict], season: str, file: Path = CHART_FILE):
    """Left: predicted against real finishing position for every driver.
    Right: the simulator's miss per race, next to the grid order's miss."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, grey, ink = "#2563eb", "#9ca3af", "#111827"
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=[1, 1.5])
    predicted = np.concatenate([row["_predicted"] for row in rows])
    actual = np.concatenate([row["_actual"] for row in rows])
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, size=(2, len(actual)))
    left.plot([1, 20], [1, 20], color=grey, linewidth=1, zorder=1)
    left.scatter(actual + jitter[0], predicted + jitter[1], s=14, color=blue, alpha=0.45,
                 linewidths=0, zorder=2)
    left.set_xlabel("Real finishing position")
    left.set_ylabel("Simulated finishing position")
    left.set_title(f"Every finisher, {season}\n(on the line = exactly right)",
                   loc="left", fontsize=11, color=ink)
    left.set_xlim(0.3, 20.7)
    left.set_ylim(0.3, 20.7)
    left.set_xticks([1, 5, 10, 15, 20])
    left.set_yticks([1, 5, 10, 15, 20])
    left.set_aspect("equal")

    names = [row["race"].replace(" Grand Prix", "") for row in rows]
    where = np.arange(len(rows))
    right.barh(where - 0.2, [row["position_miss"] for row in rows], height=0.38,
               color=blue, label="Simulator")
    right.barh(where + 0.2, [row["grid_order_miss"] for row in rows], height=0.38,
               color=grey, label="Grid order (no simulation)")
    right.set_yticks(where, names, fontsize=8)
    right.invert_yaxis()
    right.set_xlabel("Average miss in finishing position (lower is better)")
    right.set_title("Miss per race", loc="left", fontsize=11, color=ink)
    right.legend(frameon=False, fontsize=9)
    for axes in (left, right):
        axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    file.parent.mkdir(exist_ok=True)
    figure.savefig(file, dpi=130)
    plt.close(figure)
    print(f"\nChart saved to {file}")


def print_report(rows: list[dict]):
    per_race = pl.DataFrame([
        {key: (round(value, 2) if isinstance(value, float) else value)
         for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ])
    print("\nPer race (position_miss: average places a finisher is off by):")
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200):
        print(per_race)
    print("\nSummary. grid_order_miss is the same measure for 'everyone finishes"
          "\nwhere they started', which the simulator has to beat:")
    with pl.Config(tbl_cols=12, tbl_width_chars=200):
        print(summary_table(rows))


def tune(race_set, lap_model, stats):
    """Try values of the overtaking rule and show the position miss."""
    print("Overtaking advantage needed (s)  ->  position miss, order match")
    for value in [0.0, 0.3, 0.6, 1.0, 1.5, 2.5]:
        simulator.OVERTAKE_ADVANTAGE_SECONDS = value
        rows = score_all(race_set, *replay(race_set, lap_model, stats))
        dry = summarise([row for row in rows if not row["wet_qualifying"]], "")
        print(f"  {value:>4}   {dry['position_miss']:.2f}   {dry['order_match']:.2f}"
              f"   inside range {dry['inside_range']:.2f}   gap miss {dry['median_gap_miss_s']}s")


def main():
    lap_model = LapModel()
    stats = simulator.load_stats()
    race_set = RaceSet(simulator.load_races(["validation"]), lap_model, stats)
    print(f"Replaying {len(race_set.races)} dry 2024 races without red flags,"
          f" {SAMPLES} times each, on {race_set.device}.")
    if "--tune" in sys.argv:
        tune(race_set, lap_model, stats)
        return
    rows = score_all(race_set, *replay(race_set, lap_model, stats))
    print_report(rows)
    save_chart(rows, "2024 validation races")


if __name__ == "__main__":
    main()
