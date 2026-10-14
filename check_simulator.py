# check_simulator.py
#
# What it does: measures whether simulated races behave sensibly, using the
#   same test the owner ran by hand: today's grid at Interlagos.
#     1. How closely the finishing order follows the grid (rank correlation,
#        1 = exactly the grid order, 0 = no relation), compared with real
#        F1 races from 2018 to 2025.
#     2. How often the top rated driver wins when starting from pole, 5th
#        and 15th.
#     3. How often the fastest lap goes to one of the three top rated drivers.
# What it reads: web/public/data/drivers.json and settings.json (written by
#   export_web.py), web/public/data/circuits/interlagos.json, and
#   data/results.parquet for the real races.
# What it produces: a table printed to the screen. measure() is also used by
#   test_simulate.py to keep these numbers inside sensible bounds.
# Which files use it: test_simulate.py.
#
# Run:  .venv\Scripts\python.exe check_simulator.py

import json
from pathlib import Path

import polars as pl
import torch

import simulate

WEB = Path("web/public/data")
RACES = 1000


def todays_grid() -> list[dict]:
    """The 20 drivers of the latest season with the most starts, best rated first."""
    drivers = json.loads((WEB / "drivers.json").read_text(encoding="utf-8"))
    latest = max(driver["years"][1] for driver in drivers)
    field = sorted([d for d in drivers if d["years"][1] == latest], key=lambda d: -d["starts"])[:20]
    return sorted(field, key=lambda d: -d["skill"])


def rank_correlation(grid: torch.Tensor, place: torch.Tensor, crashed: torch.Tensor) -> float:
    """Average over races of the correlation between grid order and finishing order (finishers only)."""
    values = []
    for race in range(grid.shape[0]):
        keep = ~crashed[race]
        g = grid[race][keep].float().argsort().argsort().float()
        f = place[race][keep].float().argsort().argsort().float()
        g, f = g - g.mean(), f - f.mean()
        values.append(((g * f).sum() / (g.norm() * f.norm())).item())
    return sum(values) / len(values)


def measure(simulate_races, field: list[dict], lap_seconds: float, laps: int, difficulty: float,
            seed: int = 1) -> dict:
    """The four numbers for one simulator, field and circuit."""
    generator = torch.Generator(device=simulate.DEVICE).manual_seed(seed)
    skill = torch.tensor([d["skill"] for d in field], device=simulate.DEVICE).repeat(RACES, 1)
    errors = simulate.error_factor(torch.tensor([float(d["consistency"]) for d in field], device=simulate.DEVICE))

    result = simulate_races(skill, errors, lap_seconds, laps, difficulty, generator)
    best = result["best_lap"].argmin(dim=1)  # field is sorted best first, so 0, 1, 2 are the top three
    # Gaps at the flag, among cars that finished: winner to second, and to tenth.
    finish_order = torch.where(result["crashed"], torch.inf, result["race_time"]).sort(dim=1).values
    numbers = {
        "gap_to_second": (finish_order[:, 1] - finish_order[:, 0]).median().item(),
        "gap_to_tenth": (finish_order[:, 9] - finish_order[:, 0]).median().item(),
        "rank_correlation": rank_correlation(result["grid"], result["place"], result["crashed"]),
        "average_places_moved": (result["grid"] + 1 - result["place"]).abs().float().mean().item(),
        "fastest_lap_top3": (best < 3).float().mean().item(),
    }
    for slot in (0, 4, 14):
        # The top driver starts from `slot`; everyone else fills the grid in rating order.
        others = [s for s in range(len(field)) if s != slot]
        grid = torch.tensor([slot] + others, device=simulate.DEVICE)
        forced = simulate_races(skill, errors, lap_seconds, laps, difficulty, generator, grid=grid)
        numbers[f"top_wins_from_{slot + 1}"] = (forced["place"][:, 0] == 1).float().mean().item()
    return numbers


def real_rank_correlation() -> float:
    """The same correlation for real races from 2018 to 2025 (classified finishers)."""
    results = pl.read_parquet("data/results.parquet").filter(
        pl.col("Year").is_between(2018, 2025) & pl.col("Classified") & (pl.col("Grid") > 0))
    per_race = results.with_columns(
        pl.col("Grid").rank().over("Year", "Round").alias("g"),
        pl.col("Position").rank().over("Year", "Round").alias("f"),
    ).group_by("Year", "Round").agg(pl.corr("g", "f", method="spearman").alias("rho"))
    return per_race["rho"].mean()


def interlagos() -> tuple[float, int, float]:
    circuit = json.loads((WEB / "circuits" / "interlagos.json").read_text(encoding="utf-8"))
    settings = json.loads((WEB / "settings.json").read_text(encoding="utf-8"))
    return circuit["lapSeconds"], circuit["raceLaps"], settings["difficulty"]["interlagos"]


def print_numbers(label: str, numbers: dict):
    print(f"\n{label}")
    print(f"  grid vs finish rank correlation   {numbers['rank_correlation']:.3f}")
    print(f"  average places moved              {numbers['average_places_moved']:.2f}")
    for slot in (1, 5, 15):
        print(f"  top rated driver wins from {slot:<2}     {numbers[f'top_wins_from_{slot}'] * 100:5.1f}%")
    print(f"  fastest lap by a top 3 driver     {numbers['fastest_lap_top3'] * 100:5.1f}%")
    if "gap_to_second" in numbers:
        print(f"  median gap, winner to 2nd / 10th  {numbers['gap_to_second']:.1f}s / {numbers['gap_to_tenth']:.1f}s")


def main():
    field = todays_grid()
    lap_seconds, laps, difficulty = interlagos()
    print(f"Today's grid at Interlagos, {RACES} races, {laps} laps. Top rated: "
          f"{', '.join(d['name'] for d in field[:3])}")
    print(f"Real F1, 2018 to 2025: grid vs finish rank correlation {real_rank_correlation():.3f} "
          "(real cars differ, which ties the order to the grid more)")
    print_numbers("This simulator:", measure(simulate.simulate_races, field, lap_seconds, laps, difficulty))


if __name__ == "__main__":
    main()
