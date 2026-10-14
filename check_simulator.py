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
#       .venv\Scripts\python.exe check_simulator.py --race-page
#         the all-time grid checks: pole share and average grid slot of the top
#         five, and how far drivers starting 16th to 20th come through at
#         Silverstone, each next to real F1 data

import json
import sys
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


def all_time_grid() -> list[dict]:
    """The website's "All-time top 20" quick grid: the 20 best rated drivers with 50+ starts."""
    drivers = json.loads((WEB / "drivers.json").read_text(encoding="utf-8"))
    return sorted([d for d in drivers if d["starts"] >= 50], key=lambda d: -d["skill"])[:20]


def circuit(circuit_id: str) -> tuple[float, int, float]:
    """A circuit's lap time, race laps and overtaking difficulty."""
    data = json.loads((WEB / "circuits" / f"{circuit_id}.json").read_text(encoding="utf-8"))
    settings = json.loads((WEB / "settings.json").read_text(encoding="utf-8"))
    return data["lapSeconds"], data["raceLaps"], settings["difficulty"][circuit_id]


def run_field(field: list[dict], circuit_id: str, laps: int | None = None, races: int = RACES, seed: int = 3) -> dict:
    """Simulate `races` races (or, with laps=1, just qualifying) for a field at a circuit."""
    lap_seconds, race_laps, difficulty = circuit(circuit_id)
    generator = torch.Generator(device=simulate.DEVICE).manual_seed(seed)
    skill = torch.tensor([d["skill"] for d in field], device=simulate.DEVICE).repeat(races, 1)
    errors = simulate.error_factor(torch.tensor([float(d["consistency"]) for d in field], device=simulate.DEVICE))
    return simulate.simulate_races(skill, errors, lap_seconds, laps or race_laps, difficulty, generator)


def qualifying_report(field: list[dict], sessions: int = RACES) -> list[dict]:
    """How often each of the top 5 rated drivers takes pole, and their average grid slot."""
    grid = run_field(field, "silverstone", laps=1, races=sessions)["grid"] + 1  # 1 = pole
    return [{"driver": d["name"], "skill": d["skill"],
             "pole": (grid[:, i] == 1).float().mean().item(),
             "average_grid": grid[:, i].float().mean().item()} for i, d in enumerate(field[:5])]


def back_of_grid_report(field: list[dict], circuit_id: str = "silverstone", races: int = RACES) -> list[dict]:
    """For each driver, what happens when they start in the bottom five (16th to 20th)."""
    result = run_field(field, circuit_id, races=races)
    start, finish, crashed = result["grid"] + 1, result["place"], result["crashed"]
    rows = []
    for i, d in enumerate(field):  # the field is sorted best rated first
        back = (start[:, i] >= 16) & ~crashed[:, i]
        if back.sum() < 20:
            continue  # too few races starting at the back to say anything
        rows.append({"driver": d["name"], "rating_rank": i + 1, "races": int(back.sum()),
                     "average_start": start[back, i].float().mean().item(),
                     "average_finish": finish[back, i].float().mean().item()})
    return rows


def real_qualifying_reference(first: int = 2014, last: int = 2025) -> list[dict]:
    """Real F1: each season's best qualifier (best average qualifying position), their pole
    share and average position. Real cars differ, so this is the upper end of how dominant a
    top qualifier can be."""
    results = pl.read_parquet("data/results.parquet").filter(
        pl.col("Year").is_between(first, last) & (pl.col("QualiSource") == "qualifying"))
    races = results.group_by("Year").agg(pl.struct("Round").n_unique().alias("races"))
    per_driver = results.group_by("Year", "DriverName").agg(
        pl.len().alias("n"), (pl.col("QualiPosition") == 1).mean().alias("pole"),
        pl.col("QualiPosition").mean().alias("average")).join(races, on="Year")
    best = per_driver.filter(pl.col("n") >= pl.col("races") * 0.75).sort("average").group_by("Year").first()
    return best.sort("Year").to_dicts()


def real_back_of_grid_reference(first: int = 2018, last: int = 2025) -> dict:
    """Real F1: average finish of classified drivers who started 16th to 20th, all of them and
    only those in one of the season's three best cars (the nearest real case to a fast driver
    stuck at the back)."""
    results = pl.read_parquet("data/results.parquet").filter(
        pl.col("Year").is_between(first, last) & pl.col("Classified") & pl.col("Grid").is_between(16, 20))
    cars = pl.read_parquet("data/car_strengths.parquet").with_columns(
        pl.col("CarStrength").rank(descending=True).over("Year").alias("car_rank"))
    joined = results.join(cars.select("TeamId", "Year", "car_rank"), on=["TeamId", "Year"], how="left")
    top_cars = joined.filter(pl.col("car_rank") <= 3)
    return {"all": (results.height, results["Grid"].mean(), results["Position"].mean()),
            "top_three_cars": (top_cars.height, top_cars["Grid"].mean(), top_cars["Position"].mean())}


def print_race_page_checks():
    """The owner's checks on the all-time grid: qualifying spread and the back of the grid at Silverstone."""
    field = all_time_grid()
    print(f"\nAll-time top 20 grid, {RACES} simulated qualifying sessions:")
    for row in qualifying_report(field):
        print(f"  {row['driver']:<22} skill {row['skill']:+.2f}  pole {row['pole'] * 100:5.1f}%  "
              f"average grid slot {row['average_grid']:4.1f}")
    print("\nReal F1, each season's best qualifier (real cars differ):")
    for row in real_qualifying_reference():
        print(f"  {row['Year']}  {row['DriverName']:<22} pole {row['pole'] * 100:5.1f}%  average position {row['average']:4.1f}")
    print(f"\nAll-time top 20 grid at Silverstone, {RACES} races: drivers when they started 16th to 20th")
    print(f"  {'driver':<22} {'rating rank':>11} {'races':>6} {'avg start':>9} {'avg finish':>10} {'gained':>7}")
    for row in back_of_grid_report(field):
        print(f"  {row['driver']:<22} {row['rating_rank']:>11} {row['races']:>6} {row['average_start']:>9.1f} "
              f"{row['average_finish']:>10.1f} {row['average_start'] - row['average_finish']:>+7.1f}")
    real = real_back_of_grid_reference()
    for label, (count, start, finish) in (("all classified starters", real["all"]),
                                          ("in one of the 3 best cars", real["top_three_cars"])):
        print(f"  Real F1 2018-2025, {label}: {count} cases, average start {start:.1f}, "
              f"average finish {finish:.1f} (gained {start - finish:+.1f})")


def main():
    if "--race-page" in sys.argv:
        print_race_page_checks()
        return
    field = todays_grid()
    lap_seconds, laps, difficulty = interlagos()
    print(f"Today's grid at Interlagos, {RACES} races, {laps} laps. Top rated: "
          f"{', '.join(d['name'] for d in field[:3])}")
    print(f"Real F1, 2018 to 2025: grid vs finish rank correlation {real_rank_correlation():.3f} "
          "(real cars differ, which ties the order to the grid more)")
    print_numbers("This simulator:", measure(simulate.simulate_races, field, lap_seconds, laps, difficulty))


if __name__ == "__main__":
    main()
