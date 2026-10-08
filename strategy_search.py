# strategy_search.py
#
# What it does: the brute force benchmark of Phase 2. For one driver in one
#   race it tries every sensible fixed strategy (one stop or two stops, on
#   every allowed tyre combination), simulates each one many times, and keeps
#   the best. Run as a script, it does this for every finisher of every dry
#   2024 race and compares the best fixed strategy with what the team did.
# What it reads: data/laps.parquet, models/lap_time_network.pt and
#   models/race_stats.json (through simulator.py).
# What it produces: results/benchmark_2024.parquet (one row per driver per
#   race), and a summary on the screen.
# Which files use it: evaluate_strategist.py and app.py use candidate_plans(),
#   simulate_plans() and best_fixed_plan(). final_test.py runs benchmark().
#
# Run:  .venv\Scripts\python.exe strategy_search.py
#
# "Fixed" means decided before the race and followed whatever happens: it
# does not react to safety cars. That is exactly what the AI strategist of
# Phase 3 should be able to beat.

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

import simulator
from simulator import NEVER, LapModel, RaceSet, RaceSim
from train_network import COMPOUND_NAMES

RESULTS_FOLDER = Path("results")

MIN_STINT_LAPS = 5  # no stop in the first or last few laps, or this close together
ONE_STOP_STEP = 1  # try a one-stop on every lap
TWO_STOP_STEP = 3  # try two-stops on every third lap (there are far more)
SEARCH_SAMPLES = 8  # simulations per strategy in the first pass
SHORTLIST = 20  # the best strategies get a second, more careful look
FINAL_SAMPLES = 64  # simulations per strategy in the second pass
FORCED_STOP_LAPS = 3  # a real stop this early was forced by damage


def candidate_plans(total_laps: int, start_compound: int, max_laps: int) -> torch.Tensor:
    """Every fixed strategy worth trying, as plans of shape (P, max_laps).

    A plan holds, for each lap, 0 (stay out) or 1 + compound (pit at the end
    of this lap). The two compound rule is respected: at least two different
    compounds must be used, counting the starting tyres.
    """
    plans = []
    first, last = MIN_STINT_LAPS, total_laps - MIN_STINT_LAPS
    for lap in range(first, last + 1, ONE_STOP_STEP):
        for compound in range(3):
            if compound != start_compound:
                plan = np.zeros(max_laps, dtype=np.int64)
                plan[lap - 1] = 1 + compound
                plans.append(plan)
    for lap_one in range(first, last + 1, TWO_STOP_STEP):
        for lap_two in range(lap_one + MIN_STINT_LAPS, last + 1, TWO_STOP_STEP):
            for compound_one in range(3):
                for compound_two in range(3):
                    if len({start_compound, compound_one, compound_two}) >= 2:
                        plan = np.zeros(max_laps, dtype=np.int64)
                        plan[lap_one - 1] = 1 + compound_one
                        plan[lap_two - 1] = 1 + compound_two
                        plans.append(plan)
    return torch.tensor(np.stack(plans))


def describe(plan, start_compound: int) -> str:
    """A plan as text, for example 'MEDIUM, lap 18 HARD'."""
    plan = np.asarray(plan)
    parts = [COMPOUND_NAMES[start_compound]]
    parts += [f"lap {lap + 1} {COMPOUND_NAMES[plan[lap] - 1]}" for lap in np.nonzero(plan)[0]]
    return ", ".join(parts)


def simulate_plans(race_set: RaceSet, lap_model: LapModel, stats: dict, race: int, car: int,
                   plans: torch.Tensor, samples: int, real_events: bool, seed: int = 0,
                   fixed: bool = True, ages=None):
    """Simulate one car using each plan, `samples` times each.

    Every plan faces exactly the same `samples` random races (same luck, and
    the same safety cars), so differences between plans are due to the plans.
    real_events=True uses the race's real safety car laps; False invents
    them at random, as they would be unknown before the race.
    Returns (finishing positions, race times), each of shape (P, samples).
    """
    device = race_set.device
    plans = plans.to(device)
    count = len(plans)
    index = torch.full((count * samples,), race, device=device)
    group = torch.arange(samples, device=device).repeat(count)
    plan = race_set.plan[index].clone()
    plan[:, car, :] = plans.repeat_interleave(samples, dim=0)
    plan_age = race_set.plan_age[index].clone()
    plan_age[:, car, :] = 0 if ages is None else ages.to(device).repeat_interleave(samples, dim=0)
    is_fixed = torch.zeros(count * samples, simulator.CARS, dtype=torch.bool, device=device)
    is_fixed[:, car] = fixed
    status = None
    if not real_events:
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        one_each = simulator.sample_status(
            race_set, torch.full((samples,), race, device=device), stats, generator)
        status = one_each[group]
    sim = RaceSim(race_set, lap_model, stats, index, status=status, plan=plan,
                  plan_age=plan_age, fixed=is_fixed, group=group, seed=seed).run_to_end()
    shape = (count, samples)
    return (sim.final_positions()[:, car].reshape(shape).float(),
            sim.time[:, car].reshape(shape))


def best_fixed_plan(race_set, lap_model, stats, race: int, car: int, real_events: bool,
                    seed: int = 0) -> torch.Tensor:
    """Search all candidate plans; returns the best one, shape (max_laps,).

    Two passes: a quick look at every plan, then a careful look at the best
    few, so that a plan cannot win just by being lucky in a few simulations.
    Best means lowest average finishing position, then lowest race time.
    """
    info = race_set.races[race]
    plans = candidate_plans(info.total_laps, int(info.start_compound[car]), race_set.max_laps)

    def ranked(plans, samples, seed):
        positions, times = simulate_plans(race_set, lap_model, stats, race, car, plans,
                                          samples, real_events, seed)
        # Position decides; time (in fractions of a place) only breaks ties.
        score = positions.mean(dim=1) + times.mean(dim=1) / 1e5
        return plans[score.argsort().cpu()]

    shortlist = ranked(plans, SEARCH_SAMPLES, seed)[:SHORTLIST]
    return ranked(shortlist, FINAL_SAMPLES, seed + 100)[0]


def finishers(info) -> list[int]:
    """The cars that really finished (the ones worth analysing)."""
    return [car for car in range(len(info.drivers))
            if info.retire_lap[car] == NEVER and info.finish_position[car] < 99]


def benchmark_driver(race_set, lap_model, stats, race: int, car: int, samples: int) -> dict:
    """Real strategy against the best fixed strategies, for one driver."""
    info = race_set.races[race]
    start = int(info.start_compound[car])
    before_race = best_fixed_plan(race_set, lap_model, stats, race, car, real_events=False)
    hindsight = best_fixed_plan(race_set, lap_model, stats, race, car, real_events=True)
    # The final comparison: all three in the race as it really unfolded,
    # facing the same luck. Seed 999 is not used anywhere in the search.
    real_plan = race_set.plan[race, car].cpu()
    plans = torch.stack([real_plan, before_race, hindsight])
    ages = torch.stack([race_set.plan_age[race, car].cpu(),
                        torch.zeros_like(race_set.plan_age[race, car].cpu()),
                        torch.zeros_like(race_set.plan_age[race, car].cpu())])
    positions, times = simulate_plans(race_set, lap_model, stats, race, car, plans, samples,
                                      real_events=True, seed=999, ages=ages)
    position, seconds = positions.mean(dim=1).tolist(), times.mean(dim=1).tolist()
    return {
        "Year": info.year, "Round": info.round, "Race": info.name, "Circuit": info.circuit,
        "Driver": info.drivers[car], "Team": info.teams[car],
        "Grid": int(info.grid[car]), "RealFinish": int(info.finish_position[car]),
        "RealStrategy": describe(real_plan, start),
        "RealStops": int((real_plan > 0).sum()),
        # A stop in the first three laps is damage, not strategy.
        "ForcedStop": bool((real_plan[:FORCED_STOP_LAPS] > 0).any()),
        "RealPosition": position[0],
        "FixedStrategy": describe(before_race, start),
        "FixedPosition": position[1], "FixedSeconds": seconds[0] - seconds[1],
        "HindsightStrategy": describe(hindsight, start),
        "HindsightPosition": position[2], "HindsightSeconds": seconds[0] - seconds[2],
    }


def benchmark(race_set, lap_model, stats, samples: int = 200) -> pl.DataFrame:
    """Run benchmark_driver for every finisher of every race."""
    rows, started = [], time.perf_counter()
    for race, info in enumerate(race_set.races):
        for car in finishers(info):
            rows.append(benchmark_driver(race_set, lap_model, stats, race, car, samples))
        print(f"  {info.label:<34} done  ({time.perf_counter() - started:.0f}s so far)")
    return pl.DataFrame(rows)


def fair_comparisons(table: pl.DataFrame) -> pl.DataFrame:
    """Leave out drivers whose real race included a stop forced by damage:
    beating a strategy nobody chose proves nothing."""
    return table.filter(~pl.col("ForcedStop"))


def summary(table: pl.DataFrame) -> pl.DataFrame:
    """Average places and seconds gained over the real strategy.

    Positive numbers mean the alternative strategy did better than the team.
    """
    table = fair_comparisons(table)

    def row(name, position, seconds):
        gained = table["RealPosition"] - table[position]
        return {
            "strategy": name,
            "places_gained": round(gained.mean(), 2),
            "seconds_gained": round(table[seconds].mean(), 1),
            "better_than_team": f"{(gained > 0.25).mean():.0%}",
            "worse_than_team": f"{(gained < -0.25).mean():.0%}",
        }

    return pl.DataFrame([
        row("best fixed plan, chosen before the race", "FixedPosition", "FixedSeconds"),
        row("best fixed plan, chosen with hindsight", "HindsightPosition", "HindsightSeconds"),
    ])


def main():
    lap_model = LapModel()
    stats = simulator.load_stats()
    race_set = RaceSet(simulator.load_races(["validation"]), lap_model, stats)
    print(f"Brute force search for every finisher of {len(race_set.races)} dry 2024 races"
          f" on {race_set.device}:")
    table = benchmark(race_set, lap_model, stats)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    table.write_parquet(RESULTS_FOLDER / "benchmark_2024.parquet")

    print(f"\n{table.height} driver races, of which {int(table['ForcedStop'].sum())} had a"
          f" stop forced by damage and are left out of the comparison.")
    print("\nAgainst what the teams really did (all simulated in the real race conditions):")
    with pl.Config(tbl_width_chars=160, fmt_str_lengths=60):
        print(summary(table))
        print("\nStops in the real strategies against the best fixed plan:")
        fixed_stops = table["FixedStrategy"].str.count_matches("lap")
        print(pl.DataFrame({"real_stops": table["RealStops"], "fixed_stops": fixed_stops})
              .group_by("real_stops", "fixed_stops").len().sort("real_stops", "fixed_stops"))


if __name__ == "__main__":
    main()
