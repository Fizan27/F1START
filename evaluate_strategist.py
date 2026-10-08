# evaluate_strategist.py
#
# What it does: tests the trained AI strategist on races it never trained on
#   (the 2024 season). For every finisher of every dry race it lets the
#   strategist run that car's race 200 times, in the real race conditions,
#   and compares the result with (a) what the team really did and (b) the
#   best fixed strategy found by brute force. It then looks for signs that
#   the strategist is exploiting flaws in the simulator instead of being
#   clever.
# What it reads: models/strategist.pt (from train_agent.py),
#   results/benchmark_2024.parquet (from strategy_search.py), and the
#   simulator's usual files.
# What it produces: results/strategist_2024.parquet (one row per driver per
#   race), docs/strategist_2024.png and tables on the screen.
# Which files use it: final_test.py calls evaluate() and the report
#   functions on 2025. app.py reads the results file for the report card.
#
# Run:  .venv\Scripts\python.exe evaluate_strategist.py
#
# Fairness: all strategies face the same 200 random versions of each race
# (the same lap time luck and the real safety cars). Only the strategy of
# the one car being studied differs.

from pathlib import Path

import numpy as np
import polars as pl
import torch

import simulator
import strategy_search
from simulator import LapModel, RaceSet
from strategy_env import StrategyEnv
from strategy_search import RESULTS_FOLDER, describe, finishers, simulate_plans
from train_agent import drive, load_strategist
from train_network import COMPOUND_NAMES

SAMPLES = 200
SEED = 999  # the same seed strategy_search.py used for its final comparison


def evaluate_race(env: StrategyEnv, strategist, race: int) -> list[dict]:
    """The strategist against the real strategy, for every finisher."""
    race_set, info = env.set, env.set.races[race]
    cars = finishers(info)
    device = race_set.device
    agent = torch.tensor(cars, device=device).repeat_interleave(SAMPLES)
    index = torch.full((len(agent),), race, device=device)
    group = torch.arange(SAMPLES, device=device).repeat(len(cars))
    driven = drive(env, strategist, index, agent, seed=SEED, group=group)
    shape = (len(cars), SAMPLES)
    # Again, but only allowed the compounds each car's team really used.
    allowed = torch.zeros(len(cars), 3, dtype=torch.bool, device=device)
    for i, car in enumerate(cars):
        allowed[i, list(strategy_search.team_compounds(info, car))] = True
    same_tyres = drive(env, strategist, index, agent, seed=SEED, group=group,
                       allowed=allowed.repeat_interleave(SAMPLES, dim=0))
    same_positions = same_tyres["position"].reshape(shape)
    same_times = same_tyres["time"].reshape(shape)
    positions = driven["position"].reshape(shape)
    times = driven["time"].reshape(shape)
    actions = driven["actions"].reshape(*shape, -1)[:, :, :info.total_laps].cpu().numpy()
    forced = driven["forced"].reshape(shape).float()
    status = torch.as_tensor(info.status)

    rows = []
    for i, car in enumerate(cars):
        # The real strategy in the same 200 versions of the race.
        real_plan = race_set.plan[race, car].cpu()[None]
        real_positions, real_times = simulate_plans(
            race_set, env.model, env.stats, race, car, real_plan, SAMPLES, real_events=True,
            seed=SEED, ages=race_set.plan_age[race, car].cpu()[None])
        stops = actions[i] > 0  # (samples, laps)
        stop_laps = np.nonzero(stops)[1] + 1
        compound_laps = laps_per_compound(actions[i], int(info.start_compound[car]))
        rows.append({
            "Year": info.year, "Round": info.round, "Driver": info.drivers[car],
            "AgentPosition": positions[i].mean().item(),
            "AgentSeconds": (real_times.mean() - times[i].mean()).item(),
            "AgentSameTyresPosition": same_positions[i].mean().item(),
            "AgentSameTyresSeconds": (real_times.mean() - same_times[i].mean()).item(),
            "AgentStops": float(stops.sum(axis=1).mean()),
            "AgentExample": describe(actions[i][0], int(info.start_compound[car])),
            "AgentForced": forced[i].mean().item(),
            "AgentStopsUnderCaution": float(
                np.isin(stop_laps, np.nonzero(info.status > 0)[0] + 1).mean()
            ) if len(stop_laps) else 0.0,
            "AgentLongestStint": float(longest_stints(stops).mean()),
            **{f"Agent{name.title()}Laps": compound_laps[c] for c, name in enumerate(COMPOUND_NAMES)},
            "RaceHadCaution": bool((status > 0).any()),
            "CheckRealPosition": real_positions.mean().item(),
        })
    return rows


def longest_stints(stops: np.ndarray) -> np.ndarray:
    """The longest run of laps without a stop, for each simulated race."""
    longest = np.zeros(len(stops))
    current = np.zeros(len(stops))
    for lap in range(stops.shape[1]):
        current += 1
        longest = np.maximum(longest, current)
        current = np.where(stops[:, lap], 0, current)
    return longest


def laps_per_compound(actions: np.ndarray, start_compound: int) -> list[float]:
    """Share of the race spent on each compound, averaged over simulations."""
    compound = np.full(len(actions), start_compound)
    laps = np.zeros(3)
    for lap in range(actions.shape[1]):
        laps += np.bincount(compound, minlength=3)
        compound = np.where(actions[:, lap] > 0, actions[:, lap] - 1, compound)
    return (laps / laps.sum()).tolist()


def evaluate(env: StrategyEnv, strategist, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Evaluate every race and join the brute force benchmark onto it."""
    rows = []
    for race, info in enumerate(env.set.races):
        rows += evaluate_race(env, strategist, race)
        print(f"  {info.label:<34} done")
    table = benchmark.join(pl.DataFrame(rows), on=["Year", "Round", "Driver"])
    # Both files simulated the real strategy with the same seed, so the two
    # numbers must agree. If they do not, the comparison is not like for like.
    drift = (table["RealPosition"] - table["CheckRealPosition"]).abs().max()
    assert drift < 1e-3, f"real strategy results differ between files by {drift}"
    return table.drop("CheckRealPosition")


def comparison(table: pl.DataFrame, label: str = "") -> pl.DataFrame:
    """Places and seconds gained over the real team. Positive = better."""
    fair = strategy_search.fair_comparisons(table)

    def row(name, position, seconds):
        gained = fair["RealPosition"] - fair[position]
        return {
            "strategy": name, "driver_races": fair.height,
            "places_gained": round(gained.mean(), 2),
            "seconds_gained": round(fair[seconds].mean(), 1),
            "better_than_team": f"{(gained > 0.25).mean():.0%}",
            "worse_than_team": f"{(gained < -0.25).mean():.0%}",
        }

    return pl.DataFrame([
        row(f"{label}best fixed plan (before the race)", "FixedPosition", "FixedSeconds"),
        row(f"{label}best fixed plan (hindsight)", "HindsightPosition", "HindsightSeconds"),
        row(f"{label}AI strategist", "AgentPosition", "AgentSeconds"),
        row(f"{label}best fixed plan, team's tyres only", "SameTyresPosition",
            "SameTyresSeconds"),
        row(f"{label}AI strategist, team's tyres only", "AgentSameTyresPosition",
            "AgentSameTyresSeconds"),
    ])


def exploit_checks(table: pl.DataFrame, race_set: RaceSet, stats: dict) -> pl.DataFrame:
    """Compare how the strategist behaves with how real teams behave.

    Large differences are where to look for the strategist exploiting the
    simulator. Each row is one suspicion, explained in DECISIONS.md 29.
    """
    fair = strategy_search.fair_comparisons(table)
    team_stops, team_caution, team_stint, team_laps = [], [], [], np.zeros(3)
    for race in race_set.races:
        for car in finishers(race):
            plan = race.plan[car]
            if plan[:strategy_search.FORCED_STOP_LAPS].any():
                continue
            laps = np.nonzero(plan)[0] + 1
            team_stops.append(len(laps))
            team_caution += list(np.isin(laps, np.nonzero(race.status > 0)[0] + 1))
            team_stint.append(longest_stints(plan[None] > 0)[0])
            team_laps += np.array(laps_per_compound(plan[None], int(race.start_compound[car])))
    team_laps /= team_laps.sum()
    return pl.DataFrame({
        "check": ["stops per race", "share of stops made under safety car or VSC",
                  "longest stint (laps)", "share of laps on soft", "share of laps on medium",
                  "share of laps on hard", "races needing a forced last lap stop"],
        "real_teams": [np.mean(team_stops), np.mean(team_caution), np.mean(team_stint),
                       *team_laps, 0.0],
        "ai_strategist": [fair["AgentStops"].mean(), fair["AgentStopsUnderCaution"].mean(),
                          fair["AgentLongestStint"].mean(), fair["AgentSoftLaps"].mean(),
                          fair["AgentMediumLaps"].mean(), fair["AgentHardLaps"].mean(),
                          fair["AgentForced"].mean()],
    }).with_columns(pl.col("real_teams", "ai_strategist").round(3))


def save_chart(table: pl.DataFrame, season: str, file: Path):
    """Left: average places gained over the real team, by strategy.
    Right: how the strategist's gain is spread across driver races."""
    import chart_style as style

    fair = strategy_search.fair_comparisons(table)
    gain = lambda column: fair["RealPosition"] - fair[column]
    gains = {
        "Best fixed plan\n(team's tyres only)": gain("SameTyresPosition"),
        "AI strategist\n(team's tyres only)": gain("AgentSameTyresPosition"),
        "Best fixed plan\n(any tyres)": gain("FixedPosition"),
        "AI strategist\n(any tyres)": gain("AgentPosition"),
    }
    # Colour follows who chose the strategy: orange brute force, blue the AI.
    colours = [style.ORANGE, style.BLUE, style.ORANGE, style.BLUE]
    figure, (left, right) = style.new_figure(2, width=12, height=4.6, width_ratios=[1, 1.2])
    names = list(gains)
    means = [gain.mean() for gain in gains.values()]
    left.barh(names, means, color=colours, height=0.55)
    for name, mean in zip(names, means):
        left.text(mean, name, f"  {mean:+.2f}", va="center", color=style.INK)
    left.axvline(0, color=style.BASELINE, linewidth=1)
    left.invert_yaxis()
    left.set_xlim(min(0, min(means)) - 0.15, max(means) * 1.25 + 0.1)
    left.set_xlabel("Average places gained over the real team's strategy")
    style.tidy(left, f"Places gained in simulation, {season}")

    agent_gain = gains["AI strategist\n(team's tyres only)"].to_numpy()
    right.hist(agent_gain, bins=np.arange(-4.25, 6.75, 0.5), color=style.BLUE,
               edgecolor=style.SURFACE, linewidth=1.5)
    right.axvline(0, color=style.BASELINE, linewidth=1)
    right.set_xlabel("Places gained by the AI strategist over the real team (per driver race)")
    right.set_ylabel("Driver races")
    style.tidy(right, "The spread behind the average: AI strategist, team's tyres only", "y")
    style.save(figure, file)


def print_report(table: pl.DataFrame, race_set: RaceSet, stats: dict):
    with pl.Config(tbl_width_chars=170, fmt_str_lengths=60, tbl_rows=20):
        print("\n1. Against what the teams really did (simulated, real race conditions)."
              "\n   Positive means better than the team:")
        print(comparison(table))
        print("\n2. The same, split by whether the race had a safety car or VSC:")
        print(pl.concat([
            comparison(table.filter(pl.col("RaceHadCaution")), "with: "),
            comparison(table.filter(~pl.col("RaceHadCaution")), "without: "),
        ]))
        print("\n3. Exploit checks: does the strategist behave like a real team?")
        print(exploit_checks(table, race_set, stats))
        print("\n4. Examples (first simulation of each):")
        print(table.sample(8, seed=1).select(
            "Race", "Driver", "RealStrategy", "FixedStrategy", "AgentExample",
            pl.col("RealPosition", "FixedPosition", "AgentPosition").round(1)))


def main():
    lap_model = LapModel()
    stats = simulator.load_stats()
    race_set = RaceSet(simulator.load_races(["validation"]), lap_model, stats)
    env = StrategyEnv(race_set, lap_model, stats, real_events=True)
    strategist = load_strategist()
    benchmark = pl.read_parquet(RESULTS_FOLDER / "benchmark_2024.parquet")
    print(f"Evaluating the strategist on {len(race_set.races)} dry 2024 races:")
    table = evaluate(env, strategist, benchmark)
    table.write_parquet(RESULTS_FOLDER / "strategist_2024.parquet")
    print_report(table, race_set, stats)
    save_chart(table, "2024 validation races", Path("docs/strategist_2024.png"))


if __name__ == "__main__":
    main()
