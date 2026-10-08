# final_test.py
#
# What it does: the final exam, run ONCE at the end of the project. Every
#   choice (inputs, network size, calibration, simulator rules, reward) was
#   made using 2024. This file freezes those choices, retrains everything on
#   2022 to 2024, and scores it on the 2025 season, which nothing has seen.
#   The steps, in order:
#     1. lap time network and LightGBM baseline, scored on 2025 laps
#     2. pit loss and safety car statistics, measured on 2022 to 2024
#     3. replay validation of the 2025 races
#     4. the AI strategist, retrained on 2022 to 2024 races
#     5. brute force best fixed strategies for 2025
#     6. the AI strategist against real teams and fixed strategies on 2025
#     7. the small race files the website needs
# What it reads: data/laps.parquet, and models/lap_time_network.pt for the
#   two calibration numbers that were fixed on 2024.
# What it produces: models/final/ (network, statistics, strategist),
#   results/benchmark_2025.parquet, results/strategist_2025.parquet,
#   docs/replay_2025.png, docs/strategist_2025.png, app_data/races.pkl,
#   and results/final_test_done.txt as a marker that it has been run.
# Which files use it: none. app.py uses the files it produces.
#
# Run:  .venv\Scripts\python.exe final_test.py
# It refuses to run a second time (running a test repeatedly and keeping
# the best result would make it worthless) unless given --again, which is
# only for re-creating files after a crash, not for trying new ideas.

import pickle
import sys
from pathlib import Path

import polars as pl
import torch

import evaluate_strategist
import race_stats
import replay_validation
import simulator
import strategy_search
import train_agent
import train_baseline as baseline
import train_network
from simulator import LapModel, RaceSet
from strategy_env import StrategyEnv

FINAL = Path("models/final")
MARKER = Path("results/final_test_done.txt")
APP_RACES = Path("app_data/races.pkl")
SEASONS_BEFORE_TEST = ["train", "validation"]  # 2022 to 2024


def heading(text: str):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def lap_models(retrain: bool = True):
    """Step 1: retrain both lap time models on 2022 to 2024, score on 2025.

    retrain=False scores the network already saved in models/final/ again,
    without changing it (used once, to reprint a table: DECISIONS.md 32).
    """
    heading("1. Lap time models: trained on 2022 to 2024, scored on 2025")
    laps = baseline.load_clean_laps(include_test=True)
    train = laps.filter(pl.col("Split").is_in(SEASONS_BEFORE_TEST))
    test = laps.filter(pl.col("Split") == "test")
    # The two calibration numbers stay exactly as they were fixed on 2024.
    frozen = torch.load(train_network.MODEL_FILE, weights_only=False)
    factor, driver_spread = frozen["spread_factor"], frozen["driver_offset_spread"]
    file = FINAL / "lap_time_network.pt"
    if retrain:
        model, scaling = train_network.fit_network(baseline.training_laps(train), test)
    else:
        saved = torch.load(file, weights_only=False)
        scaling = saved["scaling"]
        model = train_network.LapTimeNetwork(len(scaling["circuits"])).to(train_network.DEVICE)
        model.load_state_dict(saved["weights"])
    centre, spread = train_network.predict(model, train_network.to_tensors(test, scaling))
    train_network.report(train, test, centre, spread, factor, "2025 (test)")
    if retrain:
        train_network.save_model(model, scaling, factor, driver_spread, file)


def statistics() -> dict:
    """Step 2: pit loss and safety car statistics from 2022 to 2024."""
    heading("2. Race statistics measured on 2022 to 2024")
    stats = race_stats.build_stats(SEASONS_BEFORE_TEST)
    race_stats.print_summary(stats)
    race_stats.save_stats(stats, FINAL / "race_stats.json")
    return stats


def replay(test_set, lap_model, stats):
    """Step 3: replay the 2025 races with their real strategies."""
    heading(f"3. Replay validation: {len(test_set.races)} dry 2025 races, real strategies")
    rows = replay_validation.score_all(
        test_set, *replay_validation.replay(test_set, lap_model, stats))
    replay_validation.print_report(rows)
    replay_validation.save_chart(rows, "2025 test races", Path("docs/replay_2025.png"))


def strategist(lap_model, stats):
    """Step 4: retrain the AI strategist on the 2022 to 2024 races."""
    heading("4. AI strategist retrained on 2022 to 2024 races")
    train_set = RaceSet(simulator.load_races(SEASONS_BEFORE_TEST), lap_model, stats)
    agent = train_agent.train(StrategyEnv(train_set, lap_model, stats, real_events=False))
    train_agent.save_strategist(agent, FINAL / "strategist.pt")
    return agent


def strategies(test_set, lap_model, stats, agent, search: bool = True):
    """Steps 5 and 6: brute force and the AI strategist on 2025.

    search=False reuses the brute force results already saved (the search
    does not involve the strategist, so it does not change when only the
    strategist is retrained: DECISIONS.md 33).
    """
    file = strategy_search.RESULTS_FOLDER / "benchmark_2025.parquet"
    if search:
        heading("5. Brute force best fixed strategies on 2025")
        benchmark = strategy_search.benchmark(test_set, lap_model, stats)
        strategy_search.RESULTS_FOLDER.mkdir(exist_ok=True)
        benchmark.write_parquet(file)
    else:
        benchmark = pl.read_parquet(file)

    heading("6. AI strategist on 2025, against real teams and fixed strategies")
    env = StrategyEnv(test_set, lap_model, stats, real_events=True)
    table = evaluate_strategist.evaluate(env, agent, benchmark)
    table.write_parquet(strategy_search.RESULTS_FOLDER / "strategist_2025.parquet")
    evaluate_strategist.print_report(table, test_set, stats)
    evaluate_strategist.save_chart(table, "2025 test races", Path("docs/strategist_2025.png"))


def save_app_races():
    """Step 7: the website cannot hold the full lap data, so save just the
    small description of each 2024 and 2025 race that the simulator needs."""
    heading("7. Race files for the website")
    races = simulator.load_races(["validation", "test"])
    APP_RACES.parent.mkdir(exist_ok=True)
    APP_RACES.write_bytes(pickle.dumps(races))
    print(f"Saved {len(races)} races to {APP_RACES} ({APP_RACES.stat().st_size / 1e6:.1f} MB)")


def main():
    if "--lap-table" in sys.argv:
        lap_models(retrain=False)
        return
    if "--strategist-only" in sys.argv:
        # Retrain and re-evaluate only the strategist, with the final lap
        # time network and statistics untouched. Used once, for the fix
        # described in DECISIONS.md 33.
        lap_model = LapModel(FINAL / "lap_time_network.pt")
        stats = simulator.load_stats(FINAL / "race_stats.json")
        test_set = RaceSet(simulator.load_races(["test"]), lap_model, stats)
        strategies(test_set, lap_model, stats, strategist(lap_model, stats), search=False)
        return
    if MARKER.exists() and "--again" not in sys.argv:
        raise SystemExit(f"The final test has already been run ({MARKER} exists).")
    lap_models()
    stats = statistics()
    lap_model = LapModel(FINAL / "lap_time_network.pt")
    test_set = RaceSet(simulator.load_races(["test"]), lap_model, stats)
    replay(test_set, lap_model, stats)
    agent = strategist(lap_model, stats)
    strategies(test_set, lap_model, stats, agent)
    save_app_races()
    MARKER.parent.mkdir(exist_ok=True)
    MARKER.write_text("The final 2025 test was run. See README.md for the results.\n")
    print(f"\nFinal test complete. Marker written to {MARKER}.")


if __name__ == "__main__":
    main()
