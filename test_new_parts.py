# test_new_parts.py
#
# What it does: checks the pieces added in the second round of work on small
#   made-up examples: wet race tagging, the backtest's scoring and
#   temperature fitting, the team orders flag, the wet adjustment, the chain
#   of teammates to today's grid, and the peak window.
# What it reads: nothing (no downloaded data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import math
import random

import polars as pl
import pytest

import backtest
import explain_ratings
import fit_ratings
import model
import wet_races
from test_model import NO_PACE, pair, pairs_table


# --- wet_races.py --------------------------------------------------------------

def test_a_race_counts_as_wet_when_a_tenth_of_its_laps_were_on_wet_tyres():
    laps = pl.DataFrame({
        "Year": [2020] * 20 + [2021] * 20, "Round": [1] * 20 + [2] * 20,
        "Compound": ["INTERMEDIATE"] * 2 + ["SOFT"] * 18 + ["WET"] * 1 + ["HARD"] * 19,
    })
    wet = wet_races.measured_wet_races(laps)
    assert wet.select("Year", "Round").rows() == [(2020, 1)]  # 10% is wet, 5% is not


def test_every_hand_listed_wet_race_must_match_a_real_race():
    races = pl.DataFrame({"Year": [1993], "Round": [3], "RaceName": ["European Grand Prix"]})
    saved = wet_races.WET_RACES_BEFORE_2018
    try:
        wet_races.WET_RACES_BEFORE_2018 = [(1993, "European Grand Prix")]
        assert wet_races.listed_wet_races(races).rows() == [(1993, 3, "list")]
        wet_races.WET_RACES_BEFORE_2018 = [(1993, "Made Up Grand Prix")]
        with pytest.raises(ValueError):
            wet_races.listed_wet_races(races)
    finally:
        wet_races.WET_RACES_BEFORE_2018 = saved


# --- backtest.py scoring ---------------------------------------------------------

def test_scores_reward_honest_probabilities():
    ahead = [True, False, True, True]
    assert backtest.accuracy([0.9, 0.2, 0.6, 0.5], ahead) == 0.875  # the 50/50 call counts half
    assert backtest.brier([0.5] * 4, ahead) == 0.25
    assert abs(backtest.log_loss([0.5] * 4, ahead) - math.log(2)) < 1e-12
    # Confidently wrong is punished far more than unsure.
    assert backtest.log_loss([0.99], [False]) > backtest.log_loss([0.6], [False])


def test_the_temperature_softens_an_overconfident_model():
    # The "model" says 90% every time, but the favourite only wins 60%.
    random.seed(3)
    rows = pl.DataFrame({
        "Variant": ["base"] * 400, "Year": [2015] * 400, "AAheadRace": [random.random() < 0.6 for _ in range(400)],
        "Gap": [1.0] * 400, "GapVariance": [0.0] * 400, "RaceSlope": [math.log(9)] * 400,
    })
    assert backtest.fit_temperature(rows, "base") < 0.5


# --- model.py options ------------------------------------------------------------

def test_team_orders_remove_only_race_results_that_include_the_number_one():
    pairs = pairs_table([
        {**pair(1995, 1, "herbert", "michael_schumacher", quali_gap=0.8, a_ahead_race=False), "TeamId": "benetton"},
        {**pair(1995, 1, "alesi", "berger", quali_gap=0.1, a_ahead_race=True), "TeamId": "ferrari"},
    ])
    with_orders = model.build_comparisons(pairs, NO_PACE, 1995, options=model.Options(number_one=True)).sort("DriverA")
    assert with_orders["race_ahead"].to_list() == [1.0, None]  # Benetton's race result is left out
    assert with_orders["old_quali_gap"].to_list() == [0.1, 0.8]  # qualifying still counts (pre-2006 column)


def test_the_wet_adjustment_finds_a_driver_who_is_better_in_the_rain():
    # rainman and steady are equal in the dry, but rainman wins every wet race.
    random.seed(5)
    rows, wet = [], []
    for race in range(1, 41):
        is_wet = race % 4 == 0
        rows.append(pair(2010, race, "rainman", "steady", quali_gap=random.gauss(0, 0.2),
                         a_ahead_race=True if is_wet else race % 2 == 0))
        if is_wet:
            wet.append((2010, race))
    wet_table = pl.DataFrame(wet, schema=["Year", "Round"], orient="row")
    fit = model.fit_model(pairs_table(rows), NO_PACE, 2010, wet_races=wet_table, options=model.Options(wet=True))
    adjustment, sd, races = fit.wet["rainman"]
    assert adjustment > fit.wet["steady"][0]
    assert races == 10 and sd > 0


# --- explain_ratings.py ----------------------------------------------------------

def test_the_chain_to_today_prefers_fewer_links_then_longer_partnerships():
    pairs = pl.DataFrame({
        "Year": [1990, 1990, 2000, 2000, 2000, 2025],
        "DriverA": ["old", "old", "middle", "middle", "other", "today"],
        "DriverB": ["middle", "middle", "today", "today", "today", "rookie"],
    })
    chains = explain_ratings.chains_to_today(pairs, latest=2025)
    assert [link[0] for link in chains["old"]] == ["middle", "today"]
    assert chains["today"] == []


# --- fit_ratings.py ---------------------------------------------------------------

def test_the_peak_is_the_best_run_of_three_seasons():
    assert fit_ratings.peak_window([2000, 2001, 2002, 2003, 2004], [0.1, 0.2, 0.6, 0.7, 0.5]) == (2, 5)
    assert fit_ratings.peak_window([1990, 1991], [0.3, 0.1]) == (0, 2)  # short careers use every season
