# test_improvements.py
#
# What it does: checks the improvement backtest (backtest_improvements.py)
#   and the model parts it uses, on small made-up seasons:
#   - no prediction can see its own race or any later race (the key test),
#   - the qualifying second step learns only from earlier seasons,
#   - the experience curve finds drivers who improve with experience.
# What it reads: nothing (no downloaded data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import random

import polars as pl
import pytest
import torch

import backtest_improvements as improvements
import model
from test_model import NO_PACE, pair, pairs_table


def made_up_league(seasons=(2010, 2011), races=4) -> pl.DataFrame:
    """Two teams, the same four drivers, a few seasons of qualifying gaps and results."""
    random.seed(11)
    rows = []
    for year in seasons:
        for race in range(1, races + 1):
            rows.append(pair(year, race, "ace", "mid", quali_gap=-0.4 + random.gauss(0, 0.2), a_ahead_race=race % 3 != 0))
            rows.append(pair(year, race, "fast", "slow", quali_gap=-0.3 + random.gauss(0, 0.2), a_ahead_race=race % 2 == 0))
    return pairs_table(rows)


def test_the_data_window_only_lets_in_earlier_races():
    races = pl.DataFrame({"Year": [2010, 2011, 2011, 2011, 2012], "Round": [9, 1, 2, 3, 1]})
    seen = races.filter(model.data_window(2011, before_race=(2011, 2)))
    assert seen.rows() == [(2010, 9), (2011, 1)]  # not round 2 itself, not round 3, not 2012


def test_no_in_season_prediction_can_see_its_own_race_or_later():
    pairs = made_up_league()
    predicted = improvements.predict_season(pairs, NO_PACE, 2011, "in season", model.Options())
    assert predicted.height == pairs.filter(pl.col("Year") == 2011).height
    seen = predicted["SeenYear"] * 100 + predicted["SeenRound"]
    race = predicted["Year"] * 100 + predicted["Round"]
    assert (seen < race).all()
    # Round 1 is predicted from the previous season, round 3 from rounds 1 and 2.
    by_round = dict(predicted.group_by("Round").agg(pl.col("SeenYear").first(), pl.col("SeenRound").first())
                    .select("Round", pl.struct("SeenYear", "SeenRound")).iter_rows())
    assert by_round[1] == {"SeenYear": 2010, "SeenRound": 4}
    assert by_round[3] == {"SeenYear": 2011, "SeenRound": 2}


def test_the_leakage_check_stops_a_prediction_that_saw_its_own_race():
    bad = pl.DataFrame({"Variant": ["current"], "Year": [2015], "Round": [7], "SeenYear": [2015], "SeenRound": [7]})
    with pytest.raises(RuntimeError):
        improvements.check_no_future(bad)
    fine = bad.with_columns(pl.lit(6).alias("SeenRound"))
    improvements.check_no_future(fine)  # does not raise


def test_the_hindsight_ceiling_is_allowed_to_see_its_own_season():
    pairs = made_up_league()
    predicted = improvements.predict_season(pairs, NO_PACE, 2011, "hindsight", model.Options())
    assert (predicted["SeenYear"] == 2011).all()


def test_the_second_step_learns_only_from_earlier_seasons(monkeypatch):
    seasons_used = []
    real_fit = improvements.fit_weights

    def recording_fit(x, y):
        seasons_used.append(len(y))
        return real_fit(x, y)

    rows = pl.DataFrame({
        "Variant": ["current"] * 6, "Year": [2009, 2009, 2010, 2010, 2011, 2011],
        "AAheadRace": [True, False, True, True, False, True], "AAheadQuali": [True] * 6,
        "Gap": [0.2, -0.1, 0.3, 0.1, -0.2, 0.4], "GapVariance": [0.01] * 6, "RaceSlope": [3.0] * 6,
        "QualiSlope": [3.0] * 6, "QualiGapPct": [-0.2] * 6,
    })
    monkeypatch.setattr(improvements, "FIRST_REPORTED", 2010)
    monkeypatch.setattr(improvements, "LAST_REPORTED", 2011)
    monkeypatch.setattr(improvements, "fit_weights", recording_fit)
    improvements.second_step(rows, "current", "AAheadRace", use_quali=True)
    # For 2010 it may learn from 2009 only (2 rows); for 2011 from 2009 and 2010 (4 rows).
    assert seasons_used == [2, 4]


def test_the_qualifying_residual_is_how_much_better_than_expected_a_driver_qualified():
    rows = pl.DataFrame({"Gap": [0.2], "GapVariance": [0.0], "RaceSlope": [3.0],
                         "QualiGapPct": [-0.5], "AAheadQuali": [True]})
    x = improvements.features(rows, "RaceSlope", use_quali=True)
    # Expected 0.2% quicker, was 0.5% quicker: 0.3% better than expected.
    assert abs(x[0, 1].item() - 0.3) < 1e-9
    assert x[0, 2].item() == 0.0  # a lap time existed, so "who was ahead" is not used twice


def test_the_experience_curve_finds_drivers_who_improve_with_experience():
    # Six drivers who all start in 2000, each season facing a new rookie. Their
    # advantage over the rookie grows by 0.08% for every year they have raced.
    random.seed(4)
    rows = []
    for year in range(2000, 2012):
        for n in range(6):
            advantage = 0.08 * (year - 2000)
            for race in range(1, 9):
                rows.append(pair(year, race, f"d{n}", f"rookie{year}_{n}",
                                 quali_gap=-advantage + random.gauss(0, 0.1), a_ahead_race=race % 2 == 0))
    fit = model.fit_model(pairs_table(rows), NO_PACE, 2011, options=model.Options(experience=True), with_uncertainty=False)
    curve = fit.experience_curve
    assert curve[5] > curve[0] + 0.2  # the true gain after five years is 0.4
    assert curve[5] > curve[2] > curve[0]


def test_expected_gap_matches_the_full_skill_gap():
    pairs = made_up_league()
    fit = model.fit_model(pairs, NO_PACE, 2011)
    assert abs(model.expected_gap(fit, "ace", "mid", 2012) - model.skill_gap(fit, "ace", "mid", 2012)[0]) < 1e-12


def test_warm_start_gives_the_same_answer_as_a_cold_start():
    pairs = made_up_league()
    earlier = model.fit_model(pairs, NO_PACE, 2011, before_race=(2011, 3), with_uncertainty=False)
    warm = model.fit_model(pairs, NO_PACE, 2011, before_race=(2011, 4), start=earlier, with_uncertainty=False)
    cold = model.fit_model(pairs, NO_PACE, 2011, before_race=(2011, 4), with_uncertainty=False)
    assert torch.allclose(warm.skill, cold.skill, atol=1e-4)


def test_an_effective_skill_includes_the_experience_curve_after_year_one():
    pairs = made_up_league(seasons=(2010, 2011, 2012))
    fit = model.fit_model(pairs, NO_PACE, 2012, with_uncertainty=False)  # experience curve on by default
    links = fit.data["links"]
    rookie_year = fit.index[("ace", 2010)]
    third_year = fit.index[("ace", 2012)]
    assert model.effective_positions(fit.layout, links, rookie_year) == [rookie_year]  # year 0 is the reference
    curve_start = fit.layout["experience"][0]
    assert model.effective_positions(fit.layout, links, third_year) == [third_year, curve_start + 1]
