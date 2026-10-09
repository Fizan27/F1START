# test_model.py
#
# What it does: checks model.py on tiny made-up seasons where the right answer
#   is known, because the "drivers" were invented with known skills.
# What it reads: nothing (no downloaded data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import random

import polars as pl

import model


def pair(year, race, a, b, quali_gap=None, a_ahead_race=None, a_ahead_quali=None, cars=2) -> dict:
    """One teammate comparison in a made-up race."""
    return {"Year": year, "Round": race, "DriverA": a, "DriverB": b, "TeamCars": cars,
            "AAheadRace": a_ahead_race, "AAheadQuali": a_ahead_quali, "QualiGapPct": quali_gap}


def pairs_table(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={
        "AAheadRace": pl.Boolean, "AAheadQuali": pl.Boolean, "QualiGapPct": pl.Float64})


NO_PACE = pl.DataFrame(schema={"Year": pl.Int64, "Round": pl.Int64, "DriverId": pl.String,
                               "PacePct": pl.Float64, "CleanLaps": pl.Int64})


def made_up_seasons() -> pl.DataFrame:
    """ace is 0.5% quicker than mid, who is 0.5% quicker than slow. new has 2 races."""
    random.seed(1)
    rows = []
    for race in range(1, 21):
        noise = random.gauss(0, 0.2)
        rows.append(pair(2010, race, "ace", "mid", quali_gap=-0.5 + noise, a_ahead_race=True))
        rows.append(pair(2011, race, "mid", "slow", quali_gap=-0.5 + noise, a_ahead_race=True))
    for race in (1, 2):
        rows.append(pair(2012, race, "mid", "new", quali_gap=-0.5, a_ahead_race=True))
    return pairs_table(rows)


def test_a_lap_time_gap_replaces_who_was_ahead_so_a_lap_is_not_counted_twice():
    pairs = pairs_table([
        pair(2010, 1, "aaa", "bbb", quali_gap=-0.4, a_ahead_quali=True),
        pair(2010, 2, "aaa", "bbb", quali_gap=-9.0, a_ahead_quali=True),  # a ruined lap
        pair(1980, 1, "ccc", "ddd", a_ahead_quali=False),  # grid only
    ])
    table = model.build_comparisons(pairs, NO_PACE, 2010).sort("Year", "Round")
    assert table["quali_gap"].to_list() == [None, -0.4, None]
    assert table["quali_ahead"].to_list() == [0.0, None, 1.0]


def test_seasons_after_the_cut_off_are_not_used():
    pairs = pairs_table([pair(2023, 1, "aaa", "bbb", quali_gap=-0.4),
                         pair(2024, 1, "aaa", "bbb", quali_gap=-0.4)])
    assert model.build_comparisons(pairs, NO_PACE, 2023)["Year"].to_list() == [2023]


def test_a_team_with_many_cars_does_not_count_for_more():
    pairs = pairs_table([pair(1956, 1, "aaa", "bbb", a_ahead_race=True, cars=5)])
    # Each driver is in 4 pairs that race, so each pair counts a quarter.
    assert model.build_comparisons(pairs, NO_PACE, 1956)["weight"].to_list() == [0.25]


def test_career_links_follow_each_driver_through_time():
    index = {("aaa", 2010): 0, ("aaa", 2012): 1, ("bbb", 1995): 2}
    links = model.career_links(index)
    assert links["first"].tolist() == [0, 2]
    assert links["first_era"].tolist() == [1, 0]  # decades are 1990, 2010
    assert (links["earlier"].tolist(), links["later"].tolist()) == ([0], [1])
    assert links["years"].tolist() == [2.0]  # aaa sat out 2011


def test_the_model_recovers_who_is_faster_and_by_roughly_how_much():
    fit = model.fit_model(made_up_seasons(), NO_PACE, 2012)
    gap_top, _ = model.skill_gap(fit, "ace", "mid", 2010)
    gap_chain, _ = model.skill_gap(fit, "ace", "slow", 2011)
    # The true gaps are 0.5 and 1.0. The prior pulls estimates towards zero,
    # so they come out a little smaller, never larger.
    assert 0.3 < gap_top < 0.55
    # ace and slow never raced each other: they are linked only through mid.
    assert gap_chain > gap_top


def test_a_driver_with_two_races_is_more_uncertain_than_one_with_twenty():
    fit = model.fit_model(made_up_seasons(), NO_PACE, 2012)
    # Both are measured against the same teammate, mid.
    _, twenty_races = model.skill_gap(fit, "slow", "mid", 2011)
    _, two_races = model.skill_gap(fit, "new", "mid", 2012)
    assert two_races > 2 * twenty_races


def test_an_unknown_driver_is_assumed_average_and_very_uncertain():
    fit = model.fit_model(made_up_seasons(), NO_PACE, 2012)
    gap, variance = model.skill_gap(fit, "never_seen", "also_never_seen", 2013)
    assert gap == 0.0
    assert variance == 2 * fit.skill_spread ** 2


def test_uncertainty_grows_with_every_year_since_the_last_race():
    fit = model.fit_model(made_up_seasons(), NO_PACE, 2012)
    _, soon = model.skill_gap(fit, "ace", "slow", 2012)
    _, later = model.skill_gap(fit, "ace", "slow", 2016)
    assert later > soon


def test_chance_ahead_is_fair_and_shrinks_towards_half_when_unsure():
    assert model.chance_ahead(0.0, 0.1, 3.0) == 0.5
    sure = model.chance_ahead(0.4, 0.0, 3.0)
    unsure = model.chance_ahead(0.4, 0.5, 3.0)
    assert 0.5 < unsure < sure
    # Swapping the two drivers must give the mirror image.
    assert abs(model.chance_ahead(-0.4, 0.5, 3.0) - (1 - unsure)) < 1e-12


def test_car_strength_takes_the_driver_out():
    # Two teams. Both have one strong (+0.4) and one average (0) driver, and
    # in both the strong driver scores 0.2 higher: driver weight 0.5.
    fit = model.Fit(index={("a1", 2020): 0, ("a2", 2020): 1, ("b1", 2020): 2, ("b2", 2020): 3},
                    skill=model.torch.tensor([0.4, 0.0, 0.4, 0.0]), era_level={},
                    covariance=model.torch.eye(4), scales={}, skill_spread=0.3, season_drift=0.1)
    results = pl.DataFrame({
        "Year": [2020] * 4, "TeamId": ["fast", "fast", "slow", "slow"],
        "DriverId": ["a1", "a2", "b1", "b2"], "Outcome": ["finished"] * 4,
        "PositionScore": [0.9, 0.7, 0.5, 0.3], "IsIndy500": [False] * 4,
    })
    cars, driver_weight = model.car_strengths(results, fit, 2020)
    assert round(driver_weight, 6) == 0.5
    strengths = dict(cars.select("TeamId", "CarStrength").iter_rows())
    assert round(strengths["fast"], 6) == 0.7 and round(strengths["slow"], 6) == 0.3
