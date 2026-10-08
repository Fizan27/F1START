# test_clean_laps.py
#
# What it does: checks each step in clean_laps.py on tiny hand-made tables
#   where the right answer is obvious.
# What it reads: nothing (no downloaded data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import polars as pl

import clean_laps


def one_race(**columns) -> pl.DataFrame:
    """A small table of laps that all belong to the same race."""
    height = len(next(iter(columns.values())))
    return pl.DataFrame({"Year": [2024] * height, "Round": [1] * height, **columns})


def test_clean_lap_flag_rejects_non_racing_laps():
    laps = one_race(
        LapTime=[90.0, 90.0, 90.0, 90.0, None, 90.0, 90.0],
        IsAccurate=[True] * 7,
        TrackStatus=["1", "124", "1", "1", "1", "1", "1"],
        PitInTime=[None, None, 50.0, None, None, None, None],
        PitOutTime=[None] * 7,
        LapNumber=[5, 5, 5, 1, 5, 5, 5],
        Compound=["SOFT", "SOFT", "SOFT", "SOFT", "SOFT", "INTERMEDIATE", "HYPERSOFT"],
    ).with_columns(pl.col("PitOutTime").cast(pl.Float64))
    result = clean_laps.add_clean_lap_flag(clean_laps.add_track_flags(laps))
    # The first lap is clean. The next five fail for: safety car, pit stop,
    # first lap of the race, missing lap time, and wet-weather tyre. The last
    # is clean: 2018's dry tyres had names like "HYPERSOFT".
    assert result["IsCleanLap"].to_list() == [True, False, False, False, False, False, True]


def test_pace_is_measured_against_the_same_lap_number():
    # Lap 2 is slow for everyone (heavy fuel), lap 3 is quick for everyone.
    # AAA is exactly 1% slower than the middle car on both laps.
    laps = one_race(
        Driver=["AAA", "BBB", "CCC", "DDD", "EEE"] * 2,
        LapNumber=[2] * 5 + [3] * 5,
        LapTime=[101.0, 100.0, 100.0, 99.0, 100.0] + [90.9, 90.0, 90.0, 89.0, 90.0],
        IsCleanLap=[True] * 10,
    )
    result = clean_laps.add_pace_vs_field(laps).filter(pl.col("Driver") == "AAA")
    assert [round(pct, 6) for pct in result["LapPct"].to_list()] == [1.0, 1.0]


def test_no_yardstick_when_too_few_cars_set_a_clean_lap():
    laps = one_race(Driver=["AAA", "BBB"], LapNumber=[2, 2], LapTime=[90.0, 91.0], IsCleanLap=[True, True])
    assert clean_laps.add_pace_vs_field(laps)["LapPct"].to_list() == [None, None]


def test_one_spin_does_not_ruin_the_consistency_measure():
    # Nine laps within 0.1% of each other and one spin 5% off the pace.
    laps = one_race(Driver=["AAA"] * 10, LapPct=[0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 5.0])
    row = clean_laps.summarise_per_driver(laps).row(0, named=True)
    assert row["CleanLaps"] == 10
    assert row["LapSpreadPct"] < 0.2  # an ordinary standard deviation would say 1.6
    assert row["SlowLapShare"] == 0.1


def test_driver_codes_are_matched_within_the_race_only():
    pace = pl.DataFrame({"Year": [2021, 2005], "Round": [1, 1], "Driver": ["MSC", "MSC"]})
    results = pl.DataFrame({
        "Year": [2021, 2005], "Round": [1, 1], "DriverCode": ["MSC", "MSC"],
        "DriverId": ["mick_schumacher", "michael_schumacher"], "TeamId": ["haas", "ferrari"],
        "Date": ["2021-03-28", "2005-03-06"],
    })
    matched = clean_laps.add_driver_ids(pace, results).sort("Year")
    assert matched["DriverId"].to_list() == ["michael_schumacher", "mick_schumacher"]
