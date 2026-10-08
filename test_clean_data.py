# test_clean_data.py
#
# What it does: checks each cleaning step in clean_data.py on tiny hand-made
#   tables where the right answer is obvious.
# What it reads: nothing (no real data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import polars as pl

import clean_data


def one_race(**columns) -> pl.DataFrame:
    """A small table of laps that all belong to the same race."""
    height = len(next(iter(columns.values())))
    return pl.DataFrame({"Year": [2024] * height, "Round": [1] * height, **columns})


def test_gap_ahead_is_measured_at_the_start_of_the_lap():
    # AAA leads. BBB crosses the line 2.0s behind after lap 1, 3.5s after lap 2.
    laps = one_race(
        Driver=["AAA", "AAA", "BBB", "BBB"],
        LapNumber=[1, 2, 1, 2],
        Time=[100.0, 190.0, 102.0, 193.5],
    )
    result = clean_data.add_gap_ahead(laps).sort("Driver", "LapNumber")
    bbb = result.filter(pl.col("Driver") == "BBB")["GapAhead"].to_list()
    # Lap 1 has no previous lap, so no gap. Lap 2 starts 2.0s behind, and must
    # NOT show 3.5s: that is the gap at the end of lap 2, which leaks the answer.
    assert bbb == [None, 2.0]


def test_gap_ahead_does_not_cross_between_races():
    laps = pl.DataFrame({
        "Year": [2024, 2024], "Round": [1, 2], "Driver": ["AAA", "AAA"],
        "LapNumber": [1, 1], "Time": [100.0, 105.0],
    })
    assert clean_data.add_gap_ahead(laps)["GapAhead"].to_list() == [None, None]


def test_track_flags():
    laps = one_race(
        TrackStatus=["1", "124", "67", "5"],
        PitInTime=[None, 50.0, None, None],
        PitOutTime=[None, None, 80.0, None],
    )
    result = clean_data.add_track_flags(laps)
    assert result["IsGreen"].to_list() == [True, False, False, False]
    assert result["IsSafetyCar"].to_list() == [False, True, False, False]
    assert result["IsVSC"].to_list() == [False, False, True, False]
    assert result["IsRedFlag"].to_list() == [False, False, False, True]
    assert result["IsPitInLap"].to_list() == [False, True, False, False]
    assert result["IsPitOutLap"].to_list() == [False, False, True, False]


def test_laps_remaining_counts_down_to_zero():
    laps = one_race(LapNumber=[1, 2, 3])
    result = clean_data.add_race_progress(laps)
    assert result["TotalLaps"].to_list() == [3, 3, 3]
    assert result["LapsRemaining"].to_list() == [2, 1, 0]


def test_split_keeps_seasons_apart():
    laps = pl.DataFrame({"Year": [2022, 2023, 2024, 2025]})
    result = clean_data.add_split(laps)
    assert result["Split"].to_list() == ["train", "train", "validation", "test"]


def test_weather_uses_the_latest_earlier_reading():
    laps = one_race(Time=[130.0])
    weather = one_race(
        Time=[60.0, 120.0, 180.0],
        AirTemp=[20.0, 21.0, 22.0], TrackTemp=[30.0, 31.0, 32.0],
        Humidity=[50.0, 50.0, 50.0], WindSpeed=[1.0, 1.0, 1.0],
        Rainfall=[False, False, True],
    )
    result = clean_data.add_weather(laps, weather)
    assert result["AirTemp"].to_list() == [21.0]
    assert result["Rainfall"].to_list() == [False]


def test_qualifying_pace():
    # Pole is 100s. BBB qualified 2% slower. CCC's lap was ruined (20% slower)
    # so CCC borrows team-mate BBB's gap. DDD has no lap and no team-mate.
    qualifying = one_race(
        Driver=["AAA", "BBB", "CCC", "DDD"],
        BestQualiTime=[100.0, 102.0, 120.0, float("nan")],
    )
    laps = one_race(
        Driver=["AAA", "BBB", "CCC", "DDD"],
        Team=["Red", "Blue", "Blue", "Green"],
        LapTime=[103.0, 105.0, 105.0, 105.0],
    )
    result = clean_data.add_qualifying_pace(laps, qualifying).sort("Driver")
    assert result["PoleTime"].to_list() == [100.0] * 4
    assert [round(gap, 6) if gap is not None else None
            for gap in result["QualiGapPct"].to_list()] == [0.0, 2.0, 2.0, None]
    assert [round(pct, 6) for pct in result["LapTimePct"].to_list()] == [3.0, 5.0, 5.0, 5.0]


def test_clean_lap_flag_rejects_non_racing_laps():
    laps = one_race(
        LapTime=[90.0, 90.0, 90.0, 90.0, None, 90.0],
        IsAccurate=[True] * 6,
        IsGreen=[True, False, True, True, True, True],
        IsPitInLap=[False, False, True, False, False, False],
        IsPitOutLap=[False] * 6,
        LapNumber=[5, 5, 5, 1, 5, 5],
        Compound=["SOFT", "SOFT", "SOFT", "SOFT", "SOFT", "INTERMEDIATE"],
        TyreLife=[3] * 6,
    )
    result = clean_data.add_clean_lap_flag(laps)
    # Only the first lap is clean. The others fail for: safety car, pit stop,
    # first lap of the race, missing lap time, and wet-weather tyre.
    assert result["IsCleanLap"].to_list() == [True, False, False, False, False, False]
