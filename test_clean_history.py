# test_clean_history.py
#
# What it does: checks each cleaning step in clean_history.py on tiny hand-made
#   tables where the right answer is obvious.
# What it reads: nothing (no downloaded data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import polars as pl

import clean_history


def car(driver: str, position: int, status: str = "Finished", team: str = "red", **extra) -> dict:
    """One driver's result in a made-up race (2000, round 1)."""
    row = {
        "Year": 2000, "Round": 1, "Date": "2000-03-12", "CircuitId": "monza",
        "DriverId": driver, "TeamId": team, "Grid": position, "Position": position,
        "PositionText": str(position) if status == "Finished" else "R",
        "Status": status, "QualiPosition": None, "Q1": None, "Q2": None, "Q3": None,
    }
    return {**row, **extra}


def cleaned(cars: list[dict]) -> pl.DataFrame:
    """Run the outcome steps on a list of made-up results."""
    return clean_history.add_outcome(pl.DataFrame(cars, infer_schema_length=None))


def test_statuses_are_grouped():
    assert clean_history.classify_status("Finished") == "finished"
    assert clean_history.classify_status("+2 Laps") == "finished"
    assert clean_history.classify_status("Spun off") == "driver"
    assert clean_history.classify_status("Collision") == "driver"
    assert clean_history.classify_status("Engine") == "car"
    assert clean_history.classify_status("Gearbox") == "car"
    assert clean_history.classify_status("Did not qualify") == "not_started"
    # "Retired" with no reason must not be guessed as the driver's or the car's.
    assert clean_history.classify_status("Retired") == "other"
    assert clean_history.classify_status("Disqualified") == "other"


def test_lap_time_becomes_seconds():
    assert clean_history.lap_time_seconds("1:23.456") == 83.456
    assert clean_history.lap_time_seconds("58.79") == 58.79
    assert clean_history.lap_time_seconds("") is None
    assert clean_history.lap_time_seconds(None) is None


def test_withdrawn_driver_did_not_start_even_with_an_engine_status():
    results = cleaned([car("aaa", 20, status="Engine", PositionText="W")])
    assert results["Outcome"].to_list() == ["not_started"]


def test_a_driver_listed_twice_keeps_the_best_result():
    results = pl.DataFrame([car("aaa", 9, status="Engine"), car("aaa", 2), car("bbb", 1)])
    kept = clean_history.one_row_per_driver(results).sort("DriverId")
    assert kept["DriverId"].to_list() == ["aaa", "bbb"]
    assert kept["Position"].to_list() == [2, 1]


def test_results_mean_the_same_in_every_era():
    results = cleaned([car("aaa", 1), car("bbb", 2), car("ccc", 3, status="Engine"),
                       car("ddd", 4, status="Did not qualify")])
    results = clean_history.add_era_proof_results(results).sort("Position")
    # Three cars started. The one that did not qualify gets no score.
    assert results["Starters"].to_list() == [3, 3, 3, 3]
    assert results["PositionScore"].to_list() == [1.0, 0.5, 0.0, None]
    # Today's points for 1st and 2nd; nothing for the car that retired.
    assert results["ModernPoints"].to_list() == [25, 18, 0, 0]


def test_car_failure_makes_a_race_not_comparable():
    winner = {"Outcome": "finished", "Position": 1}
    broke_down = {"Outcome": "car", "Position": 15}
    crashed = {"Outcome": "driver", "Position": 16}
    assert clean_history.compare_race(winner, broke_down) is None
    assert clean_history.compare_race(broke_down, winner) is None
    # A crash is the driver's own doing, so it counts as a loss.
    assert clean_history.compare_race(winner, crashed) is True
    assert clean_history.compare_race(crashed, winner) is False


def test_qualifying_gap_uses_the_last_part_both_drivers_reached():
    a = {"QualiPosition": 3, "Q1": 91.0, "Q2": 90.0, "Q3": 89.0}
    b = {"QualiPosition": 12, "Q1": 90.5, "Q2": 90.9, "Q3": None}
    ahead, gap = clean_history.compare_qualifying(a, b)
    assert ahead is True
    # Compared in Q2 (90.0 against 90.9), not A's Q3 lap against B's best.
    assert round(gap, 3) == round((90.0 / 90.9 - 1) * 100, 3)


def test_qualifying_position_falls_back_to_the_grid():
    results = pl.DataFrame([car("aaa", 4), car("bbb", 7)])
    qualifying = pl.DataFrame({
        "Year": [2000], "Round": [1], "DriverId": ["aaa"],
        "QualiPosition": [2], "Q1": [80.0], "Q2": [79.5], "Q3": [None],
    }, schema_overrides={"Q3": pl.Float64})
    results = results.drop("QualiPosition", "Q1", "Q2", "Q3")
    joined = clean_history.add_qualifying(results, qualifying).sort("DriverId")
    assert joined["QualiPosition"].to_list() == [2, 7]
    assert joined["QualiSource"].to_list() == ["qualifying", "grid"]
    assert joined["BestQualiTime"].to_list() == [79.5, None]


def test_three_cars_make_three_pairs_and_other_teams_are_separate():
    results = cleaned([car("aaa", 1), car("bbb", 2), car("ccc", 3, status="Engine"),
                       car("zzz", 4, team="blue")])
    pairs = clean_history.build_teammate_pairs(results)
    assert pairs.select("DriverA", "DriverB").rows() == [("aaa", "bbb"), ("aaa", "ccc"), ("bbb", "ccc")]
    # Only the pair where neither car broke down is comparable.
    assert pairs["AAheadRace"].to_list() == [True, None, None]


def test_indy_500_is_left_out_of_teammate_pairs():
    results = cleaned([car("aaa", 1, Year=1955, CircuitId="indianapolis"),
                       car("bbb", 2, Year=1955, CircuitId="indianapolis")])
    assert clean_history.build_teammate_pairs(results).height == 0


def test_linked_group_follows_chains_of_teammates():
    # aaa-bbb and bbb-ccc are linked through bbb. xxx-yyy is a separate island.
    pairs = pl.DataFrame({"DriverA": ["aaa", "bbb", "xxx"], "DriverB": ["bbb", "ccc", "yyy"]})
    assert clean_history.largest_linked_group(pairs) == (3, 5)
