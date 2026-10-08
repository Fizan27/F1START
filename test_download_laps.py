# test_download_laps.py
#
# What it does: checks the small helper functions in download_laps.py, without
#   touching the internet.
# What it reads: nothing.
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import pandas as pd
import polars as pl

from download_laps import race_name, to_polars


def test_durations_become_seconds():
    table = pd.DataFrame({"LapTime": [pd.Timedelta(minutes=1, seconds=32.5)]})
    assert to_polars(table)["LapTime"].to_list() == [92.5]


def test_missing_duration_stays_missing():
    # A lap with no recorded time must stay empty, not silently become 0.
    table = pd.DataFrame({"LapTime": [pd.Timedelta(seconds=90), pd.NaT]})
    assert to_polars(table)["LapTime"].to_list() == [90.0, None]


def test_mixed_text_column_becomes_text():
    table = pd.DataFrame({"Compound": ["SOFT", None]})
    assert to_polars(table)["Compound"].dtype == pl.String


def test_race_name_pads_round_number():
    # Padding keeps files in race order when sorted by name.
    assert race_name(2024, 5) == "2024_05"
