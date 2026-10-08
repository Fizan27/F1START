# download_laps.py
#
# What it does: downloads the lap times of every race from 2018 onwards with
#   the FastF1 library. (2018 is the first season FastF1 has lap data for.)
# What it reads: the public F1 timing data, through FastF1 (internet needed).
#   FastF1 keeps its own copy in data/fastf1_cache so nothing downloads twice.
# What it produces, in data/raw/, two files per race (for example 2024_05):
#     2024_05_laps.parquet     one row per driver per lap: lap time, tyre,
#                              pit times, track status (safety car / VSC)
#     2024_05_results.parquet  one row per driver: grid and finishing position
#   (Races downloaded for the earlier project also have a _weather file. It is
#   not used any more, but downloaded data is never deleted.)
# Which files use it: clean_laps.py reads data/raw/.
#
# Run:  .venv\Scripts\python.exe download_laps.py
# Safe to stop and run again: races already saved are skipped.
# Expect about 2 hours for the seasons not yet downloaded: FastF1 allows itself
# 500 requests per hour, so the script pauses by itself when it reaches that.

import datetime
import logging
import time
from pathlib import Path

import fastf1
import pandas as pd
import polars as pl
from fastf1.exceptions import RateLimitExceededError

FIRST_SEASON = 2018
RATE_LIMIT_WAIT_MINUTES = 10
RAW_FOLDER = Path("data/raw")
CACHE_FOLDER = Path("data/fastf1_cache")

RESULT_COLUMNS = [
    "DriverNumber", "Abbreviation", "TeamName", "GridPosition", "Position",
    "ClassifiedPosition", "Status", "Points", "Time",
]


def to_polars(table: pd.DataFrame) -> pl.DataFrame:
    """Convert a FastF1 (pandas) table to Polars, with durations as seconds.

    FastF1 stores times like "1 minute 32.5 seconds" as duration objects. A
    plain number of seconds (92.5) is much easier to subtract, average and
    feed to a model, so every duration column is converted.
    """
    table = pd.DataFrame(table).reset_index(drop=True)
    for column in table.columns:
        if pd.api.types.is_timedelta64_dtype(table[column]):
            table[column] = table[column].dt.total_seconds()
        elif table[column].dtype == object:
            # Mixed text/empty columns confuse Parquet; make them plain text.
            table[column] = table[column].astype("string")
    return pl.from_pandas(table)


def race_name(year: int, round_number: int) -> str:
    """The file name stem for a race, for example '2024_05'."""
    return f"{year}_{round_number:02d}"


def already_saved(year: int, round_number: int) -> bool:
    # The results file is written last, so if it exists the race is complete.
    return (RAW_FOLDER / f"{race_name(year, round_number)}_results.parquet").exists()


def load_race(year: int, round_number: int):
    """Download one race. Telemetry and weather are skipped: not needed."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


def save_race(session, year: int, round_number: int) -> int:
    """Save the two tables for one race. Returns how many laps were saved."""
    stem = race_name(year, round_number)
    # Stamp every row with which race it belongs to, so the files can later
    # be stacked into one big table without losing track.
    labels = [
        pl.lit(year).alias("Year"),
        pl.lit(round_number).alias("Round"),
        pl.lit(session.event["EventName"]).alias("EventName"),
        pl.lit(session.event["Location"]).alias("Circuit"),
    ]
    laps = to_polars(session.laps).with_columns(labels)
    results = to_polars(session.results[RESULT_COLUMNS]).with_columns(labels)

    laps.write_parquet(RAW_FOLDER / f"{stem}_laps.parquet")
    results.write_parquet(RAW_FOLDER / f"{stem}_results.parquet")
    return laps.height


def rounds_already_raced(year: int, today: datetime.date) -> list[int]:
    """The round numbers of a season whose race day is in the past."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    raced = schedule[schedule["EventDate"].dt.date < today]
    return [int(number) for number in raced["RoundNumber"]]


def patiently(function, *arguments):
    """Call a download function; if the hourly limit is hit, wait and retry.

    Waiting is the polite fix: the limit exists to keep a free service usable.
    """
    while True:
        try:
            return function(*arguments)
        except RateLimitExceededError:
            print(f"  hourly download limit reached, waiting {RATE_LIMIT_WAIT_MINUTES} minutes...")
            time.sleep(RATE_LIMIT_WAIT_MINUTES * 60)


def main():
    RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_FOLDER))
    logging.getLogger("fastf1").setLevel(logging.ERROR)  # hide progress chatter

    today = datetime.date.today()
    failed = []
    for year in range(FIRST_SEASON, today.year + 1):
        for round_number in patiently(rounds_already_raced, year, today):
            stem = race_name(year, round_number)
            if already_saved(year, round_number):
                continue
            try:
                session = patiently(load_race, year, round_number)
                lap_count = save_race(session, year, round_number)
                print(f"{stem}  {session.event['EventName']:<32} {lap_count:>5} laps")
            except Exception as error:
                # One bad race must not stop the whole download. Report it at
                # the end so it can be retried by running the script again.
                print(f"{stem}  FAILED: {error}")
                failed.append(stem)

    saved = len(list(RAW_FOLDER.glob("*_results.parquet")))
    print(f"\nDone. {saved} races saved in {RAW_FOLDER}.")
    if failed:
        print(f"{len(failed)} failed (run again to retry): {', '.join(failed)}")


if __name__ == "__main__":
    main()
