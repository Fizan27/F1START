# download_data.py
#
# What it does: downloads every race of the chosen seasons with the FastF1
#   library and saves the parts this project needs.
# What it reads: the public F1 timing data, through FastF1 (internet needed).
#   FastF1 keeps its own copy in data/fastf1_cache so nothing downloads twice.
# What it produces, in data/raw/, three files per race (for example 2024_05):
#     2024_05_laps.parquet     one row per driver per lap: lap time, sectors,
#                              tyre compound and age, stint, pit times,
#                              track status (safety car / VSC), position
#     2024_05_weather.parquet  about one row per minute: air and track
#                              temperature, rain, wind
#     2024_05_results.parquet  one row per driver: grid and finishing position
#   plus one file for all races together:
#     qualifying.parquet       one row per driver per race: best qualifying lap
# Which files use it: clean_data.py reads data/raw/ (the next step).
#
# Run:  .venv\Scripts\python.exe download_data.py
# Safe to stop and run again: races already saved are skipped.
# Expect about 3 hours for all four seasons: the data service allows 500
# requests per hour, so the script pauses by itself when it reaches the limit.

import logging
import time
from pathlib import Path

import fastf1
import pandas as pd
import polars as pl
from fastf1.ergast import Ergast
from fastf1.exceptions import RateLimitExceededError

SEASONS = [2022, 2023, 2024, 2025]  # why these: see docs/DECISIONS.md, number 4
RATE_LIMIT_WAIT_MINUTES = 10
RAW_FOLDER = Path("data/raw")
CACHE_FOLDER = Path("data/fastf1_cache")
QUALIFYING_FILE = RAW_FOLDER / "qualifying.parquet"

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
    """Download one race. Telemetry is skipped: it is huge and not needed."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def save_race(session, year: int, round_number: int) -> int:
    """Save the three tables for one race. Returns how many laps were saved."""
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
    weather = to_polars(session.weather_data).with_columns(labels)
    results = to_polars(session.results[RESULT_COLUMNS]).with_columns(labels)

    laps.write_parquet(RAW_FOLDER / f"{stem}_laps.parquet")
    weather.write_parquet(RAW_FOLDER / f"{stem}_weather.parquet")
    results.write_parquet(RAW_FOLDER / f"{stem}_results.parquet")
    return laps.height


def rounds_in_season(year: int) -> list[int]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return [int(number) for number in schedule["RoundNumber"]]


def qualifying_for_season(year: int) -> pl.DataFrame:
    """Each driver's best qualifying lap (seconds) for every race of a season.

    Qualifying happens the day before the race, so these times are known
    before lap 1. That makes them a safe yardstick for race pace: they cannot
    leak anything about how the race itself went.
    """
    tables = []
    page = Ergast().get_qualifying_results(season=year, limit=100)
    while page is not None:
        for (_, race), drivers in zip(page.description.iterrows(), page.content):
            # A driver's best lap can come from any of the three parts (Q1,
            # Q2, Q3); slower drivers are knocked out before the later parts.
            parts = drivers.reindex(columns=["Q1", "Q2", "Q3"])
            seconds = parts.apply(lambda part: pd.to_timedelta(part, errors="coerce").dt.total_seconds())
            tables.append(pl.DataFrame({
                "Year": year,
                "Round": int(race["round"]),
                "Driver": drivers["driverCode"].to_list(),
                "BestQualiTime": seconds.min(axis=1).to_list(),
            }, schema_overrides={"BestQualiTime": pl.Float64}))
        try:
            page = page.get_next_result_page()  # results arrive 100 rows at a time
        except ValueError:
            page = None  # no more pages
    return pl.concat(tables)


def save_qualifying():
    """Save data/raw/qualifying.parquet for all seasons (skipped if present)."""
    if QUALIFYING_FILE.exists():
        print("qualifying  already saved, skipping")
        return
    seasons = [patiently(qualifying_for_season, year) for year in SEASONS]
    qualifying = pl.concat(seasons)
    qualifying.write_parquet(QUALIFYING_FILE)
    print(f"qualifying  {qualifying.height} driver results saved")


def patiently(function, *arguments):
    """Call a download function; if the hourly limit is hit, wait and retry.

    The data service allows 500 requests per hour (about 33 races). Waiting
    is the polite fix: the limit exists to keep a free service usable.
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

    save_qualifying()

    failed = []
    for year in SEASONS:
        for round_number in patiently(rounds_in_season, year):
            stem = race_name(year, round_number)
            if already_saved(year, round_number):
                print(f"{stem}  already saved, skipping")
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
