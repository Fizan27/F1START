# clean_laps.py
#
# What it does: boils the lap times of every race since 2018 down to one row
#   per driver per race: how fast they were over the race and how consistent.
#   Race results only say who finished ahead. Lap times also say by how much,
#   and how steadily, which sharpens the ratings of modern drivers.
# What it reads: data/raw/*_laps.parquet (made by download_laps.py) and
#   data/results.parquet (made by clean_history.py, for driver and team names
#   that match the rest of the project).
# What it produces: data/race_pace.parquet, and a summary printed to the screen.
# Which files use it: the model in Phase 2 (pace gaps between teammates) and
#   the simulation in Phase 3 (how consistent each driver is).
#
# Run:  .venv\Scripts\python.exe clean_laps.py

from pathlib import Path

import polars as pl

RAW_FOLDER = Path("data/raw")
RESULTS_FILE = Path("data/results.parquet")
OUTPUT_FILE = Path("data/race_pace.parquet")

RACE = ["Year", "Round"]  # these two columns together identify one race

WET_COMPOUNDS = ["INTERMEDIATE", "WET"]

# A lap number needs at least this many clean laps across the field before
# its typical time can be trusted as a yardstick.
MIN_CARS_FOR_YARDSTICK = 5

# A clean lap this much slower than the driver's own typical lap (in percent,
# about 1 second) counts as a slow lap: a mistake, a lock-up, a trip wide.
SLOW_LAP_PCT = 1.0

LAP_COLUMNS = [
    "Year", "Round", "Driver", "LapNumber", "LapTime", "Compound",
    "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate",
]


def read_raw_laps() -> pl.DataFrame:
    """Stack the lap file of every race into one table."""
    files = sorted(RAW_FOLDER.glob("*_laps.parquet"))
    # "diagonal_relaxed" copes with a column being text in one race and
    # true/false in another, which happens when a race has empty values.
    laps = pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed")
    return laps.select(LAP_COLUMNS).with_columns(pl.col("LapNumber").cast(pl.Int32))


def add_track_flags(laps: pl.DataFrame) -> pl.DataFrame:
    """Turn the track status codes into simple yes/no columns.

    TrackStatus lists every status seen during the lap, one digit each:
    1 clear, 2 yellow flag, 4 safety car, 5 red flag, 6 virtual safety car,
    7 virtual safety car ending. So "124" means the lap started clear, then
    had a yellow flag, then a safety car.
    """
    return laps.with_columns(
        (pl.col("TrackStatus") == "1").alias("IsGreen"),
        pl.col("PitInTime").is_not_null().alias("IsPitInLap"),
        pl.col("PitOutTime").is_not_null().alias("IsPitOutLap"),
    )


def add_clean_lap_flag(laps: pl.DataFrame) -> pl.DataFrame:
    """Mark laps that show normal racing pace.

    Pit laps, safety car laps, the standing start and laps on wet-weather
    tyres are far slower for reasons that have nothing to do with how good the
    driver is, so they are left out of every pace measurement.
    """
    return laps.with_columns(
        (
            pl.col("LapTime").is_not_null()
            & pl.col("IsAccurate")  # FastF1's own check that the timing is sound
            & pl.col("IsGreen")
            & ~pl.col("IsPitInLap")
            & ~pl.col("IsPitOutLap")
            & (pl.col("LapNumber") > 1)
            & ~pl.col("Compound").is_in(WET_COMPOUNDS).fill_null(False)
        ).alias("IsCleanLap")
    )


def add_pace_vs_field(laps: pl.DataFrame) -> pl.DataFrame:
    """Add LapPct: how much slower (+) or faster (-) a lap was than the field, in percent.

    Each lap is compared with the middle (median) clean lap time of all cars
    ON THE SAME LAP of the same race. Every car gets lighter as fuel burns and
    the track gets faster as rubber goes down, so lap 50 is quicker than lap 5
    for everyone. Comparing within a lap number removes that, and percent
    makes a short circuit and a long one comparable.
    """
    same_lap = RACE + ["LapNumber"]
    clean_time = pl.col("LapTime").filter(pl.col("IsCleanLap"))
    yardstick = pl.when(clean_time.count().over(same_lap) >= MIN_CARS_FOR_YARDSTICK).then(
        clean_time.median().over(same_lap)
    )
    return laps.with_columns(
        pl.when(pl.col("IsCleanLap")).then((pl.col("LapTime") / yardstick - 1) * 100).alias("LapPct")
    )


def summarise_per_driver(laps: pl.DataFrame) -> pl.DataFrame:
    """One row per driver per race: pace, consistency and share of slow laps.

    PacePct: the driver's typical lap against the field. Negative is faster.
      It still contains the car, so it only means something between teammates.
    LapSpreadPct: how much the driver's laps vary around their own typical
      lap. Smaller is more consistent. It is built from the median distance
      to the median (scaled by 1.4826 so it reads like a standard deviation),
      which one spin cannot blow up the way an ordinary standard deviation would.
    SlowLapShare: the share of laps more than 1% off their own typical lap.
    """
    measured = laps.filter(pl.col("LapPct").is_not_null())
    own_typical = pl.col("LapPct").median().over(RACE + ["Driver"])
    return (
        measured.with_columns((pl.col("LapPct") - own_typical).alias("OffOwnPace"))
        .group_by(RACE + ["Driver"])
        .agg(
            pl.len().alias("CleanLaps"),
            pl.col("LapPct").median().alias("PacePct"),
            (pl.col("OffOwnPace").abs().median() * 1.4826).alias("LapSpreadPct"),
            (pl.col("OffOwnPace") > SLOW_LAP_PCT).mean().alias("SlowLapShare"),
        )
    )


def add_driver_ids(pace: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Swap FastF1's three letter driver codes for the ids used everywhere else.

    FastF1 says "HAM", the results history says "hamilton". The code is only
    matched within one race, because codes get reused over the decades.
    """
    names = results.select(
        *RACE, pl.col("DriverCode").alias("Driver"), "DriverId", "TeamId", "Date"
    ).drop_nulls("Driver")
    return pace.join(names, on=RACE + ["Driver"], how="left")


def build_race_pace(laps: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Run every step in order."""
    laps = add_track_flags(laps)
    laps = add_clean_lap_flag(laps)
    laps = add_pace_vs_field(laps)
    pace = add_driver_ids(summarise_per_driver(laps), results)
    return pace.sort(RACE + ["PacePct"])


def print_summary(pace: pl.DataFrame):
    summary = pace.group_by("Year").agg(
        pl.struct(RACE).n_unique().alias("races"),
        pl.len().alias("driver_races"),
        pl.col("CleanLaps").sum().alias("clean_laps"),
        pl.col("LapSpreadPct").median().round(3).alias("typical_spread_%"),
        (pl.col("SlowLapShare").mean() * 100).round(1).alias("slow_laps_%"),
        pl.col("DriverId").is_null().sum().alias("unmatched_drivers"),
    ).sort("Year")
    with pl.Config(tbl_rows=20):
        print(summary)


def main():
    pace = build_race_pace(read_raw_laps(), pl.read_parquet(RESULTS_FILE))
    pace.write_parquet(OUTPUT_FILE)
    print(f"Saved {pace.height:,} driver races to {OUTPUT_FILE}\n")
    print_summary(pace)


if __name__ == "__main__":
    main()
