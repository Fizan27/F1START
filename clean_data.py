# clean_data.py
#
# What it does: turns the raw per-race files into ONE tidy table with one row
#   per driver per lap, and works out the extra columns the models need:
#   fuel proxy, gap to the car ahead, weather on that lap, safety car flags,
#   and which laps are "clean" racing laps.
# What it reads: data/raw/*_laps.parquet, *_weather.parquet, *_results.parquet
#   and qualifying.parquet (made by download_data.py).
# What it produces: data/laps.parquet, and a summary printed to the screen.
# Which files use it: every later step reads data/laps.parquet (the lap time
#   models, the pit loss and safety car statistics, and the race replay).
#
# Run:  .venv\Scripts\python.exe clean_data.py

from pathlib import Path

import polars as pl

RAW_FOLDER = Path("data/raw")
QUALIFYING_FILE = RAW_FOLDER / "qualifying.parquet"
OUTPUT_FILE = Path("data/laps.parquet")

# Qualifying gaps to pole above this (percent) are rain or a ruined lap.
# Between dry cars the whole field is normally covered by about 3 percent.
MAX_BELIEVABLE_QUALI_GAP = 5.0

RACE = ["Year", "Round"]  # these two columns together identify one race

# Which job each season has. Why: see docs/DECISIONS.md, number 4.
SPLIT_BY_YEAR = {2022: "train", 2023: "train", 2024: "validation", 2025: "test"}

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# The same circuit sometimes gets a different name in a different season.
# The model must see one name per circuit, or it treats them as two places.
CIRCUIT_ALIASES = {"Miami Gardens": "Miami"}

LAP_COLUMNS = [
    "Year", "Round", "EventName", "Circuit", "Driver", "Team", "LapNumber",
    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound",
    "TyreLife", "FreshTyre", "Stint", "PitInTime", "PitOutTime", "Time",
    "TrackStatus", "Position", "IsAccurate",
]


def read_raw(kind: str) -> pl.DataFrame:
    """Stack every race's file of one kind ('laps', 'weather' or 'results')."""
    files = sorted(RAW_FOLDER.glob(f"*_{kind}.parquet"))
    # "diagonal_relaxed" copes with a column being text in one race and
    # true/false in another, which happens when a race has empty values.
    return pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed")


def tidy_laps(laps: pl.DataFrame) -> pl.DataFrame:
    """Keep the useful columns and give them sensible types and values."""
    return laps.select(LAP_COLUMNS).with_columns(
        pl.col("LapNumber", "TyreLife", "Stint", "Position").cast(pl.Int32),
        # Missing compounds were saved as the words "None" or "nan"; make
        # them properly empty so they cannot be mistaken for a tyre type.
        pl.when(pl.col("Compound").is_in(["None", "nan", "UNKNOWN"]))
        .then(None).otherwise(pl.col("Compound")).alias("Compound"),
        pl.col("Circuit").replace(CIRCUIT_ALIASES),
    )


def add_split(laps: pl.DataFrame) -> pl.DataFrame:
    """Label each lap train / validation / test by season."""
    # Stored in the data itself so no later script can mix the seasons up.
    return laps.with_columns(
        pl.col("Year").replace_strict(SPLIT_BY_YEAR, return_dtype=pl.String).alias("Split")
    )


def add_race_progress(laps: pl.DataFrame) -> pl.DataFrame:
    """Add TotalLaps and LapsRemaining.

    LapsRemaining is the fuel proxy: cars start full and burn fuel every lap,
    so more laps remaining means a heavier, slower car. Real fuel load is not
    public, but it falls almost in a straight line with laps completed.
    """
    return laps.with_columns(
        pl.col("LapNumber").max().over(RACE).alias("TotalLaps")
    ).with_columns((pl.col("TotalLaps") - pl.col("LapNumber")).alias("LapsRemaining"))


def add_track_flags(laps: pl.DataFrame) -> pl.DataFrame:
    """Turn the track status codes into simple yes/no columns.

    TrackStatus lists every status seen during the lap, one digit each:
    1 clear, 2 yellow flag, 4 safety car, 5 red flag, 6 virtual safety car,
    7 virtual safety car ending. So "124" means the lap started clear, then
    had a yellow flag, then a safety car.
    """
    status = pl.col("TrackStatus")
    return laps.with_columns(
        (status == "1").alias("IsGreen"),
        status.str.contains("4").alias("IsSafetyCar"),
        status.str.contains("6|7").alias("IsVSC"),
        status.str.contains("5").alias("IsRedFlag"),
        pl.col("PitInTime").is_not_null().alias("IsPitInLap"),
        pl.col("PitOutTime").is_not_null().alias("IsPitOutLap"),
    )


def add_gap_ahead(laps: pl.DataFrame) -> pl.DataFrame:
    """Add GapAhead: seconds to the car in front on the road at the lap start.

    This measures traffic. It uses the car physically in front, even if that
    car is a lap down, because dirty air does not care about race position.
    It is measured at the START of the lap (the end of the previous one) on
    purpose: the gap at the end of a lap depends on that lap's own time, and
    a model input must never contain the answer it is trying to predict.
    """
    by_time = laps.sort(RACE + ["Time"]).with_columns(
        (pl.col("Time") - pl.col("Time").shift(1).over(RACE)).alias("GapAtLapEnd")
    )
    return (
        by_time.sort(RACE + ["Driver", "LapNumber"])
        .with_columns(
            pl.col("GapAtLapEnd").shift(1).over(RACE + ["Driver"]).alias("GapAhead")
        )
        .drop("GapAtLapEnd")
    )


def add_weather(laps: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Attach the latest weather reading taken before each lap ended."""
    readings = weather.select(
        RACE + ["Time", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"]
    ).sort(RACE + ["Time"])
    # Weather is logged about once a minute and laps end at any moment, so the
    # times never match exactly. join_asof picks the nearest earlier reading.
    return laps.sort(RACE + ["Time"]).join_asof(
        readings, on="Time", by=RACE, strategy="backward", check_sortedness=False
    )


def add_results(laps: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    """Attach each driver's grid and finishing position for the race."""
    per_driver = results.select(
        RACE + [
            pl.col("Abbreviation").alias("Driver"),
            pl.col("GridPosition").cast(pl.Int32),
            pl.col("Position").cast(pl.Int32).alias("FinishPosition"),
            pl.col("Status").alias("FinishStatus"),
        ]
    )
    return laps.join(per_driver, on=RACE + ["Driver"], how="left")


def add_qualifying_pace(laps: pl.DataFrame, qualifying: pl.DataFrame) -> pl.DataFrame:
    """Add PoleTime, QualiGapPct and LapTimePct (the model's target).

    PoleTime: the fastest qualifying lap of the weekend. It is the yardstick
      that removes "which circuit is this" from the lap times.
    QualiGapPct: how much slower than pole this driver qualified, in percent.
      It tells the model how fast this car and driver are this weekend.
    LapTimePct: the race lap time as percent slower than pole.
    All three use only qualifying, which is finished before the race starts,
    so nothing about the race itself leaks into the model's inputs.
    """
    best = qualifying.with_columns(pl.col("BestQualiTime").fill_nan(None))
    pole = best.group_by(RACE).agg(pl.col("BestQualiTime").min().alias("PoleTime"))
    gap = (pl.col("BestQualiTime") / pl.col("PoleTime") - 1) * 100
    gaps = best.join(pole, on=RACE).select(
        *RACE,
        "Driver",
        # A huge gap means rain or a ruined lap, not a slow car, so it says
        # nothing about race pace. Treat it as "no representative lap".
        pl.when(gap <= MAX_BELIEVABLE_QUALI_GAP).then(gap).alias("QualiGapPct"),
    )
    return (
        laps.join(pole, on=RACE, how="left")
        .join(gaps, on=RACE + ["Driver"], how="left")
        .with_columns(
            # No usable lap: borrow the team-mate's gap, as they drive the
            # same car. If neither has one it stays empty.
            pl.col("QualiGapPct").fill_null(
                pl.col("QualiGapPct").mean().over(RACE + ["Team"])
            ),
            ((pl.col("LapTime") / pl.col("PoleTime") - 1) * 100).alias("LapTimePct"),
        )
    )


def add_clean_lap_flag(laps: pl.DataFrame) -> pl.DataFrame:
    """Mark laps that show normal racing pace.

    The lap time model should learn how tyres, fuel and traffic change pace.
    Pit laps, safety car laps, the standing start and wet tyres are far slower
    for other reasons, and would drown that signal, so they are flagged out
    here and handled by separate parts of the simulator.
    """
    return laps.with_columns(
        (
            pl.col("LapTime").is_not_null()
            & pl.col("IsAccurate")  # FastF1's own check that the timing is sound
            & pl.col("IsGreen")
            & ~pl.col("IsPitInLap")
            & ~pl.col("IsPitOutLap")
            & (pl.col("LapNumber") > 1)
            & pl.col("Compound").is_in(DRY_COMPOUNDS)
            & pl.col("TyreLife").is_not_null()
        ).alias("IsCleanLap")
    )


def build_clean_laps() -> pl.DataFrame:
    """Run every cleaning step in order."""
    laps = tidy_laps(read_raw("laps"))
    laps = add_split(laps)
    laps = add_race_progress(laps)
    laps = add_track_flags(laps)
    laps = add_gap_ahead(laps)
    laps = add_weather(laps, read_raw("weather"))
    laps = add_results(laps, read_raw("results"))
    laps = add_qualifying_pace(laps, pl.read_parquet(QUALIFYING_FILE))
    laps = add_clean_lap_flag(laps)
    return laps.sort(RACE + ["Driver", "LapNumber"])


def print_summary(laps: pl.DataFrame):
    summary = laps.group_by("Split", "Year").agg(
        pl.struct(RACE).n_unique().alias("races"),
        pl.len().alias("laps"),
        pl.col("IsCleanLap").sum().alias("clean_laps"),
        (pl.col("IsCleanLap").mean() * 100).round(1).alias("clean_%"),
        pl.col("IsSafetyCar").sum().alias("safety_car_laps"),
        pl.col("IsVSC").sum().alias("vsc_laps"),
    ).sort("Year")
    print(summary)


def main():
    laps = build_clean_laps()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    laps.write_parquet(OUTPUT_FILE)
    print(f"Saved {laps.height:,} laps with {laps.width} columns to {OUTPUT_FILE}\n")
    print_summary(laps)


if __name__ == "__main__":
    main()
