# wet_races.py
#
# What it does: makes the list of wet races, so the model can estimate how
#   much better or worse each driver was in the wet than in the dry.
#   - From 2018: measured. A race counts as wet when at least 10% of all laps
#     were driven on intermediate or full wet tyres (FastF1 lap data).
#   - Before 2018 there is no lap data, so a hand-made list of well known wet
#     races is used (WET_RACES_BEFORE_2018 below). It was written from
#     memory of well documented races, so it is incomplete: plenty of wet
#     races are missing, and races that were only wet for a few laps are left
#     out on purpose. Every entry must match exactly one race in the results,
#     or the script stops.
# What it reads: data/raw/*_laps.parquet and data/results.parquet.
# What it produces: data/wet_races.parquet (Year, Round, RaceName, Source),
#   and a count printed to the screen.
# Which files use it: model.py (the wet skill adjustment).
#
# Run:  .venv\Scripts\python.exe wet_races.py

from pathlib import Path

import polars as pl

RAW_FOLDER = Path("data/raw")
RESULTS_FILE = Path("data/results.parquet")
OUTPUT_FILE = Path("data/wet_races.parquet")

WET_TYRES = ["INTERMEDIATE", "WET"]
WET_SHARE = 0.10  # share of a race's laps on wet tyres for it to count as wet
FIRST_MEASURED_YEAR = 2018

# (season, race name as the results data spells it). Well known races run
# wholly or mostly in the wet. Hand made: see the note at the top.
WET_RACES_BEFORE_2018 = [
    (1954, "Swiss Grand Prix"),
    (1961, "German Grand Prix"),
    (1963, "Belgian Grand Prix"),
    (1966, "Belgian Grand Prix"),
    (1968, "French Grand Prix"),
    (1968, "Dutch Grand Prix"),
    (1968, "German Grand Prix"),
    (1971, "Dutch Grand Prix"),
    (1972, "Monaco Grand Prix"),
    (1975, "Austrian Grand Prix"),
    (1976, "Japanese Grand Prix"),
    (1981, "Canadian Grand Prix"),
    (1984, "Monaco Grand Prix"),
    (1985, "Portuguese Grand Prix"),
    (1989, "Australian Grand Prix"),
    (1991, "Australian Grand Prix"),
    (1993, "European Grand Prix"),
    (1994, "Japanese Grand Prix"),
    (1996, "Spanish Grand Prix"),
    (1996, "Monaco Grand Prix"),
    (1997, "Monaco Grand Prix"),
    (1998, "Belgian Grand Prix"),
    (2003, "Brazilian Grand Prix"),
    (2007, "European Grand Prix"),
    (2007, "Japanese Grand Prix"),
    (2008, "Monaco Grand Prix"),
    (2008, "British Grand Prix"),
    (2008, "Italian Grand Prix"),
    (2009, "Malaysian Grand Prix"),
    (2009, "Chinese Grand Prix"),
    (2010, "Korean Grand Prix"),
    (2011, "Canadian Grand Prix"),
    (2012, "Malaysian Grand Prix"),
    (2014, "Japanese Grand Prix"),
    (2016, "Monaco Grand Prix"),
    (2016, "British Grand Prix"),
    (2016, "Brazilian Grand Prix"),
]


def measured_wet_races(laps: pl.DataFrame) -> pl.DataFrame:
    """Races since 2018 where at least 10% of laps were on wet weather tyres."""
    share = laps.group_by("Year", "Round").agg(
        pl.col("Compound").is_in(WET_TYRES).fill_null(False).mean().alias("WetShare"))
    return share.filter(pl.col("WetShare") >= WET_SHARE).select(
        pl.col("Year", "Round").cast(pl.Int64), pl.lit("laps").alias("Source"))


def listed_wet_races(races: pl.DataFrame) -> pl.DataFrame:
    """The hand-made list, matched to races. Stops if an entry matches no race or several."""
    listed = pl.DataFrame(WET_RACES_BEFORE_2018, schema=["Year", "RaceName"], orient="row")
    matched = listed.join(races, on=["Year", "RaceName"], how="left")
    missing = matched.filter(pl.col("Round").is_null())
    if missing.height or matched.height != listed.height:
        raise ValueError(f"These listed wet races do not match exactly one race: {missing.rows()}")
    return matched.select("Year", "Round", pl.lit("list").alias("Source"))


def build_wet_races(laps: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    races = results.select("Year", "Round", "RaceName").unique()
    measured = measured_wet_races(laps.filter(pl.col("Year") >= FIRST_MEASURED_YEAR))
    wet = pl.concat([listed_wet_races(races), measured])
    return wet.join(races, on=["Year", "Round"]).sort("Year", "Round")


def main():
    columns = ["Year", "Round", "Compound"]
    laps = pl.concat([pl.read_parquet(f).select(columns) for f in sorted(RAW_FOLDER.glob("*_laps.parquet"))],
                     how="vertical_relaxed")
    wet = build_wet_races(laps, pl.read_parquet(RESULTS_FILE))
    wet.write_parquet(OUTPUT_FILE)
    print(f"Saved {wet.height} wet races to {OUTPUT_FILE}:")
    print(f"  {wet.filter(pl.col('Source') == 'list').height} from the hand-made list (before 2018)")
    print(f"  {wet.filter(pl.col('Source') == 'laps').height} measured from lap data (2018 onwards)")
    with pl.Config(tbl_rows=80):
        print(wet.filter(pl.col("Source") == "laps"))


if __name__ == "__main__":
    main()
