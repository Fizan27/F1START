# clean_history.py
#
# What it does: turns the downloaded Jolpica pages into two tidy tables.
#   1. One row per driver per race, with every retirement classified as
#      driver caused or car caused, and results made comparable across eras.
#   2. One row per pair of teammates per race: who was ahead in the race and
#      in qualifying. Teammates drive the same car, so this table is the core
#      signal the model in Phase 2 learns from.
# What it reads: data/jolpica/*.json (made by download_history.py).
# What it produces: data/results.parquet, data/teammate_pairs.parquet, and a
#   summary table printed to the screen.
# Which files use it: clean_laps.py reads data/results.parquet. The model in
#   Phase 2 reads both tables.
#
# Run:  .venv\Scripts\python.exe clean_history.py

import json
from itertools import combinations
from pathlib import Path

import polars as pl

JOLPICA_FOLDER = Path("data/jolpica")
RESULTS_FILE = Path("data/results.parquet")
PAIRS_FILE = Path("data/teammate_pairs.parquet")

RACE = ["Year", "Round"]  # these two columns together identify one race

# --- How a race ended for a driver (why these groups: DECISIONS.md 7) -------
# The data gives about 135 different reasons. Each is put into one of five
# groups. Only "finished" and "driver" say something about the driver.

DRIVER_CAUSED = {
    "Accident", "Collision", "Spun off", "Collision damage", "Fatal accident",
    "Stalled",
}
DID_NOT_START = {
    "Did not qualify", "Did not prequalify", "Did not start", "Withdrew",
    "107% Rule", "Not restarted", "Excluded",
}
# Not the car breaking, and not clearly a driving mistake either. When the
# data only says "Retired" the true reason is unknown, so it must not be
# guessed in either direction.
OTHER = {
    "Disqualified", "Retired", "Not classified", "Illness", "Injury", "Injured",
    "Driver unwell", "Physical", "Eye injury", "Safety concerns", "Safety",
    "Underweight", "Debris", "Puncture", "Tyre puncture", "Damage", "Broken wing",
}

# What every race would have paid under today's points system. Used so that
# "career points" means the same thing in 1960 (9 for a win) as in 2020 (25).
MODERN_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def classify_status(status: str) -> str:
    """Put one finishing status into: finished, driver, car, other, not_started.

    Anything not listed above is a part of the car failing ("Engine",
    "Gearbox", "Hydraulics" and about 100 more), so "car" is the fallback.
    """
    if status in ("Finished", "Lapped") or status.startswith("+"):  # "+1 Lap"
        return "finished"
    if status in DRIVER_CAUSED:
        return "driver"
    if status in DID_NOT_START:
        return "not_started"
    if status in OTHER:
        return "other"
    return "car"


def lap_time_seconds(text: str | None) -> float | None:
    """Turn a lap time like '1:23.456' into 83.456 seconds."""
    if not text:
        return None
    minutes, _, seconds = text.rpartition(":")
    try:
        return int(minutes or 0) * 60 + float(seconds)
    except ValueError:
        return None  # a few old entries hold text that is not a time


def read_races(kind: str) -> list[dict]:
    """Read every saved page of one kind ('results' or 'qualifying')."""
    races = []
    for file in sorted(JOLPICA_FOLDER.glob(f"{kind}_*.json")):
        page = json.loads(file.read_text(encoding="utf-8"))
        races.extend(page["MRData"]["RaceTable"]["Races"])
    return races


def results_rows(races: list[dict]) -> list[dict]:
    """Flatten the nested race results into one plain row per driver."""
    rows = []
    for race in races:
        for result in race["Results"]:
            driver, team = result["Driver"], result["Constructor"]
            rows.append({
                "Year": int(race["season"]),
                "Round": int(race["round"]),
                "Date": race["date"],
                "RaceName": race["raceName"],
                "CircuitId": race["Circuit"]["circuitId"],
                "CircuitName": race["Circuit"]["circuitName"],
                "Country": race["Circuit"]["Location"]["country"],
                "DriverId": driver["driverId"],
                "DriverName": f"{driver['givenName']} {driver['familyName']}",
                "DriverCode": driver.get("code"),
                "BirthDate": driver.get("dateOfBirth"),
                "TeamId": team["constructorId"],
                "TeamName": team["name"],
                "Grid": int(result["grid"]),
                "Position": int(result["position"]),
                "PositionText": result["positionText"],
                "Laps": int(result["laps"]),
                "Status": result["status"],
                "Points": float(result["points"]),
            })
    return rows


def qualifying_rows(races: list[dict]) -> list[dict]:
    """Flatten the qualifying results into one plain row per driver."""
    rows = []
    for race in races:
        for result in race["QualifyingResults"]:
            rows.append({
                "Year": int(race["season"]),
                "Round": int(race["round"]),
                "DriverId": result["Driver"]["driverId"],
                "QualiPosition": int(result["position"]),
                "Q1": lap_time_seconds(result.get("Q1")),
                "Q2": lap_time_seconds(result.get("Q2")),
                "Q3": lap_time_seconds(result.get("Q3")),
            })
    return rows


def one_row_per_driver(results: pl.DataFrame) -> pl.DataFrame:
    """Keep each driver's best result when they appear twice in one race.

    In the 1950s a driver whose car broke could take over a teammate's car, so
    the records list them twice. One race must count once per driver.
    """
    return results.sort(RACE + ["Position"]).unique(RACE + ["DriverId"], keep="first", maintain_order=True)


def add_outcome(results: pl.DataFrame) -> pl.DataFrame:
    """Add Outcome (how the race ended) and Classified (given a finishing place)."""
    by_status = pl.col("Status").map_elements(classify_status, return_dtype=pl.String)
    return results.with_columns(
        # Position text W (withdrew) or F (failed to qualify) means the driver
        # never took the start, even when the status names a reason such as
        # "Engine" or "Accident" (it happened in practice).
        pl.when(pl.col("PositionText").is_in(["W", "F"])).then(pl.lit("not_started"))
        .otherwise(by_status).alias("Outcome"),
        # Classified drivers have a number as position text; the others have a
        # letter (R retired, D disqualified, W withdrew, and so on).
        pl.col("PositionText").str.contains(r"^\d+$").alias("Classified"),
        # The Indianapolis 500 counted for the championship from 1950 to 1960
        # but was a different sport: other drivers, other cars, other rules.
        ((pl.col("CircuitId") == "indianapolis") & (pl.col("Year") <= 1960)).alias("IsIndy500"),
    )


def add_era_proof_results(results: pl.DataFrame) -> pl.DataFrame:
    """Add results that mean the same in every era.

    Starters: how many cars took the start. Fields ran from 10 to 34 cars.
    PositionScore: 1.0 for the winner down to 0.0 for last of the starters, so
      5th of 34 and 5th of 10 are no longer treated as the same result.
    ModernPoints: the points today's system would have paid for this result.
      The real points are kept too, but they are not comparable: a win was
      worth 8 points in 1950, 10 in 1991 and 25 since 2010.
    """
    started = pl.col("Outcome") != "not_started"
    points = pl.col("Position").replace_strict(
        {place: value for place, value in enumerate(MODERN_POINTS, start=1)},
        default=0, return_dtype=pl.Int32,
    )
    return results.with_columns(
        started.sum().over(RACE).alias("Starters"),
    ).with_columns(
        pl.when(started & (pl.col("Starters") > 1))
        .then(1 - (pl.col("Position") - 1) / (pl.col("Starters") - 1))
        .alias("PositionScore"),
        pl.when(pl.col("Classified")).then(points).otherwise(0).alias("ModernPoints"),
    )


def add_qualifying(results: pl.DataFrame, qualifying: pl.DataFrame) -> pl.DataFrame:
    """Attach qualifying times and position to each race result.

    Qualifying records only exist from 1994. Before that, the starting grid is
    used as the qualifying position: grid penalties were rare then, so the
    grid was the qualifying order. QualiSource says which one a row uses.
    """
    grid = pl.when(pl.col("Grid") > 0).then(pl.col("Grid"))  # grid 0 = pit lane or no start
    return results.join(qualifying, on=RACE + ["DriverId"], how="left").with_columns(
        pl.when(pl.col("QualiPosition").is_not_null()).then(pl.lit("qualifying"))
        .when(grid.is_not_null()).then(pl.lit("grid")).alias("QualiSource"),
        pl.col("QualiPosition").fill_null(grid),
        pl.min_horizontal("Q1", "Q2", "Q3").alias("BestQualiTime"),
    )


def add_teammate_count(results: pl.DataFrame) -> pl.DataFrame:
    """Add Teammates: how many other drivers started this race for the same team."""
    started = pl.col("Outcome") != "not_started"
    return results.with_columns(
        pl.when(started).then(started.sum().over(RACE + ["TeamId"]) - 1).otherwise(0).alias("Teammates")
    )


def compare_race(a: dict, b: dict) -> bool | None:
    """True if A beat B in the race, False if B beat A, None if not comparable.

    A race only counts when neither driver was stopped by something outside
    their control. So both must have either finished or gone out through
    their own crash or spin. If one car broke down, the pair says nothing
    about who is the better driver, and it is left out: this is how car caused
    retirements are kept from counting against a driver.
    """
    fair = ("finished", "driver")
    if a["Outcome"] not in fair or b["Outcome"] not in fair:
        return None
    return a["Position"] < b["Position"]


def compare_qualifying(a: dict, b: dict) -> tuple[bool | None, float | None]:
    """Who qualified ahead, and A's lap time gap to B in percent (negative = A faster).

    The gap uses the last part of qualifying that both drivers set a time in.
    Comparing a Q3 lap with a teammate's Q1 lap would be unfair, because the
    track gets faster during the hour.
    """
    ahead = None
    if a["QualiPosition"] is not None and b["QualiPosition"] is not None:
        ahead = a["QualiPosition"] < b["QualiPosition"]
    for part in ("Q3", "Q2", "Q1"):
        if a[part] and b[part]:
            return ahead, (a[part] / b[part] - 1) * 100
    return ahead, None


def build_teammate_pairs(results: pl.DataFrame) -> pl.DataFrame:
    """One row per pair of teammates per race.

    Teammates are drivers who started the same race for the same team. A team
    with three cars gives three pairs. The Indianapolis 500 is left out.
    """
    starters = results.filter((pl.col("Outcome") != "not_started") & ~pl.col("IsIndy500"))
    rows = []
    for (year, round_number, team), cars in starters.group_by(RACE + ["TeamId"], maintain_order=True):
        drivers = sorted(cars.to_dicts(), key=lambda car: car["DriverId"])
        for a, b in combinations(drivers, 2):
            quali_ahead, quali_gap = compare_qualifying(a, b)
            rows.append({
                "Year": year, "Round": round_number, "Date": a["Date"], "TeamId": team,
                "TeamCars": len(drivers),
                "DriverA": a["DriverId"], "DriverB": b["DriverId"],
                "OutcomeA": a["Outcome"], "OutcomeB": b["Outcome"],
                "PositionA": a["Position"], "PositionB": b["Position"],
                "AAheadRace": compare_race(a, b),
                "AAheadQuali": quali_ahead,
                "QualiGapPct": quali_gap,
            })
    return pl.DataFrame(rows, infer_schema_length=None)


def build_results() -> pl.DataFrame:
    """Run every cleaning step for the results table, in order."""
    results = pl.DataFrame(results_rows(read_races("results")), infer_schema_length=None)
    qualifying = pl.DataFrame(qualifying_rows(read_races("qualifying")), infer_schema_length=None)
    results = results.with_columns(pl.col("Date", "BirthDate").str.to_date(strict=False))
    results = one_row_per_driver(results)
    results = add_outcome(results)
    results = add_era_proof_results(results)
    results = add_qualifying(results, qualifying.unique(RACE + ["DriverId"], keep="first"))
    results = add_teammate_count(results)
    return results.sort(RACE + ["Position"])


def largest_linked_group(pairs: pl.DataFrame) -> tuple[int, int]:
    """How many drivers are linked to each other through chains of teammates.

    Returns (drivers in the biggest linked group, drivers with any teammate).
    Drivers outside the biggest group cannot be compared with those inside it:
    there is no chain of shared teammates connecting them.
    """
    group = {}  # driver -> a driver that stands for their whole group

    def leader(driver):
        while group.setdefault(driver, driver) != driver:
            driver = group[driver]
        return driver

    for a, b in pairs.select("DriverA", "DriverB").unique().iter_rows():
        group[leader(a)] = leader(b)  # merge the two groups
    sizes = {}
    for driver in group:
        sizes[leader(driver)] = sizes.get(leader(driver), 0) + 1
    return max(sizes.values()), len(group)


def print_summary(results: pl.DataFrame, pairs: pl.DataFrame):
    decade = (pl.col("Year") // 10 * 10).alias("Decade")
    started = results.filter(pl.col("Outcome") != "not_started")
    by_results = started.group_by(decade).agg(
        pl.col("Year").n_unique().alias("seasons"),
        pl.struct(RACE).n_unique().alias("races"),
        pl.col("DriverId").n_unique().alias("drivers"),
        pl.len().alias("starts"),
        ((pl.col("Outcome") == "car").mean() * 100).round(1).alias("car_out_%"),
        ((pl.col("Outcome") == "driver").mean() * 100).round(1).alias("driver_out_%"),
    )
    by_pairs = pairs.group_by(decade).agg(
        pl.len().alias("pairs"),
        pl.col("AAheadRace").is_not_null().sum().alias("race_comparable"),
        pl.col("QualiGapPct").is_not_null().sum().alias("quali_times"),
    )
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=140):
        print(by_results.join(by_pairs, on="Decade", how="left").sort("Decade"))

    linked, with_teammate = largest_linked_group(pairs)
    distinct_pairs = pairs.select("DriverA", "DriverB").unique().height
    print(f"\nTotals: {results['Year'].n_unique()} seasons, "
          f"{results.select(RACE).unique().height:,} races, "
          f"{started['DriverId'].n_unique():,} drivers who started a race, "
          f"{pairs.height:,} teammate pairs ({distinct_pairs:,} different pairings).")
    print(f"Linked through teammates: {linked:,} of the {with_teammate:,} drivers who ever "
          f"had a teammate are in one connected group.")
    print("\nHow races ended (all starts):")
    print(started.group_by("Outcome").agg(pl.len().alias("starts")).sort("starts", descending=True))


def main():
    results = build_results()
    pairs = build_teammate_pairs(results)
    results.write_parquet(RESULTS_FILE)
    pairs.write_parquet(PAIRS_FILE)
    print(f"Saved {results.height:,} results to {RESULTS_FILE}")
    print(f"Saved {pairs.height:,} teammate pairs to {PAIRS_FILE}\n")
    print_summary(results, pairs)


if __name__ == "__main__":
    main()
