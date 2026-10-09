# fit_ratings.py
#
# What it does: fits the model one last time on ALL the data (the validation
#   in validate_model.py is done first, on data up to 2023 only) and turns it
#   into the ratings everything else uses. For every driver:
#     - skill in every season, with uncertainty (the career curve)
#     - peak period: their best three seasons in a row, and the skill then
#     - consistency: how often they crashed out, and how steady their laps were
#     - record against each teammate
#   Plus the strength of every team's car in every season.
# What it reads: data/teammate_pairs.parquet, data/race_pace.parquet,
#   data/results.parquet.
# What it produces, in data/:
#     ratings.parquet           one row per driver
#     skill_by_season.parquet   one row per driver per season
#     teammate_records.parquet  one row per driver per teammate
#     car_strengths.parquet     one row per team per season
#     peak_covariance.pt        how uncertain each peak rating is, and how
#                               pairs of ratings move together
#   and a top 25 printed to the screen.
# Which files use it: the simulation (Phase 3) reads all of these.
#
# Run:  .venv\Scripts\python.exe fit_ratings.py     (about 30 seconds)

from pathlib import Path

import polars as pl
import torch

import model

RESULTS_FILE = Path("data/results.parquet")
OUTPUT_FOLDER = Path("data")

PEAK_SEASONS = 3  # the peak is the best run of this many seasons in a row

# The source stops giving retirement reasons in 2023 (DECISIONS.md 14), so
# crash rates can only be counted up to this season.
LAST_YEAR_WITH_REASONS = 2022

# Shrinkage: each driver's crash rate is mixed with this many imaginary
# starts at the average rate of their era. A driver with 5 starts and 1 crash
# is then not rated as crashing in 20% of races.
IMAGINARY_STARTS = 40
IMAGINARY_RACES_OF_LAPS = 10


def peak_window(seasons: list[int], skills: list[float]) -> tuple[int, int]:
    """Where a driver's best run of seasons starts and ends, as list positions.

    A window is up to three seasons the driver raced in a row in the list
    (their own seasons, even if they sat one out in between). The best
    window is the one with the highest average skill.
    """
    size = min(PEAK_SEASONS, len(seasons))
    averages = [sum(skills[start:start + size]) / size for start in range(len(seasons) - size + 1)]
    start = averages.index(max(averages))
    return start, start + size


def peak_ratings(fit: model.Fit) -> tuple[pl.DataFrame, torch.Tensor]:
    """Each driver's peak skill and its uncertainty, plus the covariance between drivers.

    The peak is an average of three of the fitted skills. The uncertainty of
    an average of uncertain numbers that move together is worked out exactly
    from the covariance: weights x covariance x weights.
    """
    careers = {}
    for (driver, year), position in fit.index.items():
        careers.setdefault(driver, []).append((year, position))

    weights = torch.zeros(len(careers), fit.covariance.shape[0], dtype=model.DTYPE, device=model.DEVICE)
    rows = []
    for row, (driver, career) in enumerate(careers.items()):
        years = [year for year, _ in career]
        skills = [fit.skill[position].item() for _, position in career]
        start, end = peak_window(years, skills)
        for _, position in career[start:end]:
            weights[row, position] = 1.0 / (end - start)
        rows.append({"DriverId": driver, "PeakFrom": years[start], "PeakTo": years[end - 1],
                     "PeakSkill": sum(skills[start:end]) / (end - start),
                     "FirstYear": years[0], "LastYear": years[-1], "Seasons": len(years)})
    covariance = weights @ fit.covariance @ weights.T
    ratings = pl.DataFrame(rows).with_columns(
        pl.Series("PeakSd", covariance.diagonal().sqrt().tolist()))
    return ratings, covariance


def crash_rates(results: pl.DataFrame) -> pl.DataFrame:
    """How often each driver put themselves out of a race, compared with their era.

    CrashRatio 1.0 = typical for the era, 0.5 = half as often, 2.0 = twice.
    Compared with the era because crashing out was three times as common in
    the 1990s as in the 2020s (gravel traps, fragile cars).
    """
    decade = (pl.col("Year") // 10 * 10).alias("Decade")
    starts = results.filter((pl.col("Outcome") != "not_started") & ~pl.col("IsIndy500")
                            & (pl.col("Year") <= LAST_YEAR_WITH_REASONS)).with_columns(decade)
    crashed = (pl.col("Outcome") == "driver")
    era = starts.group_by("Decade").agg(crashed.mean().alias("EraRate"))
    per_driver = starts.join(era, on="Decade").group_by("DriverId").agg(
        crashed.sum().alias("Crashes"),
        pl.len().alias("StartsWithReasons"),
        pl.col("EraRate").sum().alias("ExpectedCrashes"),  # what an average driver would have had
        pl.col("EraRate").mean().alias("EraRate"),
    )
    extra = IMAGINARY_STARTS * pl.col("EraRate")
    return per_driver.select(
        "DriverId", "Crashes", "StartsWithReasons",
        ((pl.col("Crashes") + extra) / (pl.col("ExpectedCrashes") + extra)).alias("CrashRatio"),
    )


def lap_steadiness(pace: pl.DataFrame) -> pl.DataFrame:
    """How steady each driver's lap times are, compared with the field (2018 onwards).

    LapSpreadRatio 1.0 = typical, below 1 = steadier laps than most.
    """
    races = pace.filter(pl.col("CleanLaps") >= model.MIN_CLEAN_LAPS).with_columns(
        (pl.col("LapSpreadPct") / pl.col("LapSpreadPct").median().over("Year", "Round")).alias("ratio"))
    return races.group_by("DriverId").agg(
        ((pl.col("ratio").sum() + IMAGINARY_RACES_OF_LAPS) / (pl.len() + IMAGINARY_RACES_OF_LAPS))
        .alias("LapSpreadRatio"),
        pl.col("LapSpreadPct").median().alias("LapSpreadPct"),
        pl.col("SlowLapShare").mean().alias("SlowLapShare"),
        pl.len().alias("RacesWithLaps"),
    )


def add_consistency(ratings: pl.DataFrame) -> pl.DataFrame:
    """One consistency score from 0 to 100, where 50 is typical and higher is better.

    It averages whichever of the two measures a driver has: crashing less
    often than their era, and (from 2018) steadier lap times than the field.
    Halving either ratio adds 25 points; doubling it takes 25 away.
    """
    def points(ratio):  # log2(0.5) = -1, log2(2) = +1
        return -25 * pl.col(ratio).log(2)

    both = pl.mean_horizontal(points("CrashRatio"), points("LapSpreadRatio"))
    return ratings.with_columns((50 + both.fill_null(0)).clip(0, 100).alias("Consistency"))


def career_totals(results: pl.DataFrame) -> pl.DataFrame:
    """Name, number of starts, wins and how many different teammates each driver had."""
    started = results.filter(pl.col("Outcome") != "not_started")
    return started.group_by("DriverId").agg(
        pl.col("DriverName").first(),
        pl.len().alias("Starts"),
        ((pl.col("Position") == 1) & pl.col("Classified")).sum().alias("Wins"),
        pl.col("TeamName").sort_by("Year").unique(maintain_order=True).alias("Teams"),
    )


def teammate_records(pairs: pl.DataFrame) -> pl.DataFrame:
    """Each driver's record against each teammate: races and qualifying, won and compared."""
    as_a = pairs.select(
        pl.col("DriverA").alias("DriverId"), pl.col("DriverB").alias("Teammate"), "Year",
        pl.col("AAheadRace").alias("ahead_race"), pl.col("AAheadQuali").alias("ahead_quali"))
    as_b = pairs.select(
        pl.col("DriverB").alias("DriverId"), pl.col("DriverA").alias("Teammate"), "Year",
        (~pl.col("AAheadRace")).alias("ahead_race"), (~pl.col("AAheadQuali")).alias("ahead_quali"))
    return pl.concat([as_a, as_b]).group_by("DriverId", "Teammate").agg(
        pl.col("Year").min().alias("FromYear"), pl.col("Year").max().alias("ToYear"),
        pl.len().alias("RacesTogether"),
        pl.col("ahead_race").sum().alias("RaceWins"),
        pl.col("ahead_race").is_not_null().sum().alias("RacesCompared"),
        pl.col("ahead_quali").sum().alias("QualiWins"),
        pl.col("ahead_quali").is_not_null().sum().alias("QualisCompared"),
    ).sort("DriverId", "RacesTogether", descending=[False, True])


def build_ratings(fit, results, pairs, pace) -> tuple[pl.DataFrame, torch.Tensor]:
    """Put every per-driver number in one table, best peak skill first."""
    ratings, covariance = peak_ratings(fit)
    teammates = teammate_records(pairs).group_by("DriverId").agg(pl.len().alias("Teammates"))
    ratings = (
        ratings.join(career_totals(results), on="DriverId", how="left")
        .join(teammates, on="DriverId", how="left")
        .join(crash_rates(results), on="DriverId", how="left")
        .join(lap_steadiness(pace), on="DriverId", how="left")
    )
    return add_consistency(ratings), covariance


def print_top(ratings: pl.DataFrame, minimum_starts: int = 50, count: int = 25):
    top = ratings.filter(pl.col("Starts") >= minimum_starts).sort("PeakSkill", descending=True).head(count)
    print(f"\nTop {count} by peak skill (at least {minimum_starts} starts). Skill is percent of lap time")
    print("quicker than an average newcomer of today, in the same car. Range = 90% interval.\n")
    print(f"{'':>3} {'Driver':<22} {'Peak':<10} {'Skill':>6} {'Range':>15} {'Consistency':>12} {'Starts':>7}")
    for place, row in enumerate(top.iter_rows(named=True), start=1):
        low, high = row["PeakSkill"] - 1.645 * row["PeakSd"], row["PeakSkill"] + 1.645 * row["PeakSd"]
        print(f"{place:>3} {row['DriverName']:<22} {row['PeakFrom']}-{row['PeakTo']:<5} {row['PeakSkill']:>6.2f} "
              f"{f'{low:+.2f} to {high:+.2f}':>15} {row['Consistency']:>12.0f} {row['Starts']:>7}")


def main():
    pairs, pace = model.load_tables()
    results = pl.read_parquet(RESULTS_FILE)
    last_year = results["Year"].max()

    fit = model.fit_model(pairs, pace, last_year)
    print(f"Fitted {len(fit.index):,} driver-seasons up to {last_year} on {model.DEVICE}")
    print("Learned scales:", {name: round(value, 3) for name, value in fit.scales.items()})
    print("Era levels:", {decade: round(level, 2) for decade, level in fit.era_level.items()})

    ratings, covariance = build_ratings(fit, results, pairs, pace)
    cars, driver_weight = model.car_strengths(results, fit, last_year)
    print(f"Driver weight in race results: {driver_weight:.3f}")

    ratings.write_parquet(OUTPUT_FOLDER / "ratings.parquet")
    model.skill_table(fit).write_parquet(OUTPUT_FOLDER / "skill_by_season.parquet")
    teammate_records(pairs).write_parquet(OUTPUT_FOLDER / "teammate_records.parquet")
    cars.write_parquet(OUTPUT_FOLDER / "car_strengths.parquet")
    torch.save({"drivers": ratings["DriverId"].to_list(), "covariance": covariance.float().cpu(),
                "scales": fit.scales, "era_level": fit.era_level, "skill_spread": fit.skill_spread, "season_drift": fit.season_drift},
               OUTPUT_FOLDER / "peak_covariance.pt")
    print(f"Saved ratings for {ratings.height} drivers in {OUTPUT_FOLDER}")
    print_top(ratings)


if __name__ == "__main__":
    main()
