# backtest.py
#
# What it does: a rolling backtest, the strongest test of the model. For every
#   season from 2010 to 2025 it fits the model on all EARLIER seasons only,
#   predicts which teammate finishes ahead (and qualifies ahead) in that
#   season, and scores the predictions against two simple baselines.
#   It does this for several versions of the model, so each optional part
#   (team orders, weak-teammate correction, wet adjustment) is kept only if it
#   does not make the predictions worse.
#   Finally it fits the calibration temperature on the 2010 to 2023
#   predictions only, so 2024 and 2025 stay untouched for the final check.
# What it reads: data/teammate_pairs.parquet, data/race_pace.parquet,
#   data/wet_races.parquet, data/results.parquet.
# What it produces: data/backtest_predictions.parquet (every prediction),
#   docs/backtest.png (accuracy by season), web/public/data/backtest.json,
#   and tables printed to the screen.
# Which files use it: none. The README and DECISIONS.md quote its numbers.
#
# Run:  .venv\Scripts\python.exe backtest.py              (about 40 minutes)
#       .venv\Scripts\python.exe backtest.py --chosen     (only the chosen model, about 8)

import json
import math
import sys
import time
from pathlib import Path

import polars as pl

import chart_style
import model
from validate_model import career_points_chance, last_season_chance

RESULTS_FILE = Path("data/results.parquet")
PREDICTIONS_FILE = Path("data/backtest_predictions.parquet")
CHART_FILE = Path("docs/backtest.png")
JSON_FILE = Path("web/public/data/backtest.json")

FIRST_TEST_SEASON = 2010
LAST_TEST_SEASON = 2025
LAST_CALIBRATION_SEASON = 2023  # the temperature is fitted on seasons up to here only
TEMPERATURES_TO_TRY = [round(0.1 + 0.05 * step, 2) for step in range(59)]  # 0.1 to 3.0

VARIANTS = {
    # experience=False: these were run before the experience curve existed (DECISIONS.md 28).
    "base": model.Options(number_one=False, career_prior=False, wet=False, experience=False),
    "team orders": model.Options(number_one=True, career_prior=False, wet=False, experience=False),
    "weak teammates": model.Options(number_one=False, career_prior=True, wet=False, experience=False),
    "wet": model.Options(number_one=False, career_prior=False, wet=True, experience=False),
    "weak teammates + wet": model.Options(number_one=False, career_prior=True, wet=True, experience=False),
}
# The version the site uses. Chosen by race log loss over all test seasons
# (DECISIONS.md 28): none of the optional parts improved it.
CHOSEN = "base"


def predict_season(fit: model.Fit, pairs: pl.DataFrame, wet_races: pl.DataFrame) -> pl.DataFrame:
    """The model's skill gap and its variance for every teammate pair of one season.

    If the model has a wet adjustment and the race was wet, the two drivers'
    adjustments are added. Whether a race is wet is known on race day, so
    this is a prediction made with the weather known, as a pundit would.
    """
    wet_keys = set(wet_races.select("Year", "Round").iter_rows()) if fit.wet else set()
    gaps, variances = [], []
    for year, race, driver_a, driver_b in pairs.select("Year", "Round", "DriverA", "DriverB").iter_rows():
        gap, variance = model.skill_gap(fit, driver_a, driver_b, year)
        if (year, race) in wet_keys:
            for driver, sign in ((driver_a, 1), (driver_b, -1)):
                adjustment, sd, _ = fit.wet.get(driver, (0.0, model.WET_SPREAD, 0))
                gap += sign * adjustment
                variance += sd ** 2
        gaps.append(gap)
        variances.append(variance)
    return pairs.with_columns(pl.Series("Gap", gaps), pl.Series("GapVariance", variances),
                              pl.lit(fit.scales["race_slope"]).alias("RaceSlope"),
                              pl.lit(model.quali_slope(fit.scales)).alias("QualiSlope"))


def chances(rows: pl.DataFrame, slope_column: str, temperature: float = 1.0) -> list[float]:
    return [model.chance_ahead(gap, variance, slope * temperature) for gap, variance, slope in
            rows.select("Gap", "GapVariance", slope_column).iter_rows()]


def run_variant(name: str, options: model.Options, pairs, pace, wet_races, results) -> pl.DataFrame:
    """Fit before each test season, predict it, and return every prediction."""
    seasons = []
    for season in range(FIRST_TEST_SEASON, LAST_TEST_SEASON + 1):
        started = time.time()
        fit = model.fit_model(pairs, pace, season - 1, wet_races=wet_races, options=options)
        test = pairs.filter(pl.col("Year") == season)
        predicted = predict_season(fit, test, wet_races).with_columns(
            career_points_chance(test, results, season - 1),
            last_season_chance(test, pairs),
            pl.lit(name).alias("Variant"),
        )
        seasons.append(predicted.select(
            "Variant", "Year", "Round", "DriverA", "DriverB", "AAheadRace", "AAheadQuali",
            "Gap", "GapVariance", "RaceSlope", "QualiSlope", "CareerPoints", "LastSeason"))
        print(f"  {name:<15} {season}  fitted in {time.time() - started:4.1f}s")
    return pl.concat(seasons)


# --- Scoring -----------------------------------------------------------------

def accuracy(chance: list[float], ahead: list[bool]) -> float:
    """Share called correctly. A 50/50 call counts as half right."""
    return sum(0.5 if p == 0.5 else float((p > 0.5) == a) for p, a in zip(chance, ahead)) / len(ahead)


def log_loss(chance: list[float], ahead: list[bool]) -> float:
    """Average surprise (lower is better; a coin flip scores 0.693)."""
    tiny = 1e-6  # a baseline that says "certain" and is wrong would otherwise score infinity
    return -sum(math.log(max(p if a else 1 - p, tiny)) for p, a in zip(chance, ahead)) / len(ahead)


def brier(chance: list[float], ahead: list[bool]) -> float:
    """Average squared miss of the probability (lower is better; a coin flip scores 0.25)."""
    return sum((p - float(a)) ** 2 for p, a in zip(chance, ahead)) / len(ahead)


def score(rows: pl.DataFrame, target: str, slope_column: str, temperature: float = 1.0) -> dict:
    """Model and baselines on one set of comparisons."""
    rows = rows.filter(pl.col(target).is_not_null())
    ahead = rows[target].to_list()
    model_chance = chances(rows, slope_column, temperature)
    fallback = rows["LastSeason"].fill_null(rows["CareerPoints"]).to_list()
    return {
        "comparisons": rows.height,
        "model": accuracy(model_chance, ahead),
        "model_log_loss": log_loss(model_chance, ahead),
        "model_brier": brier(model_chance, ahead),
        "career_points": accuracy(rows["CareerPoints"].to_list(), ahead),
        "last_season_or_points": accuracy(fallback, ahead),
    }


def by_season(predictions: pl.DataFrame, variant: str, temperature: float = 1.0) -> list[dict]:
    rows = predictions.filter(pl.col("Variant") == variant)
    table = []
    for season in range(FIRST_TEST_SEASON, LAST_TEST_SEASON + 1):
        season_rows = rows.filter(pl.col("Year") == season)
        table.append({"season": season,
                      "race": score(season_rows, "AAheadRace", "RaceSlope", temperature),
                      "quali": score(season_rows, "AAheadQuali", "QualiSlope")})
    return table


def summary(predictions: pl.DataFrame, variant: str, temperature: float = 1.0) -> dict:
    """Averages over all test seasons, pooling every comparison."""
    rows = predictions.filter(pl.col("Variant") == variant)
    return {"race": score(rows, "AAheadRace", "RaceSlope", temperature),
            "quali": score(rows, "AAheadQuali", "QualiSlope")}


def fit_temperature(predictions: pl.DataFrame, variant: str) -> float:
    """The temperature that minimises race log loss on 2010 to 2023 predictions."""
    rows = predictions.filter((pl.col("Variant") == variant) & (pl.col("Year") <= LAST_CALIBRATION_SEASON)
                              & pl.col("AAheadRace").is_not_null())
    ahead = rows["AAheadRace"].to_list()
    losses = {t: log_loss(chances(rows, "RaceSlope", t), ahead) for t in TEMPERATURES_TO_TRY}
    return min(losses, key=losses.get)


def print_summary(name: str, result: dict):
    race, quali = result["race"], result["quali"]
    print(f"{name:<15} race {race['model'] * 100:5.1f}%  (log loss {race['model_log_loss']:.4f}, "
          f"Brier {race['model_brier']:.4f})   qualifying {quali['model'] * 100:5.1f}%  "
          f"(log loss {quali['model_log_loss']:.4f})")


def draw_chart(seasons: list[dict], variant: str):
    figure, axes = chart_style.new_figure(width=10, height=4.2)
    years = [row["season"] for row in seasons]
    lines = [("model", f"Model ({variant})", chart_style.BLUE, 2.2),
             ("last_season_or_points", "Last season repeats, else career points", chart_style.BASELINE, 1.6),
             ("career_points", "More career points", "#c9c7c0", 1.6)]
    for key, label, colour, width in lines:
        axes.plot(years, [row["race"][key] * 100 for row in seasons], marker="o", markersize=4,
                  color=colour, linewidth=width, label=label)
    axes.axhline(50, color=chart_style.SECOND_INK, linewidth=0.8, linestyle=":")
    axes.set_ylim(40, 85)
    axes.set_xticks(years, [str(year)[2:] for year in years])
    axes.set_ylabel("teammate race result called correctly (%)")
    axes.legend(frameon=False, fontsize=8, loc="lower left")
    chart_style.tidy(axes, "Rolling backtest: each season predicted from earlier seasons only", grid_axis="y")
    chart_style.save(figure, CHART_FILE)


def main():
    pairs, pace = model.load_tables()
    wet_races, results = model.load_wet_races(), pl.read_parquet(RESULTS_FILE)

    if "--summarise" in sys.argv:
        new = pl.read_parquet(PREDICTIONS_FILE)  # only re-score earlier predictions
        # The first run called the combined variant "chosen"; it is the same thing.
        new = new.with_columns(pl.col("Variant").replace("chosen", "weak teammates + wet"))
    else:
        print(f"Rolling backtest {FIRST_TEST_SEASON} to {LAST_TEST_SEASON} on {model.DEVICE}")
        new = pl.concat([run_variant(name, options, pairs, pace, wet_races, results)
                         for name, options in VARIANTS.items()])
        new.write_parquet(PREDICTIONS_FILE)

    print(f"\nAll seasons {FIRST_TEST_SEASON} to {LAST_TEST_SEASON} pooled (temperature 1):")
    for name in new["Variant"].unique(maintain_order=True):
        print_summary(name, summary(new, name))
    baseline = summary(new, CHOSEN)["race"]
    print(f"{'baselines':<15} race: more career points {baseline['career_points'] * 100:.1f}%, "
          f"last season repeats else points {baseline['last_season_or_points'] * 100:.1f}%")

    temperature = fit_temperature(new, CHOSEN)
    print(f"\nTemperature fitted on {FIRST_TEST_SEASON} to {LAST_CALIBRATION_SEASON}: {temperature}")
    seasons = by_season(new, CHOSEN, temperature)
    print(f"\n{'season':>6} {'pairs':>6} {'model':>7} {'points':>7} {'last+pts':>9} {'quali model':>12}")
    for row in seasons:
        race = row["race"]
        print(f"{row['season']:>6} {race['comparisons']:>6} {race['model'] * 100:>6.1f}% "
              f"{race['career_points'] * 100:>6.1f}% {race['last_season_or_points'] * 100:>8.1f}% "
              f"{row['quali']['model'] * 100:>11.1f}%")
    draw_chart(seasons, CHOSEN)
    JSON_FILE.write_text(json.dumps({
        "temperature": temperature, "seasons": seasons,
        "chosen": CHOSEN, "summary": summary(new, CHOSEN, temperature),
        "summaryUncalibrated": summary(new, CHOSEN),
        "variants": {name: summary(new, name) for name in new["Variant"].unique(maintain_order=True)},
    }, indent=1), encoding="utf-8")
    print(f"Saved {JSON_FILE}")


if __name__ == "__main__":
    main()
