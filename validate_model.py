# validate_model.py
#
# What it does: the honest test of the model. It fits the model on results up
#   to the end of 2023, then predicts 2024 and 2025 WITHOUT refitting, and
#   compares those predictions with what really happened and with simple
#   baselines that need no model at all.
#     Test 1: which teammate finishes ahead (and which qualifies ahead).
#     Test 2: are the model's probabilities honest (calibration)?
#     Test 3: finishing positions of the whole field.
# What it reads: data/teammate_pairs.parquet, data/race_pace.parquet and
#   data/results.parquet.
# What it produces: tables printed to the screen, docs/validation.png, and
#   web/public/data/validation.json (the same numbers, for the website).
# Which files use it: none. The README quotes its numbers.
#
# Run:  .venv\Scripts\python.exe validate_model.py
#       .venv\Scripts\python.exe validate_model.py --tune   (also re-choose the
#           two settings, using only seasons up to 2023; about 5 minutes)

import json
import math
import sys
from pathlib import Path

import polars as pl

import chart_style
import model

RESULTS_FILE = Path("data/results.parquet")
CHART_FILE = Path("docs/validation.png")
JSON_FILE = Path("web/public/data/validation.json")

TRAIN_UNTIL = 2023
TEST_YEARS = [2024, 2025]

# For choosing the two settings: fit up to 2021, score on 2022 and 2023.
# 2024 and 2025 play no part in any choice.
TUNE_TRAIN_UNTIL = 2021
TUNE_YEARS = [2022, 2023]
SPREADS_TO_TRY = [0.2, 0.3, 0.4, 0.55]
DRIFTS_TO_TRY = [0.04, 0.08, 0.12, 0.18, 0.25, 0.35]

# Teams that changed their name. The car of 2023 is looked up under the old one.
EARLIER_TEAM_NAME = {"rb": "alphatauri", "sauber": "alfa"}

RACE = ["Year", "Round"]


# --- Predictions -------------------------------------------------------------

def predict_pairs(fit: model.Fit, pairs: pl.DataFrame) -> pl.DataFrame:
    """Add the model's chance that A beats B in the race and in qualifying."""
    race, quali = [], []
    slope_quali = model.AHEAD_SLOPE_PER_NOISE / fit.scales["quali_noise"]
    for driver_a, driver_b, year in pairs.select("DriverA", "DriverB", "Year").iter_rows():
        gap, variance = model.skill_gap(fit, driver_a, driver_b, year)
        race.append(model.chance_ahead(gap, variance, fit.scales["race_slope"]))
        quali.append(model.chance_ahead(gap, variance, slope_quali))
    return pairs.with_columns(pl.Series("ModelRace", race), pl.Series("ModelQuali", quali))


def career_points_chance(pairs: pl.DataFrame, results: pl.DataFrame, until: int) -> pl.Series:
    """Baseline 1: the driver with more career points (up to `until`) is ahead.

    Today's points system is applied to every era (ModernPoints) so that old
    points systems do not handicap the baseline. Returns 1.0 if it picks A,
    0.0 if B, and 0.5 when they are level (two rookies).
    """
    points = dict(results.filter(pl.col("Year") <= until).group_by("DriverId")
                  .agg(pl.col("ModernPoints").sum()).iter_rows())
    chances = []
    for driver_a, driver_b in pairs.select("DriverA", "DriverB").iter_rows():
        a, b = points.get(driver_a, 0), points.get(driver_b, 0)
        chances.append(0.5 if a == b else float(a > b))
    return pl.Series("CareerPoints", chances)


def last_season_chance(pairs: pl.DataFrame, all_pairs: pl.DataFrame) -> pl.Series:
    """Baseline 2: last season's head to head between the same two teammates repeats.

    Empty when the two were not teammates the season before, or were level.
    Note this baseline is allowed to look at 2024 when predicting 2025, which
    the model is not: it is given every advantage.
    """
    wins = all_pairs.filter(pl.col("AAheadRace").is_not_null()).group_by("Year", "DriverA", "DriverB").agg(
        pl.col("AAheadRace").sum().alias("a_wins"), pl.len().alias("races"))
    record = {(year, a, b): (a_wins, races) for year, a, b, a_wins, races in wins.iter_rows()}
    chances = []
    for year, driver_a, driver_b in pairs.select("Year", "DriverA", "DriverB").iter_rows():
        a_wins, races = record.get((year - 1, driver_a, driver_b), (0, 0))
        level = races == 0 or a_wins * 2 == races
        chances.append(None if level else float(a_wins * 2 > races))
    return pl.Series("LastSeason", chances, dtype=pl.Float64)


# --- Scoring -----------------------------------------------------------------

def accuracy(chance: pl.Series, a_was_ahead: pl.Series) -> float:
    """Share of comparisons called correctly. A 50/50 call counts as half right."""
    total = 0.0
    for p, ahead in zip(chance, a_was_ahead):
        total += 0.5 if p == 0.5 else float((p > 0.5) == ahead)
    return total / len(chance)


def log_loss(chance: pl.Series, a_was_ahead: pl.Series) -> float:
    """How surprised the model was on average (lower is better; a coin flip scores 0.693).

    Unlike accuracy, this punishes being confidently wrong, so it checks the
    probabilities and not just which side of 50% they fall on.
    """
    total = 0.0
    for p, ahead in zip(chance, a_was_ahead):
        total -= math.log(p if ahead else 1 - p)
    return total / len(chance)


def score_teammates(predicted: pl.DataFrame, target: str, model_column: str) -> dict:
    """Score the model and both baselines on one kind of teammate comparison."""
    rows = predicted.filter(pl.col(target).is_not_null())
    actual = rows[target]
    repeat = rows.filter(pl.col("LastSeason").is_not_null())
    # Last season's result where there is one, otherwise career points.
    combined = rows["LastSeason"].fill_null(rows["CareerPoints"])
    return {
        "comparisons": rows.height,
        "model": accuracy(rows[model_column], actual),
        "model_log_loss": log_loss(rows[model_column], actual),
        "career_points": accuracy(rows["CareerPoints"], actual),
        "last_season_or_points": accuracy(combined, actual),
        "same_pair_comparisons": repeat.height,
        "same_pair_last_season": accuracy(repeat["LastSeason"], repeat[target]),
        "same_pair_model": accuracy(repeat[model_column], repeat[target]),
    }


def calibration(predicted: pl.DataFrame, target: str, model_column: str) -> pl.DataFrame:
    """When the model says its favourite has a 70% chance, do they win 70% of the time?"""
    rows = predicted.filter(pl.col(target).is_not_null()).select(
        pl.when(pl.col(model_column) >= 0.5).then(pl.col(model_column))
        .otherwise(1 - pl.col(model_column)).alias("said"),
        ((pl.col(model_column) >= 0.5) == pl.col(target)).alias("favourite_won"),
    )
    return rows.with_columns(
        pl.when(pl.col("said") < 0.6).then(pl.lit("50-60%"))
        .when(pl.col("said") < 0.7).then(pl.lit("60-70%"))
        .when(pl.col("said") < 0.8).then(pl.lit("70-80%"))
        .otherwise(pl.lit("80%+")).alias("model_said")
    ).group_by("model_said").agg(
        pl.len().alias("comparisons"),
        (pl.col("said").mean() * 100).round(1).alias("average_said_%"),
        (pl.col("favourite_won").mean() * 100).round(1).alias("favourite_won_%"),
    ).sort("average_said_%")


# --- Finishing positions -----------------------------------------------------

def predict_positions(fit: model.Fit, results: pl.DataFrame, year: int) -> pl.DataFrame:
    """Predict the finishing order of every race of a test season.

    Predicted result = the team's car strength in 2023 + driver weight x the
    driver's skill at the end of 2023. Nothing from 2024 or 2025 is used.
    Two baselines are ranked the same way: the car alone, and last season's
    championship order.
    """
    cars, driver_weight = model.car_strengths(results, fit, TRAIN_UNTIL)
    car_2023 = dict(cars.filter(pl.col("Year") == TRAIN_UNTIL).select("TeamId", "CarStrength").iter_rows())
    average_car = sum(car_2023.values()) / len(car_2023)
    points_before = dict(results.filter(pl.col("Year") == year - 1).group_by("DriverId")
                         .agg(pl.col("ModernPoints").sum()).iter_rows())

    rows = results.filter((pl.col("Year") == year) & pl.col("Outcome").is_in(["finished", "driver"]))
    car, driver, championship = [], [], []
    for driver_id, team in rows.select("DriverId", "TeamId").iter_rows():
        car.append(car_2023.get(EARLIER_TEAM_NAME.get(team, team), average_car))
        season = model.latest_season(fit, driver_id, TRAIN_UNTIL)
        driver.append(fit.skill[fit.index[(driver_id, season)]].item() if season else 0.0)
        championship.append(points_before.get(driver_id, 0))
    rows = rows.with_columns(
        pl.Series("car", car), pl.Series("skill", driver), pl.Series("last_season_points", championship))

    def place(column, descending=True):  # 1 = predicted (or really) first, within each race
        return pl.col(column).rank("average", descending=descending).over(RACE)

    return rows.with_columns((pl.col("car") + driver_weight * pl.col("skill")).alias("car_and_driver")).select(
        *RACE, "DriverId",
        place("Position", descending=False).alias("actual"),
        place("car_and_driver").alias("Model (car + driver)"),
        place("car").alias("Car only"),
        place("last_season_points").alias("Last season's standings"),
    )


def score_positions(places: pl.DataFrame) -> dict:
    """Average miss in places, and rank correlation (1 = perfect order, 0 = random)."""
    scores = {}
    for method in places.columns[4:]:
        per_race = places.group_by(RACE).agg(pl.corr("actual", method).alias("correlation"))
        scores[method] = {
            "average_miss_places": float((places["actual"] - places[method]).abs().mean()),
            "rank_correlation": float(per_race["correlation"].mean()),
        }
    return scores


# --- Choosing the two settings (never uses 2024 or 2025) ---------------------

def tune(pairs: pl.DataFrame, pace: pl.DataFrame) -> tuple[float, float]:
    """Try each pair of settings: fit up to 2021, score on 2022 and 2023."""
    held_out = pairs.filter(pl.col("Year").is_in(TUNE_YEARS))
    best = None
    print(f"\nChoosing settings: fit up to {TUNE_TRAIN_UNTIL}, scored on {TUNE_YEARS}")
    print(f"{'skill spread':>12} {'season drift':>12} {'race log loss':>14} {'quali log loss':>15}")
    for spread in SPREADS_TO_TRY:
        for drift in DRIFTS_TO_TRY:
            fit = model.fit_model(pairs, pace, TUNE_TRAIN_UNTIL, spread, drift)
            predicted = predict_pairs(fit, held_out)
            race = predicted.filter(pl.col("AAheadRace").is_not_null())
            quali = predicted.filter(pl.col("AAheadQuali").is_not_null())
            race_loss = log_loss(race["ModelRace"], race["AAheadRace"])
            quali_loss = log_loss(quali["ModelQuali"], quali["AAheadQuali"])
            print(f"{spread:>12} {drift:>12} {race_loss:>14.4f} {quali_loss:>15.4f}")
            if best is None or race_loss + quali_loss < best[0]:
                best = (race_loss + quali_loss, spread, drift)
    print(f"Best: skill spread {best[1]}, season drift {best[2]}")
    return best[1], best[2]


# --- Output ------------------------------------------------------------------

def print_teammate_scores(year, kind, scores: dict):
    print(f"\n{year}, {kind}: {scores['comparisons']} teammate comparisons")
    print(f"  {'Model':<42} {scores['model'] * 100:5.1f}%   (log loss {scores['model_log_loss']:.3f}; coin flip 0.693)")
    print(f"  {'More career points':<42} {scores['career_points'] * 100:5.1f}%")
    print(f"  {'Last season repeats, else career points':<42} {scores['last_season_or_points'] * 100:5.1f}%")
    print(f"  Only the {scores['same_pair_comparisons']} where the pair were also teammates last season:")
    print(f"  {'  last season repeats':<42} {scores['same_pair_last_season'] * 100:5.1f}%")
    print(f"  {'  model':<42} {scores['same_pair_model'] * 100:5.1f}%")


def draw_chart(teammate_scores: dict, position_scores: dict):
    """Two panels: teammate accuracy against baselines, and finishing position miss."""
    figure, (left, right) = chart_style.new_figure(columns=2, width=11, height=4.2)
    bars = [("model", "Model", chart_style.BLUE),
            ("last_season_or_points", "Last season repeats", chart_style.BASELINE),
            ("career_points", "More career points", "#c9c7c0")]
    labels = [f"{year} {kind}" for year in TEST_YEARS for kind in ("race", "qualifying")]
    for offset, (key, label, colour) in enumerate(bars):
        heights = [teammate_scores[f"{year} {kind}"][key] * 100
                   for year in TEST_YEARS for kind in ("race", "qualifying")]
        left.bar([x + (offset - 1) * 0.27 for x in range(len(labels))], heights, 0.25, label=label, color=colour)
    left.axhline(50, color=chart_style.SECOND_INK, linewidth=0.8, linestyle=":")
    left.set_xticks(range(len(labels)), labels)
    left.set_ylim(40, 80)
    left.set_ylabel("teammate called correctly (%)")
    left.legend(frameon=False, fontsize=8, loc="upper left")
    chart_style.tidy(left, "Which teammate is ahead? (dotted line = coin flip)", grid_axis="y")

    methods = list(position_scores[TEST_YEARS[0]])
    colours = [chart_style.BLUE, chart_style.ORANGE, chart_style.BASELINE]
    for offset, (method, colour) in enumerate(zip(methods, colours)):
        heights = [position_scores[year][method]["average_miss_places"] for year in TEST_YEARS]
        right.bar([x + (offset - 1) * 0.27 for x in range(len(TEST_YEARS))], heights, 0.25, label=method, color=colour)
    right.set_xticks(range(len(TEST_YEARS)), [str(year) for year in TEST_YEARS])
    right.set_ylabel("average miss (places, lower is better)")
    right.set_ylim(0, 6.5)
    right.legend(frameon=False, fontsize=8, loc="upper left")
    chart_style.tidy(right, "Finishing position, predicted from 2023", grid_axis="y")
    chart_style.save(figure, CHART_FILE)


def main():
    pairs, pace = model.load_tables()
    results = pl.read_parquet(RESULTS_FILE)

    spread, drift = model.DEFAULT_SKILL_SPREAD, model.DEFAULT_SEASON_DRIFT
    if "--tune" in sys.argv:
        spread, drift = tune(pairs, pace)

    print(f"\nFitting on results up to {TRAIN_UNTIL} (skill spread {spread}, season drift {drift})...")
    fit = model.fit_model(pairs, pace, TRAIN_UNTIL, spread, drift)
    print(f"{len(fit.index):,} driver-seasons fitted on {model.DEVICE}")
    print("Learned scales:", {name: round(value, 3) for name, value in fit.scales.items()})

    test_pairs = pairs.filter(pl.col("Year").is_in(TEST_YEARS))
    predicted = predict_pairs(fit, test_pairs).with_columns(
        career_points_chance(test_pairs, results, TRAIN_UNTIL),
        last_season_chance(test_pairs, pairs),
    )

    print("\n=== Test 1: which teammate is ahead? ===")
    teammate_scores = {}
    for year in TEST_YEARS + ["both"]:
        rows = predicted if year == "both" else predicted.filter(pl.col("Year") == year)
        for kind, target, column in (("race", "AAheadRace", "ModelRace"),
                                     ("qualifying", "AAheadQuali", "ModelQuali")):
            teammate_scores[f"{year} {kind}"] = score_teammates(rows, target, column)
            print_teammate_scores(year, kind, teammate_scores[f"{year} {kind}"])

    print("\n=== Test 2: are the probabilities honest? (2024 and 2025 together) ===")
    calibrations = {}
    for kind, target, column in (("race", "AAheadRace", "ModelRace"),
                                 ("qualifying", "AAheadQuali", "ModelQuali")):
        calibrations[kind] = calibration(predicted, target, column)
        print(f"\n{kind}:")
        print(calibrations[kind])

    print("\n=== Test 3: finishing positions, predicted from 2023 only ===")
    position_scores = {}
    for year in TEST_YEARS:
        position_scores[year] = score_positions(predict_positions(fit, results, year))
        print(f"\n{year}:")
        for method, scores in position_scores[year].items():
            print(f"  {method:<26} average miss {scores['average_miss_places']:.2f} places, "
                  f"rank correlation {scores['rank_correlation']:.3f}")

    draw_chart(teammate_scores, position_scores)
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps({
        "trainedUntil": TRAIN_UNTIL, "testYears": TEST_YEARS,
        "teammates": teammate_scores,
        "calibration": {kind: table.to_dicts() for kind, table in calibrations.items()},
        "positions": {str(year): scores for year, scores in position_scores.items()},
    }, indent=1), encoding="utf-8")
    print(f"Numbers saved to {JSON_FILE}")


if __name__ == "__main__":
    main()
