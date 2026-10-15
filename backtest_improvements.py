# backtest_improvements.py
#
# What it does: finds out how far teammate prediction can honestly go, and
#   tests three improvements, all without letting a prediction see the future.
#     Ceiling:       for each season, fit WITH that season's own results and
#                    predict the same season (hindsight). No fair model can be
#                    expected to beat this, so it shows how much room there is.
#     In-season:     predict each race from all earlier seasons PLUS the
#                    earlier races of the same season, refitting before every
#                    race. Never the race being predicted, or anything after it.
#     Qualifying:    for race predictions, also use the teammates' qualifying
#                    result from the same weekend (qualifying happens before
#                    the race, so this is fair).
#     Experience:    a shared curve for how drivers typically improve or
#                    decline with their years in F1 (model.Options.experience).
#
#   The qualifying signal is added by a small second step on top of the model:
#   a logistic regression with three inputs (the model's own prediction, how
#   much better or worse the driver qualified than the model expected, and
#   who qualified ahead when no lap time exists). Its weights for each season
#   are learned only from EARLIER seasons' predictions. So that every version
#   is compared fairly, every version gets the same second step (without the
#   qualifying inputs it simply recalibrates the model's own probabilities).
#   Seasons 2007 to 2009 are predicted too, only so that 2010 has earlier
#   seasons to learn those weights from; results are reported for 2010 to 2025.
#
# What it reads: data/teammate_pairs.parquet, data/race_pace.parquet,
#   data/results.parquet.
# What it produces: data/backtest_improvements.parquet (every prediction, with
#   the last race each one's fit could see), docs/improvements.png,
#   web/public/data/improvements.json and a table printed to the screen.
# Which files use it: none. The README and DECISIONS.md quote its numbers.
#
# Run:  .venv\Scripts\python.exe backtest_improvements.py             (about 80 minutes)
#       .venv\Scripts\python.exe backtest_improvements.py --summarise (re-score saved predictions)

import json
import math
import sys
import time
from pathlib import Path

import polars as pl
import torch

import chart_style
import model
from backtest import accuracy, brier, log_loss

PREDICTIONS_FILE = Path("data/backtest_improvements.parquet")
CHART_FILE = Path("docs/improvements.png")
JSON_FILE = Path("web/public/data/improvements.json")

FIRST_PREDICTED = 2007  # 2007 to 2009 only give the second step something to learn from
FIRST_REPORTED = 2010
LAST_REPORTED = 2025
LEAKAGE_ALARM = 0.75  # an honest result above this gets checked for leakage before it is believed

# How each set of predictions is made: (mode, model options)
FITS = {
    "current": ("season start", model.Options(experience=False)),
    "in-season": ("in season", model.Options(experience=False)),
    "experience": ("season start", model.Options(experience=True)),
    "in-season + experience": ("in season", model.Options(experience=True)),
    "ceiling": ("hindsight", model.Options(experience=False)),
}
# The rows of the results table: (label, which predictions, inputs to the second step)
#   "model": the model's own prediction only; "model+quali": plus the same
#   weekend's qualifying; "quali only": just who qualified ahead (a baseline).
ROWS = [
    ("Current model", "current", "model"),
    ("+ in-season updating", "in-season", "model"),
    ("+ qualifying signal", "current", "model+quali"),
    ("+ experience curve", "experience", "model"),
    ("All combined", "in-season + experience", "model+quali"),
    ("In-season + qualifying", "in-season", "model+quali"),
    ("Baseline: whoever qualified ahead", "current", "quali only"),
    ("Ceiling (hindsight, not a fair model)", "ceiling", "model"),
    ("Ceiling + qualifying (hindsight)", "ceiling", "model+quali"),
]


# --- Making predictions --------------------------------------------------------

def last_race_seen(fit: model.Fit) -> tuple[int, int]:
    """The latest (season, round) in the data a fit was made from, read from the data itself."""
    seen = fit.data["comparisons"].select(pl.col("Year").max().alias("y"))
    year = seen["y"][0]
    race = fit.data["comparisons"].filter(pl.col("Year") == year)["Round"].max()
    return year, race


def predict_rows(rows: pl.DataFrame, mean_fit: model.Fit, variance_fit: model.Fit) -> pl.DataFrame:
    """The skill gap (from mean_fit) and its uncertainty (from variance_fit) for each comparison.

    In-season refits recompute the best estimates before every race but not
    the uncertainty (that takes several seconds more each time), so the
    uncertainty comes from the fit made at the start of the season.
    """
    gaps, variances = [], []
    year_seen, race_seen = last_race_seen(mean_fit)
    for year, driver_a, driver_b in rows.select("Year", "DriverA", "DriverB").iter_rows():
        gaps.append(model.expected_gap(mean_fit, driver_a, driver_b, year))
        variances.append(model.skill_gap(variance_fit, driver_a, driver_b, year)[1])
    return rows.with_columns(
        pl.Series("Gap", gaps), pl.Series("GapVariance", variances),
        pl.lit(mean_fit.scales["race_slope"]).alias("RaceSlope"),
        pl.lit(model.quali_slope(mean_fit.scales)).alias("QualiSlope"),
        pl.lit(year_seen).alias("SeenYear"), pl.lit(race_seen).alias("SeenRound"),
    )


def predict_season(pairs, pace, season: int, mode: str, options: model.Options) -> pl.DataFrame:
    """Every teammate comparison of one season, predicted the way `mode` says."""
    season_pairs = pairs.filter(pl.col("Year") == season)
    if mode == "hindsight":
        fit = model.fit_model(pairs, pace, season, options=options)
        return predict_rows(season_pairs, fit, fit)
    rounds = sorted(season_pairs["Round"].unique().to_list())
    start = model.fit_model(pairs, pace, season, options=options, before_race=(season, rounds[0]))
    if mode == "season start":
        return predict_rows(season_pairs, start, start)
    predicted, fit = [], start
    for race in rounds:
        if race != rounds[0]:
            fit = model.fit_model(pairs, pace, season, options=options, before_race=(season, race),
                                  start=fit, with_uncertainty=False)
        predicted.append(predict_rows(season_pairs.filter(pl.col("Round") == race), fit, start))
    return pl.concat(predicted)


def run_fit(name: str, pairs, pace) -> pl.DataFrame:
    mode, options = FITS[name]
    seasons = []
    for season in range(FIRST_PREDICTED, LAST_REPORTED + 1):
        started = time.time()
        seasons.append(predict_season(pairs, pace, season, mode, options).with_columns(pl.lit(name).alias("Variant")))
        print(f"  {name:<24} {season}  {time.time() - started:5.1f}s", flush=True)
    return pl.concat(seasons)


# --- The second step: recalibration, with or without the qualifying signal --------

def features(rows: pl.DataFrame, slope_column: str, use_quali: bool) -> torch.Tensor:
    """Inputs to the second step, one row per comparison.

    1. The model's own prediction, as a logit (the S-curve's input), already
       softened for the uncertainty of the gap.
    2. Qualifying residual: how much quicker (+) or slower (-) driver A
       qualified relative to B than the model expected, in percent. 0 when
       there is no believable lap time gap.
    3. Who qualified ahead (+1 A, -1 B) when there is no lap time gap; else 0.
    """
    inputs = {True: "model+quali", False: "model"}.get(use_quali, use_quali)
    if inputs == "quali only":
        # The simplest rule that uses qualifying: whoever qualified ahead is
        # the favourite, with a confidence learned from earlier seasons.
        ahead = rows["AAheadQuali"].cast(pl.Float64) * 2 - 1
        return torch.tensor(ahead.fill_null(0.0).to_list(), dtype=torch.float64)[:, None]
    slope, gap, variance = (torch.tensor(rows[c].to_list(), dtype=torch.float64)
                            for c in (slope_column, "Gap", "GapVariance"))
    logit = slope * gap / torch.sqrt(1 + math.pi * slope ** 2 * variance / 8)
    if inputs == "model":
        return logit[:, None]
    quali_gap = rows["QualiGapPct"]
    usable = (quali_gap.abs() <= model.MAX_BELIEVABLE_GAP).fill_null(False)
    residual = torch.tensor(pl.select(pl.when(usable).then(-quali_gap - rows["Gap"]).otherwise(0.0)).to_series().to_list(),
                            dtype=torch.float64)
    ahead = rows["AAheadQuali"].cast(pl.Float64) * 2 - 1
    sign = torch.tensor(pl.select(pl.when(~usable & ahead.is_not_null()).then(ahead).otherwise(0.0)).to_series().to_list(),
                        dtype=torch.float64)
    return torch.stack([logit, residual, sign], dim=1)


def fit_weights(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Logistic regression weights (no intercept: A and B are interchangeable), lightly regularised."""
    weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([weights], max_iter=200, line_search_fn="strong_wolfe")

    def step():
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weights, y) + 1e-4 * (weights ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(step)
    return weights.detach()


def second_step(predictions: pl.DataFrame, variant: str, target: str, use_quali: bool) -> pl.DataFrame:
    """Final probabilities for 2010 to 2025, each season's weights learned from earlier seasons only."""
    slope_column = "RaceSlope" if target == "AAheadRace" else "QualiSlope"
    rows = predictions.filter((pl.col("Variant") == variant) & pl.col(target).is_not_null())
    out = []
    for season in range(FIRST_REPORTED, LAST_REPORTED + 1):
        train = rows.filter(pl.col("Year") < season)
        test = rows.filter(pl.col("Year") == season)
        weights = fit_weights(features(train, slope_column, use_quali), torch.tensor(train[target].cast(pl.Float64).to_list()))
        chance = torch.sigmoid(features(test, slope_column, use_quali) @ weights)
        out.append(test.with_columns(pl.Series("Chance", chance.tolist())))
    return pl.concat(out)


def scores(rows: pl.DataFrame, target: str) -> dict:
    chance, ahead = rows["Chance"].to_list(), rows[target].to_list()
    return {"comparisons": rows.height, "accuracy": accuracy(chance, ahead),
            "brier": brier(chance, ahead), "log_loss": log_loss(chance, ahead)}


# --- Checks and output ---------------------------------------------------------

def check_no_future(predictions: pl.DataFrame):
    """Stop if any fair prediction was made by a fit that saw its own race or a later one."""
    fair = predictions.filter(pl.col("Variant") != "ceiling")
    leaked = fair.filter((pl.col("SeenYear") * 100 + pl.col("SeenRound")) >= (pl.col("Year") * 100 + pl.col("Round")))
    if leaked.height:
        raise RuntimeError(f"{leaked.height} predictions saw their own race or later: {leaked.head(3)}")
    print(f"Leakage check passed: all {fair.height:,} fair predictions were made only from earlier races.")


def hindsight_share(predictions: pl.DataFrame, target: str) -> pl.DataFrame:
    """The loosest reference: for each pair of teammates in each season, the
    share of that season's results driver A really won, used as A's chance in
    every one of their races. It knows the season's answers, and it is the best
    any model can do if it gives a pair one fixed chance per season (as the
    skill model does); only race-by-race information could beat it."""
    rows = predictions.filter((pl.col("Variant") == "current") & pl.col(target).is_not_null()
                              & pl.col("Year").is_between(FIRST_REPORTED, LAST_REPORTED))
    share = pl.col(target).cast(pl.Float64).mean().over("Year", "DriverA", "DriverB")
    return rows.with_columns(share.alias("Chance"))


def build_table(predictions: pl.DataFrame) -> list[dict]:
    table = []
    for label, variant, use_quali in ROWS:
        race = second_step(predictions, variant, "AAheadRace", use_quali)
        quali = second_step(predictions, variant, "AAheadQuali", "model")
        table.append({"label": label, "variant": variant, "inputs": use_quali,
                      "race": scores(race, "AAheadRace"),
                      # Qualifying cannot be predicted from qualifying itself.
                      "quali": None if use_quali == "quali only" else scores(quali, "AAheadQuali"),
                      "raceBySeason": {str(y): scores(race.filter(pl.col("Year") == y), "AAheadRace")["accuracy"]
                                       for y in range(FIRST_REPORTED, LAST_REPORTED + 1)}})
    race, quali = hindsight_share(predictions, "AAheadRace"), hindsight_share(predictions, "AAheadQuali")
    table.append({"label": "Hindsight pair-season share (loosest reference)", "variant": "current",
                  "inputs": "hindsight share", "race": scores(race, "AAheadRace"), "quali": scores(quali, "AAheadQuali"),
                  "raceBySeason": {str(y): scores(race.filter(pl.col("Year") == y), "AAheadRace")["accuracy"]
                                   for y in range(FIRST_REPORTED, LAST_REPORTED + 1)}})
    return table


def print_table(table: list[dict]):
    print(f"\nPooled {FIRST_REPORTED} to {LAST_REPORTED}. Lower Brier and log loss are better.")
    print(f"{'':<40} {'race':>7} {'Brier':>7} {'logloss':>8}   {'quali':>7} {'Brier':>7} {'logloss':>8}")
    for row in table:
        r, q = row["race"], row["quali"]
        quali_text = "      not applicable" if q is None else \
            f"{q['accuracy'] * 100:6.1f}% {q['brier']:7.4f} {q['log_loss']:8.4f}"
        print(f"{row['label']:<40} {r['accuracy'] * 100:6.1f}% {r['brier']:7.4f} {r['log_loss']:8.4f}   {quali_text}")
        best = max(r["accuracy"], 0 if q is None else q["accuracy"])
        if "indsight" not in row["label"] and best > LEAKAGE_ALARM:
            print(f"   ^ above {LEAKAGE_ALARM:.0%}: check for leakage before believing this")


def leakage_report(predictions: pl.DataFrame, table: list[dict]):
    """Extra checks for any fair result above 75%, season by season.

    1. In each such season, does the simple rule "whoever qualified ahead
       finishes ahead" score about as high? If so, qualifying was just very
       predictive that year.
    2. Shuffle test: give every comparison the qualifying result of the same
       pair's PREVIOUS race instead of its own. If the gain came from a leak
       somewhere else, it would survive; if it comes from same-weekend
       qualifying, it should vanish.
    """
    combined = second_step(predictions, "in-season + experience", "AAheadRace", "model+quali")
    rule = second_step(predictions, "current", "AAheadRace", "quali only")
    print("\nLeakage checks for fair results above 75%:")
    for row in table:
        if "indsight" in row["label"]:
            continue
        for season, share in row["raceBySeason"].items():
            if share > LEAKAGE_ALARM and row["label"] == "All combined":
                rule_share = scores(rule.filter(pl.col("Year") == int(season)), "AAheadRace")["accuracy"]
                print(f"  {season}: all combined {share * 100:.1f}%, whoever-qualified-ahead rule {rule_share * 100:.1f}%")
    rows = predictions.filter(pl.col("Variant") == "in-season + experience").sort("Year", "DriverA", "DriverB", "Round")
    shifted = rows.with_columns(
        pl.col("QualiGapPct").shift(1).over("Year", "DriverA", "DriverB"),
        pl.col("AAheadQuali").shift(1).over("Year", "DriverA", "DriverB"),
        pl.lit("shuffled").alias("Variant"))
    shuffled = scores(second_step(shifted, "shuffled", "AAheadRace", "model+quali"), "AAheadRace")["accuracy"]
    real = scores(combined, "AAheadRace")["accuracy"]
    print(f"  Shuffle test: with the previous race's qualifying instead of this weekend's, all combined scores "
          f"{shuffled * 100:.1f}% (with the real qualifying {real * 100:.1f}%).")
    return {"shuffled": shuffled, "real": real}


def draw_chart(table: list[dict]):
    figure, axes = chart_style.new_figure(width=10, height=4.2)
    years = list(range(FIRST_REPORTED, LAST_REPORTED + 1))
    lines = [("Current model", chart_style.BASELINE, 1.6), ("All combined", chart_style.BLUE, 2.2),
             ("Ceiling (hindsight, not a fair model)", chart_style.ORANGE, 1.6)]
    by_label = {row["label"]: row for row in table}
    for label, colour, width in lines:
        axes.plot(years, [by_label[label]["raceBySeason"][str(y)] * 100 for y in years],
                  marker="o", markersize=4, color=colour, linewidth=width, label=label)
    axes.axhline(50, color=chart_style.SECOND_INK, linewidth=0.8, linestyle=":")
    axes.set_ylim(40, 90)
    axes.set_xticks(years, [str(y)[2:] for y in years])
    axes.set_ylabel("teammate race result called correctly (%)")
    axes.legend(frameon=False, fontsize=8, loc="lower right")
    chart_style.tidy(axes, "How much room there is: current, improved, and the hindsight ceiling", grid_axis="y")
    chart_style.save(figure, CHART_FILE)


def main():
    pairs, pace = model.load_tables()
    if "--summarise" in sys.argv:
        predictions = pl.read_parquet(PREDICTIONS_FILE)
    else:
        done = pl.read_parquet(PREDICTIONS_FILE) if PREDICTIONS_FILE.exists() else None
        parts = [] if done is None else [done]
        for name in FITS:
            if done is not None and name in done["Variant"].unique().to_list():
                continue  # already predicted in an earlier, interrupted run
            parts.append(run_fit(name, pairs, pace))
            pl.concat(parts, how="diagonal_relaxed").write_parquet(PREDICTIONS_FILE)  # save as we go
        predictions = pl.concat(parts, how="diagonal_relaxed")
    check_no_future(predictions)
    table = build_table(predictions)
    print_table(table)
    leakage = leakage_report(predictions, table)
    draw_chart(table)
    JSON_FILE.write_text(json.dumps({"firstSeason": FIRST_REPORTED, "lastSeason": LAST_REPORTED, "table": table,
                                     "shuffleTest": leakage}, indent=1), encoding="utf-8")
    print(f"Saved {JSON_FILE}")


if __name__ == "__main__":
    main()
