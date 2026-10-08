# train_baseline.py
#
# What it does: trains the BASELINE lap time model (gradient boosting with
#   LightGBM) on the 2022 to 2023 seasons and scores it on 2024. The baseline
#   is the simple, strong model that the neural network must later beat.
#   It tries a few choices of inputs side by side, so the choice is made by
#   measurement on 2024 and not by opinion.
# What it reads: data/laps.parquet (made by clean_data.py).
# What it produces: tables printed to the screen: a comparison of the input
#   choices, the error per tyre compound and per circuit for the chosen one,
#   and how well the race pace level can be forecast before a race.
# Which files use it: none yet. Its scores are the bar for the neural network.
#
# Run:  .venv\Scripts\python.exe train_baseline.py
#
# The big idea (docs/DECISIONS.md, number 12): lap time is split in two.
#   1. Race pace level: how fast the whole field runs in this race.
#   2. Pace within the race: how this lap differs from the race's typical lap.
# The model learns part 2, because that is what strategy depends on.

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

LAPS_FILE = Path("data/laps.parquet")
RACE = ["Year", "Round"]

# What the model predicts: how much slower (+) or faster (-) than the race's
# typical lap this lap is, in percent of the pole lap.
TARGET = "PaceVsRacePct"

# Beyond about 10 seconds there is no car close enough to matter, so every
# bigger gap is treated the same. This also tames the huge gaps after red flags.
GAP_CAP_SECONDS = 10.0

# Laps this much slower than the race's typical lap are spins, damage or a
# damp track: about 1 lap in 100. They are left out of TRAINING, because the
# model cannot predict them and they pull its other predictions off. They are
# kept in the SCORING, because hiding them would flatter the model.
OUTLIER_PCT = 5.0

# In wet qualifying the pole lap is slow, so the race looks "faster than
# pole". Races below this level are left out of the circuit averages.
WET_QUALIFYING_LEVEL = 3.0

# The fixed list of tyre types, so "SOFT" means the same thing in every table.
COMPOUNDS = pl.Enum(["SOFT", "MEDIUM", "HARD"])

# Each input is known BEFORE the lap is driven and could change its pace.
BASIC_INPUTS = ["Compound", "TyreLife", "LapsRemaining", "GapAhead", "QualiGapPct"]

# The input choices to compare. Each is tried and scored on 2024.
VARIANTS = {
    "basic inputs": BASIC_INPUTS,
    "basic + temperatures": BASIC_INPUTS + ["TrackTemp", "AirTemp"],
    "basic + circuit": BASIC_INPUTS + ["Circuit"],
}

# The variant used for the detailed tables. Chosen from the comparison.
CHOSEN = "basic + circuit"


def load_clean_laps() -> pl.DataFrame:
    """Load the clean racing laps the model is allowed to see."""
    laps = pl.read_parquet(LAPS_FILE)
    # 2025 is the final exam. Dropping it here, right at the door, means
    # nothing in this file can peek at it by accident.
    laps = laps.filter(pl.col("Split") != "test")
    laps = laps.filter(pl.col("IsCleanLap"))
    circuits = pl.Enum(sorted(laps["Circuit"].unique().to_list()))
    return laps.with_columns(
        pl.col("GapAhead").clip(upper_bound=GAP_CAP_SECONDS),
        pl.col("Compound").cast(COMPOUNDS),
        pl.col("Circuit").cast(circuits),
    )


def split_by_season(laps: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train laps, validation laps)."""
    # Split by season, not at random: laps from one race are near copies of
    # each other, so a random split would let the model see part of every
    # validation race and the score would look far better than it really is.
    train = laps.filter(pl.col("Split") == "train")
    validation = laps.filter(pl.col("Split") == "validation")
    return train, validation


def to_inputs(laps: pl.DataFrame, inputs: list[str]):
    """Pick out the input columns in the form LightGBM expects."""
    return laps.select(inputs).to_pandas()


def train_model(train: pl.DataFrame, inputs: list[str]) -> lgb.LGBMRegressor:
    """Fit the gradient boosting model on the training laps."""
    train = train.filter(pl.col(TARGET) < OUTLIER_PCT)
    # 500 small trees, each making a small (0.05) correction to the ones
    # before. random_state=0 makes the result the same on every run.
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                              random_state=0, verbose=-1)
    model.fit(to_inputs(train, inputs), train[TARGET].to_numpy())
    return model


def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """The average size of the miss, ignoring whether it was high or low."""
    # Absolute, so that a lap predicted 1 too high and a lap 1 too low do not
    # cancel to zero and make a bad model look perfect.
    return np.mean(np.abs(predicted - actual))


def add_predictions(laps: pl.DataFrame, predicted) -> pl.DataFrame:
    """Add a prediction for every lap, and its error in seconds."""
    return laps.with_columns(pl.Series("Predicted", predicted)).with_columns(
        # Percent of pole is hard to picture, so the error is given in
        # seconds: one percent of a 90 second pole lap is 0.9 seconds.
        ((pl.col("Predicted") - pl.col(TARGET)) * pl.col("PoleTime") / 100)
        .alias("ErrorSeconds")
    )


def score(name: str, scored: pl.DataFrame) -> dict:
    """One row of the comparison table for one way of predicting."""
    errors = scored["ErrorSeconds"].to_numpy()
    nothing = np.zeros(len(errors))
    shanghai = scored.filter(pl.col("Circuit") == "Shanghai")["ErrorSeconds"].to_numpy()
    return {
        "model": name,
        "mean_miss_s": round(mean_absolute_error(errors, nothing), 3),
        # The median is the miss on a typical lap. The mean is pulled up by
        # the few very bad laps, so the two together show both stories.
        "median_miss_s": round(float(np.median(np.abs(errors))), 3),
        "shanghai_mean_miss_s": round(mean_absolute_error(shanghai, 0), 3),
    }


def compare_variants(train: pl.DataFrame, validation: pl.DataFrame):
    """Train every variant. Returns (comparison table, laps scored by CHOSEN)."""
    # The "know-nothing" model: every lap is the race's typical lap (0).
    # Any real model must beat this, or it has learned nothing.
    rows = [score("know-nothing (typical lap)",
                  add_predictions(validation, np.zeros(validation.height)))]
    chosen_laps = None
    for name, inputs in VARIANTS.items():
        model = train_model(train, inputs)
        scored = add_predictions(validation, model.predict(to_inputs(validation, inputs)))
        rows.append(score(name, scored))
        if name == CHOSEN:
            chosen_laps = scored
    return pl.DataFrame(rows), chosen_laps


def error_table(laps: pl.DataFrame, group: str) -> pl.DataFrame:
    """Average error for each value of one column (for example each circuit)."""
    return laps.group_by(group).agg(
        pl.len().alias("laps"),
        pl.col("ErrorSeconds").abs().mean().round(3).alias("mean_miss_s"),
        pl.col("ErrorSeconds").abs().median().round(3).alias("median_miss_s"),
        # Bias: is the model too slow (+) or too fast (-) on average here?
        pl.col("ErrorSeconds").mean().round(3).alias("bias_s"),
    ).sort("mean_miss_s")


def race_level_forecast(train: pl.DataFrame, validation: pl.DataFrame) -> pl.DataFrame:
    """Forecast each 2024 race's pace level using only earlier seasons.

    The forecast is the circuit's average level in 2022 to 2023, or the
    average of all circuits when the circuit is new. This is the part of lap
    time the lap model does NOT predict, so its error is reported separately.
    """
    def one_row_per_race(laps):
        return laps.group_by(RACE + ["Circuit"]).agg(
            pl.col("RacePacePct").first(), pl.col("PoleTime").first()
        )

    past = one_row_per_race(train).filter(pl.col("RacePacePct") > WET_QUALIFYING_LEVEL)
    by_circuit = past.group_by("Circuit").agg(pl.col("RacePacePct").mean().alias("Forecast"))
    forecast = pl.col("Forecast").fill_null(past["RacePacePct"].mean())
    return (
        one_row_per_race(validation)
        .join(by_circuit, on="Circuit", how="left")
        .with_columns(forecast)
        .with_columns(
            ((pl.col("Forecast") - pl.col("RacePacePct")) * pl.col("PoleTime") / 100)
            .round(2).alias("error_s_per_lap")
        )
        .sort("Round")
    )


def print_report(comparison, chosen_laps, train, levels):
    print("\n1. Pace within the race, on 2024 (validation). Miss in seconds per lap:")
    print(comparison)

    print(f"\n2. '{CHOSEN}' by tyre compound:")
    print(error_table(chosen_laps, "Compound"))

    seen = train["Circuit"].unique().cast(pl.String)
    by_circuit = error_table(chosen_laps, "Circuit").with_columns(
        pl.col("Circuit").cast(pl.String).is_in(seen.implode()).alias("in_training")
    )
    print(f"\n3. '{CHOSEN}' by circuit (best first):")
    with pl.Config(tbl_rows=30):
        print(by_circuit)

    misses = levels["error_s_per_lap"].abs()
    print("\n4. Race pace level, forecast before the race from earlier seasons:")
    print(f"   typical miss (median) {misses.median():.2f}s per lap,"
          f" mean {misses.mean():.2f}s per lap. The five worst races:")
    print(levels.sort(misses, descending=True).head(5).select(
        "Circuit", pl.col("RacePacePct").round(2), pl.col("Forecast").round(2),
        "error_s_per_lap",
    ))


def main():
    laps = load_clean_laps()
    train, validation = split_by_season(laps)
    print(f"Training on {train.height:,} laps, validating on {validation.height:,} laps.")
    comparison, chosen_laps = compare_variants(train, validation)
    levels = race_level_forecast(train, validation)
    print_report(comparison, chosen_laps, train, levels)


if __name__ == "__main__":
    main()
