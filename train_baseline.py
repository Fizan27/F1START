# train_baseline.py
#
# What it does: trains the BASELINE lap time model (gradient boosting with
#   LightGBM) on the 2022 to 2023 seasons and scores it on 2024. The baseline
#   is the simple, strong model that the neural network must later beat.
# What it reads: data/laps.parquet (made by clean_data.py).
# What it produces: tables printed to the screen: the overall error, and the
#   error per tyre compound and per circuit.
# Which files use it: none yet. Its scores are the bar for the neural network.
#
# Run:  .venv\Scripts\python.exe train_baseline.py
#
# This is a TEACHING file. Four places are marked TODO(human) for you to
# write, each with hints. Do them in order: 1, 2, 3, 4.

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

LAPS_FILE = Path("data/laps.parquet")

# What the model predicts: lap time as percent slower than pole.
TARGET = "LapTimePct"

# Beyond about 10 seconds there is no car close enough to matter, so every
# bigger gap is treated the same. This also tames the huge gaps after red flags.
GAP_CAP_SECONDS = 10.0

# The fixed list of tyre types, so "SOFT" means the same thing in every table.
COMPOUNDS = pl.Enum(["SOFT", "MEDIUM", "HARD"])

# TODO(human) 1: choose the model's inputs (its "features").
#   Fill this list with column names from data/laps.parquet, as text in quotes.
#   Ask of each column: "is this known BEFORE the lap is driven, and could it
#   change how fast the lap is?"
#   Candidates that pass that test:
#     "Compound", "TyreLife", "LapsRemaining", "GapAhead", "QualiGapPct",
#     "TrackTemp", "AirTemp"
#   Columns that must NOT go in, and why:
#     "LapTime", "Sector1Time" ...  they ARE the answer (or parts of it)
#     "Position", "FinishPosition"  decided by the lap times, so they leak
#     "Driver", "Team", "Circuit"   left out on purpose: DECISIONS.md 10 and 11
FEATURES = [
    "Compound", "TyreLife", "LapsRemaining", "GapAhead", "QualiGapPct",
    "TrackTemp", "AirTemp"
]


def load_clean_laps() -> pl.DataFrame:
    """Load the clean racing laps the model is allowed to see."""
    laps = pl.read_parquet(LAPS_FILE)
    # 2025 is the final exam. Dropping it here, right at the door, means
    # nothing in this file can peek at it by accident.
    laps = laps.filter(pl.col("Split") != "test")
    laps = laps.filter(pl.col("IsCleanLap"))
    return laps.with_columns(
        pl.col("GapAhead").clip(upper_bound=GAP_CAP_SECONDS),
        pl.col("Compound").cast(COMPOUNDS),
    )


def split_by_season(laps: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train laps, validation laps)."""
    # TODO(human) 2: make the two tables using the "Split" column.
    #   The column holds the text "train" or "validation" on every row.
    #   Keeping only some rows of a table looks like this:
    #       fast_laps = laps.filter(pl.col("LapTime") < 80)
    #   and "equal to" is written with two equals signs:  ==
    #   Why split by season instead of at random? Laps from the same race are
    #   near copies of each other. A random split would put some laps of a race
    #   in training and the rest in validation, and the score would look far
    #   better than the model really is on a race it has never seen.
    train = laps.filter(pl.col("Split") == "train")
    validation = laps.filter(pl.col("Split") == "validation")
    return train, validation


def to_inputs(laps: pl.DataFrame):
    """Pick out the feature columns in the form LightGBM expects."""
    return laps.select(FEATURES).to_pandas()


def train_model(train: pl.DataFrame) -> lgb.LGBMRegressor:
    """Fit the gradient boosting model on the training laps."""
    # TODO(human) 3: create the model, then fit it.
    #   Step A, create it:
    #       model = lgb.LGBMRegressor(n_estimators=..., learning_rate=...,
    #                                 random_state=0, verbose=-1)
    #     n_estimators  = how many small trees to build. Try 500.
    #     learning_rate = how big a correction each new tree makes. Try 0.05.
    #     (Many small careful steps usually beat a few big ones.)
    #     random_state=0 makes the result the same on every run.
    #   Step B, fit it. It needs the inputs and the right answers:
    #       model.fit(inputs, answers)
    #     inputs  come from to_inputs(train)
    #     answers are the target column:  train[TARGET].to_numpy()
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                 random_state=0, verbose=-1)
    model.fit(to_inputs(train), train[TARGET].to_numpy())
    return model


def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """The average size of the miss, ignoring whether it was high or low."""
    # TODO(human) 4: return the mean absolute error, in one line.
    #   1. the miss on each lap:        predicted - actual
    #   2. make every miss positive:    np.abs(...)
    #   3. average them:                np.mean(...)
    #   Why absolute? Without it, a lap predicted 1 too high and a lap 1 too
    #   low would cancel to zero and a bad model would look perfect.
    return np.mean(np.abs(predicted - actual))


def add_predictions(model, laps: pl.DataFrame) -> pl.DataFrame:
    """Add the model's prediction and its error to every lap."""
    predicted = model.predict(to_inputs(laps))
    error_pct = pl.col("Predicted") - pl.col(TARGET)
    return laps.with_columns(pl.Series("Predicted", predicted)).with_columns(
        error_pct.alias("ErrorPct"),
        # Percent of pole is hard to picture, so also give the error in
        # seconds: a percent of a 90 second pole lap is 0.9 seconds.
        (error_pct * pl.col("PoleTime") / 100).alias("ErrorSeconds"),
    )


def error_table(laps: pl.DataFrame, group: str) -> pl.DataFrame:
    """Average error for each value of one column (for example each circuit)."""
    return laps.group_by(group).agg(
        pl.len().alias("laps"),
        pl.col("ErrorSeconds").abs().mean().round(3).alias("typical_miss_s"),
        # Bias: is the model too slow (+) or too fast (-) on average here?
        pl.col("ErrorSeconds").mean().round(3).alias("bias_s"),
    ).sort("typical_miss_s")


def print_report(train: pl.DataFrame, validation: pl.DataFrame):
    actual = validation[TARGET].to_numpy()
    # The "know-nothing" model: always guess the average training lap. Any
    # real model must beat this, or it has learned nothing.
    know_nothing = np.full(len(actual), train[TARGET].mean())
    seconds_per_pct = validation["PoleTime"].mean() / 100

    print("\nOverall, on 2024 (validation):")
    for name, predicted in [
        ("always guess the average", know_nothing),
        ("LightGBM baseline", validation["Predicted"].to_numpy()),
    ]:
        miss = mean_absolute_error(predicted, actual)
        print(f"  {name:<26} typical miss {miss:.3f}% of pole"
              f"  (about {miss * seconds_per_pct:.2f}s)")

    print("\nBy tyre compound:")
    print(error_table(validation, "Compound"))

    seen = train["Circuit"].unique()
    by_circuit = error_table(validation, "Circuit").with_columns(
        pl.col("Circuit").is_in(seen.implode()).alias("in_training")
    )
    print("\nBy circuit (best first):")
    with pl.Config(tbl_rows=30):
        print(by_circuit)


def main():
    if not FEATURES:
        raise SystemExit("TODO(human) 1 is not done yet: FEATURES is empty.")
    laps = load_clean_laps()
    train, validation = split_by_season(laps)
    print(f"Training on {train.height:,} laps, validating on {validation.height:,} laps.")
    model = train_model(train)
    validation = add_predictions(model, validation)
    print_report(train, validation)


if __name__ == "__main__":
    main()
