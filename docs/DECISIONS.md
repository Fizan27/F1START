# Decisions

Each significant decision and why it was made. Newest at the bottom.

## 1. PyTorch 2.14.1 built for CUDA 13.0 (`cu130`)

The RTX 5070 Ti is an RTX 50 series card (architecture name `sm_120`). PyTorch
builds made for CUDA older than 12.8 do not include `sm_120`, so they install
without complaint but cannot run on this GPU. The installed driver (616.92)
supports CUDA up to 13.4, so the current stable `cu130` build was chosen.
`check_gpu.py` proves it: it trains a small network on the GPU and the CPU and
compares them (measured: loss 62.3 to 0.39 on the GPU, 14x faster than the CPU).

## 2. Python 3.12 in a plain virtual environment (`.venv`)

Python 3.12 has ready-made packages for every library used here. A plain
`venv` plus `pip` is the most standard setup, with nothing extra to learn.

## 3. Flat layout: code files in the top folder

The project is small. One folder means there is never a question about where a
file lives. Only `docs/` (documents) and `data/` (downloaded, not committed)
are separate. Tests sit next to the code as `test_*.py`.

## 4. Seasons: train 2022 to 2023, validate on 2024, test on 2025

All four seasons use the same car rules (the 2022 to 2025 "ground effect" era),
so tyre and lap time behaviour is comparable. They are split by time into
three parts, each with one job:

- **Train (2022 to 2023):** the models learn from these races.
- **Validation (2024):** used for every tuning and model choice (network size,
  learning rate, which inputs to keep, baseline versus network).
- **Test (2025):** touched only once, at the end, for the reported numbers.

Why three parts and not two: every time a choice is made by looking at a
score, that score becomes a little optimistic, because the choice was fitted to
it. If 2025 were used for tuning, the final numbers would flatter the model.
Keeping 2025 untouched means the reported error is an honest estimate of how
the model does on races it has truly never influenced.

Before the final 2025 evaluation the chosen model is retrained on 2022 to 2024
with the settings already fixed, so it uses all the data available before the
test season, as a real team would.

2026 brought completely new cars and engines, so it is left out of version
one. Future work: use it as a "rule change" robustness test, to measure how
much the model degrades when the cars change.

## 5. Save raw data as one Parquet file per race

`download_data.py` saves laps, weather and results for each race separately in
`data/raw/`. If one race fails to download, the others are unaffected and the
script can simply be run again: it skips races already saved. Parquet keeps
column types (unlike CSV) and Polars reads it very fast. Times are stored as
plain seconds because they are easier to do arithmetic on than time objects.

## 6. Fuel is represented by laps remaining

Real fuel load is not public. Cars start with enough fuel for the race and
burn it at a roughly steady rate, so `LapsRemaining` (total laps minus lap
number) falls in step with fuel weight. Limitation: it cannot see fuel saving
or a car that started under-fuelled.

## 7. Traffic is the gap to the car ahead on the road, at the start of the lap

`GapAhead` uses the car physically in front, even if it is a lap down, because
dirty air affects pace regardless of race position. It is measured at the
start of the lap (the end of the previous lap). The gap at the end of a lap is
partly decided by that lap's own time, so using it as an input would hand the
model part of the answer (this is called target leakage).

## 8. "Clean laps" are separated from everything else

The lap time model learns only from laps marked `IsCleanLap`: a recorded and
accurate lap time, green track, not a pit in or pit out lap, not lap 1, on a
dry tyre with a known age. About 79% of 2022 to 2024 laps qualify. The
excluded laps are slow for reasons the lap time model should not explain (pit
lane, safety car, standing start, rain); the simulator handles those
separately with pit loss and safety car models.

Known and still open: about 0.5% of clean laps are more than 7% slower than
the race's typical lap (mistakes, damage, a drying track). Whether and how to
filter them is decided in the modelling step, using 2024 validation scores.
