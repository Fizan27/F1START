# Map of the project

How the files connect, and the order things run in. Updated after every step.

## Run order

```
 one-off setup
 -------------
 check_gpu.py              proves PyTorch trains on the GPU


 Phase 1: data
 -------------

 Jolpica F1 API (internet)                 FastF1 (internet)
        |                                         |
        v                                         v
 1. download_history.py                    3. download_laps.py
        |                                         |
        v                                         v
 data/jolpica/*.json                       data/raw/*_laps.parquet
 every result since 1950,                  every lap of every race
 qualifying since 1994                     since 2018
        |                                         |
        v                                         |
 2. clean_history.py                              |
        |                                         |
        +--> data/results.parquet ----------------+
        |    one row per driver per race          |
        |          |                              v
        |          |                       4. clean_laps.py
        |          |                              |
        |          |                              v
        |          |                       data/race_pace.parquet
        |          |                       one row per driver per race
        |          |                       since 2018: pace, consistency
        |          v
        |    5. download_circuits.py  <--- FastF1 (internet)
        |          |
        |          v
        |    web/public/data/circuits/*.json
        |    one track outline per circuit, for the website
        v
 data/teammate_pairs.parquet
 one row per pair of teammates per race:
 who was ahead in the race and in qualifying


 Phase 2: the model
 ------------------

 teammate_pairs.parquet + race_pace.parquet
        |
        v
 model.py                  the model itself: skills per driver per season,
        |                  with uncertainty. Used by the two scripts below.
        |
        +--> 6. validate_model.py   fit up to 2023, predict 2024 and 2025,
        |          |                compare with baselines
        |          v
        |    docs/validation.png, web/public/data/validation.json
        |
        +--> 7. fit_ratings.py      fit on everything, for the final ratings
                   |
                   v
             data/ratings.parquet            one row per driver
             data/skill_by_season.parquet    the career curves
             data/teammate_records.parquet   record against each teammate
             data/car_strengths.parquet      each team's car, each season
             data/peak_covariance.pt         uncertainty of the ratings


 Phase 3: the simulation
 -----------------------

 simulate.py               equal car races, thousands at once on the GPU.
        |                  Run on its own it prints the calibration check.
        v
 8. export_web.py          reads the five rating files, results and laps
        |
        v
 web/public/data/drivers.json        every driver: rating, range, rank
 web/public/data/details.json        career curves, teammate records
 web/public/data/h2h.json            head to head chances
 web/public/data/championships.json  every season in equal cars
 web/public/data/settings.json       the simulation's numbers and each
                                     circuit's overtaking difficulty


 Phase 4: the website      (not started)   will read web/public/data/
```

Run the five numbered scripts in that order. Each can be stopped and run
again; anything already downloaded is skipped.

## The three tables the model will learn from

| Table | One row is | Covers | Key columns |
|---|---|---|---|
| `data/results.parquet` | one driver in one race | 1950 on | `Outcome`, `Position`, `PositionScore`, `ModernPoints`, `QualiPosition`, `Teammates` |
| `data/teammate_pairs.parquet` | two teammates in one race | 1950 on | `AAheadRace` (empty when a car failed), `AAheadQuali`, `QualiGapPct`, `TeamCars` |
| `data/race_pace.parquet` | one driver in one race | 2018 on | `PacePct`, `LapSpreadPct`, `SlowLapShare`, `CleanLaps` |

Drivers are named by `DriverId` (for example `hamilton`) and teams by `TeamId`
(for example `mercedes`) in all three, so the tables join on
`Year`, `Round`, `DriverId`.

## The time split (never mix these up)

| Seasons | Job |
|---|---|
| 1950 to 2023 | the model is fitted on these; any tuning happens inside them |
| 2024, 2025 | test only: predicted without refitting |
| 2026 (in progress) | not used for validation; joins the final refit for the website |

## Files

| File | What it is for | Reads | Produces |
|---|---|---|---|
| `check_gpu.py` | Proves training runs on the GPU | nothing | a printed report |
| `download_history.py` | Downloads results and qualifying since 1950 | Jolpica (internet) | `data/jolpica/` |
| `clean_history.py` | Classifies retirements, pairs up teammates | `data/jolpica/` | `data/results.parquet`, `data/teammate_pairs.parquet` |
| `download_laps.py` | Downloads race lap times since 2018 | FastF1 (internet) | `data/raw/` |
| `clean_laps.py` | Pace and consistency per driver per race | `data/raw/`, `data/results.parquet` | `data/race_pace.parquet` |
| `download_circuits.py` | Track outlines for the website | `data/results.parquet`, FastF1 | `web/public/data/circuits/` |
| `model.py` | The skill model (fit, uncertainty, predictions) | the pair and pace tables | nothing (used by others) |
| `validate_model.py` | Tests the model on 2024 and 2025 | via `model.py`, `data/results.parquet` | `docs/validation.png`, `web/public/data/validation.json` |
| `fit_ratings.py` | Final ratings from all data | via `model.py`, `data/results.parquet` | five files in `data/` |
| `simulate.py` | The equal car race simulation | nothing | nothing (used by others) |
| `export_web.py` | Simulates and writes the website's data | rating files, results, laps | five JSON files in `web/public/data/` |
| `chart_style.py` | Shared chart colours | nothing | nothing |
| `test_*.py` | Tests, one file per script | nothing | pass/fail |
| `requirements.txt` | Python libraries | | |
| `pytest.ini` | Tells pytest which folders to skip | | |
| `CLAUDE.md` | The rules for how this project is built | | |
| `docs/DECISIONS.md` | Each significant decision and why | | |

## Folders

- `data/` is downloaded and cleaned data. Not in git (large; rebuilt by the scripts).
- `web/` is the website. `web/public/data/` holds the small JSON files it
  reads, which are in git because the website needs them.
- `archive/` is the earlier strategy simulator project. Nothing uses it.
- `.venv/` is the Python environment. Not in git.
