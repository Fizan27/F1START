# Map of the project

How the files connect, and the order things run in. Updated every time a file
is added.

## Run order

```
 one-off setup
 -------------
 check_gpu.py            proves PyTorch trains on the GPU (reads nothing)


 Phase 1: race model
 -------------------
 the internet (FastF1)
        |
        v
 download_data.py        step 1: fetch every race of 2022 to 2025
        |
        v
 data/raw/               three Parquet files per race:
                         laps, weather, results
        |
        v
 clean_data.py           step 2: stack the races, work out the model's inputs
        |                (fuel proxy, gap ahead, weather, safety car flags,
        |                clean lap flag, train/validation/test label)
        v
 data/laps.parquet       ONE table, one row per driver per lap.
                         Everything after this reads only this file.
        |
        v
 baseline model          step 3 (NEXT, not built yet): LightGBM lap time
                         baseline, trained on 2022-23, scored on 2024
```

## The season split (never mix these up)

| Seasons | Label in `Split` column | Job |
|---|---|---|
| 2022, 2023 | `train` | models learn from these |
| 2024 | `validation` | all tuning and model choices |
| 2025 | `test` | final numbers only, touched once at the end |

## Files

| File | What it is for | Reads | Produces |
|---|---|---|---|
| `check_gpu.py` | Proves training runs on the GPU | nothing | a printed report |
| `download_data.py` | Downloads race data | FastF1 (internet) | `data/raw/*.parquet` |
| `test_download_data.py` | Tests for `download_data.py` | nothing | pass/fail |
| `clean_data.py` | Builds the one clean lap table | `data/raw/*.parquet` | `data/laps.parquet` |
| `test_clean_data.py` | Tests for `clean_data.py` | nothing | pass/fail |
| `requirements.txt` | The libraries to install | | |
| `CLAUDE.md` | The rules for how this project is built | | |
| `docs/MAP.md` | This map | | |
| `docs/DECISIONS.md` | Each significant decision and why | | |

## Folders

- `data/` is downloaded data. It is not committed to git (too big, and anyone
  can rebuild it by running `download_data.py`).
- `.venv/` is the Python environment. Also not committed.
