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
 clean_data.py           step 2 (NEXT, not built yet): one clean table of
                         laps with the model's inputs worked out
```

## Files

| File | What it is for | Reads | Produces |
|---|---|---|---|
| `check_gpu.py` | Proves training runs on the GPU | nothing | a printed report |
| `download_data.py` | Downloads race data | FastF1 (internet) | `data/raw/*.parquet` |
| `test_download_data.py` | Tests for `download_data.py` | nothing | pass/fail |
| `requirements.txt` | The libraries to install | | |
| `CLAUDE.md` | The rules for how this project is built | | |
| `docs/MAP.md` | This map | | |
| `docs/DECISIONS.md` | Each significant decision and why | | |

## Folders

- `data/` is downloaded data. It is not committed to git (too big, and anyone
  can rebuild it by running `download_data.py`).
- `.venv/` is the Python environment. Also not committed.
