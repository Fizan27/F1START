# Decisions

Each significant decision and why it was made. Newest at the bottom.

The decisions of the earlier strategy simulator project are in
`archive/docs/DECISIONS.md`. The three about the machine setup still apply and
are repeated here as numbers 1 to 3.

## 1. PyTorch 2.14.1 built for CUDA 13.0 (`cu130`)

The RTX 5070 Ti is an RTX 50 series card (architecture name `sm_120`). PyTorch
builds made for CUDA older than 12.8 do not include `sm_120`, so they install
without complaint but cannot run on this GPU. `check_gpu.py` proves the
installed build works by training a small network on the GPU and the CPU.

## 2. Python 3.12 in a plain virtual environment (`.venv`)

A plain `venv` plus `pip` is the most standard setup, with nothing extra to
learn.

## 3. Flat layout

Python files live in the top folder, with tests next to them as `test_*.py`.
Only `docs/`, `data/` (downloaded, not committed), `web/` (the website) and
`archive/` (the earlier project) are separate.

## 4. The strategy simulator was archived, not deleted (2026-10-09)

The owner changed the project to F1-EQUALIZER. Everything from the strategy
simulator moved to `archive/` in one commit, so the history stays readable and
the old results are still there. Kept in the top folder because the new
project uses them: the Python environment, `check_gpu.py`, `download_data.py`
(the FastF1 lap download), `chart_style.py` and all downloaded data.

The old `clean_data.py` was archived because most of it built inputs for a lap
time model (fuel, traffic, weather). The parts the new project needs (reading
the raw files, track status flags, the "clean lap" rule) are carried over into
the new lap cleaning script.

The old `requirements.txt` was for the Streamlit app, which no longer exists.
There is now one `requirements.txt`, for the Python side.
