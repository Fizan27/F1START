# Map of the project

How the files connect, and the order things run in.

## Run order

```
 one-off setup
 -------------
 check_gpu.py              proves PyTorch trains on the GPU


 Phase 1: race model
 -------------------
 the internet (FastF1)
        |
        v
 download_data.py          fetch every race of 2022 to 2025
        |
        v
 data/raw/                 laps, weather and results per race, qualifying
        |
        v
 clean_data.py             one clean table with the models' inputs
        |
        v
 data/laps.parquet         one row per driver per lap. Everything below
        |                  reads only this file.
        |
        +--> train_baseline.py    LightGBM baseline, scored on 2024
        |
        +--> train_network.py     the main model: a PyTorch network on the
        |          |              GPU that predicts a range of lap times
        |          v
        |    models/lap_time_network.pt
        |
        +--> race_stats.py        pit loss per circuit, safety car and VSC
                   |              likelihood, first lap spread
                   v
             models/race_stats.json


 The simulator (used by everything below)
 ----------------------------------------
 simulator.py              races lap by lap, thousands at once on the GPU.
                           Loads the network and the statistics above.
        |
        +--> replay_validation.py   Phase 1 check: replay real races with
        |                           real strategies, compare with reality
        |
        |    Phase 2
        +--> strategy_search.py     brute force: best fixed strategy
        |          |
        |          v
        |    results/benchmark_2024.parquet
        |
        +--> strategy_env.py        the race as a game for a strategist
                   |                (Gymnasium style reset / step)
                   |
                   |   Phase 3
                   +--> train_agent.py        PPO strategist, on the GPU
                   |          |
                   |          v
                   |    models/strategist.pt
                   |
                   +--> evaluate_strategist.py   strategist against real
                              |                  teams and brute force
                              v
                        results/strategist_2024.parquet


 The final exam (run once)
 -------------------------
 final_test.py             retrains everything on 2022 to 2024 and scores it
        |                  on 2025. Calls the files above.
        v
 models/final/             network, statistics and strategist
 results/*_2025.parquet    2025 results
 app_data/races.pkl        the 2024 and 2025 races, for the website


 Phase 4: website
 ----------------
 app.py                    Streamlit app. Reads app_data/, models/,
                           models/final/ and results/. Needs no GPU.
```

## The season split (never mix these up)

| Seasons | Label in `Split` column | Job |
|---|---|---|
| 2022, 2023 | `train` | models learn from these |
| 2024 | `validation` | all tuning and model choices |
| 2025 | `test` | final numbers only, touched once, by `final_test.py` |

The website shows each season using models that never saw it: 2024 with the
models in `models/` (trained on 2022 to 2023), 2025 with `models/final/`
(trained on 2022 to 2024).

## Lap time is split in two (DECISIONS.md 12)

- `RacePacePct`: the race's typical lap, percent slower than pole. One number
  per race, only known after the race. Never a model input.
- `PaceVsRacePct`: how a lap differs from that. The lap time models' target.

A simulated lap is: pole lap x (1 + (race pace + network prediction + driver
offset + luck) / 100), plus pit loss, traffic and safety car rules.

## Files

| File | What it is for | Reads | Produces |
|---|---|---|---|
| `check_gpu.py` | Proves training runs on the GPU | nothing | a printed report |
| `download_data.py` | Downloads race data | FastF1 (internet) | `data/raw/` |
| `clean_data.py` | Builds the one clean lap table | `data/raw/` | `data/laps.parquet` |
| `train_baseline.py` | LightGBM baseline lap time model | `data/laps.parquet` | printed tables |
| `train_network.py` | Network that predicts a range | `data/laps.parquet`, uses `train_baseline.py` | `models/lap_time_network.pt` |
| `race_stats.py` | Pit loss, safety car statistics | `data/laps.parquet` | `models/race_stats.json` |
| `simulator.py` | The race simulator | the two model files, `data/laps.parquet` | nothing (used by others) |
| `replay_validation.py` | Replays real races to check the simulator | via `simulator.py` | tables, `docs/replay_validation.png` |
| `strategy_search.py` | Brute force best fixed strategy | via `simulator.py` | `results/benchmark_2024.parquet` |
| `strategy_env.py` | The race as a Gymnasium style environment | via `simulator.py` | nothing (used by others) |
| `train_agent.py` | Trains the PPO strategist | via `strategy_env.py` | `models/strategist.pt` |
| `evaluate_strategist.py` | Tests the strategist, looks for exploits | strategist, benchmark | `results/strategist_2024.parquet`, chart |
| `final_test.py` | The one-off 2025 test | everything | `models/final/`, 2025 results, `app_data/` |
| `app.py` | The website | `app_data/`, `models/`, `results/` | web pages |
| `chart_style.py` | Shared chart colours | nothing | nothing |
| `test_*.py` | Tests, one file per part | models only | pass/fail |
| `requirements.txt` | Libraries for the website (CPU) | | |
| `requirements-dev.txt` | Libraries for training (GPU machine) | | |
| `CLAUDE.md` | The rules for how this project is built | | |
| `docs/DECISIONS.md` | Each significant decision and why | | |
| `docs/INTERVIEW_GUIDE.md` | How each part works, likely questions | | |

## Folders

- `data/` is downloaded data. Not in git (large; rebuild with `download_data.py`).
- `models/`, `results/` and `app_data/` are small and are in git, because the
  website needs them.
- `.venv/` is the Python environment. Not in git.
