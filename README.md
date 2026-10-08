# F1START

An F1 race strategy simulator with an AI strategist. It answers two questions:

- What if the team had pitted differently?
- What would an AI strategist have done?

Built from public timing data only, using the [FastF1](https://docs.fastf1.dev/) library.

**Status: work in progress.** Setup and data download are done. Results below
are filled in only with numbers measured by scripts in this repository.

## How it works (four phases)

1. **Race model.** A lap time model (tyre compound, tyre age, fuel, circuit,
   traffic, weather) trained on 2022 to 2023 races, tuned on 2024, and tested
   on 2025 races that are held back until the end. It predicts a range of
   likely lap times, not one number.
2. **Simulator.** A lap by lap race environment that runs thousands of races
   in parallel on the GPU.
3. **AI strategist.** A reinforcement learning agent (PPO) that decides each
   lap whether to stay out or pit, and for which tyre.
4. **Website.** A Streamlit app to replay a race, edit the strategy, and
   compare it with the AI strategist.

See [docs/MAP.md](docs/MAP.md) for how the files connect and
[docs/DECISIONS.md](docs/DECISIONS.md) for why things were done this way.

## Results

All numbers so far are on the 2024 validation season. The 2025 test season
has not been looked at. Reproduce with `train_baseline.py`.

### Lap time baseline (LightGBM), trained on 2022 to 2023

Lap time is split into the race's overall pace level and each lap's pace
within the race ([why](docs/DECISIONS.md), number 12). The model predicts the
second part. Miss in seconds per lap, on 21,354 clean laps from 2024:

| Model | Mean miss | Median miss | Shanghai (unseen circuit), mean |
|---|---|---|---|
| Know-nothing (every lap is the race's typical lap) | 1.085 | 0.894 | 1.041 |
| Basic inputs (compound, tyre age, laps remaining, gap ahead, qualifying gap) | 0.725 | 0.503 | 0.580 |
| Basic + temperatures | 0.783 | 0.556 | 0.739 |
| **Basic + circuit (chosen)** | **0.684** | **0.464** | **0.536** |

By compound for the chosen model, mean miss: hard 0.66s, soft 0.69s, medium 0.72s.

Weaknesses measured so far:

- The first version predicted lap time against the pole lap directly and was
  worse than guessing the average (2.63s against 2.09s).
- Races with changing conditions are predicted badly: Montréal 2.31s,
  Silverstone 1.38s and Monaco 1.27s mean miss, against 0.38s at Miami.
- The race pace level itself is hard to forecast before a race from earlier
  seasons: median miss 0.49s per lap but mean 1.33s, with Spa 2024 off by
  13.5s per lap because qualifying was wet. Replays of past races use the
  real level, so this does not affect them.

Still to come: the neural network with uncertainty, race replay validation,
the simulator, the AI strategist and the app.

## How to run it

Needs Python 3.12. The GPU steps need an NVIDIA GPU; RTX 50 series cards need
a PyTorch build for CUDA 12.8 or newer.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe check_gpu.py        # proves training runs on the GPU
.venv\Scripts\python.exe download_data.py    # downloads 2022 to 2025 races into data/raw/
.venv\Scripts\python.exe -m pytest           # runs the tests
```

## Data

All data comes from the public F1 timing feed through FastF1. This project is
unofficial and is not associated with Formula 1 or any team.
