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

Nothing measured yet. This section will hold tables and charts for lap time
error, replay accuracy, and the AI strategist against real team strategies,
including where the models are weak.

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
