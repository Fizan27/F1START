# F1-EQUALIZER

If every Formula 1 driver in history drove the same car, who would be fastest?

F1 results mix two things: how good the driver is and how good the car is.
This project separates them with a statistical model built on teammate
comparisons (teammates share the same car), linked across eras through shared
teammates. It then simulates races where every car is identical, and a website
lets you race drivers from any era against each other.

**Status: in progress.** Phase 1 (data) is being built. There are no model
results yet, so this README makes no claims about any driver.

| Phase | What | Status |
|---|---|---|
| 1 | Data: results since 1950, lap times since 2018, circuit outlines | in progress |
| 2 | The model: driver skill separated from car performance, validated on 2024 and 2025 | not started |
| 3 | The simulation: equal car races on the GPU | not started |
| 4 | The website (Next.js, free on Vercel) | not started |

## Data

Public data only:

- [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (the successor to
  Ergast): every race and qualifying result since 1950.
- [FastF1](https://github.com/theOehrly/Fast-F1): lap times and track
  coordinates for 2018 onwards.

## How to run it

```
.venv\Scripts\python.exe check_gpu.py     # once: proves PyTorch trains on the GPU
.venv\Scripts\python.exe -m pytest        # the tests
```

The steps of each phase are listed in [docs/MAP.md](docs/MAP.md), in the order
they run. Each decision and its reason is in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Results

None yet. This section will hold the validation results (teammate prediction
accuracy on 2024 and 2025 against simple baselines), measured by scripts in
this repository, including where the model is weak.

## The earlier project

This repository used to hold an F1 race strategy simulator. It is kept in
[archive/](archive/README.md).
