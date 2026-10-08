# F1-EQUALIZER

If every Formula 1 driver in history drove the same car, who would be fastest?

F1 results mix two things: how good the driver is and how good the car is.
This project separates them with a statistical model built on teammate
comparisons (teammates share the same car), linked across eras through shared
teammates. It then simulates races where every car is identical, and a website
lets you race drivers from any era against each other.

**Status: in progress.** Phase 1 (data) is done. There are no model results
yet, so this README makes no claims about any driver.

| Phase | What | Status |
|---|---|---|
| 1 | Data: results since 1950, lap times since 2018, circuit outlines | done |
| 2 | The model: driver skill separated from car performance, validated on 2024 and 2025 | not started |
| 3 | The simulation: equal car races on the GPU | not started |
| 4 | The website (Next.js, free on Vercel) | not started |

## Data

Public data only:

- [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (the successor to
  Ergast): every race and qualifying result since 1950.
- [FastF1](https://github.com/theOehrly/Fast-F1): lap times and track
  coordinates for 2018 onwards.

What the cleaned data holds (downloaded 2026-10-09, printed by
`clean_history.py` and `clean_laps.py`):

| Decade | Seasons | Races | Drivers | Starts | Car failure | Driver error | Teammate pairs | Pairs with a comparable race |
|---|---|---|---|---|---|---|---|---|
| 1950s | 10 | 84 | 307 | 1,834 | 36.6% | 8.0% | 2,809 | 1,120 |
| 1960s | 10 | 100 | 192 | 1,899 | 40.1% | 6.5% | 1,906 | 698 |
| 1970s | 10 | 144 | 153 | 3,427 | 32.7% | 12.3% | 2,222 | 937 |
| 1980s | 10 | 156 | 103 | 3,920 | 39.6% | 12.8% | 1,686 | 662 |
| 1990s | 10 | 162 | 96 | 3,860 | 30.6% | 17.3% | 1,844 | 940 |
| 2000s | 10 | 174 | 71 | 3,611 | 19.0% | 11.5% | 1,798 | 1,185 |
| 2010s | 10 | 198 | 66 | 4,265 | 10.2% | 7.3% | 2,118 | 1,674 |
| 2020s | 7 | 147 | 40 | 2,945 | 3.0% | 3.0% | 1,460 | 1,165 |
| **All** | **77** | **1,165** | **789** | **25,761** | | | **15,843** | **8,381** |

- 643 of the 655 drivers who ever had a teammate are linked to each other
  through chains of shared teammates. That link is what allows drivers from
  different eras to be compared.
- Lap times: 163,011 clean racing laps from 183 races since 2018.
- Track outlines: 33 circuits.

Known weaknesses of the data (details in
[docs/DECISIONS.md](docs/DECISIONS.md), number 14):

- From 2023 the source no longer says why a car retired, so those races are
  left out of teammate comparisons. A driver's own crashes therefore stop
  counting against them from 2023, which includes both test seasons. (This
  is why the 2020s row shows only 3% for each cause.)
- Before 1990 cars broke down in more than a third of starts, so well under
  half of teammate pairs give a usable race result.
- Before 1970 a "constructor" often means a car maker that sold to private
  owners, so some "teammates" did not have equal cars.
- "Collision" counts as a driver error even when the other driver caused it.
- Qualifying lap times only exist from 1994, and are patchy from 1996 to 2002.

## How to run it

```
.venv\Scripts\python.exe check_gpu.py           # once: proves PyTorch trains on the GPU
.venv\Scripts\python.exe -m pytest              # the tests

.venv\Scripts\python.exe download_history.py    # 1. results since 1950 (about 15 minutes)
.venv\Scripts\python.exe clean_history.py       # 2. results and teammate pairs
.venv\Scripts\python.exe download_laps.py       # 3. lap times since 2018 (several hours)
.venv\Scripts\python.exe clean_laps.py          # 4. pace and consistency per race
.venv\Scripts\python.exe download_circuits.py   # 5. track outlines (about 10 minutes)
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
