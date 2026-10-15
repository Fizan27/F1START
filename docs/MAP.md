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


 6. wet_races.py           tags 54 wet races (laps since 2018, a hand list before)
        |
        v
 data/wet_races.parquet


 Phase 2: the model
 ------------------

 teammate_pairs.parquet + race_pace.parquet (+ wet_races.parquet)
        |
        v
 model.py                  the model itself: skills per driver per season,
        |                  with uncertainty. Used by every script below.
        |
        +--> 7. backtest.py        every season 2010 to 2025 predicted from earlier
        |          |               seasons; tests the optional parts; fits the
        |          |               calibration temperature
        |          v
        |    docs/backtest.png, web/public/data/backtest.json
        |
        +--> 7b. backtest_improvements.py  every race 2010 to 2025 from earlier
        |          |               races only: hindsight ceiling, in-season
        |          |               updating, qualifying signal, experience curve
        |          v
        |    docs/improvements.png, web/public/data/improvements.json
        |
        +--> 8. validate_model.py  fit up to 2023, predict 2024 and 2025,
        |          |               compare with baselines, calibration before/after
        |          v
        |    docs/validation.png, web/public/data/validation.json
        |
        +--> 9. sample_posterior.py  NUTS samples of every unknown (checks Laplace)
        |          |
        |          v
        |    data/posterior_samples.pt
        |
        +--> 10. fit_ratings.py    fit on everything, for the final ratings
        |          |               (peak ranges from the samples; wet rating
        |          |               from a separate fit)
        |          v
        |    data/ratings.parquet            one row per driver
        |    data/skill_by_season.parquet    the career curves
        |    data/teammate_records.parquet   record against each teammate
        |    data/car_strengths.parquet      each team's car, each season
        |    data/peak_covariance.pt         rating uncertainty and peak samples
        |
        +--> 11. explain_ratings.py  which battles moved each rating, chain to today
        |          v
        |    web/public/data/explain.json
        |
        +--> 12. sensitivity.py    top 20 under three era assumptions
                   v
             web/public/data/sensitivity.json

 compare_ratings.py        old against new top 20, big movers, championship changes


 Phase 3: the simulation
 -----------------------

 simulate.py               equal car races, thousands at once on the GPU.
        |                  Run on its own it prints the calibration check.
        v
 13. export_web.py         reads the rating files, results and laps
        |
        v
 web/public/data/drivers.json        every driver: rating, range, rank, wet
 web/public/data/details.json        career curves, top teammate records
 web/public/data/records.json        every teammate record (for the assistant)
 web/public/data/h2h.json            head to head chances
 web/public/data/championships.json  every season in equal cars
 web/public/data/settings.json       the simulation's numbers and each
                                     circuit's overtaking difficulty

 14. check_simulator.py    grid vs finish, wins from pole/5th/15th, fastest lap


 Phase 4: the website (everything in web/)
 -----------------------------------------

 web/public/data/*.json    written by the Python scripts above
        |
        v
 lib/data.ts               loads the JSON files and describes their shape
        |
        +--> app/page.tsx                the fantasy race (home page)
        |        uses lib/sim.ts         one race, simulated in the browser:
        |                                the same rules and settings as simulate.py
        +--> app/head-to-head/page.tsx   any two drivers, chance and range
        +--> app/ranking/page.tsx        all time ranking with range bars
        +--> app/drivers/page.tsx        driver card: career curve, "Why this
        |                                rating", wet rating, teammates
        +--> app/seasons/page.tsx        every season replayed in equal cars
        +--> app/about/page.tsx          method, backtest, calibration, sensitivity

 app/layout.tsx            the frame round every page (Nav and the Pit Wall button)
 app/globals.css           every style, in one file
 components/DriverPicker.tsx   the driver search box used on three pages


 The Pit Wall assistant (also in web/)
 -------------------------------------

 components/PitWall.tsx    the chat panel, on every page
        |  POST /api/chat
        v
 app/api/chat/route.ts     the only server code: reads ANTHROPIC_API_KEY
        |
        v
 lib/chatHandler.ts        checks the key, the question and the rate limit
        |                  (lib/rateLimit.ts)
        v
 lib/pitwall.ts            asks Claude, runs the tools it asks for, then checks
        |                  every number in the answer (lib/numbers.ts)
        v
 lib/tools.ts              the seven tools, reading lib/modelData.ts
                           (the JSON files above, bundled with the server)

 eval/questions.json + eval/run-eval.ts   the 50 question evaluation
 tests/*.test.ts                          the website tests (npm test)

 To see it:   cd web ; npm install ; npm run dev   then open localhost:3000
 To build it: cd web ; npm run build
```

Run the numbered scripts in that order. The download scripts can be stopped
and run again; anything already downloaded is skipped.

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
| 2010 to 2023 | also predicted one season at a time by the backtest, which fits the test temperature |
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
| `export_web.py` | Simulates and writes the website's data | rating files, results, laps | six JSON files in `web/public/data/` |
| `wet_races.py` | Tags wet races | `data/raw/`, `data/results.parquet` | `data/wet_races.parquet` |
| `backtest.py` | Rolling backtest 2010 to 2025, model options, temperature | via `model.py` | `docs/backtest.png`, `web/public/data/backtest.json` |
| `backtest_improvements.py` | Hindsight ceiling and three improvements, race by race, no future data | via `model.py` | `docs/improvements.png`, `web/public/data/improvements.json` |
| `sample_posterior.py` | NUTS posterior samples (checks Laplace) | via `model.py` | `data/posterior_samples.pt` |
| `explain_ratings.py` | Why each driver has their rating | via `model.py`, `fit_ratings.py` | `web/public/data/explain.json` |
| `sensitivity.py` | Top 20 under three era assumptions | via `model.py` | `web/public/data/sensitivity.json` |
| `check_simulator.py` | Simulator statistics (Interlagos test) | `web/public/data/` | a printed table |
| `compare_ratings.py` | Old against new ratings and championships | saved copies in `data/` | a printed table |
| `chart_style.py` | Shared chart colours | nothing | nothing |
| `test_*.py` | Tests, one file per script | nothing | pass/fail |
| `requirements.txt` | Python libraries | | |
| `pytest.ini` | Tells pytest which folders to skip | | |
| `CLAUDE.md` | The rules for how this project is built | | |
| `docs/DECISIONS.md` | Each significant decision and why | | |
| `docs/INTERVIEW_GUIDE.md` | How to explain the project, and likely questions | | |
| `docs/DEPLOY.md` | Step by step Vercel deployment, including the API key | | |
| `web/` | The website and the assistant (see the diagram above) | `web/public/data/` | the site |

## Folders

- `data/` is downloaded and cleaned data. Not in git (large; rebuilt by the scripts).
- `web/` is the website. `web/public/data/` holds the small JSON files it
  reads, which are in git because the website needs them.
- `archive/` is the earlier strategy simulator project. Nothing uses it.
- `.venv/` is the Python environment. Not in git.
