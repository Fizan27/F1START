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

## 5. Two data sources, each used for what it is best at

- **Jolpica** (`download_history.py`): race results since 1950 and qualifying
  results since 1994, which is as far back as its qualifying records go.
- **FastF1** (`download_laps.py`): race lap times since 2018, the first season
  it has. The 2022 to 2025 races were already downloaded for the earlier
  project and are reused.

One change from the original brief: qualifying times come from Jolpica, not
from FastF1 qualifying sessions. They are the same official Q1, Q2 and Q3
times, Jolpica has them for 24 more seasons, and it avoids downloading about
190 extra sessions from a free service.

The 2026 season is in progress and is downloaded up to the latest race. It is
never used for validation (that is 2024 and 2025), only for the final ratings.

## 6. Downloads are saved exactly as received, and are polite

`download_history.py` saves each page the service sends, unchanged, and
cleaning is a separate script. A cleaning mistake then never means downloading
again. The Jolpica service is free and run by volunteers, and allows 4
requests a second and 500 an hour. The script sends one request every half
second, and when the service answers "too many requests" it waits two minutes
and carries on. Pages already saved are skipped, so it can be stopped and
restarted.

## 7. Retirements are sorted into five groups

The data has about 135 different finishing statuses. `classify_status` in
`clean_history.py` puts each into one group:

| Group | Examples | Says something about the driver? |
|---|---|---|
| `finished` | Finished, +1 Lap, Lapped | yes |
| `driver` | Accident, Collision, Spun off | yes, it counts as a bad result |
| `car` | Engine, Gearbox, Hydraulics, about 100 more | no |
| `other` | Disqualified, Puncture, Illness, plain "Retired" | no |
| `not_started` | Did not qualify, Withdrew | no |

- Anything not listed by name falls into `car`, because nearly every
  remaining status is a car part.
- A plain "Retired" with no reason goes to `other`. Guessing it as a crash
  would punish drivers for breakdowns; guessing it as a breakdown would
  excuse crashes. Neither is honest, so those races are simply not used.
- Position text W (withdrew) or F (failed to qualify) means `not_started`
  whatever the status says, because the problem happened in practice.

Known weakness: "Collision" counts against a driver even when another driver
hit them. The data does not say who was at fault.

## 8. Teammates: same team, same race, every pairing

Two drivers are teammates in a race if both started it for the same
constructor. A team with three cars gives three pairs. Each pair in each race
is one row of `data/teammate_pairs.parquet`.

Known weakness, mostly before 1970: the data names the constructor (who built
the car), not the team that ran it. Private owners bought cars from Maserati,
Cooper and Lotus, so in 1956 "Maserati" has up to a dozen cars in a race, and
they were not prepared equally. Each pair row carries `TeamCars` (how many
cars the constructor had in that race) so the model can trust these pairs
less. This is one reason uncertainty should be wider for older eras.

The Indianapolis 500 (part of the championship from 1950 to 1960) is left out
of teammate pairs. Its drivers almost never raced in Europe, and its
"constructors" were chassis makers selling to many separate teams.

When a driver appears twice in one race (in the 1950s a driver could take over
a teammate's car), only their best result is kept.

## 9. A race only counts for a pair when neither car let its driver down

`compare_race` says who was ahead only if both teammates either finished or
went out through their own crash or spin. If either car broke down, or either
result is in the `other` group, the pair is marked not comparable for that
race. This is how car caused retirements are kept from counting against a
driver. A driver's own crash does count, as finishing behind the teammate.

The cost: a driver who was leading their teammate when the teammate's engine
failed gets no credit for it. That is accepted, because nobody knows how that
race would have ended.

## 10. Results are made comparable across eras without using points

Points are not comparable: a win paid 8 points in 1950, 10 in 1991 and 25
since 2010, and seasons have grown from 7 races to 24. So the cleaned data
adds two columns and the model will use finishing order, not points.

- `PositionScore`: 1.0 for the winner down to 0.0 for the last starter. Fields
  have had anywhere from 10 to 34 starters.
- `ModernPoints`: what today's 25-18-15-12-10-8-6-4-2-1 system would have paid.
  Used only for the "more career points wins" baseline in Phase 2, so that
  baseline is not handicapped by old points systems.

## 11. Qualifying: lap times from 1994, grid position before that

Qualifying is a cleaner comparison of teammates than the race (no strategy,
no traffic, far fewer breakdowns), so it is recorded for every pair.

- From 1994 the lap times exist. The gap between teammates is measured in the
  last part of qualifying (Q3, then Q2, then Q1) that both set a time in,
  because the track gets faster during the session and comparing a Q3 lap
  with a Q1 lap would be unfair.
- Before 1994 only the starting grid exists. It is used as the qualifying
  position (`QualiSource` = "grid"); grid penalties were rare then.

Known weakness: from 2003 to 2009 parts of qualifying were run with race fuel
on board, so teammates on different strategies set different times for
reasons that were not skill.

## 12. Lap pace is measured against the field on the same lap number

`clean_laps.py` compares each clean lap with the median clean lap of all cars
on the same lap of the same race, in percent. Fuel burning off and the track
gaining grip make every car faster as the race goes on; comparing within one
lap number removes both. The "clean lap" rule is carried over from the earlier
project (green flag, not a pit lap, not lap 1, not on wet tyres).

Per driver per race it then keeps:

- `PacePct`: the median of those percentages. It still contains the car, so
  it only means something as a difference between teammates.
- `LapSpreadPct`: how much the driver's laps scatter around their own median,
  measured with the median absolute deviation, which a single spin cannot
  inflate the way a standard deviation would.
- `SlowLapShare`: the share of clean laps more than 1% off their own median.

Known weakness: tyre age and compound are not removed, so a driver on an
unusual strategy looks faster or slower, and less consistent, than they were.
Over a season this averages out between teammates; in one race it does not.

## 13. Circuit outlines: one recorded lap, 240 evenly spaced points

`download_circuits.py` takes the car positions FastF1 recorded over the
fastest lap of the most recent race at each circuit, and resamples them to 240
points the same distance apart, scaled to a 1000 x 1000 square. Evenly spaced
points mean the website can place a car that is 37% of the way round a lap at
point 0.37 x 240 without any maths. Each file is about 2 KB.

Only circuits raced since 2018 have an outline, because that is where FastF1's
position data starts. Fantasy races on the website are limited to those
circuits. The most recent layout is used when a circuit has changed.
