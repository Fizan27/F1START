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

Result: 33 circuits, 78 KB in total. All 33 were drawn on one sheet and checked
by eye. Two problems turned up and are handled in the script:

- Some races have no usable position data (Monaco 2026 has none; in Budapest
  2026 the feed froze for seconds at a time and drew the track as a polygon).
  The script rejects a lap with fewer than 150 different positions and falls
  back to the race before it at the same circuit.
- FastF1 has no map details for the two newest circuits (Madrid and the
  returning Sepang), so those two are not turned to the usual TV orientation.

## 14. What the cleaned data showed, and the weaknesses it exposed

Measured by `clean_history.py` on data downloaded 2026-10-09 (1950 to round 16
of 2026): 1,165 races, 789 drivers who started a race, 15,843 teammate pairs
across 2,828 different pairings.

- **The eras are linked.** 643 of the 655 drivers who ever had a teammate are
  in one connected group: there is a chain of shared teammates between any two
  of them. This is what makes comparing Fangio with Verstappen possible at
  all. The other 12 cannot be rated against the rest, and 134 drivers never
  had a teammate.
- **Retirement reasons stop in 2023.** From 2023 the source records only
  "Retired", with no reason (53 of 2023's retirements, and every one in 2024,
  2025 and 2026). FastF1's results have the same gap. Under decision 7 these
  are `other`, so those races are not comparable for that pair. The effect:
  from 2023 a driver's own crashes no longer count against them in race
  comparisons. This covers both test seasons, and it favours crash-prone
  current drivers slightly. It is reported, not patched: guessing reasons from
  other sources would be inventing data. Qualifying comparisons are unaffected.
- **Reliability changed enormously.** Car caused retirements were 37% of
  starts in the 1950s, 40% in the 1980s and 3% in the 2020s. So only about 40%
  of teammate pairs before 1990 give a comparable race result, against about
  80% since 2010. Older drivers have less evidence per race, which is a real
  reason for their uncertainty to be wider.
- **Many-car constructors.** 4,121 pairs come from a constructor with more
  than three cars in the race, 3,518 of them before 1970 (decision 8).
- **Qualifying times are patchy from 1996 to 2002.** Between 6% and 59% of
  results in those seasons have a lap time; from 2003 it is 94% or more.
  Qualifying position (or grid) exists throughout.
- **Test seasons.** 2024 and 2025 have 472 teammate pairs: 376 with a
  comparable race result and 451 with a qualifying gap.

Lap data, measured by `clean_laps.py`: 3,538 driver races from 183 races
(2018 to round 16 of 2026), built from 163,011 clean laps. Every driver code
matched a driver in the results history.

- Six races since 2018 have no lap pace. FastF1 has no lap data for Monza
  2018 (the download fails every time). The other five were wet from start to
  finish, so no lap passes the clean lap rule: Turkey 2020 and 2021, Spa 2021,
  Suzuka 2022 and São Paulo 2024.
- A typical driver's laps scatter by about 0.5% (roughly 0.45 seconds) around
  their own median. That number is larger than pure driver inconsistency,
  because tyre wear and traffic are still in it (decision 12).

## 15. The model: skills fitted to teammate comparisons, uncertainty from the curvature

The choice was between three kinds of model.

- **Elo style ratings** (update after each race): simple, but gives no real
  uncertainty and cannot go back and revise 1985 in the light of 1990.
- **Full Bayesian sampling (MCMC)**: the gold standard for uncertainty, but a
  black box library, slow, and hard to explain line by line.
- **Chosen: a Bayesian hierarchical model fitted by optimisation, with the
  uncertainty from a Laplace approximation.** All in about 150 lines of
  PyTorch in `model.py`, and every line can be explained.

How it works:

1. Every driver gets one unknown skill number per season (2,447 of them up to
   2023). Skill is in percent of lap time.
2. *Prior belief*: a first season is near the level of that era's newcomers
   (give or take the "skill spread"); each later season is near the one
   before (give or take the "season drift"). This is what makes it
   hierarchical: drivers are assumed to come from one common population, so a
   driver with three races is pulled towards average.
3. *Likelihood*: for each teammate pair in each race, only the skill gap
   matters. A bigger gap means a bigger expected lap time gap and a higher
   chance of being ahead (a logistic curve).
4. *Fit*: find the skills that make prior x likelihood largest (L-BFGS on the
   GPU, about 15 seconds).
5. *Uncertainty*: the second derivative (Hessian) of that function at the
   best point says how sharply the fit worsens as each skill moves. Its
   inverse is the covariance of all the skills. This is the Laplace
   approximation.

Why only teammates: any comparison between drivers in different cars needs a
model of the cars. Teammate comparisons do not, so the driver ratings do not
depend on getting the cars right. The car is estimated afterwards (number 19).

Limits of the approach: the Laplace approximation assumes the uncertainty is
bell shaped and treats the five learned scale numbers as exactly known, so the
ranges are somewhat too narrow. Calibration on the test seasons is reported in
the README as a check.

## 16. What counts as evidence, and the scale problem that had to be fixed

Four kinds of result feed the model (`build_comparisons`):

| Result | Seasons | How it is read |
|---|---|---|
| Who finished ahead in the race | all | logistic curve, slope learned |
| Qualifying lap time gap | 1994 on | expected gap = skill gap, heavy tailed noise |
| Who qualified ahead | where there is no usable lap time | logistic curve, slope tied (below) |
| Race pace gap | 2018 on | expected gap = skill gap x learned factor |

- A qualifying lap is never counted twice: where the time gap is used, "who
  was ahead" is not.
- Lap time gaps over 3% are treated as a ruined lap and only "who was ahead"
  is kept. The noise is heavy tailed (Student t) so one freak lap cannot
  drag a rating.
- Each driver's comparisons in one race are weighted to add up to 1, so a
  1950s constructor with ten cars does not count nine times.
- Qualifying before 2006 gets its own, larger noise level (learned: 0.65%
  against 0.39% from 2006), because one-lap and race-fuel formats made
  teammate gaps noisier.

**The flaw found and fixed.** In the first version the slope for "who
qualified ahead" was learned freely. It came out at 17, which says a 0.1%
skill gap means qualifying ahead 85% of the time, while the lap time data
says 58%. The cause: before 1994 there are no lap times, so nothing fixed the
size of a skill unit there. The prior prefers small skills, so the optimiser
shrank every pre-1994 skill and raised the slope to compensate. The visible
symptom was a ranking with Hülkenberg above Senna. The fix: the slope is no
longer free. It is tied to the qualifying noise (slope = 1.5 / noise) so that
a skill gap means the same chance of out-qualifying a teammate in every era.

## 17. Era levels: ratings are relative to today, and older eras are less certain

Teammate chains link the eras, but they cannot say for sure whether the
average newcomer of 1955 was as good as the average newcomer of 2015. So the
model has one extra unknown per decade: the level of that decade's newcomers.
The 2020s level is fixed at 0 (something must be the reference), and each
earlier decade may differ from the next by a typical 0.1% (`ERA_DRIFT`).

This is how uncertainty becomes honestly wider for older eras: the further
back, the more decade-to-decade steps separate a driver from the reference,
and each step adds uncertainty. Measured in the final fit, the typical
uncertainty (standard deviation) of a peak rating is 0.30% for drivers who
peaked in the 1950s against 0.16% in the 2020s.

`ERA_DRIFT` = 0.1 is an assumption, not a measurement. Nothing in the data
can pin it down. A larger value would widen every old driver's range.

## 18. The two settings were chosen on 2022 and 2023, and the choice barely matters

Skill spread and season drift cannot be learned together with the skills (the
optimiser would shrink them to zero), so they were chosen by fitting up to
2021 and scoring predictions on 2022 and 2023 (`validate_model.py --tune`).
2024 and 2025 were not involved.

Season drift below 0.12 was clearly worse. Above that the surface is flat:
the settings with drift 0.18 or 0.25 are within about 0.01 of each other in
log loss, less than the noise from about 900 comparisons. Chosen: skill spread 0.3, season drift 0.18.
Spread 0.2 scored 0.005 better and was not chosen, because it pulls drivers
with short careers harder towards average for no measurable gain. The 2024
and 2025 test was run once, with 0.3 and 0.18.

## 19. Car strength is what is left after taking the driver out

`car_strengths` treats a race result (PositionScore, 1 = win, 0 = last) as
car + weight x driver skill. The weight is estimated from teammates only
(within one team and season the car is the same), and the car is the team's
average result minus what its drivers contributed. Measured weight: 0.27, so
0.5% of driver skill is worth about 13% of the field in finishing position.

It is deliberately simple, because the car is not what this project is about.
It is used for the finishing position test and shown on the website.

## 20. Peak, consistency, and what "rating" means

- **Rating = peak skill**: the average skill over a driver's best three
  seasons in a row. Its uncertainty comes exactly from the covariance. Known
  bias: choosing the best window flatters drivers a little, more so for long
  careers and noisy eras (the best of many noisy estimates is partly luck).
- **Consistency (0 to 100, 50 = typical)** averages two measures: how often
  the driver crashed out compared with their era (1950 to 2022, because
  reasons stop in 2023), and from 2018 how steady their lap times were
  compared with the field. Both are mixed with imaginary average races so
  short careers are not extreme. It is a descriptive score, not part of the
  fitted model, and has no uncertainty range.

## 21. Validation results, and what they do and do not show

Fitted up to 2023, predicting 2024 and 2025 without refitting
(`validate_model.py`). Full tables are in the README.

- Which teammate finishes ahead: 67.0% right over 376 comparisons, against
  62.0% for "more career points". Qualifying: 70.1% against 65.0% over 472.
- The whole gain is from 2025. In 2024 nearly every team kept its 2023
  drivers, so "last season repeats" was already as good as anything can be,
  and the model only matched it (65.4% against 66.0%). In 2025 many drivers
  moved or were new, the baselines fell to 57.8% and the model held at 68.6%.
  What the model adds is the ability to compare drivers who have never been
  teammates.
- On pairs who were also teammates the season before, the model is no better
  than "last season repeats" (66.4% against 66.8%).
- Probabilities: too cautious at the top. When the model said 80% or more
  (39 race comparisons) the favourite won all of them.
- Finishing positions from 2023 information: adding the driver to the car
  improves the average miss (3.15 against 3.48 places in 2024). But for 2025
  the model (4.41) is worse than last season's standings (3.61), because it
  still uses 2023 cars and the standings baseline has seen 2024.
- With about 190 race comparisons per season, one season's accuracy has a
  margin of roughly plus or minus 7 percentage points. The 2024 result is a
  tie, not a loss; the 2025 gap is larger than that margin.

What validation cannot show: whether cross-era comparisons are right. There is
no test set for Senna against Verstappen. Only the modern end is checked.

## 22. The simulation: what is measured, what is calibrated, what is assumed

`simulate.py` races identical cars lap by lap. A lap time is the driver's pace
(from skill and the day's form) plus noise, plus an occasional mistake. A
quicker car behind only passes if it is quicker by more than the circuit's
passing margin. All the numbers are in one `SETTINGS` table, which is also
exported to the website so the browser runs exactly the same race.

| Setting | Value | Where it comes from |
|---|---|---|
| Day to day form | 0.33% | **Calibrated** so simulated drivers beat each other as often as the validated model says (table below) |
| Passing margin per circuit | 0.1s to 1.2s | The ORDER of circuits is **measured** (number 23); the two end values are assumed |
| Lap to lap noise | 0.25% | Assumed. Real laps scatter by about 0.5%, but much of that is tyres and traffic |
| Mistakes | 2% of laps, 1.5s average | Assumed |
| Crash chance | 5% per race | Close to the measured rate (driver caused retirements are 3% to 17% of starts depending on the decade) |
| Qualifying noise, grid gap, following gap | 0.20%, 0.25s, 0.4s | Assumed |

A driver's consistency score scales their noise, mistakes and crash chance
together: a score of 75 halves all three, 25 doubles them.

The calibration check (`simulate.py`, 20,000 races per row, 20 car field):

| Skill gap | Model says the better driver finishes ahead | Simulation |
|---|---|---|
| 0.1% | 57.2% | 57.0% |
| 0.2% | 64.1% | 64.1% |
| 0.4% | 76.2% | 76.2% |
| 0.6% | 85.1% | 85.4% |
| 1.0% | 94.8% | 93.3% |

So the simulation is as predictable as real racing between teammates, no more
and no less. What it leaves out: tyres, pit stops, safety cars, weather and
the start. Those mostly add luck, and the form setting stands in for them.

Speed: 20,000 races of 60 laps with 20 cars take under one second on the GPU.

## 23. Overtaking difficulty is measured from real races

For every race since 2018, `passes_per_circuit` counts how often a car gains a
place between one lap and the next, looking only at cars that did not pit on
either lap and comparing them only with each other (so a place gained because
a rival pitted does not count). Circuits are then ranked: fewest passes = 1
(hardest), most = 0. Monaco comes out hardest and Las Vegas easiest, with
Singapore and Budapest near the top, which matches what anyone who watches F1
would expect. It is a ranking, not a physical measurement.

## 24. What is precomputed for the website, and how uncertainty is carried

`export_web.py` draws 4,000 plausible versions of all 655 peak ratings from
their joint uncertainty (the covariance from the model) on the GPU.

- **Rank range**: each driver's rank in every version; the range shown is the
  middle 90%. Ranges are wide (Senna: 2nd to 27th), and that is the honest
  answer. Only Schumacher's 1st to 2nd is narrow.
- **Head to head**: in each version the chance A beats B is the model's curve
  of their skill gap. The best guess is the average over versions and the
  range is the middle 90%. Precomputed for the 213 drivers with at least 30
  starts (22,578 pairs); below that a rating is too uncertain to be worth a
  page.
- **Equal car championships**: every season from 1950 is replayed 1,000 times
  with its real calendar and the drivers who really started each race, using
  each driver's skill in THAT season (not their peak), in identical cars on a
  typical circuit. Points use today's system for both the simulated and the
  real table, so they compare like with like.

Head to heads use the model's curve directly and not a fresh race simulation
per pair, because the simulation is calibrated to that same curve (number
22) and 22,578 separate simulations would add noise without adding
information.

## 25. The website: static Next.js, the race runs in the browser

- **Static export** (`output: "export"`): the build is plain HTML, CSS and
  JavaScript. There is no server and no database, so hosting on Vercel's free
  tier costs nothing and cannot break under load. Every page fetches small
  JSON files from `web/public/data/`, all written by the Python scripts.
- **The race is simulated in the visitor's browser** by `web/lib/sim.ts`, a
  rewrite of `simulate.py` for one race at a time. The settings are not
  copied by hand: they are read from `settings.json`, which `export_web.py`
  writes from `simulate.SETTINGS`. So the two versions cannot drift apart on
  numbers, only on logic, and the logic is short enough to compare by eye.
- **The whole race is computed the moment "Start" is pressed**, then the
  animation plays it back. That keeps the animation code simple (it only
  asks "where is each car at time t?") and makes replay exact.
- Each fantasy race draws a plausible value for every driver's rating from
  its uncertainty, as well as a form for the day. A driver with a wide range
  is therefore less predictable on the site, which is the honest behaviour.
- **Plain CSS in one file, no UI or chart library.** Charts are small
  hand-written SVG. Fewer dependencies to explain and nothing to pay for.
- Twenty cars are told apart by eight colours used three ways (solid, white
  ring, hollow) and, always, by the car number on the dot and in the timing
  tower, so colour is never the only clue.
- A sixth page, Seasons, was added beyond the five in the brief, because the
  equal car championships were already computed and are one of the more
  interesting outputs.
- No images, logos or team colours are used anywhere.

Checked by driving the built site in a headless browser at desktop and phone
sizes: a full race, replay, and every page, with no horizontal scrolling on a
390 pixel wide screen.

One thing changed after watching the first race: with the first passing
margins (0.2s to 1.6s) the field finished in grid order with every gap exactly
0.4s. The margins were lowered to 0.1s to 1.2s, a stuck car now follows between
one and two following gaps behind, and the form setting was recalibrated
(0.32% to 0.33%) so the simulation still matches the model (number 22).
