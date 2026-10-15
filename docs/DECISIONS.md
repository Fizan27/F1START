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

## 22. The simulation: what is measured, what is calibrated, what is assumed (reworked in 33)

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

## 24. What is precomputed for the website, and how uncertainty is carried (rank ranges changed in 26)

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

## 26. Why Hamilton's rank range was 5 to 134 while Schumacher's was 1 to 2

The owner spotted this. Three things combined, and none of them was a fault
in the Laplace approximation itself (number 32 checks that separately):

1. **Ranks were counted among all 655 drivers**, including hundreds with a
   handful of races. Their ratings are close to pure prior: centred near an
   average newcomer, give or take 0.3%. In any one of the 4,000 plausible
   versions of the ratings, dozens of them land by chance above a driver
   rated +0.65%. That pushed the bottom of every mid-table driver's range
   far down, and Hamilton sits in the most crowded part of the list.
2. **Schumacher is an outlier.** His peak (+1.41%) was about 0.47% above
   the next driver, more than twice his own uncertainty, so almost no
   plausible version put anyone above him. A wide range for Hamilton and a
   narrow one for Schumacher were both correct answers to the question that
   was asked; the question (rank among everyone) was the problem.
3. **Hamilton's own uncertainty is larger than his 396 races suggest**,
   because the season drift setting (0.18%) lets each season's skill move,
   so each season is measured mainly from that season's teammate, and his
   peak (2015 to 2017) rests on Rosberg and then Bottas.

Fixes:

- Ranks and rank ranges are now among drivers with at least 30 starts (the
  same drivers as the head to head page). Drivers below that are shown as
  "not ranked": their ratings are mostly guesswork, and saying so is more
  honest than giving them a rank.
- The head to head explanation picked its wording from the years between
  the two drivers' peaks and could say a range was "fairly narrow" because
  they were 21 years apart, which is backwards. It now judges the width
  from the range itself, then lists the real reasons: whether they were
  teammates (which narrows it), how many years separate them (which widens
  it), and whether either driver's own rating is uncertain and why.

## 27. Calibration: one learned temperature, and an honest mixed result

The brief said the model was too cautious, from the 2024 and 2025 table
where favourites given 80% or more won 39 out of 39. To fix this properly
the scale was learned from many seasons, not from that table: every race
probability is now computed with the slope multiplied by a "temperature",
chosen to minimise log loss on the rolling backtest's predictions
(number 29).

- Fitted on 2010 to 2023 predictions only: **0.45**. Below 1 means the
  raw model was too CONFIDENT, the opposite of the brief's starting point.
- On the 2024 and 2025 test, that temperature made things worse: Brier
  score 0.2045 before, 0.2169 after; log loss 0.5939 before, 0.6250 after.
- The reason is drift over time. Fitted on separate stretches of seasons,
  the best temperature was 0.25 for 2010 to 2015, 0.40 for 2016 to 2019,
  0.90 for 2018 to 2023 and 1.10 for 2020 to 2023. Recent seasons need
  almost no softening; older ones need a lot. The 39 out of 39 bucket was
  a real sign of under-confidence in recent seasons, but not a sign that
  the model is under-confident in general.
- What is used, and why: the rule set in advance was to fit on earlier
  seasons, so the 2024 and 2025 test uses 0.45 and its worse result is
  reported as it is (`model.TEST_TEMPERATURE`). After the test, the site
  refits the temperature on every season 2010 to 2025, as the model itself
  is refitted on all data: **0.55** (`model.RACE_TEMPERATURE`). Fitting it on
  recent seasons only would probably serve modern drivers better, but that
  idea was suggested by the test result, so adopting it now would be tuning
  on the test. It is recorded here as the next thing to try.
- Where it applies: to probabilities the site states about real drivers
  (head to heads, the backtest). It does not apply to the race simulator,
  which treats each driver's skill as known and is calibrated to the
  model's fitted slope (number 33).
- Accuracy is unchanged by any temperature: it never changes which driver
  is the favourite, only how sure the model says it is.

## 28. Team orders and weak teammates: both tested, neither kept

Two ways a driver can be flattered by their teammates were each tested in
the rolling backtest (number 29), judged by race log loss pooled over all
2,728 comparisons from 2010 to 2025. A change was kept only if it did not
make that worse.

- **Documented team orders** (`NUMBER_ONE_SEASONS` in model.py): a hand
  made list of team-seasons with a contracted or publicly reported number
  one (Lotus 1978, Williams 1992, Benetton 1994 to 1995, Ferrari 1996 to
  2005 and 2010 to 2013, Red Bull 2021 to 2024). For those pairs the race
  result is left out, because it may have been arranged; qualifying still
  counts. Result: accuracy unchanged (59.6%), log loss slightly worse
  (0.6777 against 0.6765). **Not kept.** A likely reason: future races are
  run under the same team orders, so leaving those results out removes
  information that predicts them.
- **Weak teammates.** The prior assumed every newcomer starts near average,
  which flatters a driver whose teammates had short careers (they were
  probably below average, which is why their careers were short). The fix:
  a driver's expected starting level rises with the logarithm of how many
  seasons they have in the data, by a learned amount (it came out at 0.27).
  Result: accuracy up from 59.6% to 60.5%, but log loss worse (0.6812) and
  qualifying log loss worse too. Better at picking the winner, worse at
  saying how sure it is. **Not kept**, by the rule above. Worth revisiting.
- The two together with the wet adjustment scored 60.0% with log loss
  0.6835, also worse.

The weakness stays in the README: a driver measured only against weak
teammates, or against a number two, can still be flattered.

## 29. The rolling backtest: every season from 2010 to 2025

`backtest.py` fits the model on all seasons before a test season and
predicts that season's teammate results, for each of 16 seasons and each
version of the model. It is a far bigger test than 2024 and 2025 alone:
2,728 race comparisons.

Pooled over all 16 seasons, the chosen (base) model calls 59.6% of race
results correctly, against 59.3% for "last season's head to head repeats,
otherwise more career points" and 57.9% for "more career points" alone.
So over many seasons the model is only slightly better than the best simple
rule. Its advantage is concentrated in seasons with many new pairings: in
2025 it scored 65.9% against 57.8%. Season by season it beat that baseline
in 8 of the 16 seasons, tied in 1 and lost in 7 (worst: 2022, 54.4% against
62.1%). For qualifying it calls 61.3% of 3,407 comparisons, against 59.7%
and 57.5% for the two baselines. The chart is docs/backtest.png, and it is also on
the About page.

The 2025 figure differs from number 21's 68.6% because here 2025 is
predicted from a fit up to 2024, not 2023. Both are honest out-of-sample
numbers for different set-ups.

## 30. Wet weather: tagged races, a separate adjustment, and how little data there is

`wet_races.py` tags 54 wet races: 17 measured since 2018 (at least 10% of
laps on intermediate or wet tyres) and 37 from a hand-made list of well
known wet races before 2018. The list was written from memory of well
documented races; it is incomplete, and races that were wet for only a few
laps are left out on purpose. The script refuses to run if any entry does
not match exactly one race.

The model can add a per-driver wet adjustment (prior: give or take 0.25%,
an assumption) that applies to race results in wet races only. In the
backtest this made predictions slightly worse (log loss 0.6788 against
0.6765), so the main ratings do not use it. The adjustment is fitted
separately and shown on driver pages as an exploratory estimate, with its
range and the number of wet races behind it. Most drivers have fewer than
8 wet races against a teammate, so their adjustment barely moves from zero,
and the page says so.

## 31. Sensitivity: which rankings rest on the era assumption

`sensitivity.py` refits the model with three values of `ERA_DRIFT`, the one
assumption the data cannot settle (how much the level of newcomers may
change per decade): 0.03% ("hardly changed"), 0.10% (the default) and 0.25%
("changed a lot"), and compares the top 20 among drivers with 30+ starts.

A rank counts as "solid" if it moves by at most 3 places across the three.
Solid: Schumacher, Fangio, Senna, Stewart, Alonso, Frentzen, Moss, Häkkinen,
Prost and Mansell. Depends on the assumption: Clark, Rindt, Verstappen,
Ascari, Graham Hill, Surtees, Button, Massa, Hamilton and Gurney. The
pattern is what you would expect: assuming the field hardly changed favours
the 1960s drivers (Clark 2nd, Rindt 4th); assuming it changed a lot favours
modern ones (Verstappen 3rd, Hamilton 14th instead of 27th). McLaren,
Peterson and Hunt enter the top 20 only under "hardly changed"; Rosberg and
Leclerc only under "changed a lot". The table is on the About page.

## 32. Posterior sampling checks the Laplace approximation

`sample_posterior.py` draws 1,000 samples of every unknown (2,522 of them)
plus the five scale numbers with NUTS (the No-U-Turn Sampler, from the
Pyro library), using the model's own badness function, so the sampler
targets exactly the same model. It runs in coordinates where the Laplace
answer would be a plain round bell curve (a standard trick called
preconditioning), which makes it fast: 3.9 minutes on the GPU, with a
median of 1,000 effective samples per skill (lowest 868).

What it found:

- **Skills**: the sampled uncertainty is a median 1.04 times the Laplace
  value (10th to 90th percentile 0.99 to 1.09). The Laplace approximation
  was close, slightly too narrow, as expected (it ignored the uncertainty
  of the scale numbers). The ratings on the site now use the sampled
  ranges.
- **Scale numbers**: the race slope's full posterior is 2.03 to 2.32 (90%
  range, mean 2.17), while the single best fit, which the Laplace step and
  all predictions use, is 2.91. In a model with thousands of unknowns, the
  single most likely combination is not typical of the plausible ones. A
  steeper slope means more confident predictions, which is one likely
  reason the backtest found the model over-confident (number 27). Using
  the sampled slope everywhere would mean re-running the backtest with a
  sampler for every season; it is the most promising next step.
- The rank range changes in number 26 matter far more for the site than the
  switch from Laplace to sampling. Old against new top 20 (from
  `compare_ratings.py`): the same 20 drivers in the same order with the same
  best estimates; ranges a median 1.04 times as wide; rank ranges much
  narrower, for example Hamilton 5 to 134 became 7 to 90 and Fangio 2 to 37
  became 1 to 21. 99 drivers moved more than 10 places, all ranked 115th or
  lower, and all because drivers with fewer than 30 starts above them are
  no longer ranked.

## 33. The simulator, reworked so races do not just follow the grid

The owner's test at Interlagos showed races sticking to the grid: Albon
(+0.39%) won from pole while Verstappen (+0.88%) could not pass from 4th.
Measured on 1,000 races with today's grid (`check_simulator.py`), the old
simulator's finishing order had a rank correlation of 0.93 with the grid
(real F1 2018 to 2025: 0.74, and real cars differ, which makes real races
follow the grid more, not less). Causes and fixes:

- **Passing**: the old rule let a car past only if it was quicker by more
  than the circuit's margin on that single lap, so lap to lap noise decided
  passes. Now a car that catches the one ahead gets past with a probability
  that rises smoothly with its PACE advantage for the day (an S-curve,
  centred on 0.0s at the easiest circuit and 1.0s at Monaco). A clearly
  faster car gets by within a few laps at Interlagos; at Monaco it may not.
- **Qualifying**: the grid is a simulated qualifying session, shown as
  "Grid from simulated qualifying" on the race screen, unless the user
  chooses "I'll set the order" (shown as "Grid: your order"). Qualifying
  form is only partly linked to race form, by a correlation of 0.32
  MEASURED from 1,457 weekends since 2018 (how a driver's qualifying gap and
  race pace gap to their teammate move together from weekend to weekend).
  Before, qualifying used the race form itself, so the grid already was the
  race order.
- **Race distance**: the default is now the real race distance (71 laps at
  Interlagos); the old default was a 15 lap sprint.
- **Randomness** is split into a per race form (the main luck) and a small
  lap to lap noise (0.08% of a lap, down from 0.25%), so a single lap cannot
  outweigh a real skill gap.
- **Safety cars**: one random safety car in 56% of races, the rate measured
  from 2018 to 2026 races. Without it the steady form difference built up
  all race and the field finished minutes apart.
- **Calibration**: the form was retuned (now 0.40%) so simulated drivers
  still beat each other as often as the fitted model says: for a 0.4% skill
  gap, model 76.2%, simulation 75.8% (20,000 races).
- **Rating uncertainty** in fantasy races is now off by default, with a
  checkbox to turn it on. Drawing every driver's rating from its range on
  top of the form made races noisier than the calibration intends.

Before and after (1,000 races each, today's grid at Interlagos):

| | Old, 15 laps | Old, 71 laps | New, 71 laps |
|---|---|---|---|
| Grid vs finish rank correlation | 0.931 | 0.913 | 0.653 |
| Average places moved | 1.90 | 2.17 | 3.87 |
| Top rated driver wins from pole | 55.3% | 40.8% | 46.2% |
| ... from 5th | 15.1% | 22.0% | 18.8% |
| ... from 15th | 4.1% | 16.1% | 14.8% |
| Fastest lap by a top 3 rated driver | 52.1% | 52.1% | 51.7% |
| Median gap, winner to 2nd / to 10th | 2.6s / 14.4s | 9.4s / 46.6s | 6.4s / 37.1s |

What did not improve: the fastest lap still goes to a top 3 driver only
about half the time (chance would give 15%). The race form, which is
calibrated to the model, varies by 0.40% of a lap, more than the gaps
between the top drivers on today's grid, so on the day someone else is
often quickest. Making it go to the top drivers more often would make the
whole race more predictable than the validated model allows. Automated
tests in `test_simulate.py` keep these numbers inside sensible bounds.

With the reworked simulator, the equal car champion of each season changed
in only 2 of 77 seasons (1958 Hawthorn to Moss, 1983 Prost to Keke
Rosberg), both close seasons; the favourites' average title chance stayed
about 49%.

The results table and timing tower no longer show the driver's slot number,
which users read as a grid position: the results table has a "Started"
column with the change shown as "+3" or "−3", the tower shows position,
colour and driver (plus the gap, which the original brief asked for), and
the dots on the track carry the driver's three letter code.

## 34. "Why this rating": exact removal effects for a bowl-shaped fit

`explain_ratings.py` estimates, for each of the 2,828 teammate pairings,
how much each driver's peak rating would change without that pairing's
results. A first version used the one-step approximation (covariance times
the pairing's pull). It understated pairings that carry most of the
information about a driver, because the covariance it uses still contains
that pairing's own information; the result was, for example, a Senna panel
dominated by unrelated pairings. The fix uses the exact answer for a
bowl-shaped (quadratic) fit: remove the pairing's own curvature first. That
curvature has one column per season the pair raced together, so the
Woodbury identity makes it a tiny calculation per pairing. The panel shows
the four pairings that pushed a rating up and down most, and the shortest
chain of teammates to today's grid (fewest links, then longest partnerships).

A battle can matter without moving the rating: Senna's seasons with Berger
barely move his peak, because his seasons either side already put him at
the same level. Those battles make the rating more certain rather than
higher or lower. The panel's wording says "moved".

## 35. The Pit Wall assistant: a language model that may only quote the model

The assistant answers fans' questions with Claude (Anthropic's API) through
one server endpoint (`web/app/api/chat/route.ts`). Design choices:

- **Tools, not knowledge.** The language model gets seven tools that read
  the exported model data (`web/lib/tools.ts`): get_driver_rating,
  compare_drivers, simulate_race, get_teammate_record, explain_rating,
  get_wet_rating and list_top_drivers. Tools return numbers ready to quote
  (percentages as whole numbers, skill to two decimals), so the model never
  needs to do arithmetic.
- **A checker, not just an instruction.** The system prompt says every
  number must come from a tool, but instructions alone are not a guarantee.
  After each answer, `web/lib/numbers.ts` finds every number in it and
  checks it can be traced to a tool result or the question (allowing for
  rounding). If not, the model is told exactly which numbers had no source
  and asked to rewrite once; any number still untraced is shown to the
  reader with a warning.
- **Model**: Claude Opus 5.5 by default (`PIT_WALL_MODEL` can change it),
  with low effort (looking up numbers needs little deliberation) and the
  server-side refusal fallback switched on. The system prompt and tools are
  cached between questions, which makes each question after the first
  cheaper.
- **Cost control**: the API key lives only in a Vercel environment
  variable. With no key the panel says it is unavailable. A rate limiter
  allows 5 questions per visitor per 10 minutes, 20 per day, and 100 for
  the whole site per day (all adjustable by environment variable). It lives
  in server memory, so it is a brake, not a lock; the real backstop is a
  monthly spend limit set in the Anthropic Console.
- **Evaluation**: `web/eval/questions.json` has 50 questions: 15 factual
  ones whose answers are real records (Senna's 41 wins and so on, checked
  against the data by `npm run eval -- --dry`), 30 comparisons whose correct
  numbers come from the tools, and 5 trick questions (made-up drivers,
  future seasons) that must be declined. `web/eval/run-eval.ts` scores
  four things: the right tool, the expected numbers quoted, no untraceable
  numbers, and correct refusals. The live evaluation needs an API key and
  costs money; it had not been run when this was written because no key
  was available on the development machine (see the README).

## 36. Deployment: a normal Next.js build instead of a static export

The site was a static export (plain files, no server). The assistant needs
one server endpoint to keep the API key secret, so the build is now a
normal Next.js build: every page is still pre-rendered as static HTML, and
only `/api/chat` runs as a small serverless function on Vercel's free tier.
Vercel's project settings must point the Root Directory at `web`.

## 37. Race page checks on the all-time grid: qualifying and the back of the grid

The owner asked for two checks on the "All-time top 20" grid, with a change
only if the simulator was clearly out of line with real F1. Both are printed
by `check_simulator.py --race-page`. Neither needed a change.

**Qualifying** (1,000 simulated sessions):

| Driver | Skill | Pole | Average grid slot |
|---|---|---|---|
| Michael Schumacher | +1.41 | 37.2% | 3.8 |
| Juan Manuel Fangio | +0.94 | 6.1% | 8.7 |
| Ayrton Senna | +0.92 | 7.2% | 8.6 |
| Jackie Stewart | +0.92 | 6.0% | 9.1 |
| Jim Clark | +0.91 | 5.6% | 8.9 |

The top rated driver takes pole 37% of the time, above the owner's 25%
threshold. Fangio, Senna, Stewart and Clark are rated within 0.03% of each
other and of the rest of the top 20 within about 0.3%, so they share poles.
For comparison, each real season's best qualifier from 2014 to 2025 took
between 20% (Bottas, 2017) and 62.5% (Hamilton, 2020) of poles, usually with
a car advantage on top. The simulator's qualifying scatter is also consistent
with the model's measured qualifying noise: each driver's simulated lap
varies by about 0.41% (form 0.40% and lap noise 0.10% combined), and the
model's teammate qualifying gaps imply about 0.43% per driver. So qualifying
randomness was left as it is.

**The back of the grid** at Silverstone (1,000 races): drivers rated 2nd to
10th, when they started 16th to 20th (about 17.8th on average), finished
13.3th to 15.0th on average, gaining 2.9 to 4.4 places; drivers rated 11th to
20th gained 2.1 to 2.9. Real F1 from 2018 to 2025: all 699 classified
drivers who started 16th to 20th gained 4.2 places on average; the 39 of them
in one of the season's three best cars gained 11.2. The simulated gains are a
little below the real average (which also benefits from retirements ahead
and pit strategy) and far below the fast-car cases. That is expected: in
identical cars the drivers of the all-time top 20 are separated by a few
tenths of a second a lap, against about a second for a top car starting at
the back, and Silverstone's passing margin is about 0.7 seconds. A top 10
rated driver starting 18th does usually gain several places, so the
overtaking model was left as it is.

**The timing tower** was built in lineup order and moved into place with
CSS transforms, so the order on screen was right in testing but the page
itself (and so screen readers, copy and paste, and any browser that drew it
differently) was in lineup order. It is now built in race order, P1 first,
updating six times a second; a row that changes place slides from its old
position with a short animation. `web/tests/tower.test.ts` checks the order
before the start, during the race, after an overtake, with a crashed car, and
at the flag, and a browser check confirmed the order during a live race.

## 38. How far can teammate prediction go? A ceiling and two references

`backtest_improvements.py` adds reference rows that are allowed to see the
answers, so the room for improvement can be judged:

- **Ceiling (hindsight)**: for each season, fit the model WITH that season's
  results and predict the same season. Race 65.9%, qualifying 69.8%
  (2010 to 2025 pooled).
- **Ceiling + qualifying (hindsight)**: the same, plus the same weekend's
  qualifying (number 40). Race 68.8%.
- **Hindsight pair-season share**: for each pair of teammates in each season,
  use the share of that season's results driver A really won as A's chance in
  every race. Race 68.3%, qualifying 71.4%. This is the best any model can do
  if it gives a pair one fixed chance per season.

The main lesson: single races are mostly luck. Even knowing every result of
the season, a season-level skill model calls only about two thirds of
teammate race battles. The "ceiling" is not a true ceiling for a model that
uses race-by-race information: the fair combined model (68.6%) beats the
hindsight model without qualifying (65.9%), because qualifying tells it who
was quicker that particular weekend. Against the matching hindsight
reference (68.8%), the fair model is 0.2 points behind in accuracy and
0.013 behind in log loss: close to what is possible with this information.
For qualifying predictions there is more room: 66.3% fair against 69.8%.

## 39. In-season updating, and the rule that no prediction sees the future

The rolling backtest used to fit once before each season. Now each race is
predicted from all earlier seasons plus the earlier races of the same
season, refitting before every race (`model.data_window`, `before_race`).

- **The no-future rule is checked, not assumed.** Every prediction records
  the last race in the data its fit was made from, read from the fitted data
  itself, and `check_no_future` stops the script if any fair prediction's fit
  saw its own race or a later one. All 15,788 fair predictions passed.
  `test_improvements.py` checks the data window, checks that round 3 is
  predicted from rounds 1 and 2 only, and checks that the check itself stops a
  leaked prediction.
- **Speed**: about 400 refits per version. Each refit starts from the
  previous race's fit (a "warm start"), which gives the same answer
  (largest difference 0.000002; tested) a little faster. The uncertainty of
  the gap is computed once at the start of each season and reused within it,
  because recomputing it before every race would add several seconds each
  time; only the best estimates are refitted race by race.
- **Result**: race 59.6% to 62.9% (log loss 0.6715 to 0.6444), qualifying
  61.3% to 64.9%. Kept.

## 40. The qualifying signal and the experience curve: both kept, with a caution

**Qualifying signal.** Qualifying happens before the race, so a race
prediction may use it. It is added by a small second step on top of the
model: a logistic regression with three inputs (the model's own prediction,
how much better or worse than expected driver A qualified relative to B,
and who qualified ahead when no lap time exists). Each season's weights are
learned only from earlier seasons' predictions (2007 to 2009 are predicted
only to give 2010 something to learn from). So that versions are compared
fairly, every version gets the same second step; without the qualifying
inputs it simply recalibrates the model's probabilities.

- Race: 59.6% to 68.8% on its own; 68.6% with in-season updating and the
  experience curve. Kept.
- **The honest comparison**: the simple rule "whoever qualified ahead
  finishes ahead", with a confidence learned from earlier seasons, scores
  68.3% (Brier 0.2172, log loss 0.6261). So almost all of the accuracy gain
  is qualifying itself; the combined model adds 0.3 points of accuracy and
  better probabilities (Brier 0.2073, log loss 0.6035).
- **Leakage checks**, because three seasons of the combined model went above
  75% (2013: 75.7%, 2021: 76.2%, 2024: 75.4%): the qualifying data in those
  seasons comes from qualifying sessions (one grid fallback in 2021); the
  simple qualifying rule scored similarly high in the same seasons (71.1%,
  75.1%, 76.4%), so qualifying was simply very predictive then; and when each
  comparison is given the same pair's previous race qualifying instead of its
  own, the combined model falls to 63.2%, the level without qualifying. The
  gain comes from same-weekend qualifying, not a leak elsewhere.
- This improves forecasts of real teammate battles. It does not change the
  site's ratings, which describe drivers' skill, not one weekend.

**Experience curve** (`model.Options.experience`, now on). A shared curve of
how a typical driver's skill changes with years in F1, one value per year up
to 10, kept smooth (neighbouring years probably within 0.1%). Year 0 is the
reference. Learned from all data: +0.13% after one year, +0.26% at year 3,
then a slow decline to about -0.05% by year 10. When predicting a new season,
each driver's skill moves along the curve, so rookies are expected to improve
and veterans to decline.

- Backtest: race 59.6% to 60.1% (log loss 0.6715 to 0.6695), qualifying
  61.3% to 62.1% (0.6609 to 0.6591). Better on every measure, so kept. In the
  full combination it is neutral for races (log loss 0.6035 against 0.6030
  without it) and clearly better for qualifying (0.6119 against 0.6270).
- **It changes the ratings more than the backtest gain suggests.** A
  driver's skill each season is now their own skill plus the curve, so the
  peak (best three seasons) moves for drivers whose best seasons came early
  or late in their careers, and the changes travel along the teammate chains.
  68 ranked drivers moved more than 10 places. Examples: Nico Rosberg 28th
  to 13th (his peak, 2009 to 2011, includes two seasons against Michael
  Schumacher, who rose too),
  Jacques Villeneuve 21st to 11th, Max Verstappen 7th to 3rd, Stirling Moss
  10th to 20th (his best seasons were late in his career, where the curve
  expects decline). The equal car champion changed in 10 of 77 seasons.
- **It makes the ranking more dependent on the era assumption**
  (number 31): only 4 of the top 20 (Schumacher, Senna, Frentzen, Häkkinen)
  now keep their rank within 3 places across the three assumptions, down
  from 10. The backtest checks predictions within the modern era, not
  comparisons across eras, so this is a real cost that the backtest cannot
  see. It is shown on the About page.
- Knock-on numbers: on the all-time grid Schumacher now takes pole in 46.0% of
  simulated qualifying sessions (37.2% before), and on today's grid at
  Interlagos the fastest lap goes to a top 3 driver 49.1% of the time (51.7%
  before). The test bound for the fastest lap was set at 50%, right at the old
  measurement; it now guards "far above the 15% chance level" at 40%.
- The 2024 and 2025 single test (number 21) and the first rolling backtest
  (number 29) are kept as they were run, without the curve.
