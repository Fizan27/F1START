# Decisions

Each significant decision and why it was made. Newest at the bottom.

## 1. PyTorch 2.14.1 built for CUDA 13.0 (`cu130`)

The RTX 5070 Ti is an RTX 50 series card (architecture name `sm_120`). PyTorch
builds made for CUDA older than 12.8 do not include `sm_120`, so they install
without complaint but cannot run on this GPU. The installed driver (616.92)
supports CUDA up to 13.4, so the current stable `cu130` build was chosen.
`check_gpu.py` proves it: it trains a small network on the GPU and the CPU and
compares them (measured: loss 62.3 to 0.39 on the GPU, 14x faster than the CPU).

## 2. Python 3.12 in a plain virtual environment (`.venv`)

Python 3.12 has ready-made packages for every library used here. A plain
`venv` plus `pip` is the most standard setup, with nothing extra to learn.

## 3. Flat layout: code files in the top folder

The project is small. One folder means there is never a question about where a
file lives. Only `docs/` (documents) and `data/` (downloaded, not committed)
are separate. Tests sit next to the code as `test_*.py`.

## 4. Seasons: train 2022 to 2023, validate on 2024, test on 2025

All four seasons use the same car rules (the 2022 to 2025 "ground effect" era),
so tyre and lap time behaviour is comparable. They are split by time into
three parts, each with one job:

- **Train (2022 to 2023):** the models learn from these races.
- **Validation (2024):** used for every tuning and model choice (network size,
  learning rate, which inputs to keep, baseline versus network).
- **Test (2025):** touched only once, at the end, for the reported numbers.

Why three parts and not two: every time a choice is made by looking at a
score, that score becomes a little optimistic, because the choice was fitted to
it. If 2025 were used for tuning, the final numbers would flatter the model.
Keeping 2025 untouched means the reported error is an honest estimate of how
the model does on races it has truly never influenced.

Before the final 2025 evaluation the chosen model is retrained on 2022 to 2024
with the settings already fixed, so it uses all the data available before the
test season, as a real team would.

2026 brought completely new cars and engines, so it is left out of version
one. Future work: use it as a "rule change" robustness test, to measure how
much the model degrades when the cars change.

## 5. Save raw data as one Parquet file per race

`download_data.py` saves laps, weather and results for each race separately in
`data/raw/`. If one race fails to download, the others are unaffected and the
script can simply be run again: it skips races already saved. Parquet keeps
column types (unlike CSV) and Polars reads it very fast. Times are stored as
plain seconds because they are easier to do arithmetic on than time objects.

## 6. Fuel is represented by laps remaining

Real fuel load is not public. Cars start with enough fuel for the race and
burn it at a roughly steady rate, so `LapsRemaining` (total laps minus lap
number) falls in step with fuel weight. Limitation: it cannot see fuel saving
or a car that started under-fuelled.

## 7. Traffic is the gap to the car ahead on the road, at the start of the lap

`GapAhead` uses the car physically in front, even if it is a lap down, because
dirty air affects pace regardless of race position. It is measured at the
start of the lap (the end of the previous lap). The gap at the end of a lap is
partly decided by that lap's own time, so using it as an input would hand the
model part of the answer (this is called target leakage).

## 8. "Clean laps" are separated from everything else

The lap time model learns only from laps marked `IsCleanLap`: a recorded and
accurate lap time, green track, not a pit in or pit out lap, not lap 1, on a
dry tyre with a known age. About 79% of 2022 to 2024 laps qualify. The
excluded laps are slow for reasons the lap time model should not explain (pit
lane, safety car, standing start, rain); the simulator handles those
separately with pit loss and safety car models.

## 9. The target is lap time as percent slower than pole (`LapTimePct`)

Raw lap times are dominated by the circuit (about 66s at Spielberg, 105s at
Spa). A model could score well on raw times by memorising circuit lengths
while learning nothing about tyres, fuel or traffic, which are worth tenths.
Dividing by a yardstick removes the circuit. Two yardsticks were considered:

- The race's own typical lap: rejected, because it is computed from the whole
  race, including laps after the one being predicted (target leakage).
- The weekend's pole lap: chosen, because qualifying is finished before the
  race starts, so it cannot leak anything about the race.

Percent, not seconds, so that the same effect is the same size at short and
long circuits.

Measured weakness: when qualifying was wet and the race dry, pole is a bad
yardstick. In 2022 to 2024 this affects four races (Silverstone 2022, Montréal
2022 and 2023, Spa 2024), where the typical race lap is faster than pole.

## 10. Car and driver pace comes from the qualifying gap (`QualiGapPct`)

Team and driver names are not model inputs: what a name means changes between
seasons (the fastest team of 2022 to 2023 is not the fastest of 2024). A
driver's average pace in the race being predicted is not used either, because
it is built from later laps. The qualifying gap to pole is known before lap 1,
is specific to that car at that track on that weekend, and works for rookies.

Gaps above 5% are treated as missing (rain or a ruined lap, not car pace) and
filled with the team-mate's gap. About 2% of clean laps still have no value.
Limitation: one-lap pace is not race pace. Possible later addition: each
team's race pace from earlier races in the same season.

## 11. Unseen circuits: the first baseline has no circuit input

Shanghai appears in 2024 (validation) but not in 2022 to 2023 (training).
Because the target and inputs mean the same thing at every circuit, the model
can predict there without recognising the track. Circuit as an input is then
tested as an experiment on 2024, with Shanghai's error always reported
separately as the "new circuit" result.

About 1% of clean laps are far slower than the race's typical lap (mistakes,
damage, a drying track). How they are handled: number 13.

## 12. Lap time is split into a race pace level and pace within the race

The first baseline predicted `LapTimePct` (percent slower than pole) directly
and lost to always guessing the average: 2.63s typical miss against 2.09s on
2024. The circuit table showed why. On most circuits the bias was as big as
the miss, meaning the model was wrong by the same amount on every lap of a
race. How far below qualifying pace a race is run varies from about 5% to 13%
between races, and no lap-level input can explain that.

So lap time is now two parts:

- `RacePacePct`: the race's typical (median) clean lap, in percent slower than
  pole. One number per race.
- `PaceVsRacePct`: how a lap differs from that typical lap. This is the lap
  time model's target.

Why this is the right split for a strategy tool: the race pace level moves
every car by the same amount, so it does not change the finishing order, the
gaps between cars, or which strategy is best. Everything strategy depends on
(tyre wear, compound, fuel, traffic, car pace) lives in the second part.

The honesty cost, stated plainly: `RacePacePct` is computed from the whole
race, so it is only known afterwards. That is fine for replaying a past race
("what if they had pitted differently?"), which is what the app does. It
would not be fine to call the within-race error a forecast of lap times
before a race. So two numbers are always reported separately: the within-race
error, and the error of forecasting the race pace level from earlier seasons
only (the circuit's average). `RacePacePct` is never a model input.

This partly supersedes number 9: pole is still the unit (percent of pole) and
still feeds `QualiGapPct`, but it is no longer the yardstick for the target.

## 13. Outlier laps are removed from training but kept in scoring

Laps more than 5% slower than the race's typical lap (about 1 in 100) are
left out of training: the model cannot predict a spin, and trying to pulls its
other predictions off. Measured on 2024, removing them cut the mean miss from
0.85s to 0.73s. They stay in the validation scores, because dropping them
there would flatter the model. Both mean and median miss are reported: the
median shows a typical lap, the mean shows the cost of the bad laps.

## 14. Baseline inputs: circuit is in, temperatures are out

Chosen by comparison on 2024 (mean miss per lap, within the race):
know-nothing 1.085s, basic inputs 0.725s, basic plus temperatures 0.783s,
basic plus circuit 0.684s.

- Temperatures made it worse. They hardly change within a race, so they act
  as a fingerprint of the race and the trees memorise training races.
- Circuit helped, including at Shanghai, which training never saw (0.58s to
  0.54s there), so the worry in number 11 did not materialise for LightGBM.

(These figures are from before number 15 changed training to dry races only.
Current figures are in the README.)

## 15. Wet and mixed races are flagged, and dry racing is reported separately

Version one models dry racing only. On a damp track, laps on dry tyres are
slow for a reason none of the model's inputs can see, and wet-weather tyre
laps are already excluded from clean laps. So every race gets two flags in
`clean_data.py`:

- `IsWetRace`: at least 2% of the race's laps were on intermediate or wet
  tyres, or rain was recorded on at least 10% of laps. The second rule
  catches races where it rained but everyone stayed on dry tyres (Budapest
  2022, Spa 2023). The thresholds are low on purpose: a few laps of rain is
  enough to disturb the pace of the whole race. In 2022 to 2024 this flags
  11 of 68 races.
- `IsWetQualifying`: a dry race whose typical lap is less than 3% slower than
  pole. Dry races normally run 5 to 13% slower, so this only happens when
  pole was set on a wet track. It flags 4 races (Montréal 2022 and 2023,
  Silverstone 2022, Spa 2024). It matters for forecasting the race pace
  level, where it is the single biggest source of error.

How the flags are used:

- Results are reported for dry races (the headline, matching the scope) and
  for all races (showing what the wet and mixed ones cost).
- Models are trained on dry races only. Measured on 2024 with the baseline,
  this improved the dry mean miss from 0.641s to 0.632s and did not hurt the
  all-races figure (0.684s to 0.675s).

Limitation: `IsWetRace` is worked out from the whole race, so it describes
past races and could not be used to predict rain. Rain is an optional extra
for a later version.

## 16. The network predicts a centre and a spread (a bell curve per lap)

To predict a range, the network outputs two numbers per lap: a centre and a
spread, describing a bell curve (Gaussian). It is trained with the Gaussian
negative log likelihood: `log(spread) + 0.5 * ((actual - centre) / spread)^2`.
The first part punishes wide ranges and the second punishes misses measured
in spreads, so the only way to do well is an honest spread.

Why this and not the alternative (predicting fixed percentiles, "quantile
regression"): the simulator has to draw thousands of random lap times per
second on the GPU, and drawing from a bell curve is one line of code. The
cost: real lap times are lopsided (a lap can be 3 seconds slow, never 3
seconds fast), and a bell curve cannot show that. Outlier laps are left out
of training partly for this reason.

The honesty of the spreads is checked by counting how many real 2024 laps
fall inside the predicted 90% range.

## 17. Network settings: small, 20 epochs, circuits sometimes hidden

Chosen by trying values and scoring on 2024 (averaged over two random seeds):

- Two hidden layers of 64. A width of 32 was slightly worse.
- 20 epochs. At 10 the dry mean miss was 0.630s, at 20 it was 0.623s, and at
  40 it was no better while the share of laps inside the 90% range fell from
  86% to 85%: the network had started memorising training races.
- Weight decay (a common guard against memorising) made no measurable
  difference, so it is left out to keep the code simple.
- Each circuit is described by 4 learned numbers (an embedding). On 10% of
  training laps the circuit is replaced by "unknown", so the network learns a
  general answer to use at circuits it has never seen.

## 18. The network's ranges are widened by one factor (calibration)

Straight from training, 86.6% of dry 2024 laps fell inside the network's 90%
range: slightly overconfident. Every spread is now multiplied by one factor,
1.145, chosen so that exactly 90% of dry 2024 laps fall inside. The factor is
saved in the model file and the simulator always uses the widened spreads.

- Why one factor and not something cleverer: it fixes the measured problem,
  changes no prediction's centre, and is one line to explain.
- Honesty note: the factor is chosen on 2024, so 2024 coverage is 90% by
  construction and proves nothing. The real check is the 2025 test.
- For the final model (retrained on 2022 to 2024) the same 1.145 is reused,
  because 2024 is then training data and cannot also be used to calibrate.

## 19. Working mode changed: no TODOs, no approval between steps

On 2026-10-08 the owner asked for the project to be finished as fast as
possible without questions or TODO(human) lines, with decisions logged here.
CLAUDE.md was updated to match. Decisions from here on were made by the
assistant and are recorded with their reasons and measured effects.

## 20. Pit loss is measured against cars that stayed out on the same laps

A stop spreads its cost over the lap into the pits and the lap out. So the
cost is: the time the stopping car took for those two laps, minus the median
time of cars in the same race that stayed out on the same two laps. Measured
on 767 green flag stops in dry 2022 to 2023 races. Per circuit medians run
from 18.8s (Spa) to 27.4s (Lusail); circuits with fewer than 8 clean stops
use the overall median, 22.0s.

Safety car and VSC stops needed two extra steps:

- Single stops are too noisy (values from -31s to +118s, because the queue
  behind a safety car distorts the comparison). Only laps where at least 3
  cars pitted together are used. Measured this way a stop costs 72% of a
  green stop under a safety car and 86% under a VSC.
- Those are seconds on a slow clock. While the field crawls, a 15 second gap
  is a much shorter distance than 15 seconds at racing speed, and it shrinks
  back when racing resumes. The simulator keeps gaps in racing-speed
  seconds, so the measured cost is divided by how much slower the laps are
  (safety car laps are 49% of pole slower than normal, VSC laps 36%). Result:
  a stop costs 49% of a green stop under a safety car and 65% under a VSC,
  in line with what teams commonly quote.

## 21. Safety car and VSC likelihood: phase rates, scaled per circuit

The chance per lap that a safety car starts is counted per race phase over
all circuits (laps 1 to 3: 7.1%; early third: 1.0%; middle: 0.7%; late:
0.7%). Each circuit then gets a multiplier: periods seen there divided by
periods an average circuit would have had in the same laps.

With two or three races per circuit a single crash would swing that
multiplier wildly, so it is pulled towards 1 by adding 2 imaginary average
periods to both sides: (seen + 2) / (expected + 2). This is a standard trick
(shrinkage) for rates estimated from very few events. Multipliers end up
between about 0.6 (Barcelona) and 1.6 (Melbourne). Period lengths are drawn
from the real lengths seen. Based on only 19 safety car and 20 VSC periods
in 32 races, so these are rough.

## 22. Simulator rules that are assumptions, not measurements

`simulator.py` needs a few rules the data cannot give directly:

- First lap: the lap time model does not cover the standing start. Each grid
  slot costs 0.74s by the end of lap 1 plus random luck, both fitted to real
  lap 1 gaps.
- Overtaking: a car must be 1.0s a lap faster than the car in front to pass,
  otherwise it finishes the lap 0.4s behind. Values from 0 to 2.5s were
  tried in the 2024 replay; the position miss only moved between 1.96 and
  2.06 places, so a physically sensible value was chosen instead of the best
  scoring one (which would be fitting noise).
- Safety car: the leader laps slowly and every car closes to 0.6s behind the
  car in front. VSC: everyone laps at the same slow pace, so gaps freeze.
- Rivals react: when a safety car (or VSC) appears, a rival whose next
  planned stop is within 8 (or 4) laps pits at once. Without this, rivals
  would ignore safety cars and any strategy that used them would look
  unrealistically good.
- Retirements: cars that really retired drop out on the lap they really did.
  The simulator does not model crashes or breakdowns.
- Only dry races without a red flag are simulated (a red flag allows a free
  tyre change, which the simulator does not model).
- The whole pit loss is charged on the lap of the stop, and tyres age one
  lap per lap even behind a safety car.

## 23. The simulator does not trust the network on very old tyres

The network's wear curves flatten out on very old tyres, because the only
cars that ever ran 40 lap old tyres were ones whose tyres happened to be
lasting well (survivor bias). Left alone, a strategy search or an RL agent
would exploit this by never stopping. So beyond the tyre age that 99% of
real laps on that compound stayed under, the simulator adds 0.10% of pole
(about 0.09s) per extra lap. This is a guard rail, not a measurement.

## 24. Whole-race driver pace: measured, then scaled to 0.4 in replay

Part of the network's error is not lap by lap luck: a car is simply quicker
or slower all race than its qualifying gap suggested. Measured on 2024 this
whole-race offset has a spread of 0.54% of pole (about 0.44s a lap). The
simulator draws one offset per car per simulated race, and reduces the lap
by lap luck to match so the total is not counted twice.

Using all of that spread made simulated results too scattered: 98% of real
2024 finishing positions fell inside a simulated range meant to hold 90%.
The likely reason is double counting, since some of the measured offset is
traffic and strategy, which the simulator models separately. Using 0.4 of it
gives 91%, so 0.4 is used. This is tuned on 2024; 2025 is the real check.

## 25. Replay validation: what is compared, and with what

Each dry 2024 race without a red flag (19 races) is simulated 200 times with
the real grid, real strategies and real safety car laps. The simulator is
given each driver's qualifying gap but nothing about their real race pace.
Only real finishers are scored. The yardstick it has to beat is "everyone
finishes where they started" (grid order), which is a strong predictor in F1.

Result: average miss of 2.03 places against 2.43 for grid order (16% better),
winner right in 11 of 18 races, 39 of 54 podium places right. Weakness: the
simulated field is too spread out, with a median gap to the winner of about
65s against 47s in reality (median error 17s per driver). Likely causes:
real leaders manage their pace instead of pulling away, and lapped cars are
not modelled.

## 26. The strategy environment: one car per race, rivals follow history

`strategy_env.py` wraps the simulator in the standard reinforcement learning
interface (Gymnasium style: `reset()` and `step(action)`), but with tensors,
so thousands of races step together on the GPU.

- The strategist controls one car. The other 19 follow the strategies their
  teams really used (bringing a stop forward if a safety car appears, number
  22). Training a strategist against 19 other learning strategists would be
  a far harder problem and is out of scope.
- Four actions each lap: stay out, or pit at the end of this lap for soft,
  medium or hard.
- The car starts on the tyres it really started on. Choosing the starting
  tyre is not part of version one.
- Two compound rule: on the last lap where a stop is possible, a car that
  has used only one compound is forced to pit for another. Real rules
  disqualify instead; forcing keeps every simulated race legal, and the
  strategist learns that leaving it that late is expensive.
- What the strategist sees (23 numbers): race progress, tyre age and
  compound, position, gaps ahead and behind, track status this lap, stops
  made, compounds used, the circuit's pit loss, its qualifying gap, and the
  lap time model's view of its pace now and on a new set of each compound.
  Nothing identifies the circuit by name, so it can drive at circuits it
  never trained on. Giving it the tyre model's predictions mirrors a real
  pit wall, where strategists work from tyre models.

## 27. Reward: time lost to the typical car each lap, plus finishing place

- Each lap: minus (own lap time minus the median lap time of the field),
  divided by 10 seconds. A 22s pit stop costs about 2.2 at once; fresh tyres
  earn it back a little each lap.
- At the finish: minus 0.2 per finishing place.

Why two parts: finishing position is the real goal, but as the only reward
it arrives once per race, long after the decisions that caused it, which
makes learning slow. Time is a signal on every lap. Why "against the typical
car" and not the raw lap time: a safety car slows everyone, and the
strategist should not be punished for something it did not cause. Why
finishing place is still in: time alone would ignore track position, which
is what undercuts and safety car stops are really about.

Risk accepted: with time in the reward, the strategist could prefer a
slightly faster race over a better position. The evaluation reports places
and seconds separately so this would show.

## 28. PPO settings, and why its choices are sampled when it is evaluated

Own implementation of PPO in about 100 lines (`train_agent.py`): 4,096
simulated races per round, 150 rounds, two networks of 2 x 128 units, clip
0.2, GAE lambda 0.95, no discounting (races are short and always end),
learning rate 0.0003. Training takes under 5 minutes on the RTX 5070 Ti.

- Head start: a new policy starts choosing "stay out" about 98% of the time.
  An untrained network picks each of the 4 actions equally, so it would pit
  on three laps out of four, every race would be a disaster, and the useful
  region (one or two stops) would take a long time to find.
- Sampling at evaluation: the strategist's choices are drawn from its policy,
  as in training, not taken as the single most likely action. When it wants
  to stop "some time in the next few laps" it spreads that over several
  laps, and on each one staying out is still the most likely choice, so the
  most-likely rule would never stop at all. Results are averaged over 200
  simulations per driver, so the sampling noise is small.
- Training races: the 32 dry 2022 to 2023 races, any finisher's car, with
  random safety cars drawn from the measured rates.
