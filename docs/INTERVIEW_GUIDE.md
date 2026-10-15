# Interview guide

How to explain this project out loud. Read it next to `model.py`, which is the
one file you must know well. Every number here comes from a script in this
repository; if you re-run them and a number changes, trust the script.

## The 30 second version

"F1 results mix up the driver and the car. I built a model that separates
them using teammates, because teammates share a car. It is a Bayesian
hierarchical model I wrote in PyTorch: every driver gets a skill for every
season since 1950, with an uncertainty that I checked with posterior
sampling. In a rolling backtest, predicting each season from 2010 to 2025
from earlier seasons only, it is only slightly better than the best simple
rule overall, 59.6% against 59.3%, but much better when drivers change teams,
because it can compare drivers who were never teammates. Then I built a GPU
simulation of equal car races, a website, and an assistant that answers
questions using only the model's numbers, with a checker that catches any
number it invents."

## The model in five steps

**1. What is being estimated.** One number per driver per season: skill, in
percent of lap time. +0.5 means half a percent quicker than an average
newcomer of today in the same car, about 0.45 seconds a lap. There are 2,515
of these numbers in the final fit.

**2. The evidence (the likelihood).** For each pair of teammates in each race,
only the gap between their two skills matters, because the car is the same.

| Evidence | Seasons | How the model reads it |
|---|---|---|
| Who finished ahead | all | chance = 1 / (1 + exp(-slope x skill gap)), a logistic curve |
| Qualifying lap time gap | from 1994 | expected gap = skill gap, with heavy tailed noise |
| Who qualified ahead | where no lap time exists | the same logistic curve, slope tied to the qualifying noise |
| Race pace gap | from 2018 | expected gap = skill gap x a learned factor |

Races where either car broke down are left out. That is the whole mechanism
that stops car failures counting against drivers.

**3. The prior beliefs (what makes it "hierarchical").**

- A driver's first season is probably near the level of newcomers of that
  era, give or take 0.3%. So a driver with three races is pulled towards
  average and is not rated a genius on thin evidence. This is called
  shrinkage, or partial pooling.
- Each season's skill is probably near the season before, give or take 0.18%.
  This is a random walk. It lets careers rise and fall smoothly.
- The level of newcomers may differ between decades, by a typical 0.1% per
  decade. Today's level is fixed at zero as the reference.

**4. Fitting.** Add the log likelihood and the log prior, and find the skills
that make the total as large as possible. That is called the MAP estimate
(maximum a posteriori). It is done with L-BFGS on the GPU in about 15 seconds.
The function is `badness` in `model.py`, and it is about 40 lines including
`log_likelihood` and `log_prior`.

**5. Uncertainty.** At the best point, take the second derivative of the
badness in every direction (the Hessian, a 2,500 by 2,500 table). Its inverse
is the covariance of all the skills. This is the Laplace approximation.
Intuition: if the fit gets much worse when you nudge a driver's skill, that
skill is known precisely. If it barely changes, the skill is uncertain.

**6. Checking the uncertainty.** `sample_posterior.py` then draws 1,000
plausible versions of every unknown with NUTS (a Hamiltonian Monte Carlo
sampler, via Pyro), using the same badness function, and the site uses those
samples. Result: skill ranges 1.04 times as wide as Laplace said, so Laplace
was close. But the race slope's posterior (2.03 to 2.32) is well below the
single best fit (2.91), which helps explain why the model was over-confident.

## Three functions to know cold

1. **`log_likelihood` in `model.py`.** Be able to say why only the gap
   `skill[a] - skill[b]` appears: the shared car cancels.
2. **`uncertainty` in `model.py`.** Hessian, inverse, covariance. Know that
   the off-diagonal entries matter: two old teammates have a certain *gap*
   even if each is individually uncertain.
3. **`compare_race` in `clean_history.py`.** Returns "not comparable" when a
   car failed. If it returned "lost" instead, drivers of unreliable cars would
   be rated as bad drivers.

## How it was validated

**The rolling backtest (the main test).** For each season from 2010 to 2025,
fit on all earlier seasons only and predict that season: 2,728 race
comparisons. Model 59.6%, "last season repeats, else career points" 59.3%,
"more career points" 57.9%. The model beat the stronger baseline in 8 of 16
seasons, tied 1, lost 7. Three additions (team orders, a weak-teammate
correction, a wet adjustment) were tested this way; none improved log loss,
so none is used.

**How far it can go (race by race).** Predicting every race 2010 to 2025
from earlier races only: current model 59.6%, refitting before every race
62.9%, adding the same weekend's qualifying 68.8%, all combined 68.6% (log
loss 0.6035). A model fitted with the season's own results (hindsight) gets
65.9%, or 68.8% with qualifying. The simple rule "whoever qualified ahead
finishes ahead" gets 68.3%.

**The single 2024 and 2025 test.**

- Fitted on results up to the end of 2023. Predicted 2024 and 2025 without
  refitting. The two settings (0.3 and 0.18) were chosen by fitting to 2021
  and scoring on 2022 and 2023, so the test seasons had no influence on
  anything.
- Baselines: "the driver with more career points is ahead", and "last
  season's head to head between the same two repeats".

| | Comparisons | Model | Best baseline |
|---|---|---|---|
| 2024 race | 191 | 65.4% | 66.0% |
| 2025 race | 185 | 68.6% | 57.8% |
| Both, race | 376 | 67.0% | 62.0% |
| Both, qualifying | 472 | 70.1% | 65.0% |

- Finishing positions from 2023 information only: average miss 3.15 places in
  2024 (car only: 3.48). In 2025 it was 4.41, worse than using the 2024
  standings (3.61), because the model was still using 2023 cars.
- Calibration: a temperature learned on 2010 to 2023 (0.45) showed the model
  was over-confident there, but applied to 2024 and 2025 it made the Brier
  score worse (0.2045 to 0.2169). The right temperature drifts over time:
  0.25 for 2010 to 2015, about 1.0 since 2018. Reported as it is.

## The ten questions you are most likely to be asked

**1. Why teammates only? You throw away most of the data.**
Comparing drivers in different cars needs a model of the cars, and any error
in that model leaks into the driver ratings. With teammates the car cancels
exactly, so the driver ratings do not depend on getting cars right. The cost
is less data and total reliance on who your teammates were. I estimate the
car afterwards, as what is left of a team's results once its drivers are
taken out.

**2. How can you compare drivers who never raced each other?**
Through chains of shared teammates. If A beat B and B beat C, that says
something about A against C. 643 of the 655 drivers who ever had a teammate
are in one connected group. The uncertainty grows along the chain, and the
covariance matrix carries that: the model gives Senna a 56% chance against
Max Verstappen, with a range of 39% to 72%.

**3. Why is it Bayesian? What does the prior buy you?**
Three things. It handles small samples: a driver with five races is pulled
towards average. It ties seasons together, so one freak season does not
define a career. And it gives a principled uncertainty for every rating,
which a plain rating system like Elo does not.

**4. Why a Laplace approximation, and did you check it?**
It is fast (seconds) and fits in 150 lines I can explain. Then I checked it
with NUTS on the same model, using the Laplace answer to precondition the
sampler (it samples in coordinates where Laplace would be a round bell curve),
which took about 4 minutes on the GPU. The skill ranges came out 1.04 times
as wide, so Laplace was nearly right for the skills. The sampler found one
real difference: the race slope's posterior is 2.03 to 2.32, not the 2.91 of
the single best fit. In high dimensions the single most likely point is not
typical of the plausible ones. That steeper slope is one reason the model's
predictions were over-confident.

**5. Your model lost to the baseline in 2024. Is it any good?**
In 2024 almost every team kept its 2023 drivers, so "last season repeats" was
close to the ceiling, and my model matched it (65.4% against 66.0%, a
difference of one comparison in 191). In 2025 drivers moved and rookies
arrived; the baselines fell to 58% and the model stayed at 69%. The model's
value is exactly the case the baselines cannot handle: pairs with no shared
history. I would also say that 190 comparisons per season gives a margin of
about plus or minus 7 points, so I do not over-read either year. That is why
I added the rolling backtest over 16 seasons: overall the model is only
slightly better than the best simple rule, and I say so.

**6. How do you know there is no data leakage?**
The split is by time. `build_comparisons` filters on `Year <= last_year`, and
there is a test for that. Settings were tuned on 2022 and 2023 only. The test
was run once with the chosen settings. The baseline "last season repeats" is
actually given an advantage: it can see 2024 when predicting 2025, and the
model cannot.

**7. What was the hardest bug?**
A scale problem. Before 1994 there are no qualifying lap times, only who was
ahead. I had let the model learn freely how strongly a skill gap predicts
"who was ahead". Because the prior prefers small skills, the optimiser shrank
every pre-1994 skill and cranked up that slope to compensate. The fit looked
fine, but the ranking had Hülkenberg above Senna. I found it by noticing the
slope implied an 85% chance where the lap time data implied 58%. The fix was
to tie the slope to the qualifying noise so a skill gap means the same thing
in every era. The lesson: check that learned parameters are physically
sensible, not just that the loss went down.

**8. What are the biggest weaknesses?**
(a) Cross-era comparisons cannot be validated at all; only the modern end is
tested. (b) How much the level of drivers changed between decades is an
assumption, 0.1% per decade, and it sets how wide old drivers' ranges are.
(c) A driver is only measured against teammates, so team orders or a run of
weak teammates distort the rating: Frentzen coming out 9th is probably that.
I tested two fixes (dropping races under documented team orders, and assuming
short-career teammates were weaker) and neither improved the backtest, so I
kept neither and kept the weakness in the README.
(d) From 2023 the data no longer says why a car retired, so crashes stop
counting against current drivers. (e) Before 1970 some "teammates" were
private owners of the same make of car.

**9. Where does the GPU actually help?**
Three places. The fit: every one of about 25,000 comparisons is evaluated at
once each step. The Hessian: 2,500 second derivative passes, vectorised. And
the simulation: 20,000 races of 60 laps with 20 cars take under a second,
because each lap updates one table of races by drivers. The only loop over
cars is the overtaking rule, which must go front to back.

**10. How do you know the simulation is realistic?**
I do not claim it is a realistic race: it has no tyres or pit stops. I
calibrated it to the fitted model: one number, the day to day form, was tuned
until simulated drivers beat each other as often as the model says (0.4% gap:
model 76.2%, simulation 75.8%). Other parts are measured: overtaking
difficulty per circuit, the safety car rate (56% of races), and how linked
qualifying form is to race form (0.32). The first version stuck to the grid
(rank correlation 0.93, real F1 0.74), so I made passing depend on pace for
the day, gave qualifying its own form and added safety cars: now 0.65, and I
test those statistics automatically. One thing did not improve: the fastest
lap goes to a top 3 driver only half the time, and I report that.

## Questions you might get as follow-ups

- **What is shrinkage?** Pulling an estimate towards the group average, more
  strongly when there is less data. It trades a little bias for a lot less
  variance.
- **What is a random walk prior?** Each value is the previous one plus a
  small random step. It says "probably similar to last season" without
  forcing any particular shape.
- **Why heavy tailed noise (Student t)?** One ruined qualifying lap can be 2%
  off. With a normal distribution that single point would drag the rating.
  Heavy tails treat it as a rare event instead.
- **Why log loss as well as accuracy?** Accuracy only checks which side of
  50% a prediction is on. Log loss punishes being confidently wrong, so it
  tests the probabilities themselves.
- **Why L-BFGS and not Adam?** The data fits in memory and the function is
  smooth, so a full-batch second-order-style optimiser converges in far fewer
  steps and reaches the exact optimum, which the Laplace step needs.
- **What would you do next?** MCMC to check the uncertainty; model the car
  and driver jointly on full race results; use race control messages to
  recover retirement reasons after 2023; and let the era-drift assumption be
  explored with a slider on the site.

## Numbers worth remembering

| | |
|---|---|
| Races / drivers / teammate pairs | 1,165 / 789 / 15,843 |
| Drivers linked through teammates | 643 of 655 |
| Driver-seasons estimated | 2,515 |
| Rolling backtest, 2010 to 2025 | 59.6% (best baseline 59.3%), 2,728 comparisons |
| Race by race, all improvements | 68.6% (qualifying rule 68.3%, hindsight 68.8%) |
| Hindsight ceiling without qualifying | 65.9% race, 69.8% qualifying |
| Teammate race prediction, 2024 and 2025 | 67.0% (baseline 62.0%), 376 comparisons |
| Teammate qualifying prediction | 70.1% (baseline 65.0%), 472 comparisons |
| Typical uncertainty of a peak rating | 0.30% in the 1950s, 0.16% in the 2020s |
| NUTS check | ranges 1.04 times Laplace; race slope 2.03 to 2.32 against 2.91 |
| Calibration temperature | 0.45 (2010 to 2023), 0.55 (all seasons, used on the site) |
| Fit time on the GPU | about 15 seconds |
| Simulation speed | 20,000 races in under a second |
| Tests | 62 Python, 31 website |

## The Pit Wall assistant: how it avoids made-up numbers

The assistant is a chat panel that answers fans in plain English using
Claude (Anthropic's API). Three layers stop it inventing numbers:

1. **Tools.** It gets seven tools that read the model's exported numbers
   (`web/lib/tools.ts`), and the numbers come back ready to quote (whole
   percentages, skill to two decimals), so it never needs to do sums.
2. **Instructions.** The system prompt says every number must come from a
   tool, every answer must give the range, and it must decline questions the
   model cannot answer (made-up drivers, future seasons).
3. **A checker** (`web/lib/numbers.ts`). After the answer is written, every
   number in it is matched against the tool results and the question,
   allowing for rounding. An untraceable number sends the answer back once
   with the exact numbers named; anything still untraceable is shown to the
   reader with a warning. Instructions can be ignored; the checker cannot.

**Evaluation.** 50 questions in `web/eval/questions.json`: 15 factual
questions with known answers (Senna's 41 wins, checked against the data), 30
comparisons whose correct numbers come from the tools, and 5 trick questions
that must be declined. `web/eval/run-eval.ts` scores four things: the right
tool used, the expected numbers quoted, no untraceable numbers, correct
refusals. The free "dry run" (every expected tool call works, every known
fact matches) passes; the live score needs an API key (see the README).

### Five questions you are likely to be asked about it

**1. How do you stop the language model making up numbers?**
I do not trust the prompt alone. The model only gets numbers through tools,
and after it writes an answer, code extracts every number and checks each
one can be traced to a tool result or the question, within rounding. If one
cannot, the model is told which numbers failed and must rewrite once; if a
number still fails, the reader sees a warning. So the guarantee comes from
code, not from the model behaving.

**2. Why tools instead of putting all the ratings in the prompt?**
Three reasons. The data is large (655 drivers, 22,578 head to heads), so it
would be slow and expensive to send every time. A tool returns exactly the
numbers for the question, which makes tracing them simple. And some answers,
like a race simulation, have to be computed, not looked up.

**3. How did you evaluate it?**
A fixed set of 50 questions in three groups, with automatic scoring of tool
choice, number accuracy, invented numbers and refusals. Factual answers are
checked against real records, comparison answers against what the tools
return, and simulation answers against the assistant's own simulation run,
because each simulation is random. The trick questions test the most
important behaviour: saying "I don't know" for a driver who does not exist.

**4. What stops someone running up a bill on a public site?**
The API key lives only in Vercel's environment settings. Questions are
capped per visitor (5 per 10 minutes, 20 a day) and for the whole site (100 a
day), questions are limited to 500 characters, and each answer is limited in
length and in the number of tool rounds. The rate limit is in server memory,
so it is a brake, not a lock: the real backstop is a monthly spend limit in
the Anthropic Console. With no key, the panel simply says it is unavailable.

**5. What are its limits?**
The checker confirms a number came from a tool, not that it was used
correctly: the model could quote Senna's rating as Prost's. It cannot catch
wrong words around right numbers. It only knows what the tools expose. And
the live evaluation score has not been measured yet, because that needs a
paid API key; until it is, I would not claim an accuracy figure for it.

## How far can prediction go? (the third round of work)

**What was done.** Three improvements were tested race by race over 2010 to
2025, each race predicted only from earlier races, and compared with a
"ceiling": the model fitted with the season's own results.

| | Race | Log loss | Qualifying |
|---|---|---|---|
| Current model | 59.6% | 0.6715 | 61.3% |
| Refit before every race | 62.9% | 0.6444 | 64.9% |
| + same weekend's qualifying | 68.8% | 0.6137 | |
| + experience curve | 60.1% | 0.6695 | 62.1% |
| All combined | 68.6% | 0.6035 | 66.3% |
| Whoever qualified ahead (simple rule) | 68.3% | 0.6261 | |
| Hindsight ceiling / ceiling + qualifying | 65.9% / 68.8% | 0.6120 / 0.5903 | 69.8% |

### Five questions you are likely to be asked about it

**1. How do you know no prediction saw the future?**
Every prediction records the last race in the data its fit was made from,
read from the fitted data itself rather than from what I intended, and the
script stops if any of the 15,788 fair predictions saw its own race or a
later one. There are tests for the data window, for "round 3 is predicted
from rounds 1 and 2 only", and that the check really stops a leaked
prediction. The second step that adds qualifying learns each season's
weights only from earlier seasons, and a test checks that too.

**2. Some seasons went above 75%. Is that leakage?**
I checked, because a jump like that is the classic sign of a leak. Three
things. The qualifying data really comes from qualifying sessions. The
simple rule "whoever qualified ahead finishes ahead" was just as high in
those seasons (75.1% and 76.4% in 2021 and 2024), so qualifying was simply
very predictive then. And a shuffle test: give every race the previous
race's qualifying instead of its own, and the combined model falls from
68.6% to 63.2%, the level without qualifying. So the gain comes from
same-weekend qualifying and nothing else.

**3. So is the model even useful, if qualifying alone gets 68.3%?**
For calling who finishes ahead, the model adds only 0.3 points on top of
qualifying, and I say so. It adds more in two places: its probabilities are
better (log loss 0.6035 against 0.6261), and before qualifying, or for
drivers who never shared a car, there is no qualifying result to use. The
in-season refit is the biggest gain that comes from the model itself: +3.3
points for races and +3.6 for qualifying.

**4. Why is the "ceiling" below your best fair model?**
Because the ceiling, as defined, is a season-level model: it knows every
result of the season but gives each pair one skill gap for the whole year.
The fair model also knows who was quicker in that weekend's qualifying, which
is race-specific information. Compared with the matching reference (hindsight
plus qualifying, 68.8%), the fair model is 0.2 points behind. The bigger
lesson is that single races are mostly luck: even knowing the whole season,
about a third of teammate battles cannot be called.

**5. The experience curve improved predictions. Why are you cautious about it?**
It improved every backtest measure, so by my rule it stays. But it changed
the ratings more than that small gain suggests: 68 drivers moved more than
10 places, and only 4 of the top 20 now keep their rank under all three
assumptions about how the level of drivers changed between decades, down
from 10. The backtest only tests predictions within the modern era; it
cannot test comparisons across eras, so it cannot see that cost. I report
both.
