# Interview guide

How to explain this project out loud. Read it next to `model.py`, which is the
one file you must know well. Every number here comes from a script in this
repository; if you re-run them and a number changes, trust the script.

## The 30 second version

"F1 results mix up the driver and the car. I built a model that separates
them using teammates, because teammates share a car. It is a Bayesian
hierarchical model I wrote in PyTorch: every driver gets a skill for every
season since 1950, with an uncertainty. I trained it on data up to 2023 and
tested it on 2024 and 2025 without refitting. It calls the right teammate 67%
of the time against 62% for the best simple baseline, though in 2024 alone it
only matched the baseline. Then I built a GPU simulation of equal car races
and a website where you can race any drivers from any era."

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
- Calibration: when the model said 80% or more, the favourite won 39 of 39.
  The model is under-confident at the top.

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
covariance matrix carries that: the range for Senna against Verstappen is 26%
to 78%.

**3. Why is it Bayesian? What does the prior buy you?**
Three things. It handles small samples: a driver with five races is pulled
towards average. It ties seasons together, so one freak season does not
define a career. And it gives a principled uncertainty for every rating,
which a plain rating system like Elo does not.

**4. Why a Laplace approximation and not MCMC?**
It is fast (seconds), it fits in 150 lines I can explain, and for a model
that is close to bell shaped it is a good approximation. The cost: it assumes
bell shaped uncertainty and treats the five learned scale numbers as exactly
known, so the ranges are somewhat too narrow. With more time I would run MCMC
(for example NUTS) on the same model and compare the ranges.

**5. Your model lost to the baseline in 2024. Is it any good?**
In 2024 almost every team kept its 2023 drivers, so "last season repeats" was
close to the ceiling, and my model matched it (65.4% against 66.0%, a
difference of one comparison in 191). In 2025 drivers moved and rookies
arrived; the baselines fell to 58% and the model stayed at 69%. The model's
value is exactly the case the baselines cannot handle: pairs with no shared
history. I would also say that 190 comparisons per season gives a margin of
about plus or minus 7 points, so I do not over-read either year.

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
I do not claim it is a realistic race: it has no tyres, pit stops or safety
cars. What I did is calibrate it to the validated model. I tuned one number,
the day to day form, until simulated drivers beat each other as often as the
model says real teammates do: for a 0.4% skill gap the model says 76.2% and
the simulation gives 76.2%. So its randomness is the right size, even though
its mechanics are simple. The overtaking difficulty per circuit is measured
from real races since 2018, and Monaco comes out hardest, which is a good
sanity check.

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
| Teammate race prediction, 2024 and 2025 | 67.0% (baseline 62.0%), 376 comparisons |
| Teammate qualifying prediction | 70.1% (baseline 65.0%), 472 comparisons |
| Typical uncertainty of a peak rating | 0.30% in the 1950s, 0.16% in the 2020s |
| Fit time on the GPU | about 15 seconds |
| Simulation speed | 20,000 races in under a second |
| Tests | 48 |
