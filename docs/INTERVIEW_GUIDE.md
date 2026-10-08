# Interview guide

How each part of F1START works, in plain English, and the questions you are
most likely to be asked. Numbers are in the README; reasons are in
DECISIONS.md (the numbers in brackets below point there).

## The project in thirty seconds

"I built a race strategy simulator from public F1 timing data. A neural
network predicts each lap's time as a range, a simulator runs thousands of
races in parallel on the GPU, and a reinforcement learning agent makes pit
stop decisions lap by lap. I validated it by replaying real races, and I
found and documented the ways the optimisers exploited my simulator's
flaws, which taught me more than the headline numbers did."

## How each part works

**Data (`download_data.py`, `clean_data.py`).** Four seasons from FastF1 into
one table with one row per driver per lap. Seasons are split by time: train
on 2022 to 2023, tune on 2024, test once on 2025 (4).

**The target (`clean_data.py`).** Lap time is split in two: how fast the
whole field runs in that race, and how one lap differs from that. The model
predicts the second part, because that is what strategy depends on (12).

**The lap time network (`train_network.py`).** A small network with inputs:
compound, tyre age, laps remaining (a stand-in for fuel), gap to the car
ahead, qualifying gap, and circuit. It outputs a centre and a spread, trained
with a loss that punishes both wide ranges and overconfident misses (16).

**The simulator (`simulator.py`).** Every lap, for every car in every
simulated race at once: ask the network for a lap time, add luck, apply pit
loss, traffic and safety car rules. All as tensor operations, so thousands
of races advance together (22).

**Replay validation (`replay_validation.py`).** Replay real races with the
real strategies and compare the finishing order with reality, against the
baseline "everyone finishes where they started" (25).

**Brute force (`strategy_search.py`).** Try about a thousand one-stop and
two-stop plans for one car, each simulated in the same random races.

**The AI strategist (`strategy_env.py`, `train_agent.py`).** The race as a
game: each lap the agent sees 23 numbers and chooses stay out or pit for a
compound. It is trained with PPO over thousands of parallel races (26 to 28).

## Likely questions

**Why not predict the raw lap time?**
It is dominated by which circuit it is. My first model did that against the
pole lap and was worse than guessing the average, because how far below
qualifying pace a race runs varies from 5% to 13%. Splitting that race-level
part out fixed it (9, 12).

**How did you avoid data leakage?**
Three ways. Seasons are split by time, never at random, because laps of one
race are near copies of each other. Every input is known before the lap is
driven: the gap ahead is measured at the start of the lap, and car pace comes
from qualifying, not from the race. And input scaling is measured on training
laps only (4, 7, 10).

**Where does the split still leak, honestly?**
The race pace level is computed from the whole race, so it is only known
afterwards. That is fine for replaying past races, which is the use case, but
my lap time error is not a forecast made before a race. I report the
forecast error of that level separately (12).

**Why LightGBM as a baseline, and did the network beat it?**
Gradient boosting is the standard strong baseline for tabular data. The
network was about 2% better on dry laps in validation: essentially level.
The network is used because it gives a range, and because it runs on the
GPU inside the simulator (14, 16).

**How does the network predict uncertainty?**
It outputs a centre and a spread and is trained with the Gaussian negative
log likelihood: log of the spread, plus the squared miss measured in
spreads. The first term stops it claiming huge ranges, the second punishes
confident misses, so the best strategy is an honest spread (16).

**How do you know the uncertainty is right?**
Calibration: count how many real laps fall inside the 90% range. It was 87%,
so I widened every spread by one factor fitted on 2024, then checked it on
2025, which is the only honest check (18).

**Why did you add a per-driver offset in the simulator?**
Lap errors are not independent: a car is often quicker or slower all race
than qualifying suggested. Treating every lap as independent luck averages
out over 60 laps and makes simulated races far too predictable (24).

**How did you vectorise the simulator?**
State is tensors shaped (races, cars). One step computes every car's lap in
every race with one network call. The only loop is over 20 grid positions,
for the overtaking rule. Comparing strategies uses shared random numbers, so
every plan faces the same simulated races.

**What is PPO, in your own words?**
The agent plays races with its current policy, with some randomness. A
second network estimates how well a race should go from each point. Decisions
that turned out better than that estimate are made more likely. "Proximal"
means each update may only change the policy a little, which keeps learning
stable (28).

**How did you design the reward?**
Each lap: time lost to the median car, so a safety car, which slows
everyone, costs nothing. At the finish: a penalty per finishing place. Time
gives a signal every lap; position is the real goal (27).

**Did the agent beat brute force?**
No. With the same tyres as the team, brute force was ahead on average. They
were level in races with a safety car, where reacting matters. Brute force
optimises one car in one race over a thousand plans; the agent is one small
network for every car at every circuit (30).

**Did the agent exploit the simulator?**
Yes, and so did brute force. Both loved soft tyres far more than real teams
do. The cause is in the data: compound names are relative per race, teams
only run softs where they last, and drivers nurse tyres on long stints. I
tightened a guard on tyre age, which barely helped, and then added a fairer
comparison restricted to the tyres each team really used (29).

**So are real teams losing a place per race?**
No. That number mixes real inefficiency with my simulator's bias towards
fewer stops, and with things it cannot see: tyre sets available, damage,
covering rivals. The trustworthy comparison is between methods inside the
same simulator, not against reality (29).

**What would you do next?**
Get the real compound (C1 to C5) per race, which would fix the biggest flaw.
Model tyre management by giving the lap model the planned stint length.
Train the agent against rivals that also adapt. Add rain.

**What was the hardest bug?**
The first brute force result said fixed plans beat teams by two places in
90% of races. Treating a result that good as a symptom, and finding the soft
tyre bias behind it, was the most useful thing I did.

**What does the GPU actually do here?**
The lap time network is tiny; the GPU does not matter for training it. It
matters for the simulator: a search over a thousand strategies for one car,
each simulated many times, takes a few seconds, and PPO plays 4,096 races
per round.
