# model.py
#
# What it does: the core model. It gives every driver a skill number for every
#   season they raced, with an honest uncertainty, using only comparisons
#   between teammates (teammates share a car, so the car cancels out).
#
#   The idea in five lines:
#     1. Skill is measured in percent of lap time. +0.3 means "0.3% faster
#        than an average newcomer of today in the same car" (about 0.27s a lap).
#     2. A driver's skill may drift a little from one season to the next.
#     3. The bigger the skill gap between two teammates, the bigger the
#        expected lap time gap and the more often the better one is ahead.
#     4. Find the skills that best explain every teammate comparison since
#        1950 (the "fit").
#     5. Measure how far each skill could move without the explanation
#        getting much worse (the "uncertainty").
#
# What it reads: data/teammate_pairs.parquet and data/race_pace.parquet.
# What it produces: nothing on disk by itself. fit_model() returns a Fit
#   object holding the skills and their uncertainty.
# Which files use it: validate_model.py (tests it on 2024 and 2025) and
#   fit_ratings.py (the final ratings).

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

PAIRS_FILE = Path("data/teammate_pairs.parquet")
PACE_FILE = Path("data/race_pace.parquet")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 64 bit numbers: the uncertainty step inverts a large matrix, and 32 bit
# numbers are not precise enough for that.
DTYPE = torch.float64

# Lap time gaps between teammates bigger than this (percent) are a ruined
# lap, rain or damage, not skill, so only "who was ahead" is used for them.
MAX_BELIEVABLE_GAP = 3.0
MIN_CLEAN_LAPS = 15  # a race pace needs at least this many laps behind it

# How heavy the tails of the lap time gap distributions are. 4 is a common
# choice that tolerates the odd freak result without being thrown by it.
TAIL_WEIGHT = 4.0

# The two numbers that say how much drivers differ and how fast they change.
# They are chosen in validate_model.py using only seasons up to 2023.
DEFAULT_SKILL_SPREAD = 0.3  # spread of skill between drivers, in percent
DEFAULT_SEASON_DRIFT = 0.18  # typical change in a driver's skill per season

# How much the level of the average newcomer may differ from one decade to
# the next. This is an ASSUMPTION, not a measurement: see DECISIONS.md 17.
ERA_DRIFT = 0.1

# Qualifying before 2006 was run in formats (one lap each, race fuel on
# board) that made the gap between teammates noisier, so it gets its own
# noise level and does not blur the modern data.
FIRST_MODERN_QUALI_YEAR = 2006

# Where only "who qualified ahead" is known (every race before 1994), the
# chance of being ahead uses an S-curve whose slope is this number divided by
# the qualifying noise. 1.5 makes that S-curve agree with the lap time model
# (it is 4 x the height of the heavy-tailed bell curve at its centre), so a
# skill gap means the same chance of out-qualifying a teammate in every era.
# Why it must be tied and not learned freely: DECISIONS.md 16.
AHEAD_SLOPE_PER_NOISE = 1.5

KINDS = ["race_ahead", "quali_ahead", "quali_gap", "old_quali_gap", "pace_gap"]
SCALE_NAMES = ["race_slope", "pace_factor", "quali_noise", "old_quali_noise", "pace_noise"]


@dataclass
class Fit:
    """Everything a fitted model knows."""
    index: dict  # (driver, season) -> position in the skill list
    skill: torch.Tensor  # best estimate of each driver-season's skill
    era_level: dict  # decade -> level of the average newcomer (latest decade = 0)
    covariance: torch.Tensor  # how uncertain the skills are, and how they move together
    scales: dict  # the learned link between skill gaps and results
    skill_spread: float
    season_drift: float


# --- Step 1: turn the tables into comparisons the model can read -----------

def build_comparisons(pairs: pl.DataFrame, pace: pl.DataFrame, last_year: int) -> pl.DataFrame:
    """One row per teammate pair per race, up to `last_year`, with up to five results.

    race_ahead:  1 if A finished ahead, 0 if B did (empty if a car failed).
    quali_gap:   A's qualifying lap against B's, in percent (negative = A faster).
                 Before 2006 it goes in old_quali_gap instead (noisier formats).
    quali_ahead: 1 if A qualified ahead. Only used when there is no usable
                 lap time gap, so the same lap is never counted twice.
    pace_gap:    A's race pace against B's, in percent (2018 onwards).
    weight:      each driver's comparisons in one race add up to 1. Without
                 this, a 1950s constructor with 10 cars would give each of its
                 drivers 9 comparisons per race and drown out everyone else.
    """
    driver_pace = pace.filter(pl.col("CleanLaps") >= MIN_CLEAN_LAPS).select(
        "Year", "Round", "DriverId", "PacePct"
    )
    believable = (pl.col("QualiGapPct").abs() <= MAX_BELIEVABLE_GAP).fill_null(False)
    modern = pl.col("Year") >= FIRST_MODERN_QUALI_YEAR
    pace_gap = pl.col("PaceA") - pl.col("PaceB")
    return (
        pairs.filter(pl.col("Year") <= last_year)
        .join(driver_pace.rename({"DriverId": "DriverA", "PacePct": "PaceA"}),
              on=["Year", "Round", "DriverA"], how="left")
        .join(driver_pace.rename({"DriverId": "DriverB", "PacePct": "PaceB"}),
              on=["Year", "Round", "DriverB"], how="left")
        .select(
            "Year", "Round", "DriverA", "DriverB",
            pl.col("AAheadRace").cast(pl.Float64).alias("race_ahead"),
            pl.when(~believable).then(pl.col("AAheadQuali").cast(pl.Float64)).alias("quali_ahead"),
            pl.when(believable & modern).then(pl.col("QualiGapPct")).alias("quali_gap"),
            pl.when(believable & ~modern).then(pl.col("QualiGapPct")).alias("old_quali_gap"),
            pl.when(pace_gap.abs() <= MAX_BELIEVABLE_GAP).then(pace_gap).alias("pace_gap"),
            (1.0 / (pl.col("TeamCars") - 1)).alias("weight"),
        )
    )


def driver_season_index(comparisons: pl.DataFrame) -> dict:
    """Give every (driver, season) that appears a position in the skill list."""
    a = comparisons.select(pl.col("DriverA").alias("Driver"), "Year")
    b = comparisons.select(pl.col("DriverB").alias("Driver"), "Year")
    seasons = pl.concat([a, b]).unique().sort("Driver", "Year")
    return {key: position for position, key in enumerate(seasons.iter_rows())}


def to_tensors(comparisons: pl.DataFrame, index: dict) -> dict:
    """For each kind of result: which two skills are compared, the result, the weight."""
    a = [index[key] for key in comparisons.select("DriverA", "Year").iter_rows()]
    b = [index[key] for key in comparisons.select("DriverB", "Year").iter_rows()]
    table = comparisons.with_columns(pl.Series("a", a), pl.Series("b", b))
    tensors = {}
    for kind in KINDS:
        rows = table.filter(pl.col(kind).is_not_null())
        tensors[kind] = {
            "a": torch.tensor(rows["a"].to_list(), device=DEVICE, dtype=torch.long),
            "b": torch.tensor(rows["b"].to_list(), device=DEVICE, dtype=torch.long),
            "value": torch.tensor(rows[kind].to_list(), device=DEVICE, dtype=DTYPE),
            "weight": torch.tensor(rows["weight"].to_list(), device=DEVICE, dtype=DTYPE),
        }
    return tensors


def decade_of(year: int) -> int:
    return year // 10 * 10


def career_links(index: dict) -> dict:
    """Describe each career: its first season, and each step to the next season.

    Returns positions in the skill list: `first` (each driver's first season)
    with `first_era` (which decade that was, counted from the earliest), and
    `earlier` / `later` (consecutive seasons of the same driver) with `years`
    between them (2 if the driver sat out a season).
    """
    decades = sorted({decade_of(year) for _, year in index})
    first, first_era, earlier, later, years = [], [], [], [], []
    previous = None
    for (driver, year), position in index.items():  # sorted by driver, then year
        if previous is None or previous[0] != driver:
            first.append(position)
            first_era.append(decades.index(decade_of(year)))
        else:
            earlier.append(position - 1)
            later.append(position)
            years.append(year - previous[1])
        previous = (driver, year)

    def positions(values):
        return torch.tensor(values, device=DEVICE, dtype=torch.long)

    return {
        "decades": decades, "first": positions(first), "first_era": positions(first_era),
        "earlier": positions(earlier), "later": positions(later),
        "years": torch.tensor(years, device=DEVICE, dtype=DTYPE),
    }


# --- Step 2: how well does a set of skills explain the results? ------------

def heavy_tailed_log_chance(value, centre, scale):
    """Log of the chance of seeing `value` when `centre` was expected.

    A bell curve with heavier tails (a "Student t"). With an ordinary bell
    curve one freak qualifying lap would drag a driver's whole rating; with
    heavy tails a freak result is merely surprising.
    """
    return torch.distributions.StudentT(TAIL_WEIGHT, centre, scale).log_prob(value)


def quali_slope(scales: dict):
    """How reliably a skill gap turns into qualifying ahead (see AHEAD_SLOPE_PER_NOISE)."""
    return AHEAD_SLOPE_PER_NOISE / scales["quali_noise"]


def log_likelihood(skill, scales, results) -> torch.Tensor:
    """How well these skills explain every teammate comparison (higher is better).

    For each comparison the only thing that matters is the skill GAP between
    the two teammates, because they share a car.
    """
    def gap(kind):  # skill of A minus skill of B, for each comparison of this kind
        return skill[results[kind]["a"]] - skill[results[kind]["b"]]

    def total(kind, log_chance):
        return (results[kind]["weight"] * log_chance).sum()

    def ahead(kind, slope):
        # "Who was ahead" results: the chance A is ahead rises smoothly from
        # 0 to 1 as the skill gap grows (an S-shaped "logistic" curve). The
        # slope says how reliably a skill gap turns into being ahead.
        return total(kind, -torch.nn.functional.binary_cross_entropy_with_logits(
            slope * gap(kind), results[kind]["value"], reduction="none"))

    def lap_gap(kind, factor, noise):
        # Lap time gaps: A is expected to be quicker by the skill gap times a
        # factor. "Quicker" is a negative time gap, hence the minus sign.
        return total(kind, heavy_tailed_log_chance(
            results[kind]["value"], -factor * gap(kind), noise))

    return (
        ahead("race_ahead", scales["race_slope"])
        + ahead("quali_ahead", quali_slope(scales))
        # The factor is fixed at 1 for qualifying. That is what DEFINES the
        # unit of skill: 0.3 more skill = 0.3% quicker over a qualifying lap.
        + lap_gap("quali_gap", 1.0, scales["quali_noise"])
        + lap_gap("old_quali_gap", 1.0, scales["old_quali_noise"])
        # Race pace gaps follow the same skill gap, times a learned factor.
        + lap_gap("pace_gap", scales["pace_factor"], scales["pace_noise"])
    )


def log_prior(skill, era_level, links, skill_spread, season_drift) -> torch.Tensor:
    """What is believed about skills before seeing any result (higher is more believable).

    1. A driver's first season is probably near the level of newcomers in
       their era, give or take `skill_spread`. This keeps a driver with three
       races from being rated a genius or a disaster on thin evidence.
    2. Each season a driver's skill is probably near last season's, give or
       take `season_drift`. This lets careers rise and fall, but smoothly.
    3. The level of newcomers may differ between decades, but probably not by
       much from one decade to the next (`ERA_DRIFT`).
    """
    normal = torch.distributions.Normal
    expected_first = era_level[links["first_era"]]
    first_season = normal(expected_first, skill_spread).log_prob(skill[links["first"]]).sum()
    change = skill[links["later"]] - skill[links["earlier"]]
    # A two year gap allows more change than one: uncertainty adds up over time.
    drift = normal(0.0, season_drift * links["years"].sqrt()).log_prob(change).sum()
    era_steps = normal(0.0, ERA_DRIFT).log_prob(era_level[1:] - era_level[:-1]).sum()
    return first_season + drift + era_steps


def split(unknowns: torch.Tensor, skill_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the list of unknowns into skills and era levels.

    The latest decade's level is not an unknown: it is fixed at 0, because
    only differences can be measured and something has to be the reference.
    So every rating reads as "compared with a newcomer of today".
    """
    skill, earlier_levels = unknowns[:skill_count], unknowns[skill_count:]
    return skill, torch.cat([earlier_levels, earlier_levels.new_zeros(1)])


def positive_scales(raw: torch.Tensor) -> dict:
    """Unpack the learned scale numbers, forcing each to be positive."""
    return dict(zip(SCALE_NAMES, raw.exp()))  # exp() of any number is positive


def badness(unknowns, raw_scales, results, links, skill_count, skill_spread, season_drift):
    """One number to make as small as possible: minus (fit to results + prior belief)."""
    skill, era_level = split(unknowns, skill_count)
    return -(log_likelihood(skill, positive_scales(raw_scales), results)
             + log_prior(skill, era_level, links, skill_spread, season_drift))


# --- Step 3: find the best skills, then how uncertain they are -------------

def find_best_skills(results, links, skill_count, skill_spread, season_drift):
    """Find the skills, era levels and scales that make the badness smallest."""
    unknown_count = skill_count + len(links["decades"]) - 1
    unknowns = torch.zeros(unknown_count, device=DEVICE, dtype=DTYPE, requires_grad=True)
    raw_scales = torch.zeros(len(SCALE_NAMES), device=DEVICE, dtype=DTYPE, requires_grad=True)
    # L-BFGS is an optimiser for smooth problems where all the data fits in
    # memory at once. It needs far fewer steps here than Adam would.
    optimiser = torch.optim.LBFGS([unknowns, raw_scales], max_iter=1000,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def step():
        optimiser.zero_grad()
        loss = badness(unknowns, raw_scales, results, links, skill_count, skill_spread, season_drift)
        loss.backward()
        return loss

    optimiser.step(step)
    return unknowns.detach(), raw_scales.detach()


def uncertainty(unknowns, raw_scales, results, links, skill_count, skill_spread, season_drift):
    """How uncertain each skill is (the "Laplace approximation").

    Picture the badness as a valley with the best skills at the bottom. If
    the valley is steep around a driver's skill, moving it even slightly
    explains the results much worse, so that skill is known precisely. If the
    valley is flat (few races, few teammates), it could be moved a long way
    at little cost: wide uncertainty.

    The steepness in every direction at once is a table called the Hessian.
    Its inverse is the covariance: each skill's uncertainty on the diagonal,
    and off the diagonal how two ratings move together (teammates are tied
    together; drivers from distant eras are only loosely linked).
    """
    def badness_of(values):
        return badness(values, raw_scales, results, links, skill_count, skill_spread, season_drift)

    hessian = torch.autograd.functional.hessian(badness_of, unknowns, vectorize=True)
    return torch.linalg.inv(hessian)


def fit_model(pairs, pace, last_year, skill_spread=DEFAULT_SKILL_SPREAD,
              season_drift=DEFAULT_SEASON_DRIFT) -> Fit:
    """Fit the model on every teammate comparison up to and including `last_year`."""
    comparisons = build_comparisons(pairs, pace, last_year)
    index = driver_season_index(comparisons)
    results, links = to_tensors(comparisons, index), career_links(index)
    settings = (results, links, len(index), skill_spread, season_drift)
    unknowns, raw_scales = find_best_skills(*settings)
    covariance = uncertainty(unknowns, raw_scales, *settings)
    skill, era_level = split(unknowns, len(index))
    return Fit(
        index=index, skill=skill,
        era_level=dict(zip(links["decades"], era_level.tolist())),
        covariance=covariance,
        scales={name: value.item() for name, value in positive_scales(raw_scales).items()},
        skill_spread=skill_spread, season_drift=season_drift,
    )


# --- Step 4: use a fitted model ---------------------------------------------

def latest_season(fit: Fit, driver: str, year: int):
    """The driver's most recent fitted season up to `year`, or None for a newcomer."""
    seasons = [season for (name, season) in fit.index if name == driver and season <= year]
    return max(seasons) if seasons else None


def skill_gap(fit: Fit, driver_a: str, driver_b: str, year: int) -> tuple[float, float]:
    """The expected skill gap (A minus B) in a given season, and its variance.

    For a season after the last fitted one, each driver is expected to stay
    where they were, with extra uncertainty for every year that has passed.
    A driver the model has never seen is assumed to be an average newcomer
    (0), give or take the whole spread of F1 drivers.
    """
    means, positions, variance = [], [], 0.0
    for driver in (driver_a, driver_b):
        season = latest_season(fit, driver, year)
        if season is None:
            means.append(0.0)
            variance += fit.skill_spread ** 2
        else:
            position = fit.index[(driver, season)]
            positions.append(position)
            means.append(fit.skill[position].item())
            variance += fit.covariance[position, position].item()
            variance += fit.season_drift ** 2 * (year - season)
    if len(positions) == 2:
        # Two drivers whose ratings move together (old teammates) have a more
        # certain GAP than their separate uncertainties suggest.
        variance -= 2 * fit.covariance[positions[0], positions[1]].item()
    return means[0] - means[1], variance


def chance_ahead(gap: float, variance: float, slope: float) -> float:
    """The chance A is ahead of B, allowing for not knowing the gap exactly.

    With a known gap the chance is the S-curve: 1 / (1 + exp(-slope x gap)).
    When the gap is uncertain the honest answer is the average of that curve
    over every plausible gap, which pulls the chance towards 50%. The line
    below is a standard, accurate shortcut for that average.
    """
    softened = slope * gap / (1 + torch.pi * slope ** 2 * variance / 8) ** 0.5
    return torch.sigmoid(torch.tensor(softened)).item()


def skill_table(fit: Fit) -> pl.DataFrame:
    """The fitted skills as a table: one row per driver per season, with uncertainty."""
    spread = fit.covariance.diagonal().sqrt()
    rows = [(driver, year, fit.skill[position].item(), spread[position].item())
            for (driver, year), position in fit.index.items()]
    return pl.DataFrame(rows, schema=["DriverId", "Year", "Skill", "SkillSd"], orient="row")


# --- Step 5: the car, once the driver is known -----------------------------

def car_strengths(results: pl.DataFrame, fit: Fit, last_year: int) -> tuple[pl.DataFrame, float]:
    """How good each team's car was each season, with the driver taken out.

    A race result is treated as: car + driver_weight x driver skill + luck,
    where the result is PositionScore (1 = won, 0 = last starter).

    1. driver_weight is found by comparing teammates only: within one team in
       one season the car is the same, so if the driver with 0.1 more skill
       scores 0.05 higher on average, the weight is 0.5.
    2. The car is then what is left of the team's average result after taking
       out what its drivers contributed.

    Returns (one row per team per season with CarStrength, driver_weight).
    Races that ended with a car failure are left out, as everywhere else.
    """
    team_season = ["TeamId", "Year"]
    rows = (
        results.filter((pl.col("Year") <= last_year) & pl.col("Outcome").is_in(["finished", "driver"])
                       & pl.col("PositionScore").is_not_null() & ~pl.col("IsIndy500"))
        .join(skill_table(fit), on=["DriverId", "Year"], how="left")
        .with_columns(pl.col("Skill").fill_null(0.0))  # no teammate ever: assume average
    )
    centred = rows.with_columns(
        (pl.col("Skill") - pl.col("Skill").mean().over(team_season)).alias("skill_vs_team"),
        (pl.col("PositionScore") - pl.col("PositionScore").mean().over(team_season)).alias("score_vs_team"),
    )
    driver_weight = ((centred["skill_vs_team"] * centred["score_vs_team"]).sum()
                     / (centred["skill_vs_team"] ** 2).sum())
    cars = rows.group_by(team_season).agg(
        (pl.col("PositionScore") - driver_weight * pl.col("Skill")).mean().alias("CarStrength"),
        pl.len().alias("Results"),
    )
    return cars.sort("Year", "CarStrength", descending=[False, True]), driver_weight


def load_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.read_parquet(PAIRS_FILE), pl.read_parquet(PACE_FILE)


if __name__ == "__main__":
    # A quick look: fit up to 2023 and print what was learned.
    fit = fit_model(*load_tables(), last_year=2023)
    print(f"{len(fit.index):,} driver-seasons fitted on {DEVICE}")
    print("scales:", {name: round(value, 3) for name, value in fit.scales.items()})
    print("era levels:", {decade: round(level, 3) for decade, level in fit.era_level.items()})
