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
#        getting much worse (the "uncertainty"). sample_posterior.py checks
#        this step with a sampler that makes fewer assumptions.
#
#   Three optional parts, each tested by backtest.py before being switched on
#   (see Options below): documented team orders, a weak-teammate correction,
#   and a separate skill adjustment for wet races.
#
# What it reads: data/teammate_pairs.parquet, data/race_pace.parquet and
#   data/wet_races.parquet.
# What it produces: nothing on disk by itself. fit_model() returns a Fit
#   object holding the skills and their uncertainty.
# Which files use it: backtest.py, validate_model.py, fit_ratings.py,
#   sample_posterior.py and explain_ratings.py.

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch

PAIRS_FILE = Path("data/teammate_pairs.parquet")
PACE_FILE = Path("data/race_pace.parquet")
WET_FILE = Path("data/wet_races.parquet")

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
# They were chosen in validate_model.py using only seasons up to 2023.
DEFAULT_SKILL_SPREAD = 0.3  # spread of skill between drivers, in percent
DEFAULT_SEASON_DRIFT = 0.18  # typical change in a driver's skill per season

# How much the level of the average newcomer may differ from one decade to
# the next. This is an ASSUMPTION, not a measurement: see DECISIONS.md 17,
# and sensitivity.py for what happens with other values.
ERA_DRIFT = 0.1

# How far a driver's wet weather skill may plausibly differ from their dry
# skill, before seeing any wet race. An assumption: see DECISIONS.md 30.
WET_SPREAD = 0.25

# The experience curve (Options.experience) has one value per year in F1, up
# to this many; later years share the last value. Neighbouring years are
# probably close (EXPERIENCE_STEP), which keeps the curve smooth.
MAX_EXPERIENCE = 10
EXPERIENCE_STEP = 0.1

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

# How confident the race predictions should be: every race chance is
# computed with the slope multiplied by this. Below 1 means the raw model was
# too confident when predicting seasons it had not seen. See DECISIONS.md 27.
# TEST_TEMPERATURE was fitted by backtest.py on 2010 to 2023 predictions only,
# and is what the 2024 and 2025 test uses. RACE_TEMPERATURE is the same fit on
# every season 2010 to 2025, made only after that test was reported (the same
# rule as refitting the model on all data), and is what the website uses.
TEST_TEMPERATURE = 0.45
RACE_TEMPERATURE = 0.55

# Team-seasons where one driver was the documented number one, with the
# other car under contract or under publicly reported team orders to support
# them. In these, the RACE result between the two may have been arranged, so
# it can be left out (qualifying still counts). Hand made from well known,
# widely reported cases; not complete. See DECISIONS.md 28.
NUMBER_ONE_SEASONS = {
    **{(year, "lotus"): "andretti" for year in [1978]},  # Peterson's contract
    **{(year, "williams"): "mansell" for year in [1992]},
    **{(year, "benetton"): "michael_schumacher" for year in [1994, 1995]},
    **{(year, "ferrari"): "michael_schumacher" for year in range(1996, 2006)},  # Irvine, then Barrichello
    **{(year, "ferrari"): "alonso" for year in range(2010, 2014)},  # Massa ("Fernando is faster than you")
    **{(year, "red_bull"): "max_verstappen" for year in range(2021, 2025)},  # Pérez
}

KINDS = ["race_ahead", "quali_ahead", "quali_gap", "old_quali_gap", "pace_gap"]
SCALE_NAMES = ["race_slope", "pace_factor", "quali_noise", "old_quali_noise", "pace_noise"]
# The wet adjustment applies to race results only (qualifying may have been dry).
WET_KINDS = ["race_ahead", "pace_gap"]


@dataclass
class Options:
    """The optional parts of the model. Each is kept only if backtest.py shows it does not hurt."""
    number_one: bool = False  # leave out race results arranged by documented team orders
    career_prior: bool = False  # drivers with short careers start from a lower expected skill
    wet: bool = False  # a separate skill adjustment for each driver in wet races
    experience: bool = True  # a shared curve: how drivers typically improve or decline with years in F1


@dataclass
class Fit:
    """Everything a fitted model knows."""
    index: dict  # (driver, season) -> position in the skill list
    skill: torch.Tensor  # best estimate of each driver-season's skill
    era_level: dict  # decade -> level of the average newcomer (latest decade = 0)
    covariance: torch.Tensor  # how uncertain all the unknowns are, and how they move together
    scales: dict  # the learned link between skill gaps and results
    skill_spread: float
    season_drift: float
    options: Options = field(default_factory=Options)
    wet: dict = field(default_factory=dict)  # driver -> (wet adjustment, its uncertainty, wet races)
    career_weight: float = 0.0  # the learned weak-teammate correction (Options.career_prior)
    layout: dict = field(default_factory=dict)  # where each part sits in the list of unknowns
    data: dict = field(default_factory=dict)  # the comparisons, kept for sampling and explanations
    experience_curve: list = field(default_factory=list)  # skill added in year 0, 1, 2... of a career
    first_year: dict = field(default_factory=dict)  # driver -> first season in the data


# --- Step 1: turn the tables into comparisons the model can read -----------

def data_window(last_year: int, before_race: tuple[int, int] | None = None) -> pl.Expr:
    """Which races a fit is allowed to see.

    Normally every race up to the end of `last_year`. With `before_race`
    (season, round), only races strictly BEFORE that race: earlier seasons,
    plus earlier rounds of the same season. This is what stops a prediction
    from seeing its own race or anything after it.
    """
    if before_race is None:
        return pl.col("Year") <= last_year
    year, race = before_race
    return (pl.col("Year") < year) | ((pl.col("Year") == year) & (pl.col("Round") < race))


def build_comparisons(pairs: pl.DataFrame, pace: pl.DataFrame, last_year: int,
                      wet_races: pl.DataFrame | None = None, options: Options | None = None,
                      before_race: tuple[int, int] | None = None) -> pl.DataFrame:
    """One row per teammate pair per race, up to `last_year` (or before `before_race`).

    race_ahead:  1 if A finished ahead, 0 if B did (empty if a car failed, or
                 if the result was arranged by documented team orders and
                 Options.number_one is on).
    quali_gap:   A's qualifying lap against B's, in percent (negative = A faster).
                 Before 2006 it goes in old_quali_gap instead (noisier formats).
    quali_ahead: 1 if A qualified ahead. Only used when there is no usable
                 lap time gap, so the same lap is never counted twice.
    pace_gap:    A's race pace against B's, in percent (2018 onwards).
    wet:         1 if the race was wet (see wet_races.py).
    weight:      each driver's comparisons in one race add up to 1. Without
                 this, a 1950s constructor with 10 cars would give each of its
                 drivers 9 comparisons per race and drown out everyone else.
    """
    options = options or Options()
    driver_pace = pace.filter(pl.col("CleanLaps") >= MIN_CLEAN_LAPS).select(
        "Year", "Round", "DriverId", "PacePct"
    )
    believable = (pl.col("QualiGapPct").abs() <= MAX_BELIEVABLE_GAP).fill_null(False)
    modern = pl.col("Year") >= FIRST_MODERN_QUALI_YEAR
    pace_gap = pl.col("PaceA") - pl.col("PaceB")
    table = (
        pairs.filter(data_window(last_year, before_race))
        .join(driver_pace.rename({"DriverId": "DriverA", "PacePct": "PaceA"}),
              on=["Year", "Round", "DriverA"], how="left")
        .join(driver_pace.rename({"DriverId": "DriverB", "PacePct": "PaceB"}),
              on=["Year", "Round", "DriverB"], how="left")
        .with_columns(arranged_by_team_orders(pairs) if options.number_one else pl.lit(False).alias("arranged"))
    )
    wet = pl.lit(0.0)
    if wet_races is not None:
        table = table.join(wet_races.select("Year", "Round", pl.lit(1.0).alias("is_wet")),
                           on=["Year", "Round"], how="left")
        wet = pl.col("is_wet").fill_null(0.0)
    return table.select(
        "Year", "Round", "DriverA", "DriverB",
        pl.when(~pl.col("arranged")).then(pl.col("AAheadRace").cast(pl.Float64)).alias("race_ahead"),
        pl.when(~believable).then(pl.col("AAheadQuali").cast(pl.Float64)).alias("quali_ahead"),
        pl.when(believable & modern).then(pl.col("QualiGapPct")).alias("quali_gap"),
        pl.when(believable & ~modern).then(pl.col("QualiGapPct")).alias("old_quali_gap"),
        pl.when(pace_gap.abs() <= MAX_BELIEVABLE_GAP).then(pace_gap).alias("pace_gap"),
        wet.alias("wet"),
        (1.0 / (pl.col("TeamCars") - 1)).alias("weight"),
    )


def arranged_by_team_orders(pairs: pl.DataFrame) -> pl.Expr:
    """True for pairs in a documented number-one team-season that include the number one."""
    if "TeamId" not in pairs.columns:
        return pl.lit(False).alias("arranged")
    keys = [f"{year}|{team}|{driver}" for (year, team), driver in NUMBER_ONE_SEASONS.items()]

    def key(driver):
        return pl.concat_str([pl.col("Year").cast(pl.String), pl.col("TeamId"), pl.col(driver)], separator="|")

    return (key("DriverA").is_in(keys) | key("DriverB").is_in(keys)).alias("arranged")


def driver_season_index(comparisons: pl.DataFrame) -> dict:
    """Give every (driver, season) that appears a position in the skill list."""
    a = comparisons.select(pl.col("DriverA").alias("Driver"), "Year")
    b = comparisons.select(pl.col("DriverB").alias("Driver"), "Year")
    seasons = pl.concat([a, b]).unique().sort("Driver", "Year")
    return {key: position for position, key in enumerate(seasons.iter_rows())}


def to_tensors(comparisons: pl.DataFrame, index: dict, drivers: list[str]) -> dict:
    """For each kind of result: which two skills are compared, the result, the weight.

    Also which two DRIVERS (not driver-seasons) are compared and whether the
    race was wet, for the wet adjustment, which is one number per driver.
    """
    driver_position = {driver: position for position, driver in enumerate(drivers)}
    a = [index[key] for key in comparisons.select("DriverA", "Year").iter_rows()]
    b = [index[key] for key in comparisons.select("DriverB", "Year").iter_rows()]
    table = comparisons.with_columns(
        pl.Series("a", a), pl.Series("b", b),
        pl.col("DriverA").replace_strict(driver_position, return_dtype=pl.Int64).alias("driver_a"),
        pl.col("DriverB").replace_strict(driver_position, return_dtype=pl.Int64).alias("driver_b"),
    )

    def as_tensor(rows, column, dtype):
        return torch.tensor(rows[column].to_list(), device=DEVICE, dtype=dtype)

    tensors = {}
    for kind in KINDS:
        rows = table.filter(pl.col(kind).is_not_null())
        tensors[kind] = {
            "a": as_tensor(rows, "a", torch.long), "b": as_tensor(rows, "b", torch.long),
            "driver_a": as_tensor(rows, "driver_a", torch.long),
            "driver_b": as_tensor(rows, "driver_b", torch.long),
            "value": as_tensor(rows, kind, DTYPE), "weight": as_tensor(rows, "weight", DTYPE),
            "wet": as_tensor(rows, "wet", DTYPE) if kind in WET_KINDS else None,
        }
    return tensors


def decade_of(year: int) -> int:
    return year // 10 * 10


def career_links(index: dict) -> dict:
    """Describe each career: its first season, and each step to the next season.

    Returns positions in the skill list: `first` (each driver's first season)
    with `first_era` (which decade that was, counted from the earliest), and
    `earlier` / `later` (consecutive seasons of the same driver) with `years`
    between them (2 if the driver sat out a season). Also `career_length`:
    for each driver, how many seasons they have in the data, as a centred
    logarithm (used by the weak-teammate correction).
    """
    decades = sorted({decade_of(year) for _, year in index})
    first, first_era, earlier, later, years, seasons = [], [], [], [], [], []
    experience = [0] * len(index)  # years since the driver's first season, capped
    previous, first_year = None, None
    for (driver, year), position in index.items():  # sorted by driver, then year
        if previous is None or previous[0] != driver:
            first_year = year
        experience[position] = min(year - first_year, MAX_EXPERIENCE)
        if previous is None or previous[0] != driver:
            first.append(position)
            first_era.append(decades.index(decade_of(year)))
            seasons.append(1)
        else:
            earlier.append(position - 1)
            later.append(position)
            years.append(year - previous[1])
            seasons[-1] += 1
        previous = (driver, year)

    def positions(values):
        return torch.tensor(values, device=DEVICE, dtype=torch.long)

    length = torch.tensor(seasons, device=DEVICE, dtype=DTYPE).log()
    return {
        "decades": decades, "first": positions(first), "first_era": positions(first_era),
        "earlier": positions(earlier), "later": positions(later),
        "years": torch.tensor(years, device=DEVICE, dtype=DTYPE),
        "career_length": length - length.mean(),
        "experience": positions(experience),
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


def log_likelihood(skill, scales, results, wet=None) -> torch.Tensor:
    """How well these skills explain every teammate comparison (higher is better).

    For each comparison the only thing that matters is the skill GAP between
    the two teammates, because they share a car. In a wet race each driver's
    wet adjustment is added to their skill (if the model has one).
    """
    def gap(kind):  # skill of A minus skill of B, for each comparison of this kind
        rows = results[kind]
        difference = skill[rows["a"]] - skill[rows["b"]]
        if wet is not None and rows["wet"] is not None:
            difference = difference + rows["wet"] * (wet[rows["driver_a"]] - wet[rows["driver_b"]])
        return difference

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


def log_prior(parts, links, skill_spread, season_drift) -> torch.Tensor:
    """What is believed about skills before seeing any result (higher is more believable).

    1. A driver's first season is probably near the level of newcomers in
       their era, give or take `skill_spread`. This keeps a driver with three
       races from being rated a genius or a disaster on thin evidence.
       With the weak-teammate correction, drivers with longer careers start
       from a higher expected level and short careers from a lower one (by a
       learned amount), because teams keep good drivers and drop weak ones.
    2. Each season a driver's skill is probably near last season's, give or
       take `season_drift`. This lets careers rise and fall, but smoothly.
    3. The level of newcomers may differ between decades, but probably not by
       much from one decade to the next (`ERA_DRIFT`).
    4. A driver's wet adjustment is probably small (`WET_SPREAD`).
    """
    normal = torch.distributions.Normal
    skill, era_level = parts["skill"], parts["era_level"]
    expected_first = era_level[links["first_era"]]
    if parts["career_weight"] is not None:
        expected_first = expected_first + parts["career_weight"] * links["career_length"]
    first_season = normal(expected_first, skill_spread).log_prob(skill[links["first"]]).sum()
    change = skill[links["later"]] - skill[links["earlier"]]
    # A two year gap allows more change than one: uncertainty adds up over time.
    drift = normal(0.0, season_drift * links["years"].sqrt()).log_prob(change).sum()
    era_steps = normal(0.0, ERA_DRIFT).log_prob(era_level[1:] - era_level[:-1]).sum()
    total = first_season + drift + era_steps
    if parts["wet"] is not None:
        total = total + normal(0.0, WET_SPREAD).log_prob(parts["wet"]).sum()
    if parts["career_weight"] is not None:
        total = total + normal(0.0, 1.0).log_prob(parts["career_weight"]).sum()  # a weak, wide belief
    if parts["experience_curve"] is not None:
        steps = parts["experience_curve"][1:] - parts["experience_curve"][:-1]
        total = total + normal(0.0, EXPERIENCE_STEP).log_prob(steps).sum()
    return total


def unknown_layout(skill_count: int, decade_count: int, driver_count: int, options: Options) -> dict:
    """Where each part sits in the one long list of unknowns the optimiser works on."""
    layout, start = {}, 0
    for name, size in [("skill", skill_count), ("era_level", decade_count - 1),
                       ("wet", driver_count if options.wet else 0),
                       ("career_weight", 1 if options.career_prior else 0),
                       ("experience", MAX_EXPERIENCE if options.experience else 0)]:
        layout[name] = (start, start + size)
        start += size
    layout["total"] = start
    return layout


def split(unknowns: torch.Tensor, layout: dict) -> dict:
    """Split the list of unknowns into its named parts.

    The latest decade's level is not an unknown: it is fixed at 0, because
    only differences can be measured and something has to be the reference.
    So every rating reads as "compared with a newcomer of today".
    """
    def part(name):
        start, end = layout[name]
        return unknowns[start:end] if end > start else None

    earlier_levels = part("era_level")
    zero = unknowns.new_zeros(1)
    return {
        "skill": part("skill"),
        "era_level": zero if earlier_levels is None else torch.cat([earlier_levels, zero]),
        "wet": part("wet"),
        "career_weight": part("career_weight"),
        # Year 0 of a career is the reference (0); the curve says how much
        # a typical driver gains or loses from there.
        "experience_curve": None if part("experience") is None else torch.cat([zero, part("experience")]),
    }


def positive_scales(raw: torch.Tensor) -> dict:
    """Unpack the learned scale numbers, forcing each to be positive."""
    return dict(zip(SCALE_NAMES, raw.exp()))  # exp() of any number is positive


def badness(unknowns, raw_scales, data, skill_spread, season_drift):
    """One number to make as small as possible: minus (fit to results + prior belief)."""
    parts = split(unknowns, data["layout"])
    return -(log_likelihood(effective_skill(parts, data["links"]), positive_scales(raw_scales),
                            data["results"], parts["wet"])
             + log_prior(parts, data["links"], skill_spread, season_drift))


def effective_positions(layout: dict, links: dict, position: int) -> list[int]:
    """Where a driver-season's effective skill sits in the list of unknowns.

    Its own skill, plus (with the experience curve) the curve value for its
    year in F1. Year 0 is fixed at zero, so it has no position. Used to work
    out the uncertainty of anything built from effective skills, such as a peak.
    """
    positions = [position]
    start, end = layout.get("experience", (0, 0))
    year_in_f1 = int(links["experience"][position])
    if end > start and year_in_f1 >= 1:
        positions.append(start + year_in_f1 - 1)
    return positions


def effective_skill(parts, links):
    """The skill each driver-season races with: their own skill plus, if the
    model has one, the typical gain or loss for their years in F1."""
    if parts["experience_curve"] is None:
        return parts["skill"]
    return parts["skill"] + parts["experience_curve"][links["experience"]]


# --- Step 3: find the best skills, then how uncertain they are -------------

def find_best_skills(data, skill_spread, season_drift, start=None):
    """Find the skills, era levels, scales and optional parts that make the badness smallest.

    `start` is an optional (unknowns, scales) pair to begin from, such as the
    previous race's fit. It changes only how fast the answer is found, not
    the answer, because the optimiser always runs until it settles.
    """
    if start is None:
        start = (torch.zeros(data["layout"]["total"], device=DEVICE, dtype=DTYPE),
                 torch.zeros(len(SCALE_NAMES), device=DEVICE, dtype=DTYPE))
    unknowns = start[0].clone().requires_grad_(True)
    raw_scales = start[1].clone().requires_grad_(True)
    # L-BFGS is an optimiser for smooth problems where all the data fits in
    # memory at once. It needs far fewer steps here than Adam would.
    optimiser = torch.optim.LBFGS([unknowns, raw_scales], max_iter=1000,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def step():
        optimiser.zero_grad()
        loss = badness(unknowns, raw_scales, data, skill_spread, season_drift)
        loss.backward()
        return loss

    optimiser.step(step)
    return unknowns.detach(), raw_scales.detach()


def uncertainty(unknowns, raw_scales, data, skill_spread, season_drift):
    """How uncertain each unknown is (the "Laplace approximation").

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
        return badness(values, raw_scales, data, skill_spread, season_drift)

    hessian = torch.autograd.functional.hessian(badness_of, unknowns, vectorize=True)
    return torch.linalg.inv(hessian)


def prepare(pairs, pace, last_year, wet_races=None, options: Options | None = None,
            before_race: tuple[int, int] | None = None) -> tuple[dict, dict]:
    """Build everything the fit needs: the comparisons as tensors, careers, layout."""
    options = options or Options()
    if not options.wet:
        wet_races = None
    comparisons = build_comparisons(pairs, pace, last_year, wet_races, options, before_race)
    index = driver_season_index(comparisons)
    drivers = sorted({driver for driver, _ in index})
    links = career_links(index)
    layout = unknown_layout(len(index), len(links["decades"]), len(drivers), options)
    data = {"results": to_tensors(comparisons, index, drivers), "links": links,
            "layout": layout, "drivers": drivers, "comparisons": comparisons}
    return index, data


def fit_model(pairs, pace, last_year, skill_spread=DEFAULT_SKILL_SPREAD,
              season_drift=DEFAULT_SEASON_DRIFT, wet_races=None, options: Options | None = None,
              with_uncertainty: bool = True, before_race: tuple[int, int] | None = None,
              start: "Fit | None" = None) -> Fit:
    """Fit the model on every teammate comparison up to and including `last_year`.

    before_race: (season, round) to fit only on races before that race.
    start:       an earlier fit to begin from (much faster; same answer).
    """
    options = options or Options()
    index, data = prepare(pairs, pace, last_year, wet_races, options, before_race)
    initial = warm_start(start, index, data) if start is not None else None
    unknowns, raw_scales = find_best_skills(data, skill_spread, season_drift, initial)
    covariance = (uncertainty(unknowns, raw_scales, data, skill_spread, season_drift)
                  if with_uncertainty else None)
    parts = split(unknowns, data["layout"])
    data["unknowns"], data["raw_scales"] = unknowns, raw_scales
    return Fit(
        index=index, skill=parts["skill"],
        era_level=dict(zip(data["links"]["decades"], parts["era_level"].tolist())),
        covariance=covariance,
        scales={name: value.item() for name, value in positive_scales(raw_scales).items()},
        skill_spread=skill_spread, season_drift=season_drift, options=options,
        wet=wet_table(parts["wet"], covariance, data) if parts["wet"] is not None else {},
        career_weight=parts["career_weight"].item() if parts["career_weight"] is not None else 0.0,
        layout=data["layout"], data=data,
        experience_curve=[] if parts["experience_curve"] is None else parts["experience_curve"].tolist(),
        first_year=first_years(index),
    )


def first_years(index: dict) -> dict:
    """Each driver's first season in the fitted data."""
    first = {}
    for driver, year in index:
        first[driver] = min(year, first.get(driver, year))
    return first


def warm_start(start: Fit, index: dict, data: dict):
    """Starting values for a new fit, copied from an earlier fit wherever they exist.

    A driver-season the earlier fit did not have (a new season, a new driver)
    starts from that driver's latest earlier value, or from 0 for a newcomer.
    """
    unknowns = torch.zeros(data["layout"]["total"], device=DEVICE, dtype=DTYPE)
    old = start.data["unknowns"]
    latest = {}
    for (driver, year), position in start.index.items():
        latest[driver] = old[position]
    for (driver, year), position in index.items():
        if (driver, year) in start.index:
            unknowns[position] = old[start.index[(driver, year)]]
        elif driver in latest:
            unknowns[position] = latest[driver]
    for name in ("era_level", "wet", "career_weight", "experience"):
        new_start, new_end = data["layout"][name]
        old_start, old_end = start.layout.get(name, (0, 0))
        if new_end - new_start == old_end - old_start and new_end > new_start:
            unknowns[new_start:new_end] = old[old_start:old_end]
    return unknowns, start.data["raw_scales"].clone()


def wet_table(wet, covariance, data) -> dict:
    """Each driver's wet adjustment, its uncertainty, and how many wet races it rests on."""
    start, _ = data["layout"]["wet"]
    rows = data["comparisons"].filter((pl.col("wet") > 0) & pl.col("race_ahead").is_not_null())
    races = pl.concat([rows.select(pl.col("DriverA").alias("d"), "Year", "Round"),
                       rows.select(pl.col("DriverB").alias("d"), "Year", "Round")]).unique()
    counts = dict(races.group_by("d").agg(pl.len()).iter_rows())
    table = {}
    for position, driver in enumerate(data["drivers"]):
        sd = covariance[start + position, start + position].sqrt().item() if covariance is not None else None
        table[driver] = (wet[position].item(), sd, counts.get(driver, 0))
    return table


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
            means.append(fit.skill[position].item() + experience_bonus(fit, driver, year))
            variance += fit.covariance[position, position].item()
            variance += fit.season_drift ** 2 * (year - season)
    if len(positions) == 2:
        # Two drivers whose ratings move together (old teammates) have a more
        # certain GAP than their separate uncertainties suggest.
        variance -= 2 * fit.covariance[positions[0], positions[1]].item()
    return means[0] - means[1], variance


def expected_gap(fit: Fit, driver_a: str, driver_b: str, year: int) -> float:
    """Just the best estimate of the skill gap (A minus B), for a fit made without uncertainty."""
    means = []
    for driver in (driver_a, driver_b):
        season = latest_season(fit, driver, year)
        means.append(0.0 if season is None else
                     fit.skill[fit.index[(driver, season)]].item() + experience_bonus(fit, driver, year))
    return means[0] - means[1]


def experience_bonus(fit: Fit, driver: str, year: int) -> float:
    """The typical gain or loss for a driver's years in F1 in `year` (0 without the curve).

    This is how the curve helps predictions: a rookie fitted on one season is
    expected to improve the next, a veteran to decline.
    """
    if not fit.experience_curve:
        return 0.0
    first = fit.first_year.get(driver, year)
    return fit.experience_curve[min(year - first, MAX_EXPERIENCE)]


def chance_ahead(gap: float, variance: float, slope: float) -> float:
    """The chance A is ahead of B, allowing for not knowing the gap exactly.

    With a known gap the chance is the S-curve: 1 / (1 + exp(-slope x gap)).
    When the gap is uncertain the honest answer is the average of that curve
    over every plausible gap, which pulls the chance towards 50%. The line
    below is a standard, accurate shortcut for that average.
    """
    softened = slope * gap / (1 + torch.pi * slope ** 2 * variance / 8) ** 0.5
    return torch.sigmoid(torch.tensor(softened)).item()


def race_chance(fit: Fit, gap: float, variance: float, temperature: float | None = None) -> float:
    """The chance A finishes ahead of B in a race, with the calibration temperature applied."""
    temperature = RACE_TEMPERATURE if temperature is None else temperature
    return chance_ahead(gap, variance, fit.scales["race_slope"] * temperature)


def skill_table(fit: Fit) -> pl.DataFrame:
    """The fitted skills as a table: one row per driver per season, with uncertainty."""
    spread = fit.covariance.diagonal().sqrt()
    rows = [(driver, year, fit.skill[position].item() + experience_bonus(fit, driver, year), spread[position].item())
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


def load_wet_races() -> pl.DataFrame:
    return pl.read_parquet(WET_FILE)


if __name__ == "__main__":
    # A quick look: fit up to 2023 and print what was learned.
    fit = fit_model(*load_tables(), last_year=2023, wet_races=load_wet_races())
    print(f"{len(fit.index):,} driver-seasons fitted on {DEVICE}, options {fit.options}")
    print("scales:", {name: round(value, 3) for name, value in fit.scales.items()})
    print("era levels:", {decade: round(level, 3) for decade, level in fit.era_level.items()})
    print("career weight:", round(fit.career_weight, 3))
