# simulate.py
#
# What it does: the equal car race simulation. Every car is identical, so the
#   only differences are the drivers: their skill, how consistent they are,
#   and luck. It runs thousands of races at the same time on the GPU: every
#   number is a table with one row per simulated race and one column per
#   driver, and each lap updates the whole table at once.
#
#   One simulated race:
#     1. Each driver gets a "form" for the day (a good or bad weekend).
#     2. Qualifying: one lap each decides the starting grid.
#     3. Every lap, each driver's lap time = their pace + small random noise
#        + occasionally a mistake that costs time. Rarely, a crash.
#     4. A faster car behind only gets past if it is quicker by more than the
#        circuit's passing margin. Otherwise it is stuck in the queue.
#
# What it reads: nothing by itself. The numbers in SETTINGS are also written
#   to the website (by export_web.py) so the browser runs the same race.
# What it produces: nothing on disk. simulate_races() returns finishing places.
# Which files use it: export_web.py (head to heads, rankings, championships)
#   and test_simulate.py.
#
# Run:  .venv\Scripts\python.exe simulate.py
#   prints a check that simulated teammates beat each other as often as the
#   validated model says they should.

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Every number the simulation uses, in one place. Percentages are of a lap.
# Which are measured and which are assumptions: DECISIONS.md 22.
SETTINGS = {
    "formSd": 0.32,  # how much a driver's pace varies from one race to the next (%)
    "qualiNoise": 0.20,  # extra randomness of the single qualifying lap (%)
    "lapNoise": 0.25,  # lap to lap variation of a typical driver (%)
    "mistakeChance": 0.02,  # chance per lap that a typical driver makes a mistake
    "mistakeSeconds": 1.5,  # average time one mistake costs
    "crashChance": 0.05,  # chance per race that a typical driver crashes out
    "gridGapSeconds": 0.25,  # gap between grid slots as the race starts
    "followGapSeconds": 0.4,  # how close a car stuck behind another can follow
    "passMarginEasy": 0.2,  # lap time advantage (s) needed to pass at the easiest circuit
    "passMarginHard": 1.6,  # ... and at the hardest (Monaco)
}

RETIRED = 1e7  # added to a crashed car's race time, so it sorts behind every finisher


def error_factor(consistency: torch.Tensor) -> torch.Tensor:
    """Turn a consistency score (0 to 100, 50 = typical) into a multiplier for errors.

    1.0 for a typical driver, 0.5 for a score of 75 (half the noise, half the
    mistakes), 2.0 for a score of 25. This undoes the formula that built the
    score in fit_ratings.py, where halving the error rate added 25 points.
    """
    return 2 ** ((50 - consistency) / 25)


def pass_margin(difficulty: float) -> float:
    """Seconds a lap a car must be quicker to pass, for a circuit's difficulty (0 to 1)."""
    return SETTINGS["passMarginEasy"] + difficulty * (SETTINGS["passMarginHard"] - SETTINGS["passMarginEasy"])


def hold_up(race_time, lap_time, margin):
    """Add this lap to every car's race time, keeping slower cars in front where they belong.

    Cars are taken in the order they started the lap. A car that would end
    the lap ahead of the car in front of it is only allowed to if it was
    quicker by more than the passing margin. Otherwise it finishes the lap
    stuck just behind. Because the car in front may itself have been held up,
    this has to be done front to back, one position at a time: that is the
    only loop over cars, and each step still handles every simulated race at once.
    """
    order = race_time.argsort(dim=1)  # car numbers, leader first, per simulated race
    lap = lap_time.gather(1, order)
    finish = race_time.gather(1, order) + lap
    for place in range(1, order.shape[1]):
        ahead = finish[:, place - 1]
        would_pass = finish[:, place] < ahead + SETTINGS["followGapSeconds"]
        fast_enough = (lap[:, place - 1] - lap[:, place]) > margin
        stuck = would_pass & ~fast_enough
        finish[:, place] = torch.where(stuck, ahead + SETTINGS["followGapSeconds"], finish[:, place])
    return torch.empty_like(race_time).scatter_(1, order, finish)  # back to car number order


def simulate_races(skill, errors, lap_seconds, laps, margin, generator=None):
    """Simulate many equal car races at once.

    skill:   races x drivers. Each driver's skill in percent of lap time. One
             row per simulated race, so each race can use a different
             plausible value of an uncertain rating.
    errors:  one error multiplier per driver (see error_factor).
    Returns finishing place (1 = winner), races x drivers, and which cars
    crashed out.
    """
    races, drivers = skill.shape
    shape = (races, drivers)

    def bell(*size):  # random numbers from a bell curve centred on 0
        return torch.randn(*size, device=skill.device, generator=generator)

    def chance(*size):  # random numbers spread evenly between 0 and 1
        return torch.rand(*size, device=skill.device, generator=generator)

    percent = lap_seconds / 100  # one percent of a lap, in seconds
    form = bell(*shape) * SETTINGS["formSd"]
    pace = lap_seconds - (skill + form) * percent  # more skill = fewer seconds

    # Qualifying: one lap each. The quickest starts first.
    quali_lap = pace + bell(*shape) * SETTINGS["qualiNoise"] * percent
    grid_slot = quali_lap.argsort(dim=1).argsort(dim=1)  # 0 = pole
    race_time = grid_slot * SETTINGS["gridGapSeconds"]

    crashed = torch.zeros(shape, dtype=torch.bool, device=skill.device)
    crash_chance_per_lap = SETTINGS["crashChance"] / laps * errors
    for lap in range(laps):
        noise = bell(*shape) * SETTINGS["lapNoise"] * percent * errors
        made_mistake = chance(*shape) < SETTINGS["mistakeChance"] * errors
        # An exponential spread: most mistakes are small, a few are big.
        mistake = made_mistake * -torch.log(chance(*shape)) * SETTINGS["mistakeSeconds"]
        race_time = hold_up(race_time, pace + noise + mistake, margin)
        crashes_now = (chance(*shape) < crash_chance_per_lap) & ~crashed
        # A crashed car goes to the back. Crashing later still beats crashing earlier.
        race_time = race_time + crashes_now * (RETIRED - lap * 1000.0)
        crashed |= crashes_now
    place = race_time.argsort(dim=1).argsort(dim=1) + 1
    return place, crashed


def chance_a_beats_b(skill_gap: float, field_size: int = 20, races: int = 20000, seed: int = 0) -> float:
    """How often a driver beats a teammate-equivalent `skill_gap` slower, in a full field."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    skill = torch.zeros(races, field_size, device=DEVICE)
    # The other 18 cars are spread like a real grid, so the pair meets traffic.
    skill[:, 2:] = torch.linspace(-0.4, 0.6, field_size - 2, device=DEVICE)
    skill[:, 0] = skill_gap
    errors = torch.ones(field_size, device=DEVICE)
    place, _ = simulate_races(skill, errors, 90.0, 60, pass_margin(0.5), generator)
    return (place[:, 0] < place[:, 1]).float().mean().item()


def main():
    # The validated model says a skill gap g gives a chance of finishing
    # ahead of 1 / (1 + exp(-slope x g)). The simulation should agree, or its
    # races would be more (or less) predictable than real ones.
    from model import DEFAULT_SEASON_DRIFT, DEFAULT_SKILL_SPREAD, fit_model, load_tables
    fit = fit_model(*load_tables(), 2026, DEFAULT_SKILL_SPREAD, DEFAULT_SEASON_DRIFT)
    slope = fit.scales["race_slope"]
    print(f"Running on {DEVICE}. Model's race slope: {slope:.2f}\n")
    print(f"{'skill gap %':>11} {'model says':>11} {'simulation':>11}")
    for gap in (0.1, 0.2, 0.4, 0.6, 1.0):
        model_chance = 1 / (1 + torch.exp(torch.tensor(-slope * gap)).item())
        print(f"{gap:>11} {model_chance * 100:>10.1f}% {chance_a_beats_b(gap) * 100:>10.1f}%")


if __name__ == "__main__":
    main()
