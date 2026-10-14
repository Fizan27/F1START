# simulate.py
#
# What it does: the equal car race simulation. Every car is identical, so the
#   only differences are the drivers: their skill, how consistent they are,
#   and luck. It runs thousands of races at the same time on the GPU: every
#   number is a table with one row per simulated race and one column per
#   driver, and each lap updates the whole table at once.
#
#   One simulated race:
#     1. Each driver gets a "form" for the day (a good or bad weekend). This
#        is the main source of luck, and it lasts the whole race.
#     2. Qualifying: one lap each decides the starting grid, unless a grid is
#        given.
#     3. Every lap, each driver's lap time = their pace for the day + a SMALL
#        lap to lap noise + occasionally a mistake. Rarely, a crash. The lap
#        noise is kept small on purpose, so it cannot outweigh a real skill gap.
#     4. In about half the races a safety car closes up the field once.
#     5. A car that catches the one in front tries to pass. The bigger its
#        pace advantage compared with the circuit's difficulty, the more
#        likely it gets past on that lap. A clearly faster car gets by within
#        a few laps at Monza; at Monaco it may stay stuck.
#
# What it reads: nothing by itself. The numbers in SETTINGS are also written
#   to the website (by export_web.py) so the browser runs the same race.
# What it produces: nothing on disk. simulate_races() returns the results.
# Which files use it: export_web.py (championships), check_simulator.py and
#   test_simulate.py.
#
# Run:  .venv\Scripts\python.exe simulate.py
#   prints a check that simulated drivers beat each other as often as the
#   validated model says they should.

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Every number the simulation uses, in one place. Percentages are of a lap.
# Which are measured and which are assumptions: DECISIONS.md 22 and 32.
SETTINGS = {
    "formSd": 0.40,  # how much a driver's pace varies from one race to the next (%), calibrated
    "qualiNoise": 0.10,  # extra randomness of the single qualifying lap (%)
    "qualiFormLink": 0.32,  # how strongly qualifying form follows race form (0 = unrelated, 1 = the same), measured
    "lapNoise": 0.08,  # lap to lap variation of a typical driver (%): small on purpose
    "mistakeChance": 0.01,  # chance per lap that a typical driver makes a mistake
    "mistakeSeconds": 1.0,  # average time one mistake costs
    "crashChance": 0.05,  # chance per race that a typical driver crashes out
    "gridGapSeconds": 0.25,  # gap between grid slots as the race starts
    "followGapSeconds": 0.4,  # how close a car stuck behind another can follow
    "passMarginEasy": 0.0,  # pace advantage (s per lap) for an even chance of passing, easiest circuit
    "passMarginHard": 1.0,  # ... and at the hardest (Monaco)
    "passSpread": 0.15,  # how gradually the passing chance rises with the advantage (s)
    "safetyCarChance": 0.56,  # chance a race has a safety car, measured from 2018 to 2026 races
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
    """Pace advantage (seconds a lap) that gives an even chance of passing, for a difficulty 0 to 1."""
    return SETTINGS["passMarginEasy"] + difficulty * (SETTINGS["passMarginHard"] - SETTINGS["passMarginEasy"])


def pass_chance(advantage, margin):
    """The chance of getting past on one lap, given a pace advantage in seconds.

    An S-curve: 50% when the advantage equals the circuit's margin, close to
    certain when it is a few tenths more, close to zero when it is a few
    tenths less. The advantage is the difference in PACE for the day, not in
    that lap's time, so a lucky lap does not decide it.
    """
    return torch.sigmoid((advantage - margin) / SETTINGS["passSpread"])


def hold_up(race_time, lap_time, pace, margin, pass_luck, follow_luck):
    """Add this lap to every car's race time, deciding who gets past whom.

    Cars are taken in the order they started the lap. A car that would end
    the lap ahead of the car in front of it only does so if it passes (a
    random draw against pass_chance). Otherwise it finishes the lap stuck
    behind, between one and two following gaps back. Because the car in front
    may itself have been held up, this goes front to back one position at a
    time: the only loop over cars, and each step still handles every
    simulated race at once.

    pass_luck, follow_luck: random numbers between 0 and 1, same shape as
    race_time, so the GPU and browser versions can share this logic.
    """
    order = race_time.argsort(dim=1)  # car numbers, leader first, per simulated race
    lap = lap_time.gather(1, order)
    car_pace = pace.gather(1, order)
    finish = race_time.gather(1, order) + lap
    luck_pass = pass_luck.gather(1, order)
    follow = SETTINGS["followGapSeconds"] * (1 + follow_luck.gather(1, order))
    for place in range(1, order.shape[1]):
        ahead = finish[:, place - 1]
        catching = finish[:, place] < ahead + SETTINGS["followGapSeconds"]
        advantage = car_pace[:, place - 1] - car_pace[:, place]  # positive = this car is quicker
        passes = luck_pass[:, place] < pass_chance(advantage, margin)
        # A car in front that loses a lot of time on one lap (a mistake) is
        # passed whatever the circuit.
        passes |= (lap[:, place - 1] - lap[:, place]) > margin + 1.0
        stuck = catching & ~passes
        finish[:, place] = torch.where(stuck, ahead + follow[:, place], finish[:, place])
    return torch.empty_like(race_time).scatter_(1, order, finish)  # back to car number order


def bunch_up(race_time, crashed, safety_car_now):
    """A safety car: in the races where it comes out, close every gap to one following gap.

    The order stays the same; only the gaps are lost. This is what makes a
    big lead disappear and gives the cars behind another chance at the restart.
    """
    running_time = torch.where(crashed, torch.inf, race_time)
    order = running_time.argsort(dim=1)
    place = order.argsort(dim=1).float()  # 0 = leader
    leader = running_time.min(dim=1, keepdim=True).values
    bunched = leader + place * SETTINGS["followGapSeconds"]
    return torch.where(safety_car_now[:, None] & ~crashed, bunched, race_time)


def simulate_races(skill, errors, lap_seconds, laps, difficulty, generator=None, grid=None) -> dict:
    """Simulate many equal car races at once.

    skill:      races x drivers. Each driver's skill in percent of lap time.
    errors:     one error multiplier per driver (see error_factor).
    difficulty: the circuit's overtaking difficulty, 0 (easy) to 1 (Monaco).
    grid:       optional starting slot per driver (0 = pole). Without it, a
                simulated qualifying session sets the grid.
    Returns a dict of races x drivers tables: "place" (1 = winner), "grid"
    (0 = pole), "crashed", "best_lap" (quickest lap in seconds) and
    "race_time" (total seconds; crashed cars get a huge number).
    """
    races, drivers = skill.shape
    shape = (races, drivers)

    def bell(*size):  # random numbers from a bell curve centred on 0
        return torch.randn(*size, device=skill.device, generator=generator)

    def chance(*size):  # random numbers spread evenly between 0 and 1
        return torch.rand(*size, device=skill.device, generator=generator)

    percent = lap_seconds / 100  # one percent of a lap, in seconds
    margin = pass_margin(difficulty)
    form = bell(*shape) * SETTINGS["formSd"]
    pace = lap_seconds - (skill + form) * percent  # more skill = fewer seconds

    if grid is None:
        # Qualifying form is only partly the same as race form: a driver can
        # nail one lap on Saturday and be off the pace on Sunday, or the
        # reverse. Without this the grid would already be the race order.
        link = SETTINGS["qualiFormLink"]
        quali_form = link * form + (1 - link ** 2) ** 0.5 * bell(*shape) * SETTINGS["formSd"]
        quali_lap = lap_seconds - (skill + quali_form) * percent + bell(*shape) * SETTINGS["qualiNoise"] * percent
        grid_slot = quali_lap.argsort(dim=1).argsort(dim=1)  # 0 = pole
    else:
        grid_slot = torch.as_tensor(grid, device=skill.device).expand(shape)
    race_time = grid_slot * SETTINGS["gridGapSeconds"]

    crashed = torch.zeros(shape, dtype=torch.bool, device=skill.device)
    best_lap = torch.full(shape, float("inf"), device=skill.device)
    crash_chance_per_lap = SETTINGS["crashChance"] / laps * errors
    # Each race may have one safety car, on a random lap between 10% and 90% of the distance.
    has_safety_car = chance(races) < SETTINGS["safetyCarChance"]
    safety_car_lap = (laps * (0.1 + 0.8 * chance(races))).long()
    for lap in range(laps):
        noise = bell(*shape) * SETTINGS["lapNoise"] * percent * errors
        made_mistake = chance(*shape) < SETTINGS["mistakeChance"] * errors
        # An exponential spread: most mistakes are small, a few are big.
        mistake = made_mistake * -torch.log(chance(*shape)) * SETTINGS["mistakeSeconds"]
        before = race_time
        race_time = hold_up(race_time, pace + noise + mistake, pace, margin, chance(*shape), chance(*shape))
        lap_taken = torch.where(crashed, torch.inf, race_time - before)
        best_lap = torch.minimum(best_lap, lap_taken)
        crashes_now = (chance(*shape) < crash_chance_per_lap) & ~crashed
        # A crashed car goes to the back. Crashing later still beats crashing earlier.
        race_time = race_time + crashes_now * (RETIRED - lap * 1000.0)
        crashed |= crashes_now
        race_time = bunch_up(race_time, crashed, has_safety_car & (safety_car_lap == lap))
    place = race_time.argsort(dim=1).argsort(dim=1) + 1
    return {"place": place, "grid": grid_slot, "crashed": crashed, "best_lap": best_lap, "race_time": race_time}


def chance_a_beats_b(skill_gap: float, field_size: int = 20, races: int = 20000, seed: int = 0) -> float:
    """How often a driver beats one `skill_gap` slower, in a full field at a middling circuit."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    skill = torch.zeros(races, field_size, device=DEVICE)
    # The other 18 cars are spread like a real grid, so the pair meets traffic.
    skill[:, 2:] = torch.linspace(-0.4, 0.6, field_size - 2, device=DEVICE)
    skill[:, 0] = skill_gap
    errors = torch.ones(field_size, device=DEVICE)
    result = simulate_races(skill, errors, 90.0, 60, 0.5, generator)
    return (result["place"][:, 0] < result["place"][:, 1]).float().mean().item()


def main():
    # The fitted model says a skill gap g between two drivers whose skills
    # are KNOWN gives a chance of finishing ahead of 1 / (1 + exp(-slope x g)).
    # The simulation treats each driver's skill as known, so it should agree
    # with that curve. (The calibration temperature in model.py is about
    # predictions made without knowing skills exactly, so it does not apply
    # here: DECISIONS.md 27.)
    from model import fit_model, load_tables
    fit = fit_model(*load_tables(), 2026, with_uncertainty=False)
    slope = fit.scales["race_slope"]
    print(f"Running on {DEVICE}. Model's race slope: {slope:.2f}\n")
    print(f"{'skill gap %':>11} {'model says':>11} {'simulation':>11}")
    for gap in (0.1, 0.2, 0.4, 0.6, 1.0):
        model_chance = 1 / (1 + torch.exp(torch.tensor(-slope * gap)).item())
        print(f"{gap:>11} {model_chance * 100:>10.1f}% {chance_a_beats_b(gap) * 100:>10.1f}%")


if __name__ == "__main__":
    main()
