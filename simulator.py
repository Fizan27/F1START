# simulator.py
#
# What it does: simulates F1 races lap by lap, thousands at once, as PyTorch
#   tensors on the GPU (or on the CPU for the website). Every car in every
#   simulated race gets a lap time from the lap time network, plus pit stops,
#   safety cars, VSCs, traffic and the first lap.
# What it reads: models/lap_time_network.pt (from train_network.py),
#   models/race_stats.json (from race_stats.py) and data/laps.parquet.
# What it produces: nothing on disk. It provides:
#     load_races()   the real races, in the form the simulator needs
#     RaceSet        those races stacked into tensors
#     LapModel       the trained network, ready to be asked for lap times
#     RaceSim        the simulator itself: create it, then call step() once
#                    per lap, or run_to_end()
# Which files use it: replay_validation.py, strategy_env.py,
#   strategy_search.py, evaluate_strategist.py and app.py.
#
# How to read the shapes in the comments: B is the number of races being
# simulated side by side, D is the number of cars (20), L is the largest
# number of laps in any race. A tensor "(B, D)" holds one number per car per
# simulated race.

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

import race_stats
from train_network import COMPOUND_NAMES, NUMBER_INPUTS, LapTimeNetwork

LAPS_FILE = Path("data/laps.parquet")
MODEL_FILE = Path("models/lap_time_network.pt")
STATS_FILE = Path("models/race_stats.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RACE = ["Year", "Round"]

CARS = 20
GAP_CAP_SECONDS = 10.0  # must match train_baseline.GAP_CAP_SECONDS
STAY_OUT = 0  # the actions are: 0 stay out, 1 pit for soft, 2 medium, 3 hard
GREEN, VSC, SAFETY_CAR = 0, 1, 2
NEVER = 999  # "this car never retires"

# --- Simulator rules that are assumptions, not measurements. Each one is
# --- explained in docs/DECISIONS.md, number 22.
# A car needs to be this much faster over a lap to pass the car in front.
# Chosen by trying values in replay_validation.py on 2024.
OVERTAKE_ADVANTAGE_SECONDS = 1.0
# A car held up behind another finishes the lap this far behind it.
FOLLOWING_GAP_SECONDS = 0.4
# Behind the safety car the field closes up to this gap.
SAFETY_CAR_GAP_SECONDS = 0.6
# Extra slowness per lap of tyre age beyond what the data covers, in percent
# of pole. Stops the simulator trusting the network where it is guessing.
WORN_TYRE_PCT_PER_LAP = 0.30
# The share of the measured whole-race driver pace offset that the simulator
# uses. With all of it (1.0), simulated results were too scattered: 98% of
# real 2024 finishing positions fell inside a range meant to hold 90%. At 0.4
# it is 91%. Chosen in replay validation on 2024.
DRIVER_OFFSET_SHARE = 0.4
# When a safety car or VSC appears, a rival pits at once if its next planned
# stop was within this many laps.
REACT_LAPS = {SAFETY_CAR: 8, VSC: 4}


# --------------------------------------------------------------------------
# The real races, in the form the simulator needs
# --------------------------------------------------------------------------

@dataclass
class Race:
    """One real race: who started where, what they did, what happened."""
    year: int
    round: int
    name: str
    circuit: str
    split: str
    total_laps: int
    pole: float  # the pole lap, in seconds
    race_pace: float  # RacePacePct: the race's typical lap, percent off pole
    wet_qualifying: bool
    drivers: list  # three letter codes, in a fixed order
    teams: list
    grid: np.ndarray  # (D,) starting position, 1 = pole
    quali_gap: np.ndarray  # (D,) QualiGapPct
    start_compound: np.ndarray  # (D,) 0 soft, 1 medium, 2 hard
    start_age: np.ndarray  # (D,) laps already on the starting tyres
    plan: np.ndarray  # (D, total_laps) the real stops: 0, or 1 + compound
    plan_age: np.ndarray  # (D, total_laps) age of the tyres fitted at a stop
    status: np.ndarray  # (total_laps,) 0 green, 1 VSC, 2 safety car
    retire_lap: np.ndarray  # (D,) last lap completed, or NEVER
    finish_position: np.ndarray  # (D,) official result
    gap_to_winner: np.ndarray  # (D,) seconds, NaN if not on the lead lap

    @property
    def label(self) -> str:
        return f"{self.year} {self.name}"

    def stops_of(self, driver_index: int) -> list:
        """The real strategy as text, for example 'MEDIUM, lap 18 HARD'."""
        parts = [COMPOUND_NAMES[self.start_compound[driver_index]]]
        for lap in np.nonzero(self.plan[driver_index])[0]:
            parts.append(f"lap {lap + 1} {COMPOUND_NAMES[self.plan[driver_index, lap] - 1]}")
        return parts


def compound_number(name) -> int:
    """SOFT 0, MEDIUM 1, HARD 2. Anything else (unknown) counts as MEDIUM."""
    return COMPOUND_NAMES.index(name) if name in COMPOUND_NAMES else 1


def build_race(laps: pl.DataFrame, status: np.ndarray) -> Race:
    """Turn one race's laps into a Race."""
    first = laps.row(0, named=True)
    total_laps = first["TotalLaps"]
    drivers, rows = [], []
    for (driver,), driver_laps in laps.sort("LapNumber").group_by("Driver", maintain_order=True):
        drivers.append(driver)
        rows.append(driver_laps)
    # Fixed order: by grid position, with pit lane starters (grid 0) last.
    order = np.argsort([row["GridPosition"][0] or 99 for row in rows], kind="stable")
    drivers = [drivers[i] for i in order][:CARS]
    rows = [rows[i] for i in order][:CARS]
    count = len(drivers)

    plan = np.zeros((count, total_laps), dtype=np.int64)
    plan_age = np.zeros((count, total_laps), dtype=np.float32)
    start_compound = np.zeros(count, dtype=np.int64)
    start_age = np.zeros(count, dtype=np.float32)
    retire_lap = np.full(count, NEVER, dtype=np.int64)
    gap_to_winner = np.full(count, np.nan, dtype=np.float32)
    quali_gap = np.zeros(count, dtype=np.float32)
    finish = np.zeros(count, dtype=np.int64)
    winner_time = laps.filter(pl.col("LapNumber") == total_laps)["Time"].min()

    for i, driver_laps in enumerate(rows):
        lap_numbers = driver_laps["LapNumber"].to_list()
        compounds = [compound_number(name) for name in driver_laps["Compound"].to_list()]
        ages = driver_laps["TyreLife"].fill_null(1).to_list()
        pitted = driver_laps["IsPitInLap"].to_list()
        start_compound[i] = compounds[0]
        start_age[i] = max(ages[0] - 1, 0)
        for j in range(len(lap_numbers) - 1):
            if pitted[j] and lap_numbers[j] < total_laps:
                # The stop happens at the end of this lap; the next lap
                # shows which tyres were fitted and how old they were.
                plan[i, lap_numbers[j] - 1] = 1 + compounds[j + 1]
                plan_age[i, lap_numbers[j] - 1] = max(ages[j + 1] - 1, 0)
        # Cars that stop well short of the distance are treated as retired.
        if lap_numbers[-1] < 0.9 * total_laps:
            retire_lap[i] = lap_numbers[-1]
        elif lap_numbers[-1] == total_laps:
            gap_to_winner[i] = driver_laps["Time"][-1] - winner_time
        gap = driver_laps["QualiGapPct"][0]
        quali_gap[i] = np.nan if gap is None else gap
        finish[i] = driver_laps["FinishPosition"][0] or 99
    # A driver with no usable qualifying lap is assumed slowest of the field.
    quali_gap = np.where(np.isnan(quali_gap), np.nanmax(quali_gap), quali_gap)

    return Race(
        year=first["Year"], round=first["Round"], name=first["EventName"],
        circuit=first["Circuit"], split=first["Split"], total_laps=total_laps,
        pole=first["PoleTime"], race_pace=first["RacePacePct"],
        wet_qualifying=first["IsWetQualifying"], drivers=drivers,
        teams=[row["Team"][0] for row in rows],
        grid=np.arange(1, count + 1), quali_gap=quali_gap,
        start_compound=start_compound, start_age=start_age, plan=plan,
        plan_age=plan_age, status=status, retire_lap=retire_lap,
        finish_position=finish, gap_to_winner=gap_to_winner,
    )


def load_races(splits: list[str], laps_file: Path = LAPS_FILE) -> list[Race]:
    """Every dry race without a red flag in the given seasons."""
    laps = pl.read_parquet(laps_file).filter(pl.col("Split").is_in(splits))
    red_flagged = pl.col("IsRedFlag").any().over(RACE)
    # A race with no qualifying times has no pole lap to measure against, so
    # it cannot be simulated (Miami 2025: the data source has none).
    laps = laps.filter(~pl.col("IsWetRace") & ~red_flagged & pl.col("PoleTime").is_not_null())
    per_lap = race_stats.track_status_by_lap(laps)
    races = []
    for key, race_laps in laps.group_by(RACE, maintain_order=True):
        rows = per_lap.filter((pl.col("Year") == key[0]) & (pl.col("Round") == key[1]))
        status = np.zeros(race_laps["TotalLaps"][0], dtype=np.int64)
        status[rows["LapNumber"].to_numpy() - 1] = rows["Status"].to_numpy()
        races.append(build_race(race_laps, status))
    return sorted(races, key=lambda race: (race.year, race.round))


class RaceSet:
    """A list of races stacked into tensors, padded to the same size.

    Row r of every tensor belongs to race r. Shorter races are padded with
    laps that are never driven, and missing cars with cars that have
    "retired before the start".
    """

    def __init__(self, races: list[Race], lap_model: "LapModel", stats: dict,
                 device: str = DEVICE):
        self.races = races
        self.device = device
        count = len(races)
        self.max_laps = max(race.total_laps for race in races)

        def per_race(values, dtype=torch.float32):
            return torch.tensor(values, dtype=dtype, device=device)

        def per_car(name, fill, dtype=torch.float32):
            out = np.full((count, CARS), fill, dtype=np.float64)
            for r, race in enumerate(races):
                values = getattr(race, name)
                out[r, :len(values)] = values
            return torch.tensor(out, dtype=dtype, device=device)

        def per_car_lap(name, dtype):
            out = np.zeros((count, CARS, self.max_laps))
            for r, race in enumerate(races):
                values = getattr(race, name)
                out[r, :values.shape[0], :values.shape[1]] = values
            return torch.tensor(out, dtype=dtype, device=device)

        self.total_laps = per_race([race.total_laps for race in races], torch.long)
        self.pole = per_race([race.pole for race in races])
        self.race_pace = per_race([race.race_pace for race in races])
        self.circuit = per_race([lap_model.circuit_number(race.circuit) for race in races],
                                torch.long)
        self.pit_loss = per_race([
            stats["pit_loss_by_circuit"].get(race.circuit, stats["pit_loss_default"])
            for race in races
        ])
        self.car_count = per_race([len(race.drivers) for race in races], torch.long)
        self.grid = per_car("grid", CARS)
        self.quali_gap = per_car("quali_gap", 0.0)
        self.start_compound = per_car("start_compound", 1, torch.long)
        self.start_age = per_car("start_age", 0.0)
        self.retire_lap = per_car("retire_lap", 0, torch.long)  # padding never starts
        self.plan = per_car_lap("plan", torch.long)
        self.plan_age = per_car_lap("plan_age", torch.float32)
        status = np.zeros((count, self.max_laps))
        for r, race in enumerate(races):
            status[r, :race.total_laps] = race.status
        self.status = torch.tensor(status, dtype=torch.long, device=device)

    def index_of(self, year: int, round_number: int) -> int:
        for r, race in enumerate(self.races):
            if (race.year, race.round) == (year, round_number):
                return r
        raise KeyError(f"race {year} round {round_number} is not in this set")


# --------------------------------------------------------------------------
# The lap time network, ready to use
# --------------------------------------------------------------------------

class LapModel:
    """Loads the trained network and answers "how fast is this lap?"."""

    def __init__(self, file: Path = MODEL_FILE, device: str = DEVICE):
        saved = torch.load(file, map_location=device, weights_only=False)
        scaling = saved["scaling"]
        self.device = device
        self.circuits = scaling["circuits"]
        self.spread_factor = saved["spread_factor"]
        self.driver_offset_spread = saved["driver_offset_spread"]
        self.mean = torch.tensor([scaling["mean"][n] for n in NUMBER_INPUTS], device=device)
        self.std = torch.tensor([scaling["std"][n] for n in NUMBER_INPUTS], device=device)
        self.network = LapTimeNetwork(len(self.circuits)).to(device)
        self.network.load_state_dict(saved["weights"])
        self.network.eval()

    def circuit_number(self, name: str) -> int:
        """The network's number for a circuit; 0 means "never seen"."""
        return self.circuits.index(name) + 1 if name in self.circuits else 0

    @torch.no_grad()
    def predict(self, tyre_life, laps_remaining, gap_ahead, quali_gap, compound, circuit):
        """Centre and (calibrated) spread of the pace, in percent of pole.

        All inputs are tensors of the same shape; so are both outputs.
        """
        shape = tyre_life.shape
        numbers = torch.stack(
            [tyre_life, laps_remaining, gap_ahead, quali_gap], dim=-1
        ).reshape(-1, len(NUMBER_INPUTS)).float()
        numbers = (numbers - self.mean) / self.std
        centre, log_spread = self.network(numbers, compound.reshape(-1), circuit.reshape(-1))
        spread = torch.exp(log_spread) * self.spread_factor
        return centre.reshape(shape), spread.reshape(shape)


def load_stats(file: Path = STATS_FILE) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Random safety cars and VSCs
# --------------------------------------------------------------------------

def sample_status(race_set: RaceSet, race_index, stats: dict, generator) -> torch.Tensor:
    """Invent a track status for every lap of every simulated race. (B, L)

    Each green lap, a safety car or VSC may start, with the chance measured
    for that circuit and race phase. Its length is drawn from the lengths
    seen in real races.
    """
    device = race_set.device
    count = len(race_index)
    total = race_set.total_laps[race_index]
    names = [race.circuit for race in race_set.races]

    def chances(kind):
        part = stats[kind]
        by_phase = torch.tensor([part["chance_per_lap_by_phase"][p] for p in race_stats.PHASES],
                                device=device)
        multiplier = torch.tensor([part["circuit_multiplier"].get(n, 1.0) for n in names],
                                  device=device)[race_index]
        lengths = torch.tensor(part["lengths_in_laps"], device=device)
        return by_phase, multiplier, lengths

    sc_phase, sc_multiplier, sc_lengths = chances("safety_car")
    vsc_phase, vsc_multiplier, vsc_lengths = chances("vsc")
    status = torch.zeros(count, race_set.max_laps, dtype=torch.long, device=device)
    current = torch.zeros(count, dtype=torch.long, device=device)
    laps_left = torch.zeros(count, dtype=torch.long, device=device)

    def draw(lengths):
        pick = torch.randint(len(lengths), (count,), generator=generator, device=device)
        return lengths[pick]

    for lap in range(1, race_set.max_laps + 1):
        fraction = lap / total
        phase = torch.where(fraction <= 1 / 3, 1, torch.where(fraction <= 2 / 3, 2, 3))
        if lap <= race_stats.START_LAPS:
            phase = torch.zeros_like(phase)
        luck = torch.rand(count, generator=generator, device=device)
        sc_chance = sc_phase[phase] * sc_multiplier
        vsc_chance = vsc_phase[phase] * vsc_multiplier
        green = laps_left == 0
        start_sc = green & (luck < sc_chance)
        start_vsc = green & ~start_sc & (luck < sc_chance + vsc_chance)
        current = torch.where(green, 0, current)
        current = torch.where(start_sc, SAFETY_CAR, torch.where(start_vsc, VSC, current))
        laps_left = torch.where(start_sc, draw(sc_lengths), laps_left)
        laps_left = torch.where(start_vsc, draw(vsc_lengths), laps_left)
        status[:, lap - 1] = current
        laps_left = (laps_left - 1).clamp(min=0)
    return status


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------

class RaceSim:
    """Simulates B races side by side, one lap per call to step().

    race_index (B,) says which real race each simulated race is a copy of.
    By default every car follows its real strategy and the real safety cars
    happen. Pass `status` to use other track conditions, `plan` to give cars
    other strategies, and `fixed` to mark cars that must follow their plan
    exactly even when a safety car appears.

    `group` (B,) lets several simulated races share the same luck (the same
    random lap time noise). That makes comparing strategies fair: every
    strategy is tried against exactly the same set of random races.
    """

    def __init__(self, race_set: RaceSet, lap_model: LapModel, stats: dict, race_index,
                 status=None, plan=None, plan_age=None, fixed=None, group=None,
                 seed: int = 0, luck: bool = True):
        self.set, self.model, self.stats = race_set, lap_model, stats
        device = race_set.device
        self.index = torch.as_tensor(race_index, device=device)
        B = len(self.index)
        self.rows = torch.arange(B, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.group = self.rows if group is None else group
        self.groups = int(self.group.max()) + 1
        self.luck = luck

        take = lambda tensor: tensor[self.index]
        self.total_laps = take(race_set.total_laps)
        self.pole = take(race_set.pole)[:, None]
        self.race_pace = take(race_set.race_pace)[:, None]
        self.circuit = take(race_set.circuit)[:, None].expand(B, CARS)
        self.pit_loss = take(race_set.pit_loss)[:, None]
        self.grid = take(race_set.grid)
        self.quali_gap = take(race_set.quali_gap)
        self.retire_lap = take(race_set.retire_lap)
        self.status = take(race_set.status) if status is None else status
        self.plan = take(race_set.plan).clone() if plan is None else plan
        self.plan_age = (take(race_set.plan_age) if plan_age is None else plan_age).clone()
        self.fixed = (torch.zeros(B, CARS, dtype=torch.bool, device=device)
                      if fixed is None else fixed)

        self.compound = take(race_set.start_compound).clone()
        self.tyre_age = take(race_set.start_age).clone()
        self.time = torch.zeros(B, CARS, device=device)  # race time so far
        self.stops = torch.zeros(B, CARS, dtype=torch.long, device=device)
        self.used = torch.nn.functional.one_hot(self.compound, 3).bool()  # (B, D, 3)
        self.lap = 1  # the lap about to be driven
        self.last_lap_time = torch.zeros(B, CARS, device=device)
        self.last_pitted = torch.zeros(B, CARS, dtype=torch.bool, device=device)

        # Each car's pace offset for the whole race (see train_network.py,
        # driver_offset_spread): one draw per car per random race.
        self.offset_spread = lap_model.driver_offset_spread * DRIVER_OFFSET_SHARE
        self.driver_offset = self.noise() * self.offset_spread
        limits = stats["tyre_life_limit"]
        self.tyre_limit = torch.tensor([limits[name] for name in COMPOUND_NAMES],
                                       dtype=torch.float32, device=device)
        self.pit_share = torch.tensor(
            [1.0, stats["vsc_pit_share"], stats["safety_car_pit_share"]], device=device)

    # ---- small helpers ----------------------------------------------------

    def noise(self) -> torch.Tensor:
        """Random numbers (B, D), shared by simulated races in the same group."""
        if not self.luck:
            return torch.zeros(len(self.index), CARS, device=self.set.device)
        draws = torch.randn(self.groups, CARS, generator=self.generator,
                            device=self.set.device)
        return draws[self.group]

    def running(self, lap: int) -> torch.Tensor:
        """Which cars are still in the race on this lap. (B, D)"""
        return self.retire_lap >= lap

    def finished(self) -> torch.Tensor:
        """Which simulated races are over. (B,)"""
        return self.lap > self.total_laps

    def gap_ahead(self, running) -> tuple:
        """Gap to the car in front on the road, and the running order."""
        time = torch.where(running, self.time, torch.inf)
        ordered, order = time.sort(dim=1)
        gaps = torch.full_like(ordered, GAP_CAP_SECONDS)
        gaps[:, 1:] = (ordered[:, 1:] - ordered[:, :-1]).nan_to_num(GAP_CAP_SECONDS)
        gaps = gaps.clamp(0, GAP_CAP_SECONDS)
        gap = torch.empty_like(gaps).scatter_(1, order, gaps)
        if self.lap == 1:
            gap = torch.full_like(gap, GAP_CAP_SECONDS)  # nobody has a gap yet
        return gap, order

    def positions(self) -> torch.Tensor:
        """Race position of every car now (1 = leading). (B, D)"""
        if self.lap == 1:
            return self.grid.long()  # before the start: the grid order
        # Cars that have retired (or never started) are placed last.
        time = torch.where(self.running(self.lap - 1), self.time, torch.inf)
        return time.argsort(dim=1).argsort(dim=1) + 1

    def predicted_pace(self, tyre_age, compound, gap, lap=None):
        """The network's centre and spread for every car. (B, D) each."""
        lap = self.lap if lap is None else lap
        laps_remaining = (self.total_laps[:, None] - lap).expand_as(gap)
        return self.model.predict(tyre_age + 1, laps_remaining, gap, self.quali_gap,
                                  compound, self.circuit)

    # ---- choosing the actions ---------------------------------------------

    def planned_actions(self, status) -> torch.Tensor:
        """What every car's plan says to do this lap, with rivals reacting to
        a safety car or VSC by bringing their next stop forward. (B, D)"""
        lap = self.lap
        for kind, window in REACT_LAPS.items():
            ahead = self.plan[:, :, lap - 1: lap + window]  # this lap and the next few
            has_stop = (ahead > 0).any(dim=2)
            first = (ahead > 0).float().argmax(dim=2, keepdim=True)
            react = (has_stop & (first[:, :, 0] > 0) & ~self.fixed
                     & (status == kind)[:, None])
            if react.any():
                brought_forward = ahead.gather(2, first)[:, :, 0]
                age = self.plan_age[:, :, lap - 1: lap + window].gather(2, first)[:, :, 0]
                # Cancel the stop that was planned, and write it into this
                # lap of the plan instead.
                cancel = torch.zeros_like(ahead).scatter_(2, first, 1).bool() & react[:, :, None]
                self.plan[:, :, lap - 1: lap + window] = torch.where(cancel, 0, ahead)
                self.plan[:, :, lap - 1] = torch.where(
                    react, brought_forward, self.plan[:, :, lap - 1])
                self.plan_age[:, :, lap - 1] = torch.where(
                    react, age, self.plan_age[:, :, lap - 1])
        return self.plan[:, :, lap - 1].clone()

    # ---- one lap -----------------------------------------------------------

    @torch.no_grad()
    def step(self, agent=None, agent_action=None):
        """Drive one lap. Every car follows its plan, except that car
        `agent` (B,) of each race does `agent_action` (B,) if given:
        0 stay out, 1/2/3 pit at the end of this lap for soft/medium/hard."""
        lap = self.lap
        active = (lap <= self.total_laps)[:, None]  # races not yet finished
        running = self.running(lap)
        status = self.status[:, min(lap, self.status.shape[1]) - 1]
        planned = self.planned_actions(status)
        action = planned.clone()
        if agent is not None:
            action[self.rows, agent] = agent_action
        # No stops on the final lap: there is nothing left to gain.
        action = torch.where((lap >= self.total_laps)[:, None], 0, action)
        pitting = (action > 0) & running & active

        gap, order = self.gap_ahead(running)
        centre, spread = self.predicted_pace(self.tyre_age, self.compound, gap)
        # The total spread is split into the whole-race driver offset (drawn
        # once, above) and lap by lap luck. Without the split the two would
        # be counted twice.
        lap_spread = (spread ** 2 - self.offset_spread ** 2).clamp(min=0).sqrt()
        lap_spread = torch.maximum(lap_spread, 0.5 * spread)
        luck = self.noise()
        too_old = (self.tyre_age + 1 - self.tyre_limit[self.compound]).clamp(min=0)
        pace = (self.race_pace + centre + self.driver_offset + lap_spread * luck
                + too_old * WORN_TYRE_PCT_PER_LAP)
        racing_lap = self.pole * (1 + pace / 100)

        typical_lap = self.pole * (1 + self.race_pace / 100)
        if lap == 1:
            # The standing start: each grid slot costs a fixed time, plus luck.
            lap_time = (typical_lap
                        + self.stats["first_lap_seconds_per_grid_slot"] * (self.grid - 1)
                        + self.stats["first_lap_luck_seconds"] * luck)
        else:
            lap_time = racing_lap
        vsc_lap = self.pole * (1 + (self.race_pace + self.stats["vsc"]["slow_lap_pct"]) / 100)
        sc_lap = self.pole * (1 + (self.race_pace + self.stats["safety_car"]["slow_lap_pct"]) / 100)
        is_vsc = (status == VSC)[:, None]
        is_sc = (status == SAFETY_CAR)[:, None]
        lap_time = torch.where(is_vsc, vsc_lap.expand_as(lap_time), lap_time)

        new_time = self.time + lap_time
        if lap > 1:
            new_time = self.apply_traffic(new_time, lap_time, sc_lap, order, pitting,
                                          running, is_sc[:, 0], is_vsc[:, 0])
        pit_cost = self.pit_loss * self.pit_share[status][:, None]
        new_time = new_time + torch.where(pitting, pit_cost, 0.0)

        moved = active & running
        self.last_lap_time = torch.where(moved, new_time - self.time, 0.0)
        self.last_pitted = pitting
        self.time = torch.where(moved, new_time, self.time)

        # Tyres: a stop fits new tyres, otherwise they get one lap older.
        new_compound = (action - 1).clamp(min=0)
        fitted_age = torch.where(action == planned, self.plan_age[:, :, min(lap, self.plan_age.shape[2]) - 1], 0.0)
        self.compound = torch.where(pitting, new_compound, self.compound)
        self.tyre_age = torch.where(pitting, fitted_age,
                                    torch.where(moved, self.tyre_age + 1, self.tyre_age))
        self.stops = self.stops + pitting.long()
        self.used = self.used | (torch.nn.functional.one_hot(new_compound, 3).bool()
                                 & pitting[:, :, None])
        self.lap += 1

    def apply_traffic(self, new_time, lap_time, sc_lap, order, pitting, running,
                      is_sc, is_vsc):
        """Stop cars driving through each other.

        Cars are handled from the front of the field to the back. Each car
        is compared with the rearmost car ahead of it (the "blocker"):
          - Racing: to pass, the car must be OVERTAKE_ADVANTAGE_SECONDS
            faster over the lap. Otherwise it finishes the lap just behind.
          - Safety car: the leader laps slowly and everyone behind closes up
            to the car in front.
          - VSC: everyone laps at the same slow pace, so gaps freeze.
        Cars that pit this lap neither block nor get blocked.
        """
        B = new_time.shape[0]
        final = new_time.clone()
        blocker_time = torch.full((B,), -torch.inf, device=new_time.device)
        blocker_lap = torch.zeros(B, device=new_time.device)
        for place in range(CARS):
            car = order[:, place:place + 1]
            start = self.time.gather(1, car)[:, 0]
            own_lap = lap_time.gather(1, car)[:, 0]
            arrives = start + own_lap
            pits = pitting.gather(1, car)[:, 0]
            runs = running.gather(1, car)[:, 0]

            held_up = ((arrives < blocker_time + FOLLOWING_GAP_SECONDS)
                       & ((blocker_lap - own_lap) < OVERTAKE_ADVANTAGE_SECONDS) & ~pits)
            racing = torch.where(held_up, blocker_time + FOLLOWING_GAP_SECONDS, arrives)

            slow = start + sc_lap[:, 0]
            closing_up = torch.minimum(
                torch.maximum(blocker_time + SAFETY_CAR_GAP_SECONDS, arrives), slow)
            behind_safety_car = torch.where(blocker_time == -torch.inf, slow, closing_up)

            result = torch.where(is_sc, behind_safety_car, racing)
            result = torch.where(is_vsc, arrives, result)
            final.scatter_(1, car, result[:, None])

            blocks = runs & ~pits & (result > blocker_time)
            blocker_time = torch.where(blocks, result, blocker_time)
            blocker_lap = torch.where(blocks, own_lap, blocker_lap)
        return final

    def run_to_end(self):
        """Drive every remaining lap with the planned strategies."""
        while self.lap <= int(self.total_laps.max()):
            self.step()
        return self

    def final_positions(self) -> torch.Tensor:
        """Finishing position of every car (retired cars are last). (B, D)"""
        time = torch.where(self.retire_lap >= NEVER, self.time, torch.inf)
        return time.argsort(dim=1).argsort(dim=1) + 1
