# race_stats.py
#
# What it does: measures the simple facts about races that the simulator
#   needs besides the lap time model:
#     - pit lane time loss per circuit, and how much cheaper a stop is under
#       a safety car or virtual safety car (VSC)
#     - how likely a safety car or VSC is, per circuit and race phase, how
#       long they last, and how slow the laps are
#     - how spread out the field is after the first lap
# What it reads: data/laps.parquet (made by clean_data.py).
# What it produces: models/race_stats.json, and a summary on the screen.
# Which files use it: simulator.py loads the JSON. final_test.py calls
#   build_stats() again with 2024 included.
#
# Run:  .venv\Scripts\python.exe race_stats.py
#
# Everything here is counted from past seasons only (2022 to 2023 by
# default), so it can be used on later races without leaking anything.

import json
from pathlib import Path

import numpy as np
import polars as pl

LAPS_FILE = Path("data/laps.parquet")
STATS_FILE = Path("models/race_stats.json")
RACE = ["Year", "Round"]

# A circuit's own pit loss is used only if at least this many clean stops
# were seen there; otherwise the average of all circuits is used.
MIN_STOPS_PER_CIRCUIT = 8

# A stop that costs less than 10s or more than 45s is not a normal tyre stop
# (timing error, penalty served, damage repair), so it is ignored.
NORMAL_STOP_SECONDS = (10.0, 45.0)

# Safety car and VSC stop costs are only measured on laps where at least
# this many cars pitted together.
MIN_CARS_PITTING_TOGETHER = 3

# Race phases for safety car likelihood. "start" is laps 1 to 3, where first
# lap crashes make safety cars far more likely than anywhere else.
START_LAPS = 3
PHASES = ["start", "early", "middle", "late"]

# How strongly a circuit's own safety car record is pulled towards the
# average circuit. With only two or three races per circuit, a single crash
# would otherwise swing its rate wildly. DECISIONS.md, number 21.
SHRINK_EVENTS = 2.0


def usable_laps(splits: list[str]) -> pl.DataFrame:
    """All laps (not just clean ones) of dry races without a red flag."""
    laps = pl.read_parquet(LAPS_FILE).filter(pl.col("Split").is_in(splits))
    red_flagged = pl.col("IsRedFlag").any().over(RACE)
    return laps.filter(~pl.col("IsWetRace") & ~red_flagged)


# --------------------------------------------------------------------------
# Pit lane time loss
# --------------------------------------------------------------------------

def pit_stops(laps: pl.DataFrame) -> pl.DataFrame:
    """One row per pit stop, with the time it cost.

    A stop spreads its cost over two laps: the lap into the pits and the lap
    out. So the cost is measured as: the time the stopping car took for those
    two laps, minus the time cars that stayed out took for the same two laps
    of the same race. Comparing with cars on the same laps is what makes
    safety car stops come out cheaper, with no special handling: under a
    safety car the cars that stay out are slow too.
    """
    driver = RACE + ["Driver"]
    laps = laps.sort(driver + ["LapNumber"]).with_columns(
        # Time is the clock when a lap ended, so this is the duration of
        # this lap plus the next one. It works even when a pit lap has no
        # recorded lap time.
        (pl.col("Time").shift(-1).over(driver) - pl.col("Time").shift(1).over(driver))
        .alias("TwoLapTime"),
        pl.col("IsPitOutLap").shift(-1).over(driver).alias("NextIsPitOut"),
        pl.col("IsPitInLap").shift(-1).over(driver).alias("NextIsPitIn"),
        pl.col("IsSafetyCar").shift(-1).over(driver).alias("NextIsSafetyCar"),
        pl.col("IsGreen").shift(-1).over(driver).alias("NextIsGreen"),
    )
    stayed_out = ~(pl.col("IsPitInLap") | pl.col("IsPitOutLap")
                   | pl.col("NextIsPitOut") | pl.col("NextIsPitIn"))
    reference = laps.filter(stayed_out).group_by(RACE + ["LapNumber"]).agg(
        pl.col("TwoLapTime").median().alias("StayedOutTime")
    )
    condition = (
        pl.when(pl.col("IsGreen") & pl.col("NextIsGreen")).then(pl.lit("green"))
        .when(pl.col("IsSafetyCar") & pl.col("NextIsSafetyCar")).then(pl.lit("safety car"))
        .when(pl.col("IsVSC") & ~pl.col("IsSafetyCar")).then(pl.lit("vsc"))
        .otherwise(pl.lit("mixed"))
    )
    return (
        laps.filter(pl.col("IsPitInLap") & pl.col("NextIsPitOut"))
        .join(reference, on=RACE + ["LapNumber"])
        .with_columns(
            (pl.col("TwoLapTime") - pl.col("StayedOutTime")).alias("PitLoss"),
            condition.alias("Condition"),
        )
        .drop_nulls("PitLoss")
        .select(RACE + ["Circuit", "Driver", "LapNumber", "Condition", "PitLoss"])
    )


def pit_loss_stats(stops: pl.DataFrame) -> dict:
    """Pit loss per circuit under green, and the safety car / VSC discounts."""
    low, high = NORMAL_STOP_SECONDS
    green = stops.filter(
        (pl.col("Condition") == "green") & pl.col("PitLoss").is_between(low, high)
    )
    by_circuit = green.group_by("Circuit").agg(
        pl.col("PitLoss").median().alias("loss"), pl.len().alias("stops")
    ).filter(pl.col("stops") >= MIN_STOPS_PER_CIRCUIT)
    per_circuit = dict(zip(by_circuit["Circuit"], by_circuit["loss"].round(2)))
    default = round(float(np.median(list(per_circuit.values()))), 2)

    def discount(condition: str) -> tuple[float, int]:
        """Typical cost of a stop under this condition, as a share of the
        same circuit's green flag cost.

        Single stops are very noisy here (the queue behind a safety car
        distorts the comparison with cars that stayed out), so only laps
        where several cars pitted together are used, one value per lap.
        """
        groups = (
            stops.filter(pl.col("Condition") == condition)
            .group_by(RACE + ["Circuit", "LapNumber"])
            .agg(pl.col("PitLoss").median(), pl.len().alias("cars"))
            .filter(pl.col("cars") >= MIN_CARS_PITTING_TOGETHER)
            .with_columns(
                pl.col("Circuit").replace_strict(per_circuit, default=default,
                                                 return_dtype=pl.Float64).alias("Green")
            )
        )
        return round((groups["PitLoss"] / groups["Green"]).median(), 3), groups.height

    safety_car_share, safety_car_stops = discount("safety car")
    vsc_share, vsc_stops = discount("vsc")
    return {
        "pit_loss_by_circuit": per_circuit,
        "pit_loss_default": default,
        "green_stops_counted": green.height,
        "safety_car_pit_share": safety_car_share,
        "safety_car_stops_counted": safety_car_stops,
        "vsc_pit_share": vsc_share,
        "vsc_stops_counted": vsc_stops,
    }


# --------------------------------------------------------------------------
# Safety cars and VSCs
# --------------------------------------------------------------------------

def track_status_by_lap(laps: pl.DataFrame) -> pl.DataFrame:
    """One row per race lap: 0 green, 1 VSC, 2 safety car.

    Drivers are at different places when a safety car is called, so their
    laps disagree slightly. A lap counts as a safety car lap if more than
    half the drivers' laps were under it.
    """
    per_lap = laps.group_by(RACE + ["Circuit", "TotalLaps", "LapNumber"]).agg(
        (pl.col("IsSafetyCar").mean() > 0.5).alias("SafetyCar"),
        (pl.col("IsVSC").mean() > 0.5).alias("VSC"),
    ).sort(RACE + ["LapNumber"])
    status = (
        pl.when(pl.col("SafetyCar")).then(2).when(pl.col("VSC")).then(1).otherwise(0)
    )
    fraction = pl.col("LapNumber") / pl.col("TotalLaps")
    phase = (
        pl.when(pl.col("LapNumber") <= START_LAPS).then(pl.lit("start"))
        .when(fraction <= 1 / 3).then(pl.lit("early"))
        .when(fraction <= 2 / 3).then(pl.lit("middle"))
        .otherwise(pl.lit("late"))
    )
    return per_lap.with_columns(status.alias("Status"), phase.alias("Phase")).with_columns(
        pl.col("Status").shift(1).over(RACE).fill_null(0).alias("PreviousStatus")
    )


def period_lengths(per_lap: pl.DataFrame, status: int) -> list[int]:
    """How many laps each safety car (or VSC) period lasted."""
    lengths = []
    for (_, _), race in per_lap.group_by(RACE, maintain_order=True):
        run = 0
        for value in race["Status"].to_list() + [0]:
            if value == status:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
    return lengths


def start_chances(per_lap: pl.DataFrame, status: int) -> dict:
    """The chance per lap that a period of this kind starts.

    Counted per race phase over all circuits, then scaled per circuit by how
    many periods that circuit has had compared with what an average circuit
    would have had in the same number of laps.
    """
    started = (pl.col("Status") == status) & (pl.col("PreviousStatus") != status)
    # A period can only start on a lap that began green.
    could_start = per_lap.filter(pl.col("PreviousStatus") == 0).with_columns(
        started.alias("Started")
    )
    by_phase = could_start.group_by("Phase").agg(pl.col("Started").mean().alias("chance"))
    per_phase = dict(zip(by_phase["Phase"], by_phase["chance"]))
    per_phase = {phase: round(per_phase.get(phase, 0.0), 5) for phase in PHASES}

    expected = pl.col("Phase").replace_strict(per_phase, return_dtype=pl.Float64)
    by_circuit = could_start.group_by("Circuit").agg(
        pl.col("Started").sum().alias("seen"), expected.sum().alias("expected")
    )
    # Pulled towards 1 (an average circuit): see SHRINK_EVENTS above.
    multiplier = (by_circuit["seen"] + SHRINK_EVENTS) / (by_circuit["expected"] + SHRINK_EVENTS)
    return {
        "chance_per_lap_by_phase": per_phase,
        "circuit_multiplier": dict(zip(by_circuit["Circuit"], multiplier.round(3))),
        "periods_counted": int(could_start["Started"].sum()),
        "races_counted": could_start.select(pl.struct(RACE).n_unique()).item(),
    }


def slow_lap_pct(laps: pl.DataFrame, status_code: str) -> float:
    """How much slower than the race's typical lap a full lap under this
    track status is, in percent of pole."""
    full = laps.filter(
        (pl.col("TrackStatus") == status_code)
        & ~pl.col("IsPitInLap") & ~pl.col("IsPitOutLap")
    )
    return round((full["LapTimePct"] - full["RacePacePct"]).median(), 2)


def safety_car_stats(laps: pl.DataFrame) -> dict:
    per_lap = track_status_by_lap(laps)
    return {
        "safety_car": {
            **start_chances(per_lap, 2),
            "lengths_in_laps": period_lengths(per_lap, 2),
            "slow_lap_pct": slow_lap_pct(laps, "4"),
        },
        "vsc": {
            **start_chances(per_lap, 1),
            "lengths_in_laps": period_lengths(per_lap, 1),
            "slow_lap_pct": slow_lap_pct(laps, "6"),
        },
    }


# --------------------------------------------------------------------------
# The first lap
# --------------------------------------------------------------------------

def first_lap_stats(laps: pl.DataFrame) -> dict:
    """How far behind the leader each grid slot is after lap 1.

    The lap time model does not cover the standing start, so lap 1 is
    modelled simply: each grid slot further back costs a fixed amount of
    time, plus some luck.
    """
    lap_one = laps.filter(
        (pl.col("LapNumber") == 1) & (pl.col("GridPosition") >= 1) & ~pl.col("IsPitInLap")
    ).with_columns((pl.col("Time") - pl.col("Time").min().over(RACE)).alias("Gap"))
    lap_one = lap_one.filter(pl.col("Gap") < 30)  # ignore cars that crashed or spun
    slots = (lap_one["GridPosition"] - 1).to_numpy().astype(float)
    gaps = lap_one["Gap"].to_numpy()
    seconds_per_slot = float(slots @ gaps / (slots @ slots))  # best line through zero
    luck = float(np.std(gaps - seconds_per_slot * slots))
    return {
        "first_lap_seconds_per_grid_slot": round(seconds_per_slot, 3),
        "first_lap_luck_seconds": round(luck, 3),
    }


# --------------------------------------------------------------------------

def add_pit_shares_for_simulator(stats: dict, typical_race_pace_pct: float):
    """Convert the measured safety car / VSC stop costs into the form the
    simulator needs.

    The measured cost is in seconds on a slow clock: while the field crawls
    behind a safety car, a gap of 15 seconds is a much shorter distance than
    15 seconds at racing speed, and it shrinks back when racing resumes. The
    simulator keeps gaps in racing-speed seconds, so the measured cost is
    divided by how much slower the laps are. DECISIONS.md, number 20.
    """
    normal_lap = 100 + typical_race_pace_pct  # a normal lap, in percent of pole
    for kind, key in [("safety_car", "safety_car_pit_share"), ("vsc", "vsc_pit_share")]:
        slow_lap = normal_lap + stats[kind]["slow_lap_pct"]
        stats[key + "_measured"] = stats[key]
        stats[key] = round(stats[key] * normal_lap / slow_lap, 3)


def tyre_life_limits(laps: pl.DataFrame) -> dict:
    """The oldest tyre age the lap time model has real evidence for.

    99% of clean laps on each compound were driven on tyres younger than
    this. Beyond it the lap time model is guessing, so the simulator adds a
    penalty there instead of trusting it (DECISIONS.md, number 23).
    """
    clean = laps.filter(pl.col("IsCleanLap"))
    limits = clean.group_by("Compound").agg(pl.col("TyreLife").quantile(0.99).alias("limit"))
    return {"tyre_life_limit": dict(zip(limits["Compound"], limits["limit"].cast(pl.Int32)))}


def build_stats(splits: list[str]) -> dict:
    """Everything above, measured on the given seasons."""
    laps = usable_laps(splits)
    stats = {
        "measured_on": splits,
        **tyre_life_limits(laps),
        **pit_loss_stats(pit_stops(laps)),
        **safety_car_stats(laps),
        **first_lap_stats(laps),
    }
    add_pit_shares_for_simulator(stats, laps["RacePacePct"].median())
    return stats


def save_stats(stats: dict, file: Path = STATS_FILE):
    file.parent.mkdir(exist_ok=True)
    file.write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {file}")


def print_summary(stats: dict):
    print(f"Measured on: {', '.join(stats['measured_on'])} (dry races, no red flags)\n")
    print(f"Pit loss under green, from {stats['green_stops_counted']} stops. "
          f"Default for other circuits: {stats['pit_loss_default']}s")
    table = pl.DataFrame({
        "circuit": list(stats["pit_loss_by_circuit"]),
        "pit_loss_s": list(stats["pit_loss_by_circuit"].values()),
    }).sort("pit_loss_s")
    with pl.Config(tbl_rows=30):
        print(table)
    print("\nCost of a stop as a share of a green flag stop"
          " (measured on the slow clock -> in racing-speed seconds):")
    print(f"  safety car {stats['safety_car_pit_share_measured']:.0%} ->"
          f" {stats['safety_car_pit_share']:.0%}"
          f"   ({stats['safety_car_stops_counted']} laps with group stops)")
    print(f"  VSC        {stats['vsc_pit_share_measured']:.0%} ->"
          f" {stats['vsc_pit_share']:.0%}   ({stats['vsc_stops_counted']} laps)")
    for name, key in [("Safety car", "safety_car"), ("VSC", "vsc")]:
        part = stats[key]
        lengths = part["lengths_in_laps"]
        print(f"\n{name}: {part['periods_counted']} periods in {part['races_counted']} races,"
              f" typical length {np.median(lengths):.0f} laps,"
              f" laps {part['slow_lap_pct']}% of pole slower than normal.")
        print(f"  chance of one starting, per lap: {part['chance_per_lap_by_phase']}")
        ranked = sorted(part["circuit_multiplier"].items(), key=lambda item: item[1])
        print(f"  least likely circuits: {ranked[:3]}")
        print(f"  most likely circuits:  {ranked[-3:]}")
    print(f"\nFirst lap: each grid slot costs {stats['first_lap_seconds_per_grid_slot']}s,"
          f" luck of about {stats['first_lap_luck_seconds']}s.")


def main():
    stats = build_stats(["train"])
    print_summary(stats)
    save_stats(stats)


if __name__ == "__main__":
    main()
