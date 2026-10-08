# test_simulator.py
#
# What it does: checks the race simulator on a tiny made-up race where the
#   right behaviour is obvious: an extra pit stop costs time, a stop under a
#   safety car is cheaper, the field closes up behind a safety car, and the
#   same seed gives the same race.
# What it reads: models/lap_time_network.pt and models/race_stats.json (both
#   are in the repository). No race data is needed.
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import numpy as np
import torch

import simulator
from simulator import GREEN, NEVER, SAFETY_CAR, LapModel, Race, RaceSet, RaceSim

LAPS = 30
CARS = 4
DEVICE = "cpu"


def made_up_race(status=None) -> Race:
    """Four identical cars on hard tyres, nobody stopping, at a known circuit."""
    return Race(
        year=2024, round=1, name="Test Grand Prix", circuit="Sakhir", split="train",
        total_laps=LAPS, pole=90.0, race_pace=8.0, wet_qualifying=False,
        drivers=["AAA", "BBB", "CCC", "DDD"], teams=["T"] * CARS,
        grid=np.arange(1, CARS + 1), quali_gap=np.full(CARS, 1.0, dtype=np.float32),
        start_compound=np.full(CARS, 2), start_age=np.zeros(CARS, dtype=np.float32),
        plan=np.zeros((CARS, LAPS), dtype=np.int64),
        plan_age=np.zeros((CARS, LAPS), dtype=np.float32),
        status=np.zeros(LAPS, dtype=np.int64) if status is None else status,
        retire_lap=np.full(CARS, NEVER), finish_position=np.arange(1, CARS + 1),
        gap_to_winner=np.zeros(CARS, dtype=np.float32),
    )


def run(race: Race, luck: bool = False, seed: int = 0) -> RaceSim:
    lap_model = LapModel(device=DEVICE)
    stats = simulator.load_stats()
    race_set = RaceSet([race], lap_model, stats, device=DEVICE)
    return RaceSim(race_set, lap_model, stats, torch.tensor([0]), seed=seed,
                   luck=luck).run_to_end()


def test_without_luck_identical_cars_finish_in_grid_order():
    sim = run(made_up_race())
    assert sim.final_positions()[0, :CARS].tolist() == [1, 2, 3, 4]


def cost_of_stopping(status_code: int, lap: int = 12) -> float:
    """Extra time the last car takes on `lap` if it pits at the end of it."""
    def lap_time(pit: bool) -> float:
        status = np.zeros(LAPS, dtype=np.int64)
        # The safety car appears on the very lap of the planned stop, so the
        # "bring the stop forward" rule has nothing to move.
        status[lap - 1:lap + 2] = status_code
        race = made_up_race(status)
        if pit:
            race.plan[3, lap - 1] = 3
        lap_model = LapModel(device=DEVICE)
        stats = simulator.load_stats()
        race_set = RaceSet([race], lap_model, stats, device=DEVICE)
        sim = RaceSim(race_set, lap_model, stats, torch.tensor([0]), luck=False)
        for _ in range(lap):
            sim.step()
        return sim.last_lap_time[0, 3].item()

    return lap_time(pit=True) - lap_time(pit=False)


def test_a_green_flag_stop_costs_the_pit_loss():
    loss = simulator.load_stats()["pit_loss_by_circuit"]["Sakhir"]
    assert abs(cost_of_stopping(GREEN) - loss) < 0.01


def test_a_stop_under_the_safety_car_or_vsc_is_cheaper():
    stats = simulator.load_stats()
    loss = stats["pit_loss_by_circuit"]["Sakhir"]
    assert abs(cost_of_stopping(SAFETY_CAR) - loss * stats["safety_car_pit_share"]) < 0.01
    assert abs(cost_of_stopping(simulator.VSC) - loss * stats["vsc_pit_share"]) < 0.01


def test_the_field_closes_up_behind_the_safety_car():
    status = np.zeros(LAPS, dtype=np.int64)
    status[LAPS - 5:] = SAFETY_CAR  # safety car to the end
    spread_out = run(made_up_race())
    bunched = run(made_up_race(status))
    gap = lambda sim: (sim.time[0, 3] - sim.time[0, 0]).item()
    assert gap(bunched) < gap(spread_out)
    assert gap(bunched) < 3 * simulator.SAFETY_CAR_GAP_SECONDS + 0.1


def test_same_seed_same_race_and_luck_changes_the_result():
    first = run(made_up_race(), luck=True, seed=1).time
    again = run(made_up_race(), luck=True, seed=1).time
    other = run(made_up_race(), luck=True, seed=2).time
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_stops_change_the_tyres_and_are_counted():
    race = made_up_race()
    race.plan[0, 9] = 1  # car 0 pits for softs at the end of lap 10
    sim = run(race)
    assert sim.compound[0, 0].item() == 0
    assert sim.stops[0, :CARS].tolist() == [1, 0, 0, 0]
    assert sim.tyre_age[0, 0].item() == LAPS - 10
    assert sim.used[0, 0].tolist() == [True, False, True]


def test_rivals_bring_a_planned_stop_forward_under_a_safety_car():
    status = np.zeros(LAPS, dtype=np.int64)
    status[9:13] = SAFETY_CAR  # safety car on laps 10 to 13
    race = made_up_race(status)
    race.plan[2, 14] = 1  # planned stop on lap 15, two laps after it ends
    lap_model = LapModel(device=DEVICE)
    stats = simulator.load_stats()
    race_set = RaceSet([race], lap_model, stats, device=DEVICE)
    sim = RaceSim(race_set, lap_model, stats, torch.tensor([0]), luck=False)
    for _ in range(10):
        sim.step()
    # It pitted on lap 10, the first safety car lap, and only once in total.
    assert sim.stops[0, 2].item() == 1
    assert sim.run_to_end().stops[0, 2].item() == 1


def test_sampled_track_status_is_valid():
    lap_model = LapModel(device=DEVICE)
    stats = simulator.load_stats()
    race_set = RaceSet([made_up_race()], lap_model, stats, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    index = torch.zeros(500, dtype=torch.long)
    status = simulator.sample_status(race_set, index, stats, generator)
    assert status.shape == (500, LAPS)
    assert set(status.unique().tolist()) <= {0, 1, 2}
    # Across 500 random races there are some safety cars, but most laps are green.
    assert 0.02 < (status == SAFETY_CAR).float().mean().item() < 0.3
