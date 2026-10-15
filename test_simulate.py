# test_simulate.py
#
# What it does: checks the race simulation in simulate.py and the summary
#   maths in export_web.py on small cases where the right answer is known,
#   and keeps the simulator's behaviour inside sensible bounds using the same
#   statistics as check_simulator.py (today's grid at Interlagos).
# What it reads: web/public/data/ (for the bounds tests).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import pytest
import torch

import check_simulator
import export_web
import simulate

DEVICE = simulate.DEVICE


def field(races: int, skills: list[float]) -> torch.Tensor:
    """The same grid of drivers repeated for every simulated race."""
    return torch.tensor(skills, device=DEVICE).repeat(races, 1)


def run(skills, races=4000, difficulty=0.5, seed=0, grid=None, laps=50):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    errors = torch.ones(len(skills), device=DEVICE)
    return simulate.simulate_races(field(races, skills), errors, 90.0, laps, difficulty, generator, grid=grid)


def test_every_race_gives_each_place_exactly_once():
    place = run([0.0] * 10, races=200)["place"]
    assert (place.sort(dim=1).values == torch.arange(1, 11, device=DEVICE)).all()


def test_equal_drivers_win_equally_often():
    share = (run([0.0, 0.0])["place"][:, 0] == 1).float().mean().item()
    assert 0.46 < share < 0.54


def test_the_more_skilled_driver_wins_more_often_but_not_always():
    share = (run([0.5, 0.0])["place"][:, 0] == 1).float().mean().item()
    assert 0.7 < share < 0.95


def test_the_same_seed_gives_the_same_races():
    first = run([0.3, 0.1, 0.0, -0.2], races=100, seed=7)["place"]
    second = run([0.3, 0.1, 0.0, -0.2], races=100, seed=7)["place"]
    assert (first == second).all()


def test_a_given_grid_is_used():
    result = run([0.0, 0.5, 0.2], races=10, grid=[2, 0, 1])
    assert (result["grid"] == torch.tensor([2, 0, 1], device=DEVICE)).all()


def no_luck(shape):
    """Middle-of-the-road random numbers, so the passing rule can be checked exactly."""
    return torch.full(shape, 0.5, device=DEVICE)


def test_a_slightly_quicker_car_stays_stuck_where_passing_is_hard():
    # Car 1 is 0.5s a lap quicker, but the margin is 1.0s: its chance of
    # passing is tiny, so it ends the lap one following gap behind.
    race_time = torch.tensor([[0.0, 1.0]], device=DEVICE)
    pace = torch.tensor([[90.0, 89.5]], device=DEVICE)
    for _ in range(5):
        race_time = simulate.hold_up(race_time, pace, pace, 1.0, no_luck((1, 2)), torch.zeros(1, 2, device=DEVICE))
    assert race_time[0, 0] < race_time[0, 1]
    assert abs((race_time[0, 1] - race_time[0, 0]).item() - 0.4) < 1e-3


def test_a_clearly_quicker_car_gets_past():
    race_time = torch.tensor([[0.0, 1.0]], device=DEVICE)
    pace = torch.tensor([[90.0, 88.0]], device=DEVICE)  # 2s a lap quicker
    race_time = simulate.hold_up(race_time, pace, pace, 1.0, no_luck((1, 2)), no_luck((1, 2)))
    assert race_time[0, 1] < race_time[0, 0]


def test_the_passing_chance_rises_with_the_advantage():
    chances = simulate.pass_chance(torch.tensor([0.0, 0.5, 1.0]), 0.5)
    assert chances[0] < 0.1 and abs(chances[1].item() - 0.5) < 1e-6 and chances[2] > 0.9


def test_a_fast_driver_starting_behind_wins_less_often_where_passing_is_hard():
    skills = [0.3] + [0.0] * 9
    grid = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]  # the quick driver starts 6th
    easy = run(skills, races=4000, difficulty=0.0, grid=grid)["place"]
    hard = run(skills, races=4000, difficulty=1.0, grid=grid)["place"]
    assert (hard[:, 0] == 1).float().mean() < (easy[:, 0] == 1).float().mean() - 0.05


def test_crashed_cars_finish_behind_everyone_who_did_not():
    result = run([0.0] * 8, races=3000)
    crashed, place = result["crashed"], result["place"]
    assert crashed.any()
    finishers = (~crashed).sum(dim=1, keepdim=True)
    assert (place[crashed] > finishers.expand_as(place)[crashed]).all()


def test_every_finisher_has_a_fastest_lap_near_the_lap_time():
    result = run([0.0, 0.2, 0.4], races=200)
    best = result["best_lap"][~result["crashed"]]
    assert ((best > 85) & (best < 91)).all()


def test_consistency_score_becomes_an_error_multiplier():
    assert simulate.error_factor(torch.tensor([50.0, 75.0, 25.0])).tolist() == [1.0, 0.5, 2.0]


def test_monaco_needs_a_bigger_advantage_than_the_easiest_circuit():
    assert simulate.pass_margin(1.0) == simulate.SETTINGS["passMarginHard"]
    assert simulate.pass_margin(0.0) == simulate.SETTINGS["passMarginEasy"]


# --- The owner's Interlagos test, with sensible bounds (see check_simulator.py) ---

@pytest.fixture(scope="module")
def interlagos_numbers():
    lap_seconds, laps, difficulty = check_simulator.interlagos()
    return check_simulator.measure(simulate.simulate_races, check_simulator.todays_grid(), lap_seconds, laps, difficulty)


def test_finishing_order_follows_the_grid_but_not_slavishly(interlagos_numbers):
    # Real F1 2018 to 2025 scores 0.74, with real cars that differ. The old
    # simulator scored 0.93 here: nearly everyone finished where they started.
    assert 0.5 < interlagos_numbers["rank_correlation"] < 0.88
    assert interlagos_numbers["average_places_moved"] > 2.0


def test_the_top_driver_wins_often_from_pole_and_sometimes_from_further_back(interlagos_numbers):
    from_pole, from_5th, from_15th = (interlagos_numbers[f"top_wins_from_{s}"] for s in (1, 5, 15))
    assert from_pole > 0.4
    assert from_pole > from_5th > from_15th
    assert from_5th > 0.12
    assert from_15th > 0.03


def test_the_fastest_lap_usually_goes_to_a_quick_driver(interlagos_numbers):
    # With 20 drivers, chance alone would give the top three 15%. The measured
    # value is about half (51.7% with the base model, 49.1% once the experience
    # curve reshuffled today's ratings; DECISIONS.md 33 and 40), so the bound
    # guards "far above chance", not one particular measurement.
    assert interlagos_numbers["fastest_lap_top3"] > 0.4


# --- Summary maths in export_web.py ---

def test_rank_ranges_are_narrow_when_ratings_are_certain():
    versions = torch.tensor([[1.0, 0.5, 0.0], [1.1, 0.4, 0.1], [0.9, 0.6, -0.1]])
    assert export_web.rank_ranges(versions) == ([1, 2, 3], [1, 2, 3])


def test_head_to_head_is_fifty_fifty_for_equal_drivers_and_mirrors():
    versions = torch.tensor([[0.5, 0.5, 0.0]] * 100)
    best, low, high = export_web.head_to_heads(versions, race_slope=3.0)
    assert abs(best[0, 1].item() - 0.5) < 1e-6
    assert abs(best[0, 2].item() + best[2, 0].item() - 1) < 1e-6
    assert best[0, 2] > 0.8


def test_upper_triangle_lists_each_pair_once():
    table = torch.tensor([[0.5, 0.7, 0.9], [0.3, 0.5, 0.6], [0.1, 0.4, 0.5]])
    assert export_web.upper_triangle(table) == [70, 90, 60]
