# test_simulate.py
#
# What it does: checks the race simulation in simulate.py and the summary
#   maths in export_web.py on small cases where the right answer is known.
# What it reads: nothing (no downloaded data needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import torch

import export_web
import simulate

DEVICE = simulate.DEVICE


def field(races: int, skills: list[float]) -> torch.Tensor:
    """The same grid of drivers repeated for every simulated race."""
    return torch.tensor(skills, device=DEVICE).repeat(races, 1)


def run(skills, races=4000, margin=0.5, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    errors = torch.ones(len(skills), device=DEVICE)
    return simulate.simulate_races(field(races, skills), errors, 90.0, 50, margin, generator)


def test_every_race_gives_each_place_exactly_once():
    place, _ = run([0.0] * 10, races=200)
    assert (place.sort(dim=1).values == torch.arange(1, 11, device=DEVICE)).all()


def test_equal_drivers_win_equally_often():
    place, _ = run([0.0, 0.0])
    share = (place[:, 0] == 1).float().mean().item()
    assert 0.46 < share < 0.54


def test_the_more_skilled_driver_wins_more_often_but_not_always():
    place, _ = run([0.5, 0.0])
    share = (place[:, 0] == 1).float().mean().item()
    assert 0.7 < share < 0.95


def test_the_same_seed_gives_the_same_races():
    first, _ = run([0.3, 0.1, 0.0, -0.2], races=100, seed=7)
    second, _ = run([0.3, 0.1, 0.0, -0.2], races=100, seed=7)
    assert (first == second).all()


def test_a_slower_car_in_front_holds_up_a_slightly_quicker_one():
    # Car 0 leads by 1s. Car 1 is 0.5s a lap quicker: enough to catch it, but
    # less than the 1.0s margin needed to pass, so it ends 0.4s behind.
    race_time = torch.tensor([[0.0, 1.0]], device=DEVICE)
    lap_time = torch.tensor([[90.0, 89.5]], device=DEVICE)
    for _ in range(5):
        race_time = simulate.hold_up(race_time, lap_time, margin=1.0)
    assert race_time[0, 0] < race_time[0, 1]
    assert abs((race_time[0, 1] - race_time[0, 0]).item() - 0.4) < 1e-3


def test_a_much_quicker_car_gets_past():
    race_time = torch.tensor([[0.0, 1.0]], device=DEVICE)
    lap_time = torch.tensor([[90.0, 88.0]], device=DEVICE)  # 2s a lap quicker
    race_time = simulate.hold_up(race_time, lap_time, margin=1.0)
    assert race_time[0, 1] < race_time[0, 0]


def test_hard_circuits_keep_the_grid_order_more_than_easy_ones():
    # One quicker driver among nine equal ones. When he qualifies badly he has
    # to overtake to win, so he wins less often where passing is hard.
    skills = [0.3] + [0.0] * 9
    easy, _ = run(skills, races=8000, margin=simulate.pass_margin(0.0))
    hard, _ = run(skills, races=8000, margin=simulate.pass_margin(1.0))
    assert (hard[:, 0] == 1).float().mean() < (easy[:, 0] == 1).float().mean() - 0.02


def test_crashed_cars_finish_behind_everyone_who_did_not():
    place, crashed = run([0.0] * 8, races=3000)
    assert crashed.any()
    finishers = (~crashed).sum(dim=1, keepdim=True)
    assert (place[crashed] > finishers.expand_as(place)[crashed]).all()


def test_consistency_score_becomes_an_error_multiplier():
    scores = torch.tensor([50.0, 75.0, 25.0])
    assert simulate.error_factor(scores).tolist() == [1.0, 0.5, 2.0]


def test_monaco_needs_a_bigger_advantage_than_the_easiest_circuit():
    assert simulate.pass_margin(1.0) == simulate.SETTINGS["passMarginHard"]
    assert simulate.pass_margin(0.0) == simulate.SETTINGS["passMarginEasy"]


def test_rank_ranges_are_narrow_when_ratings_are_certain():
    # Three drivers, clearly separated in every plausible version.
    versions = torch.tensor([[1.0, 0.5, 0.0], [1.1, 0.4, 0.1], [0.9, 0.6, -0.1]])
    low, high = export_web.rank_ranges(versions)
    assert (low, high) == ([1, 2, 3], [1, 2, 3])


def test_head_to_head_is_fifty_fifty_for_equal_drivers_and_mirrors():
    versions = torch.tensor([[0.5, 0.5, 0.0]] * 100)
    best, low, high = export_web.head_to_heads(versions, race_slope=3.0)
    assert abs(best[0, 1].item() - 0.5) < 1e-6
    assert abs(best[0, 2].item() + best[2, 0].item() - 1) < 1e-6
    assert best[0, 2] > 0.8


def test_upper_triangle_lists_each_pair_once():
    table = torch.tensor([[0.5, 0.7, 0.9], [0.3, 0.5, 0.6], [0.1, 0.4, 0.5]])
    assert export_web.upper_triangle(table) == [70, 90, 60]
