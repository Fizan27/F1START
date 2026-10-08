# test_strategy.py
#
# What it does: checks the strategy environment, the brute force search and
#   the PPO pieces on a tiny made-up race: the view has the right shape, a
#   pit stop costs reward, the two compound rule is enforced, every
#   candidate plan is legal, and decisions are scored correctly.
# What it reads: models/lap_time_network.pt and models/race_stats.json.
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import numpy as np
import torch

import simulator
import strategy_search
import train_agent
from simulator import LapModel, RaceSet
from strategy_env import OBSERVATION_NAMES, StrategyEnv
from test_simulator import CARS, DEVICE, LAPS, made_up_race


def new_env(races: int = 2) -> tuple:
    lap_model = LapModel(device=DEVICE)
    stats = simulator.load_stats()
    race_set = RaceSet([made_up_race()], lap_model, stats, device=DEVICE)
    env = StrategyEnv(race_set, lap_model, stats, real_events=True)
    race = torch.zeros(races, dtype=torch.long)
    agent = torch.full((races,), 1)  # the strategist drives the second car
    view, _ = env.reset(race, agent)
    return env, view


def test_the_view_has_one_number_per_name():
    env, view = new_env()
    assert view.shape == (2, len(OBSERVATION_NAMES))
    assert torch.isfinite(view).all()


def test_a_pit_stop_costs_reward_on_the_lap_it_happens():
    env, _ = new_env()
    for _ in range(9):
        env.step(torch.tensor([0, 0]))
    # Race 0 stays out on lap 10, race 1 pits for mediums.
    _, reward, _, _, info = env.step(torch.tensor([0, 2]))
    assert info["action"].tolist() == [0, 2]
    loss = simulator.load_stats()["pit_loss_by_circuit"]["Sakhir"]
    assert reward[1] < reward[0] - 0.9 * loss / 10


def test_the_two_compound_rule_is_enforced_at_the_last_chance():
    env, _ = new_env()
    stay = torch.tensor([0, 0])
    finished = False
    for lap in range(1, LAPS + 1):
        _, _, terminated, _, info = env.step(stay)
        if lap == LAPS - 1:
            # Never stopped, started on hard: forced onto mediums.
            assert info["action"].tolist() == [2, 2]
        finished = bool(terminated.all())
    assert finished
    assert env.forced_stops.tolist() == [1, 1]
    assert env.sim.used[0, 1].sum().item() == 2


def test_no_forced_stop_when_the_rule_is_already_met():
    env, _ = new_env()
    for lap in range(1, LAPS + 1):
        action = torch.tensor([1, 1]) if lap == 10 else torch.tensor([0, 0])
        env.step(action)
    assert env.forced_stops.tolist() == [0, 0]
    assert env.sim.stops[0, 1].item() == 1


def test_every_candidate_plan_is_legal():
    for start in range(3):
        plans = strategy_search.candidate_plans(LAPS, start, max_laps=LAPS).numpy()
        for plan in plans:
            laps = np.nonzero(plan)[0] + 1
            used = {start} | {int(plan[lap - 1]) - 1 for lap in laps}
            assert 1 <= len(laps) <= 2
            assert len(used) >= 2  # the two compound rule
            assert laps.min() >= strategy_search.MIN_STINT_LAPS
            assert laps.max() <= LAPS - strategy_search.MIN_STINT_LAPS


def test_describe_a_plan():
    plan = np.zeros(LAPS, dtype=np.int64)
    plan[17] = 3
    assert strategy_search.describe(plan, 1) == "MEDIUM, lap 18 HARD"


def test_decisions_are_scored_against_what_was_expected():
    # Two laps, one race. Nothing was expected (value 0) and the only reward
    # is -1 on the last lap, so both decisions turned out 1 worse than
    # expected, the first slightly discounted by LAMBDA.
    memory = {
        "reward": torch.tensor([[0.0], [-1.0]]),
        "value": torch.zeros(2, 1),
        "racing": torch.ones(2, 1, dtype=torch.bool),
    }
    advantage, outcome = train_agent.score_decisions(memory)
    assert torch.allclose(advantage[:, 0], torch.tensor([-train_agent.LAMBDA, -1.0]))
    assert outcome[1, 0].item() == -1.0


def test_a_new_strategist_mostly_stays_out():
    torch.manual_seed(0)
    strategist = train_agent.Strategist()
    choices, value = strategist(torch.zeros(5, len(OBSERVATION_NAMES)))
    assert (choices.probs[:, 0] > 0.95).all()
    assert value.shape == (5,)
