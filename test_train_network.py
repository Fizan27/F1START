# test_train_network.py
#
# What it does: checks the key pieces of train_network.py on tiny examples
#   where the right answer is obvious: the loss, the range check, the shape
#   of the network's output, and that one training step really learns.
# What it reads: nothing (no real data and no GPU needed).
# What it produces: pass/fail results when run with pytest.
# Which files use it: none.  Run:  .venv\Scripts\python.exe -m pytest

import math

import numpy as np
import torch

import train_network as network


def loss_for_spread(spread: float) -> float:
    """The loss when every prediction misses by exactly 1, at a given spread."""
    centre, actual = torch.zeros(4), torch.ones(4)
    log_spread = torch.full((4,), math.log(spread))
    return network.range_loss(centre, log_spread, actual).item()


def test_range_loss_rewards_an_honest_spread():
    # Misses are all 1, so a spread of 1 is honest. Claiming 0.1 is
    # overconfident and claiming 10 is uselessly vague: both must cost more.
    assert loss_for_spread(1.0) < loss_for_spread(0.1)
    assert loss_for_spread(1.0) < loss_for_spread(10.0)


def test_share_inside_range():
    centre = np.zeros(4)
    spread = np.ones(4)
    # The range is -1.645 to +1.645, so two of these four laps are inside.
    actual = np.array([0.0, 1.0, 2.0, -3.0])
    assert network.share_inside_range(centre, spread, actual) == 0.5


def small_batch() -> dict:
    torch.manual_seed(0)
    return {
        "numbers": torch.randn(8, len(network.NUMBER_INPUTS)),
        "compound": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        "circuit": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        "actual": torch.randn(8),
    }


def test_network_gives_a_centre_and_spread_for_every_lap():
    batch = small_batch()
    model = network.LapTimeNetwork(circuit_count=3)
    centre, log_spread = model(batch["numbers"], batch["compound"], batch["circuit"])
    assert centre.shape == (8,)
    assert log_spread.shape == (8,)


def test_training_steps_reduce_the_loss():
    batch = small_batch()
    model = network.LapTimeNetwork(circuit_count=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first = network.train_step(model, optimizer, batch)
    for _ in range(50):
        last = network.train_step(model, optimizer, batch)
    assert last < first


def test_hidden_circuits_become_unknown():
    network_share = network.HIDE_CIRCUIT_SHARE
    circuit = torch.full((10_000,), 5)
    hidden = network.hide_some_circuits(circuit)
    # Every value is either untouched (5) or "unknown" (0), in about the
    # intended share.
    assert set(hidden.tolist()) == {0, 5}
    assert abs((hidden == 0).float().mean().item() - network_share) < 0.02
