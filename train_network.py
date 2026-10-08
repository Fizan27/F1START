# train_network.py
#
# What it does: trains the MAIN lap time model, a PyTorch neural network, on
#   the GPU. Unlike the baseline it predicts a RANGE of likely lap times: a
#   centre (its best guess) and a spread (how unsure it is). The simulator
#   needs the spread to make races play out differently each time, the way
#   real ones do.
# What it reads: data/laps.parquet (made by clean_data.py).
# What it produces: models/lap_time_network.pt (the trained network plus the
#   numbers needed to prepare its inputs), and a comparison with the LightGBM
#   baseline on 2024 printed to the screen.
# Which files use it: the simulator (Phase 2) will load the saved model.
#   This file uses train_baseline.py for loading laps and for scoring, so
#   both models are judged in exactly the same way.
#
# Run:  .venv\Scripts\python.exe train_network.py
#
# The four parts worth understanding well, each explained where it appears:
#   1. the layers (LapTimeNetwork)      3. one training step (train_step)
#   2. the loss for a range (range_loss) 4. the check against the baseline
#                                           (score_network, share_inside_range)

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

import train_baseline as baseline

MODEL_FILE = Path("models/lap_time_network.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = baseline.TARGET

# The same inputs as the chosen baseline: four numbers, the tyre compound and
# the circuit.
NUMBER_INPUTS = ["TyreLife", "LapsRemaining", "GapAhead", "QualiGapPct"]
COMPOUND_NAMES = ["SOFT", "MEDIUM", "HARD"]

# Each circuit is described to the network by this many numbers, which it
# learns by itself (for example "hard on tyres", "easy to follow"). Small on
# purpose: there are only about 20 circuits to learn from.
CIRCUIT_NUMBERS = 4

# During training the circuit is hidden on this share of laps and replaced
# by "unknown circuit". That teaches the network a sensible general answer
# for a circuit it has never seen, such as Shanghai in 2024.
HIDE_CIRCUIT_SHARE = 0.1

# Training settings, chosen by trying values and scoring on 2024.
# 20 epochs: by 40 the network starts memorising training races and its
# ranges become overconfident (docs/DECISIONS.md, number 17).
EPOCHS = 20  # how many times the network sees every training lap
BATCH_SIZE = 1024  # how many laps it looks at before each update
LEARNING_RATE = 0.001  # how big each update is
SEED = 0  # makes the run repeatable

# A range of "centre plus or minus 1.645 spreads" should contain 90% of real
# laps, IF the network's spreads are honest. That is checked in TODO 4.
RANGE_SPREADS = 1.645
RANGE_TARGET = 0.90


# --------------------------------------------------------------------------
# Preparing the inputs (written for you)
# --------------------------------------------------------------------------

def learn_scaling(train: pl.DataFrame) -> dict:
    """Measure, on training laps only, what is needed to prepare any lap.

    Networks learn best when every input is a small number around zero, so
    each number input is shifted by its average and divided by its spread.
    The averages MUST come from training laps only: using 2024 laps here
    would let information from the validation season leak into the model.
    """
    return {
        "fill": {name: train[name].median() for name in NUMBER_INPUTS},
        "mean": {name: train[name].mean() for name in NUMBER_INPUTS},
        "std": {name: train[name].std() for name in NUMBER_INPUTS},
        # Circuit number 0 is reserved for "unknown circuit".
        "circuits": sorted(train["Circuit"].cast(pl.String).unique().to_list()),
    }


def to_tensors(laps: pl.DataFrame, scaling: dict) -> dict:
    """Turn a table of laps into tensors (arrays PyTorch can use) on the GPU."""
    numbers = laps.select(
        # A missing value becomes the typical training value, so it says
        # "nothing unusual here" instead of crashing the network.
        (pl.col(name).fill_null(scaling["fill"][name]) - scaling["mean"][name])
        / scaling["std"][name]
        for name in NUMBER_INPUTS
    ).to_numpy()
    circuit_number = {name: i + 1 for i, name in enumerate(scaling["circuits"])}
    circuits = [circuit_number.get(name, 0) for name in laps["Circuit"].cast(pl.String)]
    compounds = [COMPOUND_NAMES.index(name) for name in laps["Compound"].cast(pl.String)]
    return {
        "numbers": torch.tensor(numbers, dtype=torch.float32, device=DEVICE),
        "compound": torch.tensor(compounds, device=DEVICE),
        "circuit": torch.tensor(circuits, device=DEVICE),
        "actual": torch.tensor(laps[TARGET].to_numpy(), dtype=torch.float32, device=DEVICE),
    }


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------

class LapTimeNetwork(nn.Module):
    """Predicts a centre and a spread for one lap's pace."""

    def __init__(self, circuit_count: int):
        super().__init__()
        # One learned list of CIRCUIT_NUMBERS numbers per circuit (+1 for
        # "unknown circuit"). PyTorch calls this an embedding.
        self.circuit_numbers = nn.Embedding(circuit_count + 1, CIRCUIT_NUMBERS)
        input_size = len(NUMBER_INPUTS) + len(COMPOUND_NAMES) + CIRCUIT_NUMBERS

        # A layer, nn.Linear(inputs, outputs), multiplies its inputs by
        # weights and adds them up: alone it can only draw straight lines.
        # nn.ReLU() after it turns negative values into zero. That small
        # kink is what lets a stack of layers learn curves, like tyres
        # wearing slowly at first and then falling off a cliff.
        # nn.Sequential runs the layers one after another.
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            # Two outputs: the centre and the (log) spread. No ReLU here,
            # because the centre must be free to go negative (a lap faster
            # than the race's typical lap).
            nn.Linear(64, 2),
        )

    def forward(self, numbers, compound, circuit):
        """Returns (centre, log_spread) for every lap given."""
        # Join the three kinds of input into one row of numbers per lap. The
        # compound becomes three 0/1 switches (for example MEDIUM = 0, 1, 0).
        switches = nn.functional.one_hot(compound, len(COMPOUND_NAMES)).float()
        joined = torch.cat([numbers, switches, self.circuit_numbers(circuit)], dim=1)
        output = self.layers(joined)
        centre = output[:, 0]
        # The network outputs the LOG of the spread. A spread must be positive
        # and a raw output can be anything; taking exp() of it later always
        # gives a positive number. The clamp stops absurd values early on.
        log_spread = output[:, 1].clamp(-5, 3)
        return centre, log_spread


# --------------------------------------------------------------------------
# The loss: how wrong was a prediction of a range?
# --------------------------------------------------------------------------

def range_loss(centre, log_spread, actual):
    """Average loss over a batch of laps. Lower is better."""
    # With one number, the loss is simple: how far off was it. With a range,
    # the network could cheat in two ways, and the loss has two parts that
    # block one cheat each:
    #   wide_penalty: punishes wide ranges. Without it the network would say
    #     "the lap is somewhere between 0 and 1000 seconds" and never be wrong.
    #   miss_penalty: the miss, measured in spreads, squared. Missing by 1
    #     second is a disaster if you claimed a spread of 0.1 (10 spreads
    #     away) but fine if you claimed 2 (half a spread). It punishes
    #     overconfidence.
    # The only way to make both small is an honest spread: narrow where laps
    # are predictable, wide where they are not.
    # (Its proper name is the "Gaussian negative log likelihood".)
    spread = torch.exp(log_spread)
    wide_penalty = log_spread
    miss_penalty = 0.5 * ((actual - centre) / spread) ** 2
    return (wide_penalty + miss_penalty).mean()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_step(model, optimizer, batch) -> float:
    """Learn from one batch of laps. Returns the loss, to watch it fall."""
    # The four steps that every neural network training uses.
    # 1. PREDICT.
    centre, log_spread = model(batch["numbers"], batch["compound"], batch["circuit"])
    # 2. MEASURE the error.
    loss = range_loss(centre, log_spread, batch["actual"])
    # 3. FIND each weight's share of the blame. First clear the blame left
    #    over from the previous batch, then work it out for this one.
    #    backward() is "backpropagation": it works backwards through the
    #    layers to find how much each weight contributed to the error.
    optimizer.zero_grad()
    loss.backward()
    # 4. UPDATE every weight a small step in the direction that reduces it.
    optimizer.step()
    return loss.item()


def hide_some_circuits(circuit):
    """Replace a random share of circuits with 0, the "unknown circuit"."""
    hidden = torch.rand(circuit.shape, device=circuit.device) < HIDE_CIRCUIT_SHARE
    return torch.where(hidden, torch.zeros_like(circuit), circuit)


def average_loss(model, data) -> float:
    """The loss on a whole table of laps, without learning from it."""
    model.eval()
    with torch.no_grad():  # no blame is worked out: this is only a check
        centre, log_spread = model(data["numbers"], data["compound"], data["circuit"])
        return range_loss(centre, log_spread, data["actual"]).item()


def train_network(train_data: dict, validation_data: dict, circuit_count: int):
    """Train for EPOCHS passes over the training laps. Returns the model."""
    torch.manual_seed(SEED)
    model = LapTimeNetwork(circuit_count).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lap_count = len(train_data["actual"])

    print(f"\nTraining on {DEVICE} for {EPOCHS} epochs. Both losses should fall:")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # A fresh random order each epoch, so batches are never the same.
        order = torch.randperm(lap_count, device=DEVICE)
        for start in range(0, lap_count, BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            batch = {name: values[rows] for name, values in train_data.items()}
            batch["circuit"] = hide_some_circuits(batch["circuit"])
            train_step(model, optimizer, batch)
        if epoch == 1 or epoch % 5 == 0:
            # If the training loss keeps falling while the validation loss
            # stops falling or rises, the network has started memorising
            # training races (overfitting) and more epochs will not help.
            print(f"  epoch {epoch:>3}   training loss {average_loss(model, train_data):.3f}"
                  f"   validation loss {average_loss(model, validation_data):.3f}")
    return model


def predict(model, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centre, spread) for every lap, as ordinary arrays."""
    model.eval()
    with torch.no_grad():
        centre, log_spread = model(data["numbers"], data["compound"], data["circuit"])
    return centre.cpu().numpy(), torch.exp(log_spread).cpu().numpy()


# --------------------------------------------------------------------------
# Checking against the baseline
# --------------------------------------------------------------------------

def score_network(validation: pl.DataFrame, centre: np.ndarray) -> dict:
    """Score the network's centre exactly as the baseline was scored."""
    # Both steps reuse functions from train_baseline.py, so the two models
    # are judged by the same code and the comparison cannot be skewed.
    scored = baseline.add_predictions(validation, centre)
    return baseline.score("neural network", scored)


def share_inside_range(centre, spread, actual) -> float:
    """What share of real laps fell inside the predicted range?"""
    # This checks whether the spreads are honest. If they are, the answer is
    # close to 0.90. Well below means the network is overconfident; well
    # above means it is too cautious.
    low = centre - RANGE_SPREADS * spread
    high = centre + RANGE_SPREADS * spread
    # The mean of True/False values is the share that are True.
    return np.mean((actual >= low) & (actual <= high))


def baseline_row(train: pl.DataFrame, validation: pl.DataFrame) -> dict:
    """Train the chosen LightGBM baseline and score it, for comparison."""
    inputs = baseline.VARIANTS[baseline.CHOSEN]
    model = baseline.train_model(train, inputs)
    predicted = model.predict(baseline.to_inputs(validation, inputs))
    return baseline.score("LightGBM baseline", baseline.add_predictions(validation, predicted))


def range_table(validation: pl.DataFrame, centre, spread) -> pl.DataFrame:
    """How honest the ranges are, for dry races and for all races."""
    laps = validation.with_columns(
        pl.Series("Centre", centre), pl.Series("Spread", spread)
    )
    rows = []
    for name, part in [("dry races", laps.filter(~pl.col("IsWetRace"))), ("all races", laps)]:
        inside = share_inside_range(
            part["Centre"].to_numpy(), part["Spread"].to_numpy(), part[TARGET].to_numpy()
        )
        width = 2 * RANGE_SPREADS * part["Spread"] * part["PoleTime"] / 100
        rows.append({
            "laps": name,
            "target_share_inside": RANGE_TARGET,
            "actual_share_inside": round(float(inside), 3),
            "typical_range_width_s": round(width.median(), 2),
        })
    return pl.DataFrame(rows)


def save_model(model, scaling: dict):
    """Save the weights and the input preparation numbers together."""
    # The simulator must prepare its inputs in exactly the same way, so the
    # scaling numbers travel with the weights in one file.
    MODEL_FILE.parent.mkdir(exist_ok=True)
    torch.save({"weights": model.state_dict(), "scaling": scaling}, MODEL_FILE)
    print(f"\nSaved the trained network to {MODEL_FILE}")


def main():
    laps = baseline.load_clean_laps()
    train, validation = baseline.split_by_season(laps)
    learn_from = baseline.training_laps(train)  # dry races, no outlier laps

    scaling = learn_scaling(learn_from)
    train_data = to_tensors(learn_from, scaling)
    validation_data = to_tensors(validation, scaling)

    # The loss printed during training uses the same kind of laps on both
    # sides (dry, no outliers), so the two columns can be compared. The
    # scores afterwards use every 2024 lap.
    like_for_like = to_tensors(baseline.training_laps(validation), scaling)

    model = train_network(train_data, like_for_like, len(scaling["circuits"]))
    centre, spread = predict(model, validation_data)

    print("\n1. Pace within the race, on 2024 (validation). Miss in seconds per lap:")
    print(pl.DataFrame([baseline_row(train, validation), score_network(validation, centre)]))
    print("\n2. Are the predicted ranges honest?")
    print(range_table(validation, centre, spread))
    save_model(model, scaling)


if __name__ == "__main__":
    main()
